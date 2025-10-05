import streamlit as st
import re
import io
from PIL import Image
import numpy as np
import pandas as pd
from openai import OpenAI
import cv2
import keras_ocr

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- تحميل نموذج OCR (يتم تخزينه في الكاش لسرعة الأداء) ---
@st.cache_resource
def load_ocr_model():
    """
    تحميل نموذج keras-ocr مرة واحدة فقط وتخزينه في الكاش.
    """
    return keras_ocr.pipeline.Pipeline()

# --- قواعد البيانات والفحوصات (بدون تغيير) ---
NORMAL_RANGES = {
    "wbc": {"range": (4.0, 11.0), "unit": "x10^9/L", "name_ar": "كريات الدم البيضاء", "type":"blood"},
    "rbc": {"range": (4.1, 5.9), "unit": "x10^12/L", "name_ar": "كريات الدم الحمراء", "type":"blood"},
    "hemoglobin": {"range": (13.0, 18.0), "unit": "g/dL", "name_ar": "الهيموغلوبين", "type":"blood"},
    "hematocrit": {"range": (40, 54), "unit": "%", "name_ar": "الهيماتوكريت", "type":"blood"},
    "platelets": {"range": (150, 450), "unit": "x10^9/L", "name_ar": "الصفائح الدموية", "type":"blood"},
    "glucose": {"range": (70, 100), "unit": "mg/dL", "name_ar": "الجلوكوز (صائم)", "type":"blood"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين", "type":"blood"},
    "alt": {"range": (7, 56), "unit": "U/L", "name_ar": "إنزيم ALT", "type":"liver"},
    "ast": {"range": (10, 40), "unit": "U/L", "name_ar": "إنزيم AST", "type":"liver"},
    "crp": {"range": (0, 10), "unit": "mg/L", "name_ar": "بروتين سي التفاعلي (CRP)", "type":"blood"},
    "sodium": {"range": (135, 145), "unit": "mEq/L", "name_ar": "الصوديوم", "type":"blood"},
    "potassium": {"range": (3.5, 5.0), "unit": "mEq/L", "name_ar": "البوتاسيوم", "type":"blood"},
    "urine_ph": {"range": (4.5, 8.0), "unit": "", "name_ar": "حموضة البول", "type":"urine"},
    "pus_cells": {"range": (0, 5), "unit": "/HPF", "name_ar": "خلايا الصديد في البول", "type":"urine"},
    "rbcs_urine": {"range": (0, 2), "unit": "/HPF", "name_ar": "كريات دم حمراء في البول", "type":"urine"},
    "protein_urine": {"range": (0, 0.15), "unit": "g/L", "name_ar": "بروتين في البول", "type":"urine"},
    "stool_occult": {"range": (0, 0), "unit": "positive/negative", "name_ar": "دم خفي في البراز", "type":"stool"},
    "stool_parasite": {"range": (0, 0), "unit": "positive/negative", "name_ar": "طفيليات البراز", "type":"stool"},
}

RECOMMENDATIONS = {
    "wbc": {"Low": "ضعف المناعة محتمل.", "High": "وجود عدوى محتملة."},
    "hemoglobin": {"Low": "قد يشير لفقر دم.", "High": "ارتفاع قد يدل جفاف."},
    "platelets": {"Low": "خطر نزيف.", "High": "خطر جلطة."},
    "glucose": {"High": "ارتفاع السكر محتمل."},
    "creatinine": {"High": "قد يشير لضعف الكلى."},
    "alt": {"High": "إصابة بالكبد محتملة."},
    "ast": {"High": "إصابة بالكبد محتملة."},
    "sodium": {"High": "ارتفاع الصوديوم قد يشير لجفاف."},
    "urine_ph": {"High": "الحمضية مرتفعة، احتمال التهاب بولي."},
    "pus_cells": {"High": "وجود التهاب بولي محتمل."},
    "rbcs_urine": {"High": "وجود دم في البول يحتاج متابعة."},
    "stool_occult": {"High": "وجود دم في البراز، قد يحتاج مناظير."},
    "stool_parasite": {"High": "وجود طفيليات، يتطلب علاج."},
}

# --- دوال المعالجة ---

# تم استبدال دالة pytesseract بهذه الدالة الجديدة
def extract_text_from_image(pipeline, image_bytes):
    """
    يستخدم keras-ocr لاستخراج النص من الصورة.
    """
    try:
        image = keras_ocr.tools.read(image_bytes)
        prediction_groups = pipeline.recognize([image])
        
        recognized_text = ""
        predictions = prediction_groups[0]
        sorted_predictions = sorted(predictions, key=lambda x: x[1][:, 1].min())
        
        lines = []
        current_line = []
        if sorted_predictions:
            avg_height = np.mean([ (p[1][:,1].max() - p[1][:,1].min()) for p in sorted_predictions])
            last_y = sorted_predictions[0][1][:, 1].min()

            for pred in sorted_predictions:
                current_y = pred[1][:, 1].min()
                if current_y - last_y > avg_height * 0.8:
                    lines.append(sorted(current_line, key=lambda x: x[1][:, 0].min()))
                    current_line = []
                current_line.append(pred)
                last_y = current_y
            lines.append(sorted(current_line, key=lambda x: x[1][:, 0].min()))

            final_text = []
            for line in lines:
                line_text = " ".join([pred[0] for pred in line])
                final_text.append(line_text)
            
            recognized_text = "\n".join(final_text)

        return recognized_text, None
    except Exception as e:
        return None, f"Keras-OCR Error: {e}"

# دالة تحليل النص (بدون تغيير)
def analyze_text_robust(text):
    if not text:
        return []

    text_lower = text.lower()
    number_pattern = re.compile(r'(\d+\.?\d*)')
    found_numbers = [(m.group(1), m.start()) for m in number_pattern.finditer(text_lower)]

    found_tests = []
    for key, details in NORMAL_RANGES.items():
        pattern = re.compile(rf'\b{key}\b', re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            found_tests.append({'key': key, 'pos': match.end()})

    results = []
    processed_tests = set()
    found_tests.sort(key=lambda x: x['pos'])

    for test in found_tests:
        key = test['key']
        if key in processed_tests: continue
        best_candidate = None
        min_distance = float('inf')
        for num_val, num_pos in found_numbers:
            distance = num_pos - test['pos']
            if 0 < distance < min_distance:
                min_distance = distance
                best_candidate = num_val
        if best_candidate:
            try:
                value = float(best_candidate)
                details = NORMAL_RANGES[key]
                low, high = details["range"]
                status = "طبيعي"
                if value < low: status = "منخفض"
                elif value > high: status = "مرتفع"
                recommendation = RECOMMENDATIONS.get(key, {}).get(status, "")
                results.append({
                    "name": f"{details['name_ar']}",
                    "value": value,
                    "status": status,
                    "recommendation": recommendation,
                    "type": details["type"]
                })
                processed_tests.add(key)
            except:
                continue
    return results

# --- عرض النتائج (الدالة المعدلة والمستقرة) ---
def display_results(results):
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return

    grouped = {}
    for res in results:
        cat_type = res.get("type", "other")
        if cat_type not in grouped:
            grouped[cat_type] = []
        grouped[cat_type].append(res)

    categories_to_display = [cat for cat in ["blood", "urine", "stool", "liver"] if cat in grouped]
    
    if not categories_to_display:
        st.warning("تم العثور على نتائج ولكن لا تنتمي لأي فئة معروفة.")
        return

    cols = st.columns(len(categories_to_display))

    for i, category in enumerate(categories_to_display):
        with cols[i]:
            st.markdown(f"### 🔬 {category.replace('_', ' ').capitalize()}")
            st.markdown("---")
            
            items = grouped[category]
            for r in items:
                status_color = "green" if r['status'] == 'طبيعي' else "orange" if r['status'] == 'منخفض' else "red"
                st.markdown(f"**{r['name']}**")
                st.markdown(f"النتيجة: **{r['value']}** | الحالة: <span style='color:{status_color};'>{r['status']}</span>", unsafe_allow_html=True)
                if r['recommendation']:
                    st.info(f"💡 {r['recommendation']}")
                st.markdown("---")

# --- الواجهة الرئيسية ---
st.title("🩺 المحلل الطبي الذكي Pro")
st.sidebar.header("⚙️ الإعدادات")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API", type="password")

mode = st.sidebar.radio("اختر الخدمة:", ["🔬 تحليل التقارير الطبية", "💬 استشارة حسب الأعراض"])
st.sidebar.info("هذا التطبيق لا يغني عن استشارة الطبيب المختص.")

if mode == "🔬 تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])
    
    if uploaded_file:
        # تحميل نموذج OCR
        pipeline = load_ocr_model()
        file_bytes = uploaded_file.getvalue()
        
        with st.spinner("🧠 العين القوية (Keras-OCR) تقرأ التقرير..."):
            text, err = extract_text_from_image(pipeline, file_bytes)

        if err:
            st.error(f"خطأ في قراءة الصورة: {err}")
        elif text:
            with st.expander("📄 عرض النص الخام المستخرج من الصورة (للتشخيص)"):
                st.text_area("النص الذي تم استخراجه:", text, height=250)

            results = analyze_text_robust(text)
            display_results(results)
        else:
            st.warning("لم تتمكن العين القوية من قراءة أي نص في الصورة.")


elif mode == "💬 استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    symptoms = st.text_area("📝 صف الأعراض هنا:", height=200)
    if st.button("تحليل الأعراض بالذكاء الاصطناعي"):
        if not api_key_input:
            st.error("يرجى إدخال مفتاح OpenAI API في الشريط الجانبي أولاً.")
        elif not symptoms.strip():
            st.warning("يرجى وصف الأعراض أولاً.")
        else:
            try:
                client = OpenAI(api_key=api_key_input)
                prompt = f'''أنت طبيب خبير وودود. المريض يصف الأعراض التالية: "{symptoms}".
                بناءً على هذه الأعراض، قم بالمهام التالية باللغة العربية:
                1. ابدأ بعبارة لطيفة.
                2. حلل الأعراض بشكل مبسط.
                3. اقترح بعض الفحوصات المخبرية الأولية المفيدة.
                4. قدم نصائح عامة أولية.
                5. اختتم بنصيحة **مهمة جدًا** تؤكد فيها أن هذه مجرد استشارة أولية وأن التشخيص الدقيق يتطلب زيارة طبيب حقيقي.'''
                
                with st.spinner("🧠 الذكاء الاصطناعي يحلل الأعراض..."):
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "أنت طبيب استشاري خبير تتحدث العربية بأسلوب ودود ومتعاطف."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.markdown(response.choices[0].message.content)
            except Exception as e:
                if "authentication" in str(e).lower():
                    st.error("❌ خطأ: مفتاح OpenAI API غير صحيح أو انتهت صلاحيته. يرجى التحقق منه.")
                else:
                    st.error(f"❌ حدث خطأ أثناء الاتصال بالذكاء الاصطناعي: {e}")

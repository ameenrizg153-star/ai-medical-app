import streamlit as st
import re
import io
import numpy as np
import pandas as pd
from openai import OpenAI
import cv2
import easyocr  # استيراد المكتبة الجديدة
import os

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- تحميل نموذج OCR (الآن يستخدم EasyOCR) ---
@st.cache_resource
def load_ocr_model():
    """
    تحميل قارئ EasyOCR مرة واحدة فقط.
    """
    # تحديد اللغات (الإنجليزية والعربية)
    reader = easyocr.Reader(['en', 'ar'])
    return reader

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
    "wbc": {"Low": "قد يشير إلى ضعف المناعة.", "High": "قد يشير إلى وجود عدوى."},
    "rbc": {"Low": "قد يكون مؤشرًا على فقر الدم.", "High": "قد يشير إلى الجفاف."},
    "hemoglobin": {"Low": "مؤشر أساسي على فقر الدم.", "High": "قد يشير إلى الجفاف."},
    "hematocrit": {"Low": "قد يشير إلى فقر الدم.", "High": "قد يشير إلى الجفاف الشديد."},
    "platelets": {"Low": "قد يزيد من خطر النزيف.", "High": "قد يزيد من خطر الجلطات."},
    "glucose": {"Low": "انخفاض السكر قد يسبب دوخة.", "High": "ارتفاع السكر قد يكون مؤشرًا على السكري."},
    "creatinine": {"High": "قد يشير إلى انخفاض كفاءة الكلى."},
    "alt": {"High": "مؤشر على وجود ضرر في الكبد."},
    "ast": {"High": "قد يشير إلى ضرر في الكبد أو العضلات."},
    "crp": {"High": "مؤشر على وجود التهاب حاد."},
    "sodium": {"Low": "قد يسبب ضعفًا وتعبًا.", "High": "قد يشير إلى الجفاف."},
    "potassium": {"Low": "قد يسبب ضعفًا في العضلات.", "High": "خطير على القلب."},
    "urine_ph": {"Low": "زيادة حمضية البول.", "High": "قلوية البول قد تشير لالتهاب."},
    "pus_cells": {"High": "علامة على وجود التهاب بولي."},
    "rbcs_urine": {"High": "وجود دم في البول يتطلب استشارة."},
    "protein_urine": {"High": "قد يكون علامة على مشاكل في الكلى."},
    "stool_occult": {"High": "وجود دم خفي يتطلب فحوصات إضافية."},
    "stool_parasite": {"High": "وجود طفيليات يتطلب علاجًا."}
}

# --- دوال المعالجة ---

# تم تعديل الدالة لاستخدام EasyOCR
def extract_text_from_image(reader, image_bytes):
    """
    يستخدم EasyOCR لاستخراج النص من الصورة.
    """
    try:
        # EasyOCR يقرأ مباشرة من الـ bytes
        result = reader.readtext(image_bytes, detail=0, paragraph=True)
        # result هو قائمة من الفقرات، ندمجها في نص واحد
        return "\n".join(result), None
    except Exception as e:
        return None, f"EasyOCR Error: {e}"

# باقي الدوال (analyze_text_robust, display_results, etc.) تبقى كما هي بدون تغيير

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
                if value < low: status = "Low"
                elif value > high: status = "High"
                recommendation = RECOMMENDATIONS.get(key, {}).get(status, "")
                results.append({
                    "name": f"{details['name_ar']}",
                    "value": value,
                    "status": "منخفض" if status == "Low" else "مرتفع" if status == "High" else "طبيعي",
                    "recommendation": recommendation,
                    "type": details["type"]
                })
                processed_tests.add(key)
            except:
                continue
    return results

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

def get_ai_interpretation(api_key, results):
    abnormal_results = [r for r in results if r['status'] != 'طبيعي']
    if not abnormal_results:
        return "✅ كل الفحوصات التي تم التعرف عليها ضمن النطاق الطبيعي. لا توجد ملاحظات خاصة."
    prompt_text = "أنت طبيب استشاري خبير ومهمتك هي تفسير نتائج التحاليل الطبية التالية لمريض يتحدث العربية. النتائج غير الطبيعية هي:\n\n"
    for r in abnormal_results:
        prompt_text += f"- {r['name']}: النتيجة هي {r['value']}، وهي تعتبر **{r['status']}**.\n"
    prompt_text += """
\nبناءً على هذه النتائج فقط، قم بالمهام التالية بأسلوب واضح وبسيط ومطمئن:
1.  ابدأ بملخص عام للحالة.
2.  فسّر كل نتيجة غير طبيعية على حدة.
3.  اشرح كيف يمكن أن ترتبط هذه النتائج ببعضها البعض إن أمكن.
4.  قدم بعض النصائح العامة جدًا.
5.  اختتم بفقرة **مهمة جدًا** تؤكد فيها أن هذا التفسير هو مجرد أداة مساعدة أولية.
"""
    try:
        client = OpenAI(api_key=api_key)
        with st.spinner("🧠 الذكاء الاصطناعي يكتب التفسير الشامل..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "أنت طبيب استشاري خبير، تتحدث العربية بأسلوب واضح ومطمئن للمرضى."},
                    {"role": "user", "content": prompt_text}
                ]
            )
            return response.choices[0].message.content
    except Exception as e:
        if "authentication" in str(e).lower():
            return "❌ **خطأ:** مفتاح OpenAI API غير صحيح."
        return f"❌ حدث خطأ: {e}"

# --- الواجهة الرئيسية ---
st.title("🩺 المحلل الطبي الذكي Pro (يعمل بـ EasyOCR)")
st.sidebar.header("⚙️ الإعدادات")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API", type="password")
mode = st.sidebar.radio("اختر الخدمة:", ["🔬 تحليل التقارير الطبية", "🩺 مدقق الأعراض الذكي"])
st.sidebar.info("هذا التطبيق لا يغني عن استشارة الطبيب المختص.")

if mode == "🔬 تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])
    if uploaded_file:
        reader = load_ocr_model()
        file_bytes = uploaded_file.getvalue()
        
        with st.spinner("🚀 EasyOCR يقرأ التقرير بسرعة ودقة..."):
            text, err = extract_text_from_image(reader, file_bytes)

        if err:
            st.error(f"خطأ في قراءة الصورة: {err}")
        elif text:
            with st.expander("📄 عرض النص الخام المستخرج من الصورة"):
                st.text_area("النص الذي تم استخراجه:", text, height=250)
            results = analyze_text_robust(text)
            display_results(results)
            st.markdown("---")
            if st.button("🔬 طلب تفسير شامل من الذكاء الاصطناعي"):
                if not api_key_input:
                    st.error("يرجى إدخال مفتاح OpenAI API أولاً.")
                elif not results:
                    st.warning("لا توجد نتائج لتحليلها.")
                else:
                    interpretation = get_ai_interpretation(api_key_input, results)
                    st.subheader("📜 التفسير الشامل للنتائج")
                    st.markdown(interpretation)
        else:
            st.warning("لم يتمكن EasyOCR من قراءة أي نص في الصورة.")

# قسم مدقق الأعراض يبقى كما هو بدون تغيير
elif mode == "🩺 مدقق الأعراض الذكي":
    st.header("🩺 مدقق الأعراض الذكي (نموذج مدرب محليًا)")
    
    @st.cache_data
    def load_symptom_data():
        try:
            symptom_data = pd.read_csv('Training.csv')
            symptom_model = joblib.load('symptom_checker_model.joblib')
            symptoms_list = symptom_data.columns[:-1].tolist()
            return symptom_model, symptoms_list, symptom_data
        except FileNotFoundError:
            return None, None, None
            
    symptom_model, symptoms_list, symptom_data = load_symptom_data()

    if symptom_model is None or symptoms_list is None:
        st.error("خطأ: لم يتم العثور على ملفات النموذج ('symptom_checker_model.joblib') أو البيانات ('Training.csv').")
    else:
        st.info("اختر الأعراض التي تشعر بها من القائمة أدناه.")
        selected_symptoms = st.multiselect("حدد الأعراض:", options=symptoms_list)
        if st.button("🔬 تشخيص الأعراض"):
            if not selected_symptoms:
                st.warning("يرجى تحديد عرض واحد على الأقل.")
            else:
                input_vector = [0] * len(symptoms_list)
                for symptom in selected_symptoms:
                    if symptom in symptoms_list:
                        index = symptoms_list.index(symptom)
                        input_vector[index] = 1
                input_df = pd.DataFrame([input_vector], columns=symptoms_list)
                with st.spinner("...النموذج المحلي يحلل الأعراض..."):
                    prediction = symptom_model.predict(input_df)
                    predicted_diagnosis = prediction[0]
                st.subheader("📜 التشخيص الأولي المحتمل")
                st.success(f"بناءً على الأعراض، قد يكون التشخيص المحتمل هو: **{predicted_diagnosis}**")
                st.warning("**تنبيه هام:** هذا التشخيص هو مجرد تنبؤ أولي ولا يغني عن استشارة الطبيب.")

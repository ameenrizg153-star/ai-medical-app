import streamlit as st
import re
import io
import numpy as np
import pandas as pd
from openai import OpenAI
import cv2
import easyocr
# import pytesseract # <-- تم حذفه
import joblib
from PIL import Image
import os
from tensorflow.keras.models import load_model
import altair as alt

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Suite",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- تحميل النماذج (EasyOCR فقط) ---
@st.cache_resource
def load_ocr_model():
    # استخدام اللغة الإنجليزية فقط لضمان الدقة
    return easyocr.Reader(['en'])

# ... (باقي دوال التحميل وقاعدة المعرفة تبقى كما هي تمامًا) ...
@st.cache_data
def load_symptom_checker():
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_ecg_analyzer():
    try:
        ecg_model = load_model("ecg_classifier_model.h5")
        ecg_signals = np.load("sample_ecg_signals.npy", allow_pickle=True).item()
        return ecg_model, ecg_signals
    except FileNotFoundError:
        return None, None

KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "aliases": ["w.b.c", "white blood cells"], "range": (4.0, 11.0), "unit": "x10^9/L", "category": "الالتهابات والمناعة", "recommendation_high": "ارتفاع قد يشير إلى عدوى بكتيرية.", "recommendation_low": "انخفاض قد يشير إلى ضعف مناعي."},
    "rbc": {"name_ar": "كريات الدم الحمراء", "aliases": ["r.b.c", "red blood cells"], "range": (4.1, 5.9), "unit": "x10^12/L", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض قد يشير إلى فقر دم."},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "aliases": ["hb", "hgb"], "range": (13.0, 18.0), "unit": "g/dL", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض هو مؤشر أساسي على فقر الدم."},
    "platelets": {"name_ar": "الصفائح الدموية", "aliases": ["plt"], "range": (150, 450), "unit": "x10^9/L", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يزيد من خطر الجلطات.", "recommendation_low": "انخفاض قد يزيد من خطر النزيف."},
    "color": {"name_ar": "لون البول", "aliases": ["colour"], "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "لون داكن قد يشير لجفاف، لون أحمر قد يشير لوجود دم.", "recommendation_low": ""},
    "appearance": {"name_ar": "عكارة البول", "aliases": ["clarity"], "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "عكارة قد تشير لوجود التهاب أو أملاح.", "recommendation_low": ""},
    "ph": {"name_ar": "حموضة البول (pH)", "aliases": ["p.h", "p h"], "range": (4.5, 8.0), "unit": "", "category": "تحليل البول", "recommendation_high": "قلوية البول قد تشير لالتهاب.", "recommendation_low": "حمضية البول قد ترتبط بحصوات معينة."},
    "sg": {"name_ar": "الكثافة النوعية (SG)", "aliases": ["specific gravity", "gravity"], "range": (1.005, 1.030), "unit": "", "category": "تحليل البول", "recommendation_high": "ارتفاع الكثافة قد يشير إلى الجفاف.", "recommendation_low": "انخفاض الكثافة قد يشير إلى شرب كميات كبيرة من الماء."},
    "leukocytes": {"name_ar": "كريات الدم البيضاء", "aliases": ["leukocyte", "leu"], "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "وجودها هو علامة قوية على التهاب المسالك البولية.", "recommendation_low": ""},
    "pus": {"name_ar": "خلايا الصديد", "aliases": ["pus cells"], "range": (0, 5), "unit": "/HPF", "category": "تحليل البول", "recommendation_high": "ارتفاع عددها يؤكد وجود التهاب بولي.", "recommendation_low": ""},
    "rbcs": {"name_ar": "كريات الدم الحمراء", "aliases": ["rbc's", "red blood cells", "blood"], "range": (0, 2), "unit": "/HPF", "category": "تحليل البول", "recommendation_high": "وجود دم في البول يتطلب استشارة طبية لمعرفة السبب.", "recommendation_low": ""},
}
def analyze_text_robust(text):
    if not text: return []
    results = []
    text_lower = text.lower()
    found_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+\.?\d*)', text_lower)]
    found_tests = []
    for key, details in KNOWLEDGE_BASE.items():
        search_terms = [key] + details.get("aliases", [])
        for term in search_terms:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            for match in pattern.finditer(text_lower):
                found_tests.append({'key': key, 'pos': match.end()})
                break
            else:
                continue
            break
    found_tests.sort(key=lambda x: x['pos'])
    unique_found_keys = []
    for test in found_tests:
        if test['key'] not in [t['key'] for t in unique_found_keys]:
             unique_found_keys.append(test)
    for test in unique_found_keys:
        key = test['key']
        best_candidate_val = None
        min_distance = float('inf')
        for num_val, num_pos in found_numbers:
            distance = num_pos - test['pos']
            if 0 < distance < 50:
                if num_pos + len(num_val) < len(text_lower) and text_lower[num_pos + len(num_val)].isalpha():
                    continue
                min_distance = distance
                best_candidate_val = num_val
                break
        if best_candidate_val:
            try:
                value = float(best_candidate_val)
                details = KNOWLEDGE_BASE[key]
                low, high = details["range"]
                status = "طبيعي"
                if value < low: status = "منخفض"
                elif value > high: status = "مرتفع"
                results.append({
                    "name": details['name_ar'], "value": value, "status": status,
                    "recommendation": details.get(f"recommendation_{status.lower()}", details.get("recommendation_high", "") if status == "مرتفع" else ""),
                    "category": details.get("category", "عام")
                })
            except (ValueError, KeyError):
                continue
    return results
def display_results(results):
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return
    grouped = {}
    for res in results:
        cat = res.get("category", "عام")
        if cat not in grouped: grouped[cat] = []
        grouped[cat].append(res)
    sorted_categories = sorted(grouped.keys())
    for category in sorted_categories:
        st.subheader(f"📁 {category}")
        for r in results:
            if r['category'] == category:
                status_color = "green" if r['status'] == 'طبيعي' else "orange" if r['status'] == 'منخفض' else "red"
                st.markdown(f"**{r['name']}**: {r['value']}  <span style='color:{status_color}; font-weight:bold;'>({r['status']})</span>", unsafe_allow_html=True)
                if r['recommendation']:
                    st.info(f"💡 {r['recommendation']}")
        st.markdown("---")
def plot_signal(signal, title):
    df = pd.DataFrame({'Time': range(len(signal)), 'Amplitude': signal})
    chart = alt.Chart(df).mark_line(color='#FF4B4B').encode(x=alt.X('Time', title='الزمن'), y=alt.Y('Amplitude', title='السعة'), tooltip=['Time', 'Amplitude']).properties(title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- الواجهة الرئيسية (تم تبسيطها للاعتماد على EasyOCR فقط) ---
st.title("⚕️ المجموعة الطبية الذكية")
st.sidebar.header("اختر الأداة المطلوبة")
mode = st.sidebar.radio("الأدوات المتاحة:", ("🔬 تحليل التقارير الطبية (OCR)", "🩺 مدقق الأعراض الذكي", "💓 محلل إشارات ECG"))
st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API (اختياري)", type="password")

if mode == "🔬 تحليل التقارير الطبية (OCR)":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])
    
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        text = None
        
        # --- استراتيجية جديدة: الاعتماد الكامل على EasyOCR للموثوقية ---
        with st.spinner("المحرك المتقدم (EasyOCR) يحلل الصورة الآن... (قد يستغرق بعض الوقت)"):
            try:
                # معالجة الصورة لتقليل الذاكرة
                img = Image.open(io.BytesIO(file_bytes)).convert('L')
                img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_bytes_processed = buffered.getvalue()
                
                reader = load_ocr_model()
                raw_results = reader.readtext(img_bytes_processed, detail=0, paragraph=True)
                text = "\n".join(raw_results)
                st.success("تم تحليل الصورة بنجاح!")
            except Exception as e:
                st.error(f"حدث خطأ فادح أثناء التحليل: {e}")
                text = None

        # عرض النتائج النهائية
        if text:
            with st.expander("📄 عرض النص الخام المستخرج"):
                st.text_area("النص:", text, height=250)
            
            final_results = analyze_text_robust(text)
            display_results(final_results)
        elif text is None:
            # لا تفعل شيئًا، فقد تم عرض رسالة الخطأ بالفعل
            pass
        else:
            st.error("لم يتمكن المحرك من قراءة أي نص في الصورة.")

elif mode == "🩺 مدقق الأعراض الذكي":
    # ... (الكود الكامل لهذا الوضع موجود في الردود السابقة) ...
    pass
elif mode == "💓 محلل إشارات ECG":
    # ... (الكود الكامل لهذا الوضع موجود في الردود السابقة) ...
    pass

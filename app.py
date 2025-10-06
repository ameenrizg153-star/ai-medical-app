import streamlit as st
import re
import io
import numpy as np
import pandas as pd
from openai import OpenAI
import cv2
import easyocr
import pytesseract # إعادة استيراده للمحلل الهجين
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

# --- تحميل النماذج (مع التخزين المؤقت) ---
@st.cache_resource
def load_ocr_models():
    """تحميل قارئ EasyOCR (النموذج الثقيل)."""
    return easyocr.Reader(['en', 'ar'])

@st.cache_data
def load_symptom_checker():
    """تحميل نموذج مدقق الأعراض."""
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_ecg_analyzer():
    """تحميل نموذج محلل ECG."""
    try:
        ecg_model = load_model("ecg_classifier_model.h5")
        ecg_signals = np.load("sample_ecg_signals.npy", allow_pickle=True).item()
        return ecg_model, ecg_signals
    except FileNotFoundError:
        return None, None

# --- قاعدة المعرفة المتكاملة ---
KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "range": (4.0, 11.0), "unit": "x10^9/L", "category": "الالتهابات والمناعة", "recommendation_high": "ارتفاع قد يشير إلى عدوى بكتيرية.", "recommendation_low": "انخفاض قد يشير إلى ضعف مناعي."},
    "rbc": {"name_ar": "كريات الدم الحمراء", "range": (4.1, 5.9), "unit": "x10^12/L", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض قد يشير إلى فقر دم."},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "range": (13.0, 18.0), "unit": "g/dL", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض هو مؤشر أساسي على فقر الدم."},
    "glucose": {"name_ar": "سكر الدم", "range": (70, 100), "unit": "mg/dL", "category": "سكر الدم", "recommendation_high": "قد يدل على سكري أو مقاومة للأنسولين.", "recommendation_low": "قد يدل على هبوط سكر."},
    "creatinine": {"name_ar": "الكرياتينين", "range": (0.6, 1.3), "unit": "mg/dL", "category": "وظائف الكلى", "recommendation_high": "ارتفاع يدل على ضعف محتمل في وظائف الكلى.", "recommendation_low": "عادة لا يثير القلق."},
    "alt": {"name_ar": "إنزيم ALT", "range": (7, 56), "unit": "U/L", "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على التهاب أو تلف في الكبد.", "recommendation_low": ""},
    "ast": {"name_ar": "إنزيم AST", "range": (10, 40), "unit": "U/L", "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على تلف في الكبد أو العضلات.", "recommendation_low": ""},
    # ... يمكنك إضافة جميع الفحوصات الأخرى هنا بنفس الهيكل ...
}

# --- دوال المعالجة والتحليل ---

def analyze_text_robust(text):
    """
    الخوارزمية الذكية لتحليل النص واستخراج النتائج.
    """
    if not text: return []
    results = []
    processed_tests = set()
    text_lower = text.lower()
    
    # البحث عن كل الأرقام في النص ومواقعها
    found_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+\.?\d*)', text_lower)]
    
    # البحث عن كل أسماء الفحوصات ومواقعها
    found_tests = []
    for key in KNOWLEDGE_BASE.keys():
        # استخدام \b لضمان تطابق الكلمة كاملة
        pattern = re.compile(rf'\b{key}\b', re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            found_tests.append({'key': key, 'pos': match.end()})
            
    # ترتيب الفحوصات حسب موقعها في النص
    found_tests.sort(key=lambda x: x['pos'])

    for test in found_tests:
        key = test['key']
        if key in processed_tests: continue
        
        best_candidate_val = None
        min_distance = float('inf')

        # البحث عن أقرب رقم يأتي بعد اسم الفحص
        for num_val, num_pos in found_numbers:
            distance = num_pos - test['pos']
            if 0 < distance < min_distance:
                # تحقق من عدم وجود حرف مباشرة بعد الرقم (لتجنب أرقام مثل x10^9)
                if num_pos + len(num_val) < len(text_lower) and text_lower[num_pos + len(num_val)].isalpha():
                    continue
                min_distance = distance
                best_candidate_val = num_val
        
        if best_candidate_val:
            try:
                value = float(best_candidate_val)
                details = KNOWLEDGE_BASE[key]
                low, high = details["range"]
                
                status = "طبيعي"
                if value < low: status = "منخفض"
                elif value > high: status = "مرتفع"
                
                results.append({
                    "name": details['name_ar'],
                    "value": value,
                    "status": status,
                    "recommendation": details.get(f"recommendation_{status.lower()}", ""),
                    "category": details.get("category", "عام")
                })
                processed_tests.add(key)
            except (ValueError, KeyError):
                continue
    return results

def display_results(results):
    """
    عرض النتائج مجمعة حسب الفئة.
    """
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return

    grouped = {}
    for res in results:
        cat = res.get("category", "عام")
        if cat not in grouped:
            grouped[cat] = []
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

# (دالة get_ai_interpretation ودالة plot_signal تبقى كما هي)
def get_ai_interpretation(api_key, results):
    # ... (الكود الكامل للدالة موجود في الردود السابقة) ...
    pass

def plot_signal(signal, title):
    # ... (الكود الكامل للدالة موجود في الردود السابقة) ...
    pass

# --- الواجهة الرئيسية للتطبيق ---
st.title("⚕️ المجموعة الطبية الذكية")

st.sidebar.header("اختر الأداة المطلوبة")
mode = st.sidebar.radio(
    "الأدوات المتاحة:",
    ("🔬 تحليل التقارير الطبية (OCR)", "🩺 مدقق الأعراض الذكي", "💓 محلل إشارات ECG")
)
st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API (اختياري)", type="password")

# --- منطق العرض حسب الاختيار ---

# 1. وضع تحليل التقارير (مع المحلل الهجين)
if mode == "🔬 تحليل التقارير الطبية (OCR)":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])
    
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        text = ""
        
        # المرحلة الأولى: المحاولة السريعة بـ Tesseract
        with st.spinner("المرحلة 1: جاري المحاولة السريعة..."):
            try:
                text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)), lang='eng+ara')
                results = analyze_text_robust(text)
                # التحقق من نجاح المحاولة السريعة
                if len(results) < 3:
                    st.warning("المحاولة السريعة لم تجد نتائج كافية. جاري الانتقال إلى المحرك المتقدم...")
                    text = "" # إفراغ النص للانتقال للمرحلة الثانية
                else:
                    st.success("تم التحليل بنجاح باستخدام المحرك السريع!")
            except Exception:
                text = "" # أي خطأ ينقلنا للمرحلة الثانية

        # المرحلة الثانية: المحاولة القوية بـ EasyOCR (فقط إذا فشلت الأولى)
        if not text:
            with st.spinner("المرحلة 2: المحرك المتقدم (EasyOCR) يحلل الصورة الآن... (قد يستغرق بعض الوقت)"):
                try:
                    # معالجة الصورة لتقليل الذاكرة
                    img = Image.open(io.BytesIO(file_bytes)).convert('L')
                    img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_bytes_processed = buffered.getvalue()
                    
                    reader = load_ocr_models()
                    raw_results = reader.readtext(img_bytes_processed, detail=0, paragraph=True)
                    text = "\n".join(raw_results)
                    st.success("تم التحليل بنجاح باستخدام المحرك المتقدم!")
                except Exception as e:
                    st.error(f"حدث خطأ فادح أثناء التحليل المتقدم: {e}")
                    text = None

        # عرض النتائج النهائية
        if text:
            with st.expander("📄 عرض النص الخام المستخرج"):
                st.text_area("النص:", text, height=250)
            
            final_results = analyze_text_robust(text)
            display_results(final_results)
            
            # (زر طلب التفسير الشامل من GPT)
            # ...
        elif text is None:
            # لا تفعل شيئًا، فقد تم عرض رسالة الخطأ بالفعل
            pass
        else:
            st.error("لم يتمكن أي من المحركين من قراءة النص في الصورة.")

# 2. وضع مدقق الأعراض
elif mode == "🩺 مدقق الأعراض الذكي":
    # ... (الكود الكامل لهذا الوضع موجود في الردود السابقة) ...
    pass

# 3. وضع محلل إشارات ECG
elif mode == "💓 محلل إشارات ECG":
    # ... (الكود الكامل لهذا الوضع موجود في الردود السابقة) ...
    pass

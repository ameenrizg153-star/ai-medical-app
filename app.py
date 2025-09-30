import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import joblib
import os
import urllib.request
from pathlib import Path

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- إعدادات النموذج الجديد (TFLite) ---
MODEL_URL = "https://github.com/tulasiram58827/ocr_tflite/raw/main/models/keras_ocr_float16.tflite"
MODEL_LOCAL_PATH = Path("models/keras_ocr_float16.tflite")

# --- إصلاح المسارات لبيئة Streamlit Cloud ---
pytesseract.pytesseract.tesseract_cmd = 'tesseract'
POPPLER_PATH = '/usr/bin'

# --- قاعدة بيانات الفحوصات (القاموس الرئيسي) ---
NORMAL_RANGES = {
    "wbc": {"range": (4.0, 11.0), "unit": "x10^9/L", "name_ar": "كريات الدم البيضاء"},
    "rbc": {"range": (4.1, 5.7), "unit": "x10^12/L", "name_ar": "كريات الدم الحمراء"},
    "hemoglobin": {"range": (13.0, 18.0), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "hematocrit": {"range": (40, 54), "unit": "%", "name_ar": "الهيماتوكريت"},
    "platelets": {"range": (150, 450), "unit": "x10^9/L", "name_ar": "الصفائح الدموية"},
    "neutrophils": {"range": (40, 70), "unit": "%", "name_ar": "العدلات"},
    "lymphocytes": {"range": (20, 45), "unit": "%", "name_ar": "اللمفاويات"},
    "monocytes": {"range": (2, 10), "unit": "%", "name_ar": "الوحيدات"},
    "eosinophils": {"range": (0, 6), "unit": "%", "name_ar": "الحمضات"},
    "basophils": {"range": (0, 1), "unit": "%", "name_ar": "القعدات"},
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
    "esr": {"range": (0, 15), "unit": "mm/hr", "name_ar": "سرعة الترسيب"},
}

# --- قاموس الأسماء البديلة (القلب النابض للذكاء) ---
ALIASES = {
    "hb": "hemoglobin", "hgb": "hemoglobin",
    "pcv": "hematocrit", "hct": "hematocrit",
    "t.wbc": "wbc", "w.b.c": "wbc", "wbc count": "wbc", "t.w.b.c": "wbc",
    "rbc count": "rbc", "r.b.c": "rbc",
    "platelats": "platelets", "plt": "platelets", "platelet count": "platelets",
    "neutrophil": "neutrophils", "neu": "neutrophils",
    "lymphocyte": "lymphocytes", "lym": "lymphocytes",
    "monocyte": "monocytes", "mono": "monocytes",
    "eosinophil": "eosinophils", "eos": "eosinophils",
    "basophil": "basophils", "baso": "basophils",
    "blood sugar": "glucose", "sugar": "glucose",
}

# --- تحميل نموذج الذكاء (وهمي حاليًا) ---
@st.cache_resource
def load_model():
    model_path = "symptom_checker_model.joblib"
    if not os.path.exists(model_path):
        from sklearn.dummy import DummyClassifier
        dummy_model = DummyClassifier(strategy="most_frequent")
        dummy_model.fit([[0]], [0])
        joblib.dump(dummy_model, model_path)
        return dummy_model
    try:
        return joblib.load(model_path)
    except Exception:
        return None

model = load_model()

# --- دوال النموذج الجديد (TFLite OCR) ---
@st.cache_resource
def download_model(url=MODEL_URL, local_path=MODEL_LOCAL_PATH):
    if local_path.exists(): return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner(f"Downloading TFLite OCR model (~23MB)..."):
            urllib.request.urlretrieve(url, str(local_path))
        return True
    except Exception as e:
        st.error(f"Failed to download TFLite model: {e}")
        return False

@st.cache_resource
def init_keras_ocr():
    import keras_ocr
    return keras_ocr.pipeline.Pipeline()

def run_keras_ocr(image, pipeline):
    prediction_groups = pipeline.recognize([np.array(image)])
    return " ".join([pred[0] for pred in prediction_groups[0]])

# --- دوال المعالجة والتحليل ---
def extract_text_from_image(file_bytes, engine='pytesseract'):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        if engine == 'TFLite (أكثر دقة)':
            if download_model():
                pipeline = init_keras_ocr()
                text = run_keras_ocr(img, pipeline)
                return text, None
            else:
                return None, "Could not use TFLite model."
        else:
            text = pytesseract.image_to_string(img, lang="eng+ara")
            return text, None
    except Exception as e:
        return None, f"OCR Error: {e}"

def analyze_text(text):
    results = []
    if not text: return results
    text_lower = text.lower()
    processed_tests = set()
    for key, details in NORMAL_RANGES.items():
        if key in processed_tests: continue
        aliases = [k for k, v in ALIASES.items() if v == key]
        search_keys = [key] + aliases
        pattern_keys = '|'.join([re.escape(k).replace(r"\_", "_").replace(".", r"\.?") for k in search_keys])
        pattern = re.compile(rf"\b({pattern_keys})\b\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        matches = pattern.finditer(text_lower)
        for m in matches:
            try:
                value = float(m.group(2))
                if key in processed_tests: continue
                low, high = details["range"]
                status = "Normal"
                if value < low: status = "Low"
                elif value > high: status = "High"
                results.append({
                    "الفحص": details["name_ar"], "القيمة": value, "الوحدة": details["unit"],
                    "الحالة": status, "النطاق الطبيعي": f"{low}-{high}"
                })
                processed_tests.add(key)
                break 
            except: continue
    return results

# --- واجهة التطبيق ---
st.title("🩺 المحلل الطبي الذكي")

st.sidebar.header("📌 القائمة")
mode = st.sidebar.radio("اختر الخدمة:", ["تحليل التقارير الطبية", "استشارة حسب الأعراض"])

if mode == "تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    ocr_engine = st.sidebar.selectbox("اختر محرك قراءة النصوص (OCR):", ["pytesseract (سريع)", "TFLite (أكثر دقة)"])
    uploaded_file = st.sidebar.file_uploader("📂 ارفع ملف صورة", type=["png","jpg","jpeg"])
    
    if uploaded_file:
        with st.spinner(f"جاري التحليل باستخدام {ocr_engine}..."):
            file_bytes = uploaded_file.getvalue()
            text, err = extract_text_from_image(file_bytes, engine=ocr_engine)
            
            if err:
                st.error(err)
            elif text:
                st.subheader("📄 النص المستخرج:")
                st.text_area("Extracted Text", text, height=200)
                results = analyze_text(text)
                if results:
                    df = pd.DataFrame(results)
                    st.subheader("📊 نتائج التحليل:")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("لم يتم التعرف على أي فحوصات.")
            else:
                st.warning("لم يتم استخراج أي نص.")

elif mode == "استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    symptoms = st.text_area("📝 صف الأعراض هنا بالتفصيل:", height=150)
    if st.button("تحليل الأعراض"):
        if symptoms:
            if model:
                st.success("✅ تم تحميل نموذج الذكاء بنجاح.")
                st.info("⚠️ هذا النموذج هو نموذج تجريبي. في المستقبل، سيتم استخدام الأعراض المدخلة للتنبؤ بالحالة الصحية.")
                st.write(f"الأعراض المدخلة: {symptoms}")
            else:
                st.error("🚨 خطأ: لم يتم تحميل نموذج الذكاء (symptom_checker_model.joblib).")
        else:
            st.warning("يرجى إدخال الأعراض أولاً.")

st.sidebar.markdown("---")
st.sidebar.info("تم التطوير بواسطة فريق Manus بالتعاون معك.")

import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
from pdf2image import convert_from_bytes
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

# --- إعدادات النموذج (TFLite) ---
MODEL_URL = "https://github.com/tulasiram58827/ocr_tflite/raw/main/models/keras_ocr_float16.tflite"
MODEL_LOCAL_PATH = Path("models/keras_ocr_float16.tflite")

# --- إصلاح المسارات لبيئة Streamlit Cloud ---
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# --- قاعدة بيانات الفحوصات (القاموس الرئيسي) ---
NORMAL_RANGES = {
    "wbc": {"range": (4.0, 11.0), "unit": "x10^9/L", "name_ar": "كريات الدم البيضاء", "icon": "⚪"},
    "rbc": {"range": (4.1, 5.7), "unit": "x10^12/L", "name_ar": "كريات الدم الحمراء", "icon": "🔴"},
    "hemoglobin": {"range": (13.0, 18.0), "unit": "g/dL", "name_ar": "الهيموغلوبين", "icon": "🩸"},
    "hematocrit": {"range": (40, 54), "unit": "%", "name_ar": "الهيماتوكريت", "icon": "📊"},
    "platelets": {"range": (150, 450), "unit": "x10^9/L", "name_ar": "الصفائح الدموية", "icon": "🩹"},
    "neutrophils": {"range": (40, 70), "unit": "%", "name_ar": "العدلات", "icon": "🔬"},
    "lymphocytes": {"range": (20, 45), "unit": "%", "name_ar": "اللمفاويات", "icon": "🔬"},
    "monocytes": {"range": (2, 10), "unit": "%", "name_ar": "الوحيدات", "icon": "🔬"},
    "eosinophils": {"range": (0, 6), "unit": "%", "name_ar": "الحمضات", "icon": "🔬"},
    "basophils": {"range": (0, 1), "unit": "%", "name_ar": "القعدات", "icon": "🔬"},
    "esr": {"range": (0, 15), "unit": "mm/hr", "name_ar": "سرعة الترسيب", "icon": "⏳"},
    "glucose": {
        "range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)", "icon": "🍬",
        "alt_units": {"mmol/L": {"range": (3.9, 7.8), "factor": 18.018}}
    },
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين", "icon": "💧"},
    "urea": {"range": (15, 45), "unit": "mg/dL", "name_ar": "اليوريا", "icon": "💧"},
    "uric_acid": {"range": (3.5, 7.2), "unit": "mg/dL", "name_ar": "حمض اليوريك", "icon": "💎"},
    "total_cholesterol": {"range": (120, 200), "unit": "mg/dL", "name_ar": "الكوليسترول الكلي", "icon": "🧈"},
    "triglycerides": {"range": (50, 150), "unit": "mg/dL", "name_ar": "الدهون الثلاثية", "icon": "🧈"},
    "hdl_cholesterol": {"range": (40, 60), "unit": "mg/dL", "name_ar": "الكوليسترول الجيد (HDL)", "icon": "✅"},
    "ldl_cholesterol": {"range": (70, 130), "unit": "mg/dL", "name_ar": "الكوليسترول الضار (LDL)", "icon": "❌"},
    "ast": {"range": (10, 40), "unit": "U/L", "name_ar": "إنزيم AST", "icon": "🌿"},
    "alt": {"range": (7, 56), "unit": "U/L", "name_ar": "إنزيم ALT", "icon": "🌿"},
}

# --- قاموس الأسماء البديلة ---
ALIASES = {
    "hb": "hemoglobin", "hgb": "hemoglobin", "pcv": "hematocrit", "hct": "hematocrit",
    "t.wbc": "wbc", "w.b.c": "wbc", "wbc count": "wbc", "white blood cells": "wbc",
    "rbc count": "rbc", "r.b.c": "rbc", "red blood cells": "rbc",
    "platelats": "platelets", "plt": "platelets", "platelet count": "platelets",
    "neutrophil": "neutrophils", "neu": "neutrophils", "lymphocyte": "lymphocytes", "lym": "lymphocytes",
    "monocyte": "monocytes", "mono": "monocytes", "eosinophil": "eosinophils", "eos": "eosinophils",
    "basophil": "basophils", "baso": "basophils", "blood sugar": "glucose", "sugar": "glucose",
    "uric acid": "uric_acid", "cholesterol": "total_cholesterol", "trig": "triglycerides",
    "hdl": "hdl_cholesterol", "ldl": "ldl_cholesterol", "sgot": "ast", "sgpt": "alt",
}

# --- قاموس التوصيات الأولية ---
RECOMMENDATIONS = {
    "wbc": {"Low": "قد يشير إلى ضعف المناعة أو عدوى فيروسية.", "High": "قد يشير إلى وجود عدوى بكتيرية أو التهاب."},
    "hemoglobin": {"Low": "قد يشير إلى فقر الدم (الأنيميا).", "High": "قد يكون بسبب التدخين أو العيش في المرتفعات."},
    "platelets": {"Low": "قد يزيد من خطر النزيف.", "High": "قد يزيد من خطر تكوّن الجلطات."},
    "glucose": {"Low": "انخفاض سكر الدم قد يسبب دوخة وإرهاق.", "High": "ارتفاع سكر الدم قد يكون مؤشرًا على السكري."},
    "creatinine": {"High": "قد يشير إلى وجود مشكلة في وظائف الكلى."},
    "alt": {"High": "قد يشير إلى وجود التهاب أو ضرر في خلايا الكبد."},
    "ast": {"High": "قد يشير إلى ضرر في الكبد أو أعضاء أخرى كالقلب."},
    "ldl_cholesterol": {"High": "ارتفاعه يزيد من خطر أمراض القلب."},
    "hdl_cholesterol": {"Low": "انخفاضه قد يزيد من خطر أمراض القلب."}
}

# --- دوال المعالجة والتحليل ---

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

def extract_text_from_image(file_bytes, engine='pytesseract'):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        if engine == 'TFLite (أكثر دقة)':
            if download_model():
                pipeline = init_keras_ocr()
                return run_keras_ocr(img, pipeline), None
            else:
                return None, "Could not use TFLite model."
        else:
            return pytesseract.image_to_string(img, lang="eng+ara"), None
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
        pattern = re.compile(rf"({pattern_keys})\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/µL^0-9]*)", re.IGNORECASE | re.DOTALL)
        
        for m in pattern.finditer(text_lower):
            try:
                value = float(m.group(2))
                unit_found = m.group(3).strip().lower()
                if key in processed_tests: continue

                current_range, current_unit = details["range"], details["unit"]
                
                if 'alt_units' in details:
                    for alt_unit, info in details['alt_units'].items():
                        if alt_unit.lower() in unit_found:
                            value /= info['factor']
                            break

                low, high = current_range
                status = "Normal"
                if value < low: status = "Low"
                elif value > high: status = "High"

                recommendation = RECOMMENDATIONS.get(key, {}).get(status, "")

                results.append({
                    "key": key, "name": f"{details['icon']} {details['name_ar']}",
                    "value_str": f"{m.group(2)} {unit_found}", "status": status,
                    "range_str": f"{low} - {high} {current_unit}", "recommendation": recommendation
                })
                processed_tests.add(key)
                break 
            except Exception:
                continue
    return results

# --- دوال الواجهة ---

def get_status_indicator(status):
    """Returns a colored HTML indicator based on the status."""
    colors = {"Normal": "#2E8B57", "Low": "#DAA520", "High": "#DC143C"}
    color = colors.get(status, "#808080")
    return f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <div style="width: 12px; height: 12px; background-color: {color}; border-radius: 50%; margin-right: 8px;"></div>
        <span style="color: {color}; font-weight: bold;">{status}</span>
    </div>
    """

def display_results_as_cards(results):
    """Displays analysis results in a card-based layout."""
    st.subheader("📊 نتائج التحليل")
    
    # Custom CSS for cards
    st.markdown("""
    <style>
    .result-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #007bff;
    }
    .result-card h4 {
        margin-top: 0;
        margin-bottom: 10px;
        color: #003366;
    }
    .recommendation {
        background-color: #e1ecf4;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

    for res in results:
        with st.container():
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"<h4>{res['name']}</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**النتيجة:** {res['value_str']}")
                st.markdown(f"**النطاق الطبيعي:** {res['range_str']}")
            with col2:
                st.markdown(get_status_indicator(res['status']), unsafe_allow_html=True)

            if res["recommendation"]:
                st.markdown(f"<div class='recommendation'>💡 **توصية أولية:** {res['recommendation']}</div>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("") # Adds a little space

# --- واجهة التطبيق الرئيسية ---
st.title("🩺 المحلل الطبي الذكي")
st.markdown("---")

st.sidebar.header("📌 القائمة")
mode = st.sidebar.radio("اختر الخدمة:", ["تحليل التقارير الطبية", "استشارة حسب الأعراض (قريباً)"])

if mode == "تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي")
    st.info("👋 أهلاً بك! ارفع صورة تقريرك الطبي وسأقوم بتحليلها وتقديم توصيات أولية لك.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ الإعدادات")
        ocr_engine = st.selectbox("اختر محرك قراءة النصوص (OCR):", ["pytesseract (سريع)", "TFLite (أكثر دقة)"])
        uploaded_file = st.file_uploader("📂 ارفع ملف صورة", type=["png","jpg","jpeg"])

    if uploaded_file:
        with st.spinner(f"جاري التحليل باستخدام {ocr_engine}..."):
            file_bytes = uploaded_file.getvalue()
            
            with col2:
                st.subheader("🖼️ الصورة المرفوعة")
                st.image(file_bytes, use_column_width=True)

            text, err = extract_text_from_image(file_bytes, engine=ocr_engine)
            
            if err:
                st.error(f"حدث خطأ أثناء قراءة الصورة: {err}")
            elif text:
                results = analyze_text(text)
                
                if results:
                    display_results_as_cards(results)
                    st.markdown("---")
                    st.warning(
                        "**⚠️ تنبيه هام:** هذه التوصيات هي لأغراض إرشادية فقط ولا تعتبر تشخيصًا طبيًا. "
                        "يجب عرض النتائج على طبيب مختص لتفسيرها بشكل دقيق."
                    )
                else:
                    st.warning("لم يتم التعرف على أي فحوصات مدعومة في النص المستخرج. قد تكون جودة الصورة منخفضة أو أن الفحوصات غير موجودة في قاعدة البيانات.")
                
                with st.expander("📄 عرض النص المستخرج من التقرير"):
                    st.text_area("Extracted Text", text, height=150)
            else:
                st.warning("لم يتمكن النظام من استخراج أي نص من الصورة. يرجى التأكد من وضوح الصورة.")

elif mode == "استشارة حسب الأعراض (قريباً)":
    st.header("💬 استشارة أولية حسب الأعراض")
    st.info("هذه الميزة قيد التطوير حاليًا وسيتم إطلاقها قريبًا!")
    st.text_area("📝 صف الأعراض هنا بالتفصيل:", height=150, disabled=True)
    st.button("تحليل الأعراض", disabled=True)

st.sidebar.markdown("---")
st.sidebar.info("تم التطوير بواسطة فريق Manus بالتعاون معك.")

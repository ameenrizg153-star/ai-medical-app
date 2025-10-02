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
    # ... (نفس القاموس من الإصدار السابق، لا تغيير هنا)
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
    # ... (نفس القاموس من الإصدار السابق، لا تغيير هنا)
    "hb": "hemoglobin", "hgb": "hemoglobin",
    "pcv": "hematocrit", "hct": "hematocrit",
    "t.wbc": "wbc", "w.b.c": "wbc", "wbc count": "wbc", "t.w.b.c": "wbc", "white blood cells": "wbc",
    "rbc count": "rbc", "r.b.c": "rbc", "red blood cells": "rbc",
    "platelats": "platelets", "plt": "platelets", "platelet count": "platelets",
    "neutrophil": "neutrophils", "neu": "neutrophils",
    "lymphocyte": "lymphocytes", "lym": "lymphocytes",
    "monocyte": "monocytes", "mono": "monocytes",
    "eosinophil": "eosinophils", "eos": "eosinophils",
    "basophil": "basophils", "baso": "basophils",
    "blood sugar": "glucose", "sugar": "glucose", "fasting blood sugar": "glucose",
    "uric acid": "uric_acid",
    "cholesterol": "total_cholesterol", "total cholesterol": "total_cholesterol",
    "trig": "triglycerides",
    "hdl": "hdl_cholesterol", "hdl-c": "hdl_cholesterol",
    "ldl": "ldl_cholesterol", "ldl-c": "ldl_cholesterol",
    "sgot": "ast", "aspartate aminotransferase": "ast",
    "sgpt": "alt", "alanine aminotransferase": "alt",
}

# --- الخطوة 3: إضافة قاموس التوصيات الأولية ---
RECOMMENDATIONS = {
    "wbc": {
        "Low": "قد يشير إلى ضعف المناعة أو عدوى فيروسية. يُنصح بمراجعة الطبيب.",
        "High": "قد يشير إلى وجود عدوى بكتيرية أو التهاب في الجسم."
    },
    "hemoglobin": {
        "Low": "قد يشير إلى فقر الدم (الأنيميا). يُنصح بزيادة مصادر الحديد وفيتامين B12.",
        "High": "قد يكون بسبب التدخين أو العيش في المرتفعات. نادرًا ما يكون خطيرًا."
    },
    "platelets": {
        "Low": "قد يزيد من خطر النزيف. يجب استشارة الطبيب لمعرفة السبب.",
        "High": "قد يزيد من خطر تكوّن الجلطات. يجب استشارة الطبيب."
    },
    "glucose": {
        "Low": "انخفاض سكر الدم. قد يسبب دوخة وإرهاق. يجب تناول شيء سكري.",
        "High": "ارتفاع سكر الدم. قد يكون مؤشرًا على مرض السكري أو مقدماته."
    },
    "creatinine": {
        "High": "قد يشير إلى وجود مشكلة في وظائف الكلى. ضروري مراجعة الطبيب."
    },
    "alt": {
        "High": "قد يشير إلى وجود التهاب أو ضرر في خلايا الكبد."
    },
    "ast": {
        "High": "قد يشير إلى ضرر في الكبد أو أعضاء أخرى مثل القلب أو العضلات."
    },
    "ldl_cholesterol": {
        "High": "يُعرف بالكوليسترول 'الضار'. ارتفاعه يزيد من خطر أمراض القلب."
    },
    "hdl_cholesterol": {
        "Low": "يُعرف بالكوليسترول 'الجيد'. انخفاضه قد يزيد من خطر أمراض القلب."
    }
}

# --- دوال المعالجة والتحليل (مع التحديثات) ---

@st.cache_resource
def download_model(url=MODEL_URL, local_path=MODEL_LOCAL_PATH):
    # ... (لا تغيير هنا)
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
    # ... (لا تغيير هنا)
    import keras_ocr
    return keras_ocr.pipeline.Pipeline()

def run_keras_ocr(image, pipeline):
    # ... (لا تغيير هنا)
    prediction_groups = pipeline.recognize([np.array(image)])
    return " ".join([pred[0] for pred in prediction_groups[0]])

def extract_text_from_image(file_bytes, engine='pytesseract'):
    # ... (لا تغيير هنا)
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
        pattern = re.compile(rf"({pattern_keys})\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/µL^0-9]*)", re.IGNORECASE | re.DOTALL)
        
        matches = pattern.finditer(text_lower)
        for m in matches:
            try:
                value = float(m.group(2))
                unit_found = m.group(3).strip().lower()
                if key in processed_tests: continue

                current_range = details["range"]
                current_unit = details["unit"]
                
                if 'alt_units' in details:
                    for alt_unit_name, alt_unit_info in details['alt_units'].items():
                        if alt_unit_name.lower() in unit_found:
                            value = value / alt_unit_info['factor']
                            break

                low, high = current_range
                status = "Normal"
                if value < low: status = "Low"
                elif value > high: status = "High"

                # *** الجزء الجديد: إضافة التوصية ***
                recommendation = ""
                if status in ["Low", "High"]:
                    recommendation = RECOMMENDATIONS.get(key, {}).get(status, "")

                results.append({
                    "الفحص": f"{details['icon']} {details['name_ar']}",
                    "القيمة": f"{m.group(2)} {unit_found}",
                    "الحالة": status,
                    "النطاق الطبيعي": f"{low} - {high} {current_unit}",
                    "توصية أولية": recommendation # إضافة العمود الجديد
                })
                processed_tests.add(key)
                break 
            except Exception:
                continue
    return results

def style_status(df):
    def color_status(val):
        color = 'white'
        if 'Normal' in val: color = '#2E8B57'
        elif 'Low' in val: color = '#DAA520' # لون أصفر أغمق
        elif 'High' in val: color = '#DC143C'
        return f'background-color: {color}; color: white; border-radius: 5px; text-align: center; padding: 5px;'
    
    # تطبيق التنسيق على عمود "الحالة" فقط
    return df.style.apply(lambda s: s.map(color_status), subset=['الحالة'])

# --- واجهة التطبيق ---
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
                st.subheader("📄 النص المستخرج من التقرير")
                st.text_area("Extracted Text", text, height=150, help="هذا هو النص الذي تمكن الذكاء الاصطناعي من قراءته من الصورة.")
                
                results = analyze_text(text)
                
                if results:
                    st.subheader("📊 نتائج التحليل")
                    df = pd.DataFrame(results)
                    
                    # إعادة ترتيب الأعمدة لجعل التوصية في النهاية
                    df = df[["الفحص", "القيمة", "الحالة", "النطاق الطبيعي", "توصية أولية"]]
                    
                    st.dataframe(style_status(df), use_container_width=True, height=len(df) * 38 + 38)
                    
                    # *** الجزء الجديد: إضافة ملاحظة هامة ***
                    st.markdown("---")
                    st.warning(
                        "**⚠️ تنبيه هام:** هذه التوصيات هي لأغراض إرشادية فقط ولا تعتبر تشخيصًا طبيًا. "
                        "النتائج يجب أن تُعرض على طبيب مختص لتفسيرها بشكل دقيق واتخاذ الإجراءات اللازمة."
                    )
                else:
                    st.warning("لم يتم التعرف على أي فحوصات مدعومة في النص المستخرج. قد تكون جودة الصورة منخفضة أو أن الفحوصات غير موجودة في قاعدة البيانات.")
            else:
                st.warning("لم يتمكن النظام من استخراج أي نص من الصورة. يرجى التأكد من وضوح الصورة.")

elif mode == "استشارة حسب الأعراض (قريباً)":
    st.header("💬 استشارة أولية حسب الأعراض")
    st.info("هذه الميزة قيد التطوير حاليًا وسيتم إطلاقها قريبًا!")
    symptoms = st.text_area("📝 صف الأعراض هنا بالتفصيل:", height=150, disabled=True)
    if st.button("تحليل الأعراض", disabled=True):
        pass

st.sidebar.markdown("---")
st.sidebar.info("تم التطوير بواسطة فريق Manus بالتعاون معك.")

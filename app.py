import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import pandas as pd
import os
from openai import OpenAI

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- إصلاح المسارات لبيئة Streamlit Cloud ---
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# --- قواعد البيانات والفحوصات الموسعة ---
NORMAL_RANGES = {
    # CBC - الدم
    "wbc": {"range": (4.0, 11.0), "unit": "x10^9/L", "name_ar": "كريات الدم البيضاء", "icon": "⚪"},
    "rbc": {"range": (4.1, 5.7), "unit": "x10^12/L", "name_ar": "كريات الدم الحمراء", "icon": "🔴"},
    "hemoglobin": {"range": (13.0, 18.0), "unit": "g/dL", "name_ar": "الهيموغلوبين", "icon": "🩸"},
    "hematocrit": {"range": (40, 54), "unit": "%", "name_ar": "الهيماتوكريت", "icon": "📊"},
    "platelets": {"range": (150, 450), "unit": "x10^9/L", "name_ar": "الصفائح الدموية", "icon": "🩹"},
    # كيمياء الدم
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)", "icon": "🍬"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين", "icon": "💧"},
    "alt": {"range": (7, 56), "unit": "U/L", "name_ar": "إنزيم ALT", "icon": "🌿"},
    "ast": {"range": (10, 40), "unit": "U/L", "name_ar": "إنزيم AST", "icon": "🌿"},
    # البول
    "ph": {"range": (4.5, 8.0), "unit": "", "name_ar": "حموضة البول (pH)", "icon": "💦"},
    "specific_gravity": {"range": (1.005, 1.030), "unit": "", "name_ar": "الكثافة النوعية للبول", "icon": "💧"},
    "pus_cells": {"range": (0, 5), "unit": "/HPF", "name_ar": "خلايا الصديد (Pus)", "icon": "🧫"},
    "rbc_urine": {"range": (0, 2), "unit": "/HPF", "name_ar": "كريات الدم الحمراء (بول)", "icon": "🔴"},
    # البراز
    "occult_blood": {"range": (0, 0), "unit": "", "name_ar": "دم خفي في البراز", "icon": "🟤"},
}

ALIASES = {
    "hb": "hemoglobin", "hgb": "hemoglobin", "pcv": "hematocrit", "hct": "hematocrit",
    "w.b.c": "wbc", "wbc count": "wbc", "white blood cells": "wbc",
    "r.b.c": "rbc", "red blood cells": "rbc", "plt": "platelets", "platelet count": "platelets",
    "blood sugar": "glucose", "sugar": "glucose", "sgot": "ast", "sgpt": "alt",
}

RECOMMENDATIONS = {
    "wbc": {"Low": "قد يشير إلى ضعف المناعة.", "High": "قد يشير إلى وجود عدوى أو التهاب."},
    "hemoglobin": {"Low": "قد يشير إلى فقر الدم (الأنيميا)."},
    "platelets": {"Low": "قد يزيد من خطر النزيف.", "High": "قد يزيد من خطر تكوّن الجلطات."},
    "glucose": {"High": "ارتفاع سكر الدم قد يكون مؤشرًا على السكري."},
    "creatinine": {"High": "قد يشير إلى مشكلة في وظائف الكلى."},
    "alt": {"High": "قد يشير إلى ضرر في خلايا الكبد."},
}

# --- دوال المعالجة ---
def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
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
        pattern_keys = '|'.join([re.escape(k).replace(".", r"\.?") for k in search_keys])
        pattern = re.compile(rf"({pattern_keys})\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            try:
                value = float(m.group(2))
                if key in processed_tests: continue
                low, high = details["range"]
                status = "Normal"
                if value < low: status = "Low"
                elif value > high: status = "High"
                recommendation = RECOMMENDATIONS.get(key, {}).get(status, "")
                results.append({
                    "name": f"{details['icon']} {details['name_ar']}",
                    "value_str": m.group(2), "status": status,
                    "range_str": f"{low} - {high} {details['unit']}", "recommendation": recommendation
                })
                processed_tests.add(key)
                break
            except Exception:
                continue
    return results

# --- الذكاء الاصطناعي ---
def get_ai_symptom_analysis(api_key, symptoms):
    try:
        client = OpenAI(api_key=api_key)
        prompt = f\"\"\"أنت طبيب استشاري ذكي. المريض يصف الأعراض: "{symptoms}".
        قم بتحليل الأعراض، اقترح فحوصات أولية، وأعطِ نصائح عامة باللغة العربية.
        \"\"\"
        with st.spinner("🤖 الذكاء الاصطناعي يحلل الأعراض..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "أنت طبيب خبير وودود."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
    except Exception as e:
        if "authentication" in str(e).lower():
            return "❌ مفتاح OpenAI API غير صحيح."
        return f"❌ خطأ في الاتصال بالذكاء الاصطناعي: {e}"

# --- عرض النتائج ---
def display_results_as_cards(results):
    st.subheader("📊 نتائج التحليل")
    st.markdown(\"\"\"<style>
    .result-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 5px solid #007bff; }
    .result-card h4 { margin-top: 0; margin-bottom: 10px; color: #003366; }
    .recommendation { background-color: #e1ecf4; border-radius: 5px; padding: 10px; margin-top: 10px; font-size: 0.9em; color: #333; }
    </style>\"\"\", unsafe_allow_html=True)
    for res in results:
        with st.container():
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"<h4>{res['name']}</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**النتيجة:** {res['value_str']}")
                st.markdown(f"**النطاق الطبيعي:** {res['range_str']}")
            with col2:
                colors = {"Normal": "#2E8B57", "Low": "#DAA520", "High": "#DC143C"}
                color = colors.get(res['status'], "#808080")
                st.markdown(f'<div style="color: {color}; font-weight: bold;">الحالة: {res["status"]}</div>', unsafe_allow_html=True)
            if res["recommendation"]:
                st.markdown(f"<div class='recommendation'>💡 **ملاحظة أولية:** {res['recommendation']}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- الواجهة الرئيسية ---
st.title("🩺 المحلل الطبي الذكي")
st.sidebar.header("⚙️ الإعدادات")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API", type="password")
mode = st.sidebar.radio("اختر الخدمة:", ["تحليل التقارير الطبية", "استشارة حسب الأعراض"])
st.sidebar.markdown("---")

if mode == "تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        text, err = extract_text_from_image(file_bytes)
        if err: st.error(err)
        elif text:
            results = analyze_text(text)
            if results: display_results_as_cards(results)
            else: st.warning("لم يتم التعرف على أي فحوصات مدعومة.")
            with st.expander("📄 عرض النص الخام المستخرج من الصورة"):
                st.text_area("", text, height=150)

elif mode == "استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    symptoms = st.text_area("📝 صف الأعراض هنا:", height=150)
    if st.button

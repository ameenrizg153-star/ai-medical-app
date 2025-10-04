import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
from openai import OpenAI
import cv2

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- قواعد البيانات والفحوصات ---
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
    # أضف باقي الفحوصات بنفس الشكل...
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
def preprocess_image_for_ocr(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        cv_image = np.array(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(thresh)
    except:
        return Image.open(io.BytesIO(file_bytes))

def extract_text_from_image(processed_img):
    try:
        return pytesseract.image_to_string(processed_img, lang="eng+ara"), None
    except Exception as e:
        return None, str(e)

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

# --- عرض النتائج مع نصائح ذكية ---
def display_results(results):
    if not results:
        st.error("لم يتم التعرف على أي فحوصات.")
        return
    grouped = {"blood":[], "urine":[], "stool":[]}
    for res in results:
        grouped[res["type"]].append(res)
    
    for category, items in grouped.items():
        if not items: continue
        st.subheader(f"🔬 {category.capitalize()}")
        for r in items:
            st.markdown(f"- {r['name']} — {r['value']} — {r['status']}")
            if r['recommendation']:
                st.markdown(f"  - 💡 {r['recommendation']}")

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
        file_bytes = uploaded_file.getvalue()
        processed_img = preprocess_image_for_ocr(file_bytes)
        text, err = extract_text_from_image(processed_img)
        if err:
            st.error(err)
        elif text:
            results = analyze_text_robust(text)
            display_results(results)
            with st.expander("📄 النص المستخرج"):
                st.text_area("", text, height=250)

elif mode == "💬 استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    symptoms = st.text_area("📝 صف الأعراض هنا:", height=200)
    if st.button("تحليل الأعراض بالذكاء الاصطناعي"):
        if not symptoms:
            st.warning("يرجى وصف الأعراض أولاً.")
        else:
            client = OpenAI(api_key=api_key_input)
            prompt = f'''أنت طبيب خبير. المريض يصف الأعراض: "{symptoms}".
قدم نصائح أولية بشكل نقاط.'''
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "أنت طبيب خبير وودود."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown(response.choices[0].message.content)

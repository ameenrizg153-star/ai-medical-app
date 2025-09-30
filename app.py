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

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- قاعدة بيانات الفحوصات ---
NORMAL_RANGES = {
    "wbc": {"range": (4000, 11000), "unit": "cells/mcL", "name_ar": "كريات الدم البيضاء"},
    "hemoglobin": {"range": (12.0, 17.5), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
    "vitamin_d": {"range": (30, 100), "unit": "ng/mL", "name_ar": "فيتامين D"},
    "crp": {"range": (0, 5), "unit": "mg/L", "name_ar": "بروتين سي التفاعلي (CRP)"},
}

ALIASES = {
    "blood sugar": "glucose",
    "hb": "hemoglobin",
    "sugar": "glucose"
}

# --- تحميل نموذج الذكاء ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("symptom_checker_model.joblib")
        return model
    except:
        return None

model = load_model()

# --- معالجة الصورة قبل OCR ---
def preprocess_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

# --- قراءة صورة ---
def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = preprocess_image(img)
        text = pytesseract.image_to_string(img, lang="eng+ara")
        return text, None
    except Exception as e:
        return None, f"OCR Error: {e}"

# --- قراءة PDF (نص + صور) ---
def extract_text_from_pdf(file_bytes):
    texts = []
    errors = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)

        if not texts:
            pages = convert_from_bytes(file_bytes)
            for i, page_img in enumerate(pages):
                try:
                    page_img = preprocess_image(page_img)
                    txt = pytesseract.image_to_string(page_img, lang="eng+ara")
                    texts.append(txt)
                except Exception as e:
                    errors.append(f"Error OCR page {i+1}: {e}")
    except Exception as e:
        return None, f"PDF Error: {e}"

    return "\n".join(texts), (errors if errors else None)

# --- تحليل النصوص الطبية ---
def analyze_text(text):
    results = []
    if not text:
        return results
    text_lower = text.lower()
    for key, details in NORMAL_RANGES.items():
        aliases = [k for k, v in ALIASES.items() if v == key]
        search_keys = [key] + aliases
        pattern = re.compile(rf"\b({'|'.join(search_keys)})\b[:\-= ]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        matches = pattern.finditer(text_lower)
        for m in matches:
            try:
                value = float(m.group(2))
                low, high = details["range"]
                status = "Normal"
                if value < low:
                    status = "Low"
                elif value > high:
                    status = "High"
                results.append({
                    "الفحص": details["name_ar"],
                    "القيمة": value,
                    "الوحدة": details["unit"],
                    "الحالة": status,
                    "النطاق الطبيعي": f"{low}-{high}"
                })
            except:
                continue
    return results

# --- واجهة ---
st.sidebar.header("📌 القائمة")
mode = st.sidebar.radio("اختر الخدمة:", ["تحليل التقارير الطبية", "استشارة حسب الأعراض"])

if mode == "تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (PDF أو صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف", type=["png","jpg","jpeg","pdf"])
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        text, err = (extract_text_from_pdf(file_bytes) if "pdf" in uploaded_file.type
                     else extract_text_from_image(file_bytes))
        if err: st.error(err)
        if text:
            st.subheader("📄 النص المستخرج:")
            st.text_area("Extracted Text", text, height=200)
            results = analyze_text(text)
            if results:
                df = pd.DataFrame(results)
                st.subheader("📊 نتائج التحليل:")
                st.dataframe(df, use_container_width=True)
                abnormal = df[df["الحالة"] != "Normal"]
                if not abnormal.empty:
                    st.error("⚠️ يوجد فحوصات غير طبيعية تحتاج متابعة طبية.")
            else:
                st.warning("لم يتم التعرف على أي فحوصات.")
elif mode == "استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    symptoms = st.text_area("📝 صف الأعراض:", height=150)
    if st.button("تحليل الأعراض"):
        if symptoms:
            if model:
                st.success("✅ تم تحميل نموذج الذكاء.")
                st.info("⚠️ مبدئيًا، النموذج يتنبأ بالحالة حسب البيانات المدخلة.")
            else:
                st.warning("🚨 لم يتم العثور على نموذج الذكاء (symptom_checker_model.joblib).")
        else:
            st.warning("يرجى إدخال الأعراض.")

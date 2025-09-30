import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pdf2image import convert_from_bytes

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- قاعدة بيانات الفحوصات (اختصاراً وضعت بعضها فقط، أضفناها في النسخة السابقة) ---
NORMAL_RANGES = {
    "wbc": {"range": (4000, 11000), "unit": "cells/mcL", "name_ar": "كريات الدم البيضاء"},
    "hemoglobin": {"range": (12.0, 17.5), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
    "vitamin_d": {"range": (30, 100), "unit": "ng/mL", "name_ar": "فيتامين D"},
    "crp": {"range": (0, 5), "unit": "mg/L", "name_ar": "بروتين سي التفاعلي (CRP)"},
}

ALIASES = {"blood sugar": "glucose", "hb": "hemoglobin", "sugar": "glucose"}

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
        text = pytesseract.image_to_string(img, lang='eng+ara')
        return text, None
    except Exception as e:
        return None, f"OCR Error: {e}"

# --- قراءة PDF (نص + صور) ---
def extract_text_from_pdf(file_bytes):
    texts = []
    errors = []
    try:
        # أولاً نحاول استخراج النصوص مباشرة
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)

        # إذا لم نجد نصوص أو الملف يحتوي على صور
        if not texts:
            pages = convert_from_bytes(file_bytes)
            for i, page_img in enumerate(pages):
                try:
                    page_img = preprocess_image(page_img)
                    txt = pytesseract.image_to_string(page_img, lang='eng+ara')
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
                if value < low: status = "Low"
                elif value > high: status = "High"
                results.append({
                    "الفحص": details["name_ar"],
                    "القيمة": value,
                    "الوحدة": details["unit"],
                    "الحالة": status,
                    "النطاق الطبيعي": f"{low}-{high}"
                })
            except: continue
    return results

# --- استشارة حسب الأعراض ---
RULE_KB = {
    "fever": {"conds": {"Infection": 0.8, "Flu": 0.6}, "advice_ar": ["قس الحرارة بانتظام", "اشرب سوائل كثيرة"]},
    "cough": {"conds": {"Bronchitis": 0.5, "Flu": 0.6}, "advice_ar": ["انتبه لضيق التنفس", "راجع الطبيب إذا استمر السعال"]},
    "chest pain": {"conds": {"Cardiac": 0.9}, "advice_ar": ["اطلب الرعاية الطبية فورًا إذا الألم شديد"]},
}
def rule_based_consult(symptoms: str):
    txt = symptoms.lower()
    cond_scores = {}
    advices = set()
    for kw, info in RULE_KB.items():
        if kw in txt:
            for cond, w in info["conds"].items():
                cond_scores[cond] = cond_scores.get(cond, 0) + w
            for a in info["advice_ar"]:
                advices.add(a)
    return cond_scores, list(advices)

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
            st.subheader("📄 النص المستخرج من التقرير:")
            st.text_area("Extracted Text", text, height=200)

            results = analyze_text(text)
            if results:
                df = pd.DataFrame(results)
                st.subheader("📊 نتائج التحليل:")
                st.dataframe(df, use_container_width=True)
                abnormal = df[df["الحالة"] != "Normal"]
                if not abnormal.empty:
                    st.subheader("📈 الفحوصات غير الطبيعية:")
                    fig, ax = plt.subplots()
                    ax.barh(abnormal["الفحص"], abnormal["القيمة"], color="red")
                    ax.set_xlabel("القيمة")
                    st.pyplot(fig)
            else:
                st.warning("لم يتم التعرف على أي فحوصات.")
elif mode == "استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    symptoms = st.text_area("📝 صف الأعراض التي تشعر بها:", height=150)
    if st.button("تحليل الأعراض"):
        if symptoms:
            conds, advices = rule_based_consult(symptoms)
            if conds:
                st.write("**الأمراض المحتملة:**")
                for c, score in conds.items():
                    st.write(f"- {c} (احتمالية: {score:.1f})")
            if advices:
                st.write("**نصائح:**")
                for a in advices: st.write(f"- {a}")
            st.info("⚠️ هذه الاستشارة أولية ولا تغني عن مراجعة الطبيب.")
        else:
            st.warning("يرجى إدخال الأعراض.")

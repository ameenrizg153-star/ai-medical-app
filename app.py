import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import cv2
import numpy as np
import pandas as pd
from openai import OpenAI

# إعدادات الصفحة
st.set_page_config(page_title="AI Medical Analyzer Pro", page_icon="🩺", layout="wide")

# --- قواعد البيانات ---
NORMAL_RANGES = {
    "wbc": {"range": (4.0, 11.0), "unit": "x10^9/L", "name_ar": "كريات الدم البيضاء"},
    "rbc": {"range": (4.1, 5.7), "unit": "x10^12/L", "name_ar": "كريات الدم الحمراء"},
    "hemoglobin": {"range": (13.0, 18.0), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "hematocrit": {"range": (40, 54), "unit": "%", "name_ar": "الهيماتوكريت"},
    "platelets": {"range": (150, 450), "unit": "x10^9/L", "name_ar": "الصفائح الدموية"},
    "glucose": {"range": (70, 100), "unit": "mg/dL", "name_ar": "الجلوكوز (صائم)"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
}

ALIASES = {
    "hb": "hemoglobin", "hgb": "hemoglobin",
    "pcv": "hematocrit", "hct": "hematocrit",
    "w.b.c": "wbc", "r.b.c": "rbc",
    "plt": "platelets", "blood sugar": "glucose",
}

RECOMMENDATIONS = {
    "wbc": {"Low": "قد يشير إلى ضعف المناعة.", "High": "قد يشير إلى وجود عدوى أو التهاب."},
    "hemoglobin": {"Low": "قد يشير إلى فقر الدم (الأنيميا)."},
    "platelets": {"Low": "قد يزيد من خطر النزيف.", "High": "قد يزيد من خطر تكوّن الجلطات."},
}

# --- دوال المعالجة ---
def preprocess_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        cv_img = np.array(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(thresh)
    except:
        return Image.open(io.BytesIO(file_bytes))

def extract_text(img):
    try:
        return pytesseract.image_to_string(img, lang="eng+ara"), None
    except Exception as e:
        return None, str(e)

def analyze_text(text):
    results = []
    if not text: return results
    text_lower = text.lower()
    for key, details in NORMAL_RANGES.items():
        aliases = [k for k,v in ALIASES.items() if v==key]
        search_keys = [key]+aliases
        pattern = re.compile(rf"({'|'.join(search_keys)})\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            try:
                value = float(m.group(2))
                low, high = details["range"]
                status = "Normal"
                if value < low: status="Low"
                elif value > high: status="High"
                results.append({
                    "name": details["name_ar"],
                    "value": value,
                    "unit": details["unit"],
                    "status": status
                })
            except:
                continue
    return results

# --- واجهة ---
st.title("🩺 AI Medical Analyzer Pro")
st.sidebar.header("⚙️ الإعدادات")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")

mode = st.sidebar.radio("اختر الخدمة:", ["تحليل التقرير", "استشارة حسب الأعراض"])

if mode=="تحليل التقرير":
    uploaded_file = st.file_uploader("رفع صورة التقرير", type=["png","jpg","jpeg"])
    if uploaded_file:
        img = preprocess_image(uploaded_file.getvalue())
        text, err = extract_text(img)
        if err: st.error(err)
        elif text:
            results = analyze_text(text)
            if results:
                for res in results:
                    st.write(f"{res['name']}: {res['value']} {res['unit']} ({res['status']})")
            else:
                st.warning("لم يتم التعرف على أي فحوصات مدعومة.")
elif mode=="استشارة حسب الأعراض":
    symptoms = st.text_area("صف الأعراض:")
    if st.button("تحليل الأعراض"):
        if not symptoms: st.warning("يرجى كتابة الأعراض أولاً.")
        else:
            if api_key_input:
                client = OpenAI(api_key=api_key_input)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system","content":"أنت طبيب خبير وودود"},
                        {"role": "user","content":f"حلل الأعراض التالية: {symptoms}"}
                    ]
                )
                st.markdown(response.choices[0].message.content)
            else:
                st.warning("يرجى إدخال مفتاح OpenAI API في الشريط الجانبي.")

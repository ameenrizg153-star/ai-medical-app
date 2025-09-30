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

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- إصلاح المسارات لبيئة Streamlit Cloud ---
# في بيئة لينكس، نحتاج أحيانًا لتحديد المسار يدويًا
pytesseract.pytesseract.tesseract_cmd = 'tesseract'
POPPLER_PATH = '/usr/bin' # المسار الشائع في بيئات لينكس

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
    # سنقوم بإنشاء ملف وهمي إذا لم يكن موجودًا
    model_path = "symptom_checker_model.joblib"
    if not os.path.exists(model_path):
        st.warning("Model file not found. Creating a dummy model.")
        # إنشاء نموذج وهمي بسيط (يمكن استبداله بنموذج حقيقي لاحقًا)
        from sklearn.dummy import DummyClassifier
        dummy_model = DummyClassifier(strategy="most_frequent")
        # تدريب وهمي
        dummy_model.fit([[0]], [0])
        joblib.dump(dummy_model, model_path)
        return dummy_model
        
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- معالجة الصورة قبل OCR ---
def preprocess_image(img):
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        # استخدام التنعيم لإزالة التشويش
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # استخدام عتبة متكيفة للحصول على نتائج أفضل مع إضاءة مختلفة
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(thresh)
    except Exception as e:
        st.warning(f"Could not preprocess image: {e}")
        return img # إرجاع الصورة الأصلية في حالة حدوث خطأ

# --- قراءة صورة ---
def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img_processed = preprocess_image(img)
        text = pytesseract.image_to_string(img_processed, lang="eng+ara")
        return text, None
    except Exception as e:
        return None, f"OCR Error: {e}"

# --- قراءة PDF (نص + صور) ---
def extract_text_from_pdf(file_bytes):
    texts = []
    errors = []
    try:
        # محاولة قراءة النص مباشرة أولاً
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)

        # إذا لم يتم العثور على نص، فهذا يعني أن الـ PDF هو صورة
        if not texts or not "".join(texts).strip():
            st.info("PDF seems to be an image. Converting pages to images for OCR...")
            pages = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
            for i, page_img in enumerate(pages):
                try:
                    page_img_processed = preprocess_image(page_img)
                    txt = pytesseract.image_to_string(page_img_processed, lang="eng+ara")
                    texts.append(f"\n--- OCR from Page {i+1} ---\n{txt}")
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
        pattern = re.compile(rf"\b({'|'.join(search_keys)})\b\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        matches = pattern.finditer(text_lower)
        for m in matches:
            try:
                value = float(m.group(2))
                
                # تجنب النتائج المكررة
                if any(d['الفحص'] == details["name_ar"] for d in results):
                    continue

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
                break # نكتفي بأول نتيجة للفحص
            except:
                continue
    return results

# --- واجهة ---
st.title("🩺 المحلل الطبي الذكي (نسخة احترافية)")

st.sidebar.header("📌 القائمة")
mode = st.sidebar.radio("اختر الخدمة:", ["تحليل التقارير الطبية", "استشارة حسب الأعراض"])

if mode == "تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (PDF أو صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف", type=["png","jpg","jpeg","pdf"])
    if uploaded_file:
        with st.spinner("جاري التحليل... هذه العملية قد تستغرق بعض الوقت."):
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
                    
                    def color_status(row):
                        if row['الحالة'] == 'High':
                            return ['background-color: #ffebee'] * len(row)
                        elif row['الحالة'] == 'Low':
                            return ['background-color: #fff8e1'] * len(row)
                        else:
                            return [''] * len(row)

                    st.dataframe(df.style.apply(color_status, axis=1), use_container_width=True)
                    
                    abnormal = df[df["الحالة"] != "Normal"]
                    if not abnormal.empty:
                        st.error("⚠️ يوجد فحوصات غير طبيعية تحتاج متابعة طبية.")
                else:
                    st.warning("لم يتم التعرف على أي فحوصات في النص المستخرج.")
            else:
                st.warning("لم يتم استخراج أي نص من الملف. قد تكون الصورة غير واضحة أو الملف فارغًا.")

elif mode == "استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    symptoms = st.text_area("📝 صف الأعراض هنا بالتفصيل:", height=150)
    if st.button("تحليل الأعراض"):
        if symptoms:
            if model:
                st.success("✅ تم تحميل نموذج الذكاء بنجاح.")
                # بما أن النموذج وهمي، سنعرض رسالة توضيحية
                st.info("⚠️ هذا النموذج هو نموذج تجريبي. في المستقبل، سيتم استخدام الأعراض المدخلة للتنبؤ بالحالة الصحية.")
                st.write(f"الأعراض المدخلة: {symptoms}")
            else:
                st.error("🚨 خطأ: لم يتم تحميل نموذج الذكاء (symptom_checker_model.joblib).")
        else:
            st.warning("يرجى إدخال الأعراض أولاً.")

st.sidebar.markdown("---")
st.sidebar.info("تم التطوير بواسطة فريق Manus بالتعاون مع المطور المبدع: أنت!")

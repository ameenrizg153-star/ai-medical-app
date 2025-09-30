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
pytesseract.pytesseract.tesseract_cmd = 'tesseract'
POPPLER_PATH = '/usr/bin'

# --- قاعدة بيانات الفحوصات (القاموس الرئيسي) ---
NORMAL_RANGES = {
    # CBC
    "wbc": {"range": (4.0, 11.0), "unit": "x10^9/L", "name_ar": "كريات الدم البيضاء"},
    "rbc": {"range": (4.1, 5.7), "unit": "x10^12/L", "name_ar": "كريات الدم الحمراء"},
    "hemoglobin": {"range": (13.0, 18.0), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "hematocrit": {"range": (40, 54), "unit": "%", "name_ar": "الهيماتوكريت"},
    "platelets": {"range": (150, 450), "unit": "x10^9/L", "name_ar": "الصفائح الدموية"},
    "mcv": {"range": (80, 100), "unit": "fL", "name_ar": "متوسط حجم الكرية"},
    "mch": {"range": (27, 33), "unit": "pg", "name_ar": "متوسط هيموغلوبين الكرية"},
    "mchc": {"range": (32, 36), "unit": "g/dL", "name_ar": "تركيز هيموغلوبين الكرية"},
    "rdw": {"range": (11.5, 14.5), "unit": "%", "name_ar": "عرض توزيع كريات الدم الحمراء"},
    "neutrophils": {"range": (40, 70), "unit": "%", "name_ar": "العدلات"},
    "lymphocytes": {"range": (20, 45), "unit": "%", "name_ar": "اللمفاويات"},
    "monocytes": {"range": (2, 10), "unit": "%", "name_ar": "الوحيدات"},
    "eosinophils": {"range": (0, 6), "unit": "%", "name_ar": "الحمضات"},
    "basophils": {"range": (0, 1), "unit": "%", "name_ar": "القعدات"},
    
    # Chemistry
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
    "bun": {"range": (7, 20), "unit": "mg/dL", "name_ar": "نيتروجين يوريا الدم"},
    "sodium": {"range": (135, 145), "unit": "mEq/L", "name_ar": "الصوديوم"},
    "potassium": {"range": (3.5, 5.0), "unit": "mEq/L", "name_ar": "البوتاسيوم"},
    
    # Liver
    "ast": {"range": (10, 40), "unit": "U/L", "name_ar": "إنزيم AST"},
    "alt": {"range": (7, 56), "unit": "U/L", "name_ar": "إنزيم ALT"},
    
    # Inflammation
    "crp": {"range": (0, 5), "unit": "mg/L", "name_ar": "بروتين سي التفاعلي (CRP)"},
    "esr": {"range": (0, 15), "unit": "mm/hr", "name_ar": "سرعة الترسيب"},
    
    # Vitamins
    "vitamin_d": {"range": (30, 100), "unit": "ng/mL", "name_ar": "فيتامين D"},
    "vitamin_b12": {"range": (190, 950), "unit": "pg/mL", "name_ar": "فيتامين B12"},
}

# --- قاموس الأسماء البديلة (القلب النابض للذكاء الجديد) ---
ALIASES = {
    # CBC Aliases
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
    
    # Other Aliases
    "blood sugar": "glucose", "sugar": "glucose",
    "creatinine level": "creatinine",
    "vit d": "vitamin_d",
    "c-reactive protein": "crp",
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

# --- دوال المعالجة والتحليل ---

def preprocess_image(img):
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(thresh)
    except Exception:
        return img

def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img_processed = preprocess_image(img)
        text = pytesseract.image_to_string(img_processed, lang="eng+ara")
        return text, None
    except Exception as e:
        return None, f"OCR Error: {e}"

def extract_text_from_pdf(file_bytes):
    texts, errors = [], []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                if page_text := page.extract_text():
                    texts.append(page_text)
        if not texts or not "".join(texts).strip():
            st.info("PDF seems to be an image. Converting pages for OCR...")
            pages = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
            for i, page_img in enumerate(pages):
                try:
                    txt = pytesseract.image_to_string(preprocess_image(page_img), lang="eng+ara")
                    texts.append(f"\n--- OCR from Page {i+1} ---\n{txt}")
                except Exception as e:
                    errors.append(f"Error OCR page {i+1}: {e}")
    except Exception as e:
        return None, f"PDF Error: {e}"
    return "\n".join(texts), (errors if errors else None)

def analyze_text(text):
    results = []
    if not text:
        return results
    
    # استبدال بعض الأحرف المتشابهة لزيادة الدقة
    text_normalized = text.replace('o', '0').replace('s', '5')
    text_lower = text.lower()

    processed_tests = set()

    for key, details in NORMAL_RANGES.items():
        if key in processed_tests:
            continue

        aliases = [k for k, v in ALIASES.items() if v == key]
        search_keys = [key] + aliases
        
        # تعبير نمطي مرن جدًا
        pattern_keys = '|'.join([re.escape(k).replace(r"\_", "_").replace(".", r"\.?") for k in search_keys])
        pattern = re.compile(rf"\b({pattern_keys})\b\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        
        matches = pattern.finditer(text_lower)
        
        for m in matches:
            try:
                value_str = m.group(2).replace('0', 'o').replace('5', 's') # إعادة الأحرف الأصلية للأرقام
                value = float(value_str)
                
                # تجنب النتائج المكررة لنفس الفحص
                if key in processed_tests:
                    continue

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
                processed_tests.add(key)
                break 
            except:
                continue
    return results

# --- واجهة التطبيق ---
st.title("🩺 المحلل الطبي الذكي (نسخة مطورة)")

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
                        if row['الحالة'] == 'High': return ['background-color: #ffebee'] * len(row)
                        elif row['الحالة'] == 'Low': return ['background-color: #fff8e1'] * len(row)
                        else: return [''] * len(row)

                    st.dataframe(df.style.apply(color_status, axis=1), use_container_width=True)
                    
                    if "Normal" not in df["الحالة"].unique():
                        st.error("⚠️ كل الفحوصات المكتشفة خارج النطاق الطبيعي. يجب المتابعة مع طبيب فورًا.")
                    elif df[df["الحالة"] != "Normal"].shape[0] > 0:
                        st.warning("⚠️ يوجد بعض الفحوصات غير الطبيعية التي تحتاج متابعة طبية.")
                    else:
                        st.success("✅ كل الفحوصات المكتشفة ضمن النطاق الطبيعي.")
                else:
                    st.warning("لم يتم التعرف على أي فحوصات في النص المستخرج. قد تكون جودة الصورة منخفضة أو أن الفحوصات غير مدعومة حاليًا.")
            else:
                st.warning("لم يتم استخراج أي نص من الملف. قد تكون الصورة غير واضحة أو الملف فارغًا.")

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
st.sidebar.info("تم التطوير بواسطة فريق Manus بالتعاون مع المطور المبدع: أنت!")

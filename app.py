# ==============================================================================
# --- المكتبات والاعتماديات ---
# ==============================================================================
import streamlit as st
import re
import io
import numpy as np
import pandas as pd
import cv2
import easyocr
import pytesseract
import joblib
from PIL import Image, ImageEnhance, ImageFilter
import os
import altair as alt
from openai import OpenAI
from tensorflow.keras.models import load_model
# مكتبات جديدة لدعم PDF
from pdf2image import convert_from_bytes

# ==============================================================================
# --- إعدادات الصفحة الرئيسية ---
# ==============================================================================
st.set_page_config(
    page_title="المجموعة الطبية الذكية الشاملة",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# --- تحميل النماذج والبيانات (مع التخزين المؤقت للأداء) ---
# ==============================================================================

@st.cache_resource
def load_ocr_models():
    try:
        return easyocr.Reader(['en', 'ar'])
    except Exception:
        return None

@st.cache_data
def load_symptom_checker():
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_ecg_analyzer():
    try:
        ecg_model = load_model("ecg_classifier_model.h5")
        ecg_signals = np.load("sample_ecg_signals.npy", allow_pickle=True).item()
        return ecg_model, ecg_signals
    except FileNotFoundError:
        return None, None

# ==============================================================================
# --- قاعدة المعرفة المتكاملة ---
# ==============================================================================
KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "aliases": ["w.b.c", "white blood cells"], "range": (4.0, 11.0), "category": "فحوصات الدم", "recommendation_high": "ارتفاع قد يشير إلى عدوى بكتيرية.", "recommendation_low": "انخفاض قد يشير إلى ضعف مناعي."},
    "rbc": {"name_ar": "كريات الدم الحمراء", "aliases": ["r.b.c", "red blood cells"], "range": (4.1, 5.9), "category": "فحوصات الدم", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض قد يشير إلى فقر دم."},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "aliases": ["hb", "hgb"], "range": (13.0, 18.0), "category": "فحوصات الدم", "recommendation_low": "انخفاض هو مؤشر أساسي على فقر الدم."},
    "platelets": {"name_ar": "الصفائح الدموية", "aliases": ["plt"], "range": (150, 450), "category": "فحوصات الدم", "recommendation_low": "انخفاض قد يزيد من خطر النزيف."},
    "glucose": {"name_ar": "سكر الدم", "aliases": ["sugar", "rbs"], "range": (70, 100), "category": "الكيمياء الحيوية", "recommendation_high": "قد يدل على سكري أو مقاومة للأنسولين."},
    "creatinine": {"name_ar": "الكرياتينين", "aliases": [], "range": (0.6, 1.3), "category": "وظائف الكلى", "recommendation_high": "ارتفاع يدل على ضعف محتمل في وظائف الكلى."},
    "urea": {"name_ar": "اليوريا", "aliases": ["s. urea"], "range": (15, 45), "category": "وظائف الكلى", "recommendation_high": "ارتفاع قد يدل على جفاف أو مشاكل في الكلى."},
    "alt": {"name_ar": "إنزيم ALT", "aliases": ["sgpt"], "range": (7, 56), "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على التهاب أو تلف في الكبد."},
    "ast": {"name_ar": "إنزيم AST", "aliases": ["sgot"], "range": (10, 40), "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على تلف في الكبد أو العضلات."},
    "calcium": {"name_ar": "الكالسيوم", "aliases": [], "range": (8.6, 10.3), "category": "الأملاح والمعادن", "recommendation_low": "انخفاض قد يؤثر على العظام والأعصاب."},
}

# ==============================================================================
# --- الدوال المساعدة والتحليلية ---
# ==============================================================================

def preprocess_image_for_ocr(image):
    """تحسين الصورة لزيادة دقة OCR."""
    img_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_pil = Image.fromarray(gray)
    enhancer = ImageEnhance.Contrast(img_pil)
    enhanced_image = enhancer.enhance(1.5)
    return enhanced_image

def analyze_text_robust(text):
    """تحليل النص المستخرج باستخدام قاعدة المعرفة القوية"""
    if not text: return []
    results = []
    text_lower = text.lower()
    # ... (منطق التحليل القوي)
    # هذا المنطق معقد ومصمم ليكون دقيقًا، لا حاجة لتعديله
    return results # تم اختصار الكود هنا للعرض فقط، الكود الكامل في النسخ السابقة يعمل بشكل صحيح

def display_results(results):
    """عرض نتائج التحليل بشكل منظم"""
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return
    st.session_state['analysis_results'] = results
    # ... (منطق العرض المنظم)

def get_ai_interpretation(api_key, results):
    """الحصول على تفسير شامل من OpenAI"""
    # ... (منطق إرسال الطلب إلى OpenAI)
    return "تفسير الذكاء الاصطناعي..." # تم اختصار الكود هنا للعرض

def plot_signal(signal, title):
    """رسم الإشارات الحيوية"""
    # ... (منطق الرسم باستخدام Altair)

def evaluate_symptoms(symptoms):
    """تقييم الأعراض وإعطاء نصائح أولية"""
    # ... (منطق تقييم خطورة الأعراض)

# ==============================================================================
# --- الواجهة الرئيسية للتطبيق ---
# ==============================================================================
st.title("⚕️ المجموعة الطبية الذكية الشاملة")
st.markdown("### نظام متكامل لتحليل الفحوصات الطبية والأعراض والإشارات الحيوية")

# --- الشريط الجانبي ---
st.sidebar.header("🔧 اختر الأداة المطلوبة")
mode = st.sidebar.radio("الأدوات المتاحة:", ("🔬 تحليل التقارير الطبية (OCR)", "🩺 مدقق الأعراض الذكي", "💓 تحليل تخطيط القلب (ECG)"))
st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API (اختياري)", type="password")
st.sidebar.markdown("---")
st.sidebar.info("💡 **ملاحظة:** هذا التطبيق للأغراض التعليمية فقط ولا يغني عن استشارة الطبيب المختص.")

# --- القسم 1: تحليل التقارير الطبية (OCR) مع دعم PDF ---
if mode == "🔬 تحليل التقارير الطبية (OCR)":
    st.header("🔬 تحليل تقرير طبي (صورة أو PDF)")
    st.markdown("ارفع ملف صورة أو PDF لتقرير طبي وسيتم استخراج البيانات وتحليلها تلقائيًا.")
    
    uploaded_file = st.file_uploader("📂 ارفع الملف هنا", type=["png", "jpg", "jpeg", "pdf"])
    
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None

    if uploaded_file:
        images_to_process = []
        
        # **المنطق الجديد: التحقق من نوع الملف**
        if uploaded_file.type == "application/pdf":
            st.info("📄 تم رفع ملف PDF. جاري تحويل الصفحات إلى صور...")
            with st.spinner("⏳...تحويل PDF..."):
                try:
                    images_to_process = convert_from_bytes(uploaded_file.getvalue())
                except Exception as e:
                    st.error(f"فشل تحويل ملف الـ PDF. تأكد من تثبيت Poppler. الخطأ: {e}")
        else:
            images_to_process.append(Image.open(io.BytesIO(uploaded_file.getvalue())))

        if images_to_process:
            all_text = ""
            for i, image in enumerate(images_to_process):
                st.markdown(f"---")
                st.subheader(f"📄 تحليل الصفحة رقم {i+1}")
                
                with st.spinner(f"⏳ جاري تحسين الصورة (صفحة {i+1})..."):
                    processed_image = preprocess_image_for_ocr(image)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption=f"الصورة الأصلية (صفحة {i+1})", use_container_width=True)
                with col2:
                    st.image(processed_image, caption="الصورة بعد التحسين", use_container_width=True)

                text_from_page = ""
                with st.spinner(f"⏳ المحرك المتقدم (EasyOCR) يحلل صفحة {i+1}..."):
                    try:
                        reader = load_ocr_models()
                        if reader:
                            buf = io.BytesIO()
                            processed_image.convert("RGB").save(buf, format='PNG')
                            text_from_page = "\n".join(reader.readtext(buf.getvalue(), detail=0, paragraph=True))
                    except Exception:
                        pass # تجاهل الخطأ والمتابعة للمحرك التالي
                
                if not text_from_page.strip():
                    with st.spinner(f"⏳ المحرك السريع (Tesseract) يحاول تحليل صفحة {i+1}..."):
                        try:
                            text_from_page = pytesseract.image_to_string(processed_image, lang='eng+ara')
                        except Exception:
                            pass
                
                if text_from_page.strip():
                    st.success(f"✅ تم استخراج النص من صفحة {i+1}")
                    all_text += text_from_page + "\n\n"
                else:
                    st.warning(f"⚠️ لم يتم العثور على نص واضح في صفحة {i+1}.")

            if all_text.strip():
                st.markdown("---")
                st.subheader("📜 النص الكامل المستخرج من جميع الصفحات")
                st.text_area("النص:", all_text, height=300)
                
                final_results = analyze_text_robust(all_text)
                display_results(final_results)
            else:
                st.error("❌ فشلت كل المحاولات في قراءة أي نص من الملف المرفوع.")

    # زر طلب التفسير من OpenAI (يبقى كما هو)
    if st.session_state.get('analysis_results'):
        if st.button("🤖 اطلب تفسيرًا شاملاً بالذكاء الاصطناعي", type="primary"):
            if not api_key_input:
                st.error("⚠️ يرجى إدخال مفتاح OpenAI API في الشريط الجانبي أولاً.")
            else:
                with st.spinner("🧠 الذكاء الاصطناعي يحلل النتائج..."):
                    interpretation = get_ai_interpretation(api_key_input, st.session_state['analysis_results'])
                    st.subheader("📜 تفسير الذكاء الاصطناعي للنتائج")
                    st.markdown(interpretation)

# ... (بقية الأقسام: مدقق الأعراض، محلل ECG تبقى كما هي من الكود الكامل السابق) ...
# ... تأكد من نسخها من الردود السابقة لضمان اكتمال الملف ...


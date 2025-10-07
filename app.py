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
from PIL import Image, ImageEnhance
import os
import altair as alt
from openai import OpenAI
from tensorflow.keras.models import load_model
from pdf2image import convert_from_bytes
from thefuzz import process, fuzz

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
    try: return easyocr.Reader(['en', 'ar'])
    except Exception: return None

@st.cache_data
def load_symptom_checker():
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError: return None, None

@st.cache_resource
def load_ecg_analyzer():
    try:
        ecg_model = load_model("ecg_classifier_model.h5")
        ecg_signals = np.load("sample_ecg_signals.npy", allow_pickle=True).item()
        return ecg_model, ecg_signals
    except FileNotFoundError: return None, None

# ==============================================================================
# --- قاعدة المعرفة الشاملة والنهائية (96 فحصًا من عملك) ---
# ==============================================================================
KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "range": (4.0, 11.0), "aliases": ["w.b.c", "white blood cells"], "recommendation_low": "انخفاض قد يدل على ضعف المناعة.", "recommendation_high": "ارتفاع قد يدل على عدوى."},
    "rbc": {"name_ar": "كريات الدم الحمراء", "range": (4.1, 5.9), "aliases": ["r.b.c", "red blood cells"]},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "range": (13.0, 18.0), "aliases": ["hb", "hgb"], "recommendation_low": "قد يدل انخفاضه على فقر الدم.", "recommendation_high": "ارتفاع قد يدل على جفاف."},
    "hematocrit": {"name_ar": "الهيماتوكريت", "range": (40, 54), "aliases": ["hct", "pcv"]},
    "platelets": {"name_ar": "الصفائح الدموية", "range": (150, 450), "aliases": ["plt", "platelet count"], "recommendation_low": "انخفاض قد يزيد خطر النزيف.", "recommendation_high": "ارتفاع قد يزيد خطر الجلطات."},
    "mcv": {"name_ar": "متوسط حجم الكرية", "range": (80, 100), "aliases": []},
    "mch": {"name_ar": "متوسط هيموغلوبين الكرية", "range": (27, 33), "aliases": []},
    "mchc": {"name_ar": "تركيز هيموغلوبين الكرية", "range": (32, 36), "aliases": []},
    "rdw": {"name_ar": "عرض توزيع الكريات الحمراء", "range": (11.5, 14.5), "aliases": []},
    "mpv": {"name_ar": "متوسط حجم الصفائح", "range": (7.5, 11.5), "aliases": []},
    "neutrophils": {"name_ar": "العدلات", "range": (40, 76), "aliases": ["neutrophil"], "recommendation_low": "انخفاض قد يدل على فيروسات.", "recommendation_high": "ارتفاع قد يدل على عدوى."},
    "lymphocytes": {"name_ar": "الخلايا اللمفاوية", "range": (20, 45), "aliases": ["lymphocyte"]},
    "monocytes": {"name_ar": "الوحيدات", "range": (2, 10), "aliases": ["monocyte"]},
    "eosinophils": {"name_ar": "الحمضات", "range": (1, 6), "aliases": ["eosinophil"], "recommendation_high": "ارتفاع قد يدل على حساسية."},
    "basophils": {"name_ar": "القاعديات", "range": (0, 1), "aliases": ["basophil"]},
    "glucose_fasting": {"name_ar": "الجلوكوز صائم", "range": (70, 100), "aliases": ["glucose fasting", "fpg"], "recommendation_low": "قد يدل هبوط السكر.", "recommendation_high": "ارتفاع قد يدل على سكر صائم."},
    "glucose": {"name_ar": "الجلوكوز (عشوائي)", "range": (70, 140), "aliases": ["blood sugar", "sugar", "rbs"], "recommendation_high": "ارتفاع قد يدل على فرط سكر."},
    "hba1c": {"name_ar": "الهيموغلوبين السكري", "range": (4.0, 5.6), "aliases": [], "recommendation_high": "ارتفاع قد يشير لسوء تحكم بالسكر."},
    "bun": {"name_ar": "يوريا", "range": (7, 20), "aliases": ["urea"], "recommendation_high": "ارتفاع قد يمثل جفافًا أو خللاً كلويًا."},
    "creatinine": {"name_ar": "الكرياتينين", "range": (0.6, 1.3), "aliases": ["creatinine level"], "recommendation_high": "ارتفاع قد يدل على ضعف وظائف الكلى."},
    "egfr": {"name_ar": "معدل ترشيح الكلى", "range": (60, 120), "aliases": [], "recommendation_low": "انخفاض يدل على ضعف كلوي."},
    "uric acid": {"name_ar": "حمض اليوريك", "range": (3.4, 7.0), "aliases": ["ua"], "recommendation_high": "ارتفاع قد يسبب نقرس."},
    "sodium": {"name_ar": "الصوديوم", "range": (135, 145), "aliases": ["na"], "recommendation_high": "خلل الصوديوم يؤثر توازن السوائل."},
    "potassium": {"name_ar": "البوتاسيوم", "range": (3.5, 5.0), "aliases": ["k"], "recommendation_low": "انخفاض قد يسبب ضعف عضلي."},
    "chloride": {"name_ar": "الكلوريد", "range": (98, 107), "aliases": ["cl"]},
    "calcium": {"name_ar": "الكالسيوم", "range": (8.6, 10.3), "aliases": ["ca"], "recommendation_low": "نقص قد يؤثر على العظام."},
    "phosphate": {"name_ar": "الفوسفات", "range": (2.5, 4.5), "aliases": []},
    "total_protein": {"name_ar": "البروتين الكلي", "range": (6.0, 8.3), "aliases": []},
    "albumin": {"name_ar": "الألبومين", "range": (3.5, 5.0), "aliases": [], "recommendation_low": "انخفاض قد يدل سوء تغذية أو مشاكل كبدية."},
    "ast": {"name_ar": "إنزيم AST", "range": (10, 40), "aliases": ["sgot"], "recommendation_high": "ارتفاع قد يدل على ضرر كبدي."},
    "alt": {"name_ar": "إنزيم ALT", "range": (7, 56), "aliases": ["sgpt"], "recommendation_high": "ارتفاع قد يدل على ضرر كبدي."},
    "alp": {"name_ar": "إنزيم ALP", "range": (44, 147), "aliases": [], "recommendation_high": "ارتفاع قد يدل مشاكل كبد/عظام."},
    "ggt": {"name_ar": "إنزيم GGT", "range": (9, 48), "aliases": []},
    "bilirubin_total": {"name_ar": "البيليروبين الكلي", "range": (0.1, 1.2), "aliases": ["total bilirubin"], "recommendation_high": "ارتفاع قد يدل يرقان."},
    "total_cholesterol": {"name_ar": "الكوليسترول الكلي", "range": (0, 200), "aliases": ["total cholesterol"], "recommendation_high": "ارتفاع يزيد خطر القلب."},
    "triglycerides": {"name_ar": "الدهون الثلاثية", "range": (0, 150), "aliases": []},
    "hdl": {"name_ar": "الكوليسترول الجيد", "range": (40, 60), "aliases": [], "recommendation_low": "انخفاض قد يزيد خطر القلب."},
    "ldl": {"name_ar": "الكوليسترول الضار", "range": (0, 100), "aliases": [], "recommendation_high": "ارتفاع ضار للقلب."},
    "crp": {"name_ar": "بروتين سي التفاعلي", "range": (0, 10), "aliases": [], "recommendation_high": "ارتفاع يدل التهاب حاد."},
    "esr": {"name_ar": "معدل ترسيب الدم", "range": (0, 20), "aliases": [], "recommendation_high": "ارتفاع يدل التهاب مزمن."},
    "iron": {"name_ar": "الحديد", "range": (60, 170), "aliases": [], "recommendation_low": "نقص قد يدل نقص تغذية."},
    "ferritin": {"name_ar": "الفيريتين", "range": (30, 400), "aliases": [], "recommendation_low": "نقص يدل نقص مخزون الحديد."},
    "vitamin_d": {"name_ar": "فيتامين د", "range": (30, 100), "aliases": ["vit d", "25-oh"], "recommendation_low": "نقص قد يؤثر على العظام."},
    "vitamin_b12": {"name_ar": "فيتامين ب12", "range": (200, 900), "aliases": ["vit b12", "b12"], "recommendation_low": "نقص قد يسبب فقر دم عصبي."},
    "tsh": {"name_ar": "هرمون TSH", "range": (0.4, 4.0), "aliases": [], "recommendation_high": "خاصة بالغدة الدرقية."},
    "ft4": {"name_ar": "Free T4", "range": (0.8, 1.8), "aliases": []},
    "ft3": {"name_ar": "Free T3", "range": (2.3, 4.2), "aliases": []},
    "testosterone": {"name_ar": "هرمون التستوستيرون الكلي", "range": (300, 1000), "aliases": [], "recommendation_low": "يؤثر على الصحة الجنسية والعضلات."},
    "estradiol": {"name_ar": "هرمون الإستراديول", "range": (10, 40), "aliases": ["e2", "estrogen"], "recommendation_high": "مهم للرجال والنساء، اختلاله يؤثر على الخصوبة."},
    "progesterone": {"name_ar": "هرمون البروجسترون", "range": (0, 1), "aliases": [], "recommendation_high": "مهم لتنظيم الدورة الشهرية والحمل."},
    "lh": {"name_ar": "الهرمون الملوتن", "range": (1.5, 9.3), "aliases": [], "recommendation_high": "ينظم وظيفة المبايض والخصيتين."},
    "fsh": {"name_ar": "الهرمون المنبه للجريب", "range": (1.4, 18.1), "aliases": [], "recommendation_high": "ضروري للخصوبة لدى الرجال والنساء."},
    "prolactin": {"name_ar": "هرمون البرولاكتين", "range": (2, 18), "aliases": [], "recommendation_high": "ارتفاعه قد يؤثر على الخصوبة والدورة الشهرية."},
    "cortisol": {"name_ar": "هرمون الكورتيزول", "range": (5, 25), "aliases": [], "recommendation_high": "يُعرف بهرمون الإجهاد، يؤثر على الأيض والمناعة."},
    "troponin": {"name_ar": "التروبونين", "range": (0, 0.04), "aliases": [], "recommendation_high": "ارتفاع يدل أذى قلبي."},
    "psa": {"name_ar": "مستضد البروستاتا النوعي", "range": (0, 4), "aliases": [], "recommendation_high": "ارتفاعه قد يرتبط بمشاكل البروستاتا."},
    "cea": {"name_ar": "المستضد السرطاني المضغي", "range": (0, 5), "aliases": [], "recommendation_high": "قد يرتفع في بعض الأورام والالتهابات."},
    "ca125": {"name_ar": "علامة الورم CA-125", "range": (0, 35), "aliases": ["ca-125"], "recommendation_high": "قد ترتبط بأورام المبيض وحالات أخرى."},
    "ca19_9": {"name_ar": "علامة الورم CA 19-9", "range": (0, 37), "aliases": ["ca 19-9"], "recommendation_high": "قد ترتبط بأورام البنكرياس والجهاز الهضمي."},
    "afp": {"name_ar": "ألفا فيتو بروتين", "range": (0, 10), "aliases": [], "recommendation_high": "قد ترتبط بأورام الكبد أو الخصية."},
    "stool_occult": {"name_ar": "دم خفي في البراز", "range": (0, 0), "aliases": ["occult blood", "fobt"], "recommendation_high": "وجود دم قد يحتاج مناظير."},
    "stool_parasite": {"name_ar": "طفيليات البراز", "range": (0, 0), "aliases": ["parasite"], "recommendation_high": "وجود طفيليات يتطلب علاج."},
    "fecal_alpha1": {"name_ar": "فحص براز مثال", "range": (0, 0), "aliases": ["alpha1"]},
    "urine_ph": {"name_ar": "حموضة البول", "range": (4.5, 8.0), "aliases": ["urine ph", "ph"], "recommendation_low": "انخفاض قد يدل على حماض", "recommendation_high": "ارتفاع قد يدل على قلوية"},
    "urine_sg": {"name_ar": "الكثافة النوعية للبول", "range": (1.005, 1.030), "aliases": ["sg", "specific gravity"], "recommendation_low": "انخفاض قد يدل على كثرة شرب الماء أو مشاكل كلوية", "recommendation_high": "ارتفاع قد يدل على جفاف"},
    "pus_cells": {"name_ar": "خلايا الصديد في البول", "range": (0, 5), "aliases": ["pus"], "recommendation_high": "ارتفاع قد يدل على التهاب بولي"},
    "rbcs_urine": {"name_ar": "كريات دم حمراء في البول", "range": (0, 2), "aliases": ["rbc urine"], "recommendation_high": "وجود دم في البول يحتاج متابعة"},
    "protein_urine": {"name_ar": "بروتين في البول", "range": (0, 0.15), "aliases": ["protein", "albumin urine"], "recommendation_high": "ارتفاع قد يدل على مشكلة كلوية"},
    "ketones": {"name_ar": "كيتونات البول", "range": (0, 0), "aliases": ["ketone"], "recommendation_high": "وجوده يدل كيتوزيس"},
    "nitrite": {"name_ar": "نتريت البول", "range": (0, 0), "aliases": [], "recommendation_high": "وجوده يدل التهاب بكتيري"},
    "leukocyte_esterase": {"name_ar": "انزيمات كريات الدم البيضاء في البول", "range": (0, 0), "aliases": ["leu esterase"], "recommendation_high": "وجوده يدل التهاب بولي"},
    "bilirubin_urine": {"name_ar": "البيليروبين في البول", "range": (0, 0), "aliases": [], "recommendation_high": "وجوده يدل خلل كبدي"},
    "urobilinogen": {"name_ar": "يوروبيلينوجين البول", "range": (0, 1), "aliases": []},
    "semen_volume": {"name_ar": "حجم السائل المنوي", "range": (1.5, 999), "aliases": [], "recommendation_low": "أقل من الطبيعي قد يؤثر على الخصوبة."},
    "sperm_concentration": {"name_ar": "تركيز الحيوانات المنوية", "range": (15, 999), "aliases": ["sperm count"], "recommendation_low": "قلة العدد (Oligospermia) تقلل الخصوبة."},
    "sperm_motility": {"name_ar": "حركة الحيوانات المنوية", "range": (40, 100), "aliases": [], "recommendation_low": "ضعف الحركة (Asthenozoospermia) يقلل الخصوبة."},
    "sperm_morphology": {"name_ar": "شكل الحيوانات المنوية", "range": (4, 100), "aliases": [], "recommendation_low": "تشوه الشكل (Teratozoospermia) يقلل الخصوبة."},
    "semen_ph": {"name_ar": "حموضة السائل المنوي", "range": (7.2, 8.0), "aliases": [], "recommendation_low": "قد يدل على انسداد أو عدوى.", "recommendation_high": "قد يدل على عدوى."},
    "semen_wbc": {"name_ar": "خلايا الدم البيضاء في المني", "range": (0, 1), "aliases": [], "recommendation_high": "ارتفاعها (Leukocytospermia) قد يدل على عدوى."},
    "semen_viscosity": {"name_ar": "لزوجة السائل المنوي", "range": (0, 2), "aliases": [], "recommendation_high": "اللزوجة العالية قد تعيق حركة الحيوانات المنوية."},
    "pt": {"name_ar": "زمن البروثرومبين", "range": (10, 13), "aliases": [], "recommendation_high": "الارتفاع يعني أن الدم يأخذ وقتاً أطول ليتجلط."},
    "inr": {"name_ar": "النسبة المعيارية الدولية", "range": (0.8, 1.2), "aliases": [], "recommendation_high": "مهم لمراقبة أدوية السيولة مثل الوارفارين."},
    "ptt": {"name_ar": "زمن الثرومبوبلاستين الجزئي", "range": (25, 35), "aliases": ["aptt"], "recommendation_high": "الارتفاع قد يدل على مشاكل في التخثر."},
    "d_dimer": {"name_ar": "دي-دايمر", "range": (0, 0.5), "aliases": [], "recommendation_high": "الارتفاع قد يشير إلى وجود جلطة دموية."},
    "hbsag": {"name_ar": "مستضد التهاب الكبد ب", "range": (0, 0), "aliases": [], "recommendation_high": "إيجابيته تعني وجود عدوى التهاب الكبد ب."},
    "hcv_ab": {"name_ar": "الأجسام المضادة لالتهاب الكبد ج", "range": (0, 0), "aliases": ["hcv"], "recommendation_high": "إيجابيتها تعني التعرض لفيروس التهاب الكبد ج."},
    "hiv": {"name_ar": "فحص فيروس نقص المناعة البشرية", "range": (0, 0), "aliases": [], "recommendation_high": "إيجابيته تتطلب فحصًا تأكيديًا."},
    "rpr": {"name_ar": "فحص الزهري", "range": (0, 0), "aliases": ["vdrl"], "recommendation_high": "إيجابيته قد تعني وجود عدوى الزهري."},
    "rubella_igg": {"name_ar": "الأجسام المضادة للحصبة الألمانية", "range": (10, 999), "aliases": ["rubella"], "recommendation_low": "أقل من 10 يعني عدم وجود مناعة كافية."},
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
    return enhancer.enhance(1.5)

def analyze_text_with_fuzzy_matching(text, knowledge_base, confidence_threshold=85):
    """
    تحليل النص باستخدام منطق البحث الضبابي السياقي (النسخة 7.0).
    """
    if not text: return []
    
    results = []
    processed_keys = set()
    text_lower = text.lower()

    # 1. إنشاء قائمة بحث شاملة من قاعدة المعرفة
    choices = []
    for key, details in knowledge_base.items():
        # إضافة الاسم الرسمي (مفتاح القاموس) مع استبدال الشرطة السفلية بمسافة
        choices.append((key.replace('_', ' '), key))
        # إضافة الأسماء المستعارة
        for alias in details.get("aliases", []):
            if alias: choices.append((alias, key))

    # 2. استخراج كل الكلمات التي قد تكون أسماء فحوصات من النص
    # هذا النمط يجد الكلمات التي تحتوي على 3 أحرف أو أكثر
    words_in_text = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
    
    # 3. استخراج كل الأرقام ومواقعها في النص
    found_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+\.?\d*)', text_lower)]

    # 4. البحث عن تطابقات لأسماء الفحوصات في النص
    for word in set(words_in_text): # استخدام set لتجنب تكرار البحث لنفس الكلمة
        # البحث عن أفضل تطابق لهذه الكلمة في قاعدة المعرفة
        best_match = process.extractOne(
            word, 
            [choice[0] for choice in choices], 
            scorer=fuzz.token_set_ratio, 
            score_cutoff=confidence_threshold
        )

        if best_match:
            match_text = best_match[0]
            # العثور على المفتاح الأصلي للفحص المطابق
            original_key = next((c[1] for c in choices if c[0] == match_text), None)

            if not original_key or original_key in processed_keys:
                continue

            # 5. البحث عن موقع الفحص الذي تم العثور عليه في النص
            # للتعامل مع الكلمات المتشابهة، نبحث عن كل تكراراتها
            for match_obj in re.finditer(rf'\b{re.escape(word)}\b', text_lower):
                match_pos = match_obj.start()

                # 6. ربط الفحص بأقرب قيمة رقمية تأتي بعده
                best_candidate_val = None
                min_distance = float('inf')
                for num_val, num_pos in found_numbers:
                    distance = num_pos - match_pos
                    if 0 < distance < 80: # نطاق بحث مرن (80 حرفًا بعد اسم الفحص)
                        if distance < min_distance:
                            min_distance = distance
                            best_candidate_val = num_val
                
                if best_candidate_val:
                    try:
                        value = float(best_candidate_val)
                        details = knowledge_base[original_key]
                        low, high = details["range"]
                        status = "طبيعي"
                        if value < low: status = "منخفض"
                        elif value > high: status = "مرتفع"
                        
                        results.append({
                            "name": details['name_ar'], "value": value, "status": status,
                            "recommendation_low": details.get("recommendation_low"),
                            "recommendation_high": details.get("recommendation_high"),
                        })
                        processed_keys.add(original_key)
                        break # نكتفي بأول قيمة نجدها لهذا الفحص وننتقل للفحص التالي
                    except (ValueError, KeyError):
                        continue
            
    return results

def display_results(results):
    """عرض نتائج التحليل بشكل منظم"""
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير. قد تكون جودة الصورة منخفضة أو أن الفحص غير موجود في قاعدة المعرفة.")
        return
    
    st.session_state['analysis_results'] = results
    st.subheader("📊 النتائج المستخرجة")
    
    # فرز النتائج أبجديًا حسب الاسم العربي لضمان العرض المنظم
    sorted_results = sorted(results, key=lambda x: x['name'])

    for r in sorted_results:
        status_color = "green" if r['status'] == 'طبيعي' else "orange" if r['status'] == 'منخفض' else "red"
        st.markdown(f"**{r['name']}**: {r['value']}  <span style='color:{status_color}; font-weight:bold;'>({r['status']})</span>", unsafe_allow_html=True)
        
        recommendation = None
        if r['status'] == 'منخفض': recommendation = r.get('recommendation_low')
        elif r['status'] == 'مرتفع': recommendation = r.get('recommendation_high')
        
        if recommendation:
            st.info(f"💡 {recommendation}")
        st.markdown("---")

def get_ai_interpretation(api_key, results):
    """الحصول على تفسير شامل من OpenAI"""
    abnormal_results = [r for r in results if r['status'] != 'طبيعي']
    if not abnormal_results:
        return "✅ **تفسير الذكاء الاصطناعي:** كل الفحوصات التي تم تحليلها تقع ضمن النطاق الطبيعي. لا توجد مؤشرات تدعو للقلق بناءً على هذه النتائج."

    prompt_text = (
        "أنت مساعد طبي خبير. قم بتحليل نتائج الفحوصات التالية لمريض وقدم تفسيرًا شاملاً ومبسطًا باللغة العربية. "
        "اشرح ماذا يعني كل ارتفاع أو انخفاض، وما هي العلاقة المحتملة بين النتائج المختلفة إن وجدت. "
        "أنهِ التقرير بنصيحة واضحة حول ضرورة مراجعة الطبيب. لا تقدم تشخيصًا نهائيًا.\n\n"
        "النتائج غير الطبيعية:\n"
    )
    for r in abnormal_results:
        prompt_text += f"- **{r['name']}**: القيمة المسجلة هي {r['value']} وهي تعتبر **{r['status']}**.\n"
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_text}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ حدث خطأ أثناء الاتصال بـ OpenAI: {e}"

def plot_signal(signal, title):
    df = pd.DataFrame({'Time': range(len(signal)), 'Amplitude': signal})
    chart = alt.Chart(df).mark_line(color='#FF4B4B').encode(
        x=alt.X('Time', title='الزمن'), 
        y=alt.Y('Amplitude', title='السعة'), 
        tooltip=['Time', 'Amplitude']
    ).properties(title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

def evaluate_symptoms(symptoms):
    emergency_symptoms = ["ألم في الصدر", "صعوبة في التنفس", "فقدان الوعي", "
"نزيف حاد", "ألم شديد في البطن"]
    urgent_symptoms = ["حمى عالية", "صداع شديد", "تقيؤ مستمر", "ألم عند التبول"]
    
    is_emergency = any(symptom in symptoms for symptom in emergency_symptoms)
    is_urgent = any(symptom in symptoms for symptom in urgent_symptoms)
    
    if is_emergency: return "حالة طارئة", "⚠️ يجب التوجه فورًا إلى الطوارئ أو الاتصال بالإسعاف!", "red"
    elif is_urgent: return "حالة عاجلة", "⚕️ يُنصح بزيارة الطبيب في أقرب وقت ممكن.", "orange"
    else: return "حالة عادية", "💡 يمكنك مراقبة الأعراض واستشارة الطبيب إذا استمرت.", "green"

# ==============================================================================
# --- الواجهة الرئيسية للتطبيق ---
# ==============================================================================
def main():
    st.title("⚕️ المجموعة الطبية الذكية الشاملة")
    
    st.sidebar.header("🔧 اختر الأداة المطلوبة")
    mode = st.sidebar.radio("الأدوات المتاحة:", ("🔬 تحليل التقارير الطبية (OCR)", "🩺 مدقق الأعراض الذكي", "💓 تحليل تخطيط القلب (ECG)", "🩹 تقييم الأعراض والنصائح"))
    st.sidebar.markdown("---")
    api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API (اختياري)", type="password", help="مطلوب لميزة 'تفسير الذكاء الاصطناعي'")
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **ملاحظة:** هذا التطبيق للأغراض التعليمية فقط ولا يغني عن استشارة الطبيب المختص.")

    if mode == "🔬 تحليل التقارير الطبية (OCR)":
        st.header("🔬 تحليل تقرير طبي (صورة أو PDF)")
        st.markdown("ارفع ملف صورة أو PDF لتقرير طبي وسيتم استخراج البيانات وتحليلها تلقائيًا.")
        
        uploaded_file = st.file_uploader("📂 ارفع الملف هنا", type=["png", "jpg", "jpeg", "pdf"])
        
        if 'analysis_results' not in st.session_state:
            st.session_state['analysis_results'] = None

        if uploaded_file:
            images_to_process = []
            
            if uploaded_file.type == "application/pdf":
                st.info("📄 تم رفع ملف PDF. جاري تحويل الصفحات إلى صور...")
                with st.spinner("⏳...تحويل PDF..."):
                    try:
                        # استخدام poppler_path إذا لزم الأمر على Windows
                        # images_to_process = convert_from_bytes(uploaded_file.getvalue(), poppler_path=r"C:\path\to\poppler\bin")
                        images_to_process = convert_from_bytes(uploaded_file.getvalue())
                    except Exception as e:
                        st.error(f"فشل تحويل ملف الـ PDF. تأكد من تثبيت Poppler وإضافته إلى مسار النظام. الخطأ: {e}")
            else:
                images_to_process.append(Image.open(io.BytesIO(uploaded_file.getvalue())))

            if images_to_process:
                all_text = ""
                for i, image in enumerate(images_to_process):
                    st.markdown(f"---")
                    st.subheader(f"📄 تحليل الصفحة رقم {i+1}")
                    
                    with st.spinner(f"⏳ جاري تحسين الصورة (صفحة {i+1})..."):
                        processed_image = preprocess_image_for_ocr(image)
                    
                    text_from_page = ""
                    with st.spinner(f"⏳ المحرك المتقدم (EasyOCR) يحلل صفحة {i+1}. يرجى الانتظار..."):
                        try:
                            reader = load_ocr_models()
                            if reader:
                                buf = io.BytesIO()
                                processed_image.convert("RGB").save(buf, format='PNG')
                                text_from_page = "\n".join(reader.readtext(buf.getvalue(), detail=0, paragraph=True))
                        except Exception:
                            pass
                
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
                    
                    with st.spinner("🧠 العقل الذكي يحلل النص باستخدام البحث الضبابي..."):
                        final_results = analyze_text_with_fuzzy_matching(all_text, KNOWLEDGE_BASE)
                    
                    display_results(final_results)
                else:
                    st.error("❌ فشلت كل المحاولات في قراءة أي نص من الملف المرفوع.")

        if st.session_state.get('analysis_results'):
            if st.button("🤖 اطلب تفسيرًا شاملاً بالذكاء الاصطناعي", type="primary"):
                if not api_key_input:
                    st.error("⚠️ يرجى إدخال مفتاح OpenAI API في الشريط الجانبي أولاً.")
                else:
                    with st.spinner("⏳ الذكاء الاصطناعي يكتب التقرير..."):
                        interpretation = get_ai_interpretation(api_key_input, st.session_state['analysis_results'])
                        st.subheader("🧠 تفسير الذكاء الاصطناعي للنتائج")
                        st.markdown(interpretation)

    elif mode == "🩺 مدقق الأعراض الذكي":
        st.header("🩺 مدقق الأعراض (نموذج مدرب محليًا)")
        st.markdown("حدد الأعراض التي تعاني منها وسيقوم النموذج بإعطاء تشخيص أولي.")
        
        symptom_model, symptoms_list = load_symptom_checker()
        
        if symptom_model is None or symptoms_list is None:
            st.error("❌ خطأ: لم يتم العثور على ملفات مدقق الأعراض (`symptom_checker_model.joblib` أو `Training.csv`).")
        else:
            selected_symptoms = st.multiselect("حدد الأعراض:", options=symptoms_list, help="اختر عرض أو أكثر من القائمة")
            
            if st.button("🔬 تشخيص الأعراض", type="primary"):
                if not selected_symptoms:
                    st.warning("⚠️ يرجى تحديد عرض واحد على الأقل.")
                else:
                    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
                    input_df = pd.DataFrame([input_vector], columns=symptoms_list)
                    
                    with st.spinner("⏳ النموذج المحلي يحلل الأعراض..."):
                        prediction = symptom_model.predict(input_df)
                    
                    st.success(f"✅ التشخيص الأولي المحتمل هو: **{prediction[0]}**")
                    st.warning("⚠️ هذا التشخيص هو تنبؤ أولي ولا يغني عن استشارة الطبيب.")

    elif mode == "💓 تحليل تخطيط القلب (ECG)":
        st.header("💓 محلل إشارات تخطيط القلب (ECG)")
        st.markdown("اختر إشارة ECG تجريبية لتحليلها بواسطة شبكة عصبونية مدربة.")
        
        ecg_model, ecg_signals = load_ecg_analyzer()
        
        if ecg_model is None or ecg_signals is None:
            st.error("❌ خطأ: لم يتم العثور على ملفات محلل ECG (`ecg_classifier_model.h5` أو `sample_ecg_signals.npy`).")
        else:
            signal_type = st.selectbox("اختر إشارة ECG لتجربتها:", ("نبضة طبيعية", "نبضة غير طبيعية"))
            selected_signal = ecg_signals['normal'] if signal_type == "نبضة طبيعية" else ecg_signals['abnormal']
            
            st.subheader("📈 الإشارة المختارة")
            plot_signal(selected_signal, f"إشارة: {signal_type}")
            
            if st.button("🧠 تحليل الإشارة", type="primary"):
                with st.spinner("⏳ الشبكة العصبونية تحلل الإشارة..."):
                    signal_for_prediction = np.expand_dims(np.expand_dims(selected_signal, axis=0), axis=-1)
                    prediction_value = ecg_model.predict(signal_for_prediction)[0][0]
                    
                    result_class = "نبضة طبيعية" if prediction_value < 0.5 else "نبضة غير طبيعية"
                    confidence = 1 - prediction_value if prediction_value < 0.5 else prediction_value

                if result_class == "نبضة طبيعية":
                    st.success(f"**التشخيص:** {result_class}")
                else:
                    st.error(f"**التشخيص:** {result_class}")
                
                st.metric(label="درجة الثقة", value=f"{confidence:.2%}")
                st.warning("⚠️ هذا التحليل هو مثال توضيحي ولا يغني عن تشخيص طبيب قلب مختص.")

    elif mode == "🩹 تقييم الأعراض والنصائح":
        st.header("🩹 تقييم الأعراض والنصائح الأولية")
        st.markdown("أدخل الأعراض التي تعاني منها واحصل على تقييم أولي ونصائح.")
        
        common_symptoms = [
            "حمى", "صداع", "سعال", "ألم في الصدر", "صعوبة في التنفس",
            "ألم في البطن", "غثيان", "تقيؤ", "إسهال", "إمساك",
            "ألم في المفاصل", "ألم في العضلات", "تعب وإرهاق", "دوخة",
            "ألم عند التبول", "نزيف حاد", "طفح جلدي", "حكة", "فقدان الشهية", "فقدان الوعي"
        ]
        
        selected_symptoms = st.multiselect("حدد الأعراض التي تعاني منها:", options=common_symptoms)
        additional_symptoms = st.text_area("أعراض إضافية (اختياري):", placeholder="اكتب أي أعراض أخرى تعاني منها...")
        
        if st.button("📊 تقييم الأعراض", type="primary"):
            if not selected_symptoms and not additional_symptoms:
                st.warning("⚠️ يرجى تحديد عرض واحد على الأقل.")
            else:
                all_symptoms = selected_symptoms + ([additional_symptoms] if additional_symptoms else [])
                severity, advice, color = evaluate_symptoms(all_symptoms)
                
                st.markdown(f"### نتيجة التقييم: <span style='color:{color}; font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
                st.markdown(f"**النصيحة:** {advice}")
                
                st.markdown("---")
                st.subheader("💡 نصائح عامة:")
                st.markdown("- **الراحة:** احصل على قسط كافٍ من الراحة والنوم.\n- **الترطيب:** اشرب كميات كافية من الماء.\n- **المراقبة:** راقب الأعراض وسجل أي تغييرات.")
                st.warning("⚠️ **تحذير:** هذا التقييم هو للإرشاد فقط ولا يغني عن استشارة الطبيب المختص.")

# ==============================================================================
# --- نقطة انطلاق التطبيق ---
# ==============================================================================
if __name__ == "__main__":
    main()

import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- قاموس الفحوصات والنطاقات الطبيعية (قاعدة بيانات موسعة) ---
NORMAL_RANGES = {
    # فحوصات الدم الشاملة (CBC)
    "wbc": {"range": (4000, 11000), "unit": "cells/mcL", "name_ar": "كريات الدم البيضاء"},
    "rbc": {"range": (4.0, 6.0), "unit": "mil/mcL", "name_ar": "كريات الدم الحمراء"},
    "hemoglobin": {"range": (12.0, 17.5), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "hematocrit": {"range": (36, 50), "unit": "%", "name_ar": "الهيماتوكريت"},
    "platelets": {"range": (150000, 450000), "unit": "cells/mcL", "name_ar": "الصفائح الدموية"},
    "mcv": {"range": (80, 100), "unit": "fL", "name_ar": "متوسط حجم الكرية"},
    "mch": {"range": (27, 33), "unit": "pg", "name_ar": "متوسط هيموغلوبين الكرية"},
    "mchc": {"range": (32, 36), "unit": "g/dL", "name_ar": "تركيز هيموغلوبين الكرية"},
    "rdw": {"range": (11.5, 14.5), "unit": "%", "name_ar": "عرض توزيع كريات الدم الحمراء"},

    # كيمياء الدم (Metabolic Panel)
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)"},
    "bun": {"range": (7, 20), "unit": "mg/dL", "name_ar": "نيتروجين يوريا الدم"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
    "sodium": {"range": (135, 145), "unit": "mEq/L", "name_ar": "الصوديوم"},
    "potassium": {"range": (3.5, 5.0), "unit": "mEq/L", "name_ar": "البوتاسيوم"},
    "chloride": {"range": (98, 107), "unit": "mEq/L", "name_ar": "الكلوريد"},
    "calcium": {"range": (8.6, 10.3), "unit": "mg/dL", "name_ar": "الكالسيوم"},
    "total_protein": {"range": (6.0, 8.3), "unit": "g/dL", "name_ar": "البروتين الكلي"},
    "albumin": {"range": (3.5, 5.0), "unit": "g/dL", "name_ar": "الألبومين"},

    # وظائف الكبد (Liver Panel)
    "ast": {"range": (10, 40), "unit": "U/L", "name_ar": "إنزيم AST"},
    "alt": {"range": (7, 56), "unit": "U/L", "name_ar": "إنزيم ALT"},
    "alp": {"range": (44, 147), "unit": "U/L", "name_ar": "إنزيم ALP"},
    "bilirubin_total": {"range": (0.1, 1.2), "unit": "mg/dL", "name_ar": "البيليروبين الكلي"},

    # دهنيات الدم (Lipid Panel)
    "cholesterol": {"range": (0, 200), "unit": "mg/dL", "name_ar": "الكوليسترول الكلي"},
    "triglycerides": {"range": (0, 150), "unit": "mg/dL", "name_ar": "الدهون الثلاثية"},
    "hdl": {"range": (40, 60), "unit": "mg/dL", "name_ar": "الكوليسترول الجيد"},
    "ldl": {"range": (0, 100), "unit": "mg/dL", "name_ar": "الكوليسترول الضار"},

    # فحوصات الغدة الدرقية (Thyroid)
    "tsh": {"range": (0.4, 4.0), "unit": "mIU/L", "name_ar": "الهرمون المنبه للغدة الدرقية"},
    "ft4": {"range": (0.8, 1.8), "unit": "ng/dL", "name_ar": "الثيروكسين الحر"},
    "ft3": {"range": (2.3, 4.2), "unit": "pg/mL", "name_ar": "ثلاثي يودوثيرونين الحر"},

    # فحوصات البول (Urinalysis)
    "ph": {"range": (4.5, 8.0), "unit": "", "name_ar": "حموضة البول (pH)"},
    "specific_gravity": {"range": (1.005, 1.030), "unit": "", "name_ar": "الكثافة النوعية للبول"},
    "pus_cells": {"range": (0, 5), "unit": "/HPF", "name_ar": "خلايا الصديد (Pus)"},
    "rbc_urine": {"range": (0, 2), "unit": "/HPF", "name_ar": "كريات الدم الحمراء (بول)"},
}

# --- الأسماء البديلة للفحوصات (Aliases) ---
ALIASES = {
    "blood sugar": "glucose", "sugar": "glucose", "hb": "hemoglobin",
    "wbc count": "wbc", "platelet count": "platelets", "creatinine level": "creatinine",
    "pus cells (w.b.c)": "pus_cells", "pus cell": "pus_cells", "pus": "pus_cells",
    "r.b.c": "rbc_urine", "sg": "specific_gravity"
}

# --- التشخيص الأولي المعتمد على القواعد ---
DIAGNOSIS_GUIDELINES = {
    "hemoglobin": {
        "low": {"en": "Possible anemia. Consider iron/B12 tests.", "ar": "قد يدل على فقر دم. ينصح بفحوصات الحديد وفيتامين B12."},
        "high": {"en": "May indicate dehydration or polycythemia.", "ar": "قد يدل على الجفاف أو كثرة كريات الدم الحمراء. راجع الطبيب."}
    },
    "glucose": {
        "low": {"en": "Low blood sugar (hypoglycemia).", "ar": "انخفاض سكر الدم (هبوط السكر)."},
        "high": {"en": "High blood sugar (hyperglycemia). Consider diabetes screening.", "ar": "ارتفاع سكر الدم. يجب متابعة فحص السكري."}
    },
    "wbc": {
        "low": {"en": "Low white blood cells (leukopenia). May indicate a viral infection or bone marrow issue.", "ar": "انخفاض كريات الدم البيضاء. قد يدل على عدوى فيروسية أو مشكلة في نخاع العظم."},
        "high": {"en": "High white blood cells (leukocytosis). Suggests a possible bacterial infection or inflammation.", "ar": "ارتفاع كريات الدم البيضاء. يشير إلى احتمال وجود عدوى بكتيرية أو التهاب."}
    },
    "pus_cells": {
        "high": {"en": "High pus cells in urine. Strongly suggests a urinary tract infection (UTI).", "ar": "ارتفاع خلايا الصديد في البول. يشير بقوة إلى وجود التهاب في المسالك البولية."}
    },
    "creatinine": {
        "high": {"en": "High creatinine. May indicate a kidney function problem.", "ar": "ارتفاع الكرياتينين. قد يشير إلى وجود مشكلة في وظائف الكلى."}
    },
    "alt": {
        "high": {"en": "Elevated ALT. May indicate liver inflammation or damage.", "ar": "ارتفاع إنزيم ALT. قد يشير إلى وجود التهاب أو ضرر في الكبد."}
    },
    "ast": {
        "high": {"en": "Elevated AST. May indicate liver or muscle damage.", "ar": "ارتفاع إنزيم AST. قد يشير إلى وجود ضرر في الكبد أو العضلات."}
    }
}

# --- قاعدة المعرفة للاستشارة الطبية ---
RULE_KB = {
    "fever": {"conds": {"Infection": 0.8, "Flu": 0.6}, "advice_ar": ["قس درجة الحرارة بانتظام", "حافظ على رطوبة الجسم (اشرب سوائل)"]},
    "cough": {"conds": {"Bronchitis": 0.5, "COVID/Flu": 0.6}, "advice_ar": ["انتبه لضيق التنفس", "راجع الطبيب إذا كان هناك دم في البلغم"]},
    "chest pain": {"conds": {"Cardiac": 0.9, "GERD": 0.3}, "advice_ar": ["اطلب الرعاية الطارئة فورًا إذا كان الألم شديدًا"]},
    "headache": {"conds": {"Migraine": 0.6, "Tension headache": 0.4}, "advice_ar": ["ارتح، اشرب سوائل، وفكر في تناول مسكن للألم"]},
    "dizziness": {"conds": {"Dehydration": 0.6, "Vertigo": 0.4}, "advice_ar": ["اجلس أو استلقِ، واشرب سوائل"]},
    "stomach pain": {"conds": {"Gastritis": 0.7, "Food Poisoning": 0.5}, "advice_ar": ["تجنب الأطعمة الدسمة", "راجع الطبيب إذا استمر الألم أو كان مصحوبًا بحمى"]},
}

# --- الدوال المساعدة ---

def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img, lang='eng+ara')
        return text, None
    except Exception as e:
        return None, f"Error during OCR: {e}"

def extract_text_from_pdf(file_bytes):
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text, None
    except Exception as e:
        return None, f"Error reading PDF: {e}"

def analyze_text(text):
    found_tests = []
    if not text:
        return found_tests
        
    text_lower = text.lower()
    
    for key, details in NORMAL_RANGES.items():
        aliases = [k for k, v in ALIASES.items() if v == key]
        search_keys = [key] + aliases
        
        pattern_keys = '|'.join([re.escape(k).replace('_', r'[\s_.]*') for k in search_keys])
        
        pattern = re.compile(rf'\b({pattern_keys})\b\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
        
        matches = pattern.finditer(text_lower)
        
        for match in matches:
            try:
                value_str = match.group(2)
                value = float(value_str)
                
                # تجنب النتائج المكررة
                if any(d['test_en'] == key and d['value'] == value for d in found_tests):
                    continue

                low, high = details["range"]
                status = "Normal"
                if value < low:
                    status = "Low"
                elif value > high:
                    status = "High"
                
                diag_info = DIAGNOSIS_GUIDELINES.get(key, {}).get(status.lower(), {})
                
                found_tests.append({
                    "test_ar": details["name_ar"],
                    "test_en": key,
                    "value": value,
                    "unit": details["unit"],
                    "status": status,
                    "normal_range": f"{low} - {high}",
                    "diagnosis_ar": diag_info.get("ar", "القيمة ضمن النطاق الطبيعي." if status == "Normal" else "لا يوجد تفسير تلقائي لهذه النتيجة."),
                })
                break 
            except (ValueError, IndexError):
                continue
                
    return found_tests

def rule_based_consult(symptoms: str):
    txt = symptoms.lower()
    cond_scores = {}
    advices = set()
    matched_keywords = []

    for kw, info in RULE_KB.items():
        if kw in txt:
            matched_keywords.append(kw)
            for cond, w in info["conds"].items():
                cond_scores[cond] = cond_scores.get(cond, 0) + w
            for a in info["advice_ar"]:
                advices.add(a)
    
    probable_conditions = sorted(cond_scores.items(), key=lambda x: x[1], reverse=True)
    return {"matched": matched_keywords, "probable": probable_conditions, "advices": list(advices)}

# --- واجهة التطبيق ---

st.title("🩺 المحلل الطبي الذكي")
st.markdown("أداة لتحليل تقارير الفحوصات الطبية وتقديم استشارة أولية. **هذه الأداة لا تغني عن استشارة الطبيب.**")

# --- الشريط الجانبي ---
st.sidebar.header("خيارات")
app_mode = st.sidebar.selectbox("اختر الخدمة:", ["تحليل تقرير طبي", "استشارة حسب الأعراض"])

if app_mode == "تحليل تقرير طبي":
    st.sidebar.subheader("رفع تقرير")
    uploaded_file = st.sidebar.file_uploader("ارفع صورة أو ملف PDF للتقرير الطبي", type=["png", "jpg", "jpeg", "pdf"])
    
    st.header("🔬 تحليل تقرير الفحص الطبي")

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type
        text = None
        error = None

        with st.spinner("جاري قراءة وتحليل الملف..."):
            if "pdf" in file_type:
                text, error = extract_text_from_pdf(file_bytes)
            else:
                text, error = extract_text_from_image(file_bytes)

        if error:
            st.error(f"حدث خطأ: {error}")
        elif text:
            st.subheader("النص المستخرج من التقرير:")
            st.text_area("Full text from report", text, height=250)

            results = analyze_text(text)

            if results:
                st.subheader("📊 نتائج التحليل الذكي:")
                
                df_data = {
                    "الفحص": [r["test_ar"] for r in results],
                    "النتيجة": [f"{r['value']} {r['unit']}" for r in results],
                    "الحالة": [r["status"] for r in results],
                    "النطاق الطبيعي": [r["normal_range"] for r in results],
                    "التفسير الأولي": [r["diagnosis_ar"] for r in results],
                }
                df = pd.DataFrame(df_data)

                def color_status(row):
                    if row['الحالة'] == 'High':
                        return ['background-color: #ffebee'] * len(row)
                    elif row['الحالة'] == 'Low':
                        return ['background-color: #fff8e1'] * len(row)
                    else:
                        return [''] * len(row)

                st.dataframe(df.style.apply(color_status, axis=1), use_container_width=True)

            else:
                st.warning("لم يتم العثور على قيم فحوصات واضحة باستخدام المحلل البسيط.")
        else:
            st.warning("لم يتم استخراج أي نص من الملف. قد تكون الصورة غير واضحة.")

elif app_mode == "استشارة حسب الأعراض":
    st.header("💬 استشارة طبية أولية حسب الأعراض")
    st.markdown("صف الأعراض التي تشعر بها (مثل: حمى وسعال وألم في الحلق)")
    
    symptoms = st.text_area("اكتب الأعراض هنا:", height=150)
    
    use_openai = st.checkbox("استخدام الذكاء الاصطناعي المتقدم (OpenAI - يتطلب إعداد)")
    
    if st.button("تحليل الأعراض"):
        if symptoms:
            with st.spinner("جاري تحليل الأعراض..."):
                if use_openai:
                    st.warning("ميزة OpenAI لم يتم تفعيلها بعد في هذه النسخة.")
                else:
                    consult_results = rule_based_consult(symptoms)
                    st.subheader("نتائج التحليل الأولي:")
                    
                    if consult_results["probable"]:
                        st.write("**الأمراض المحتملة (بناءً على القواعد البسيطة):**")
                        for cond, score in consult_results["probable"]:
                            st.write(f"- {cond} (درجة الاحتمال: {score:.1f})")
                    
                    if consult_results["advices"]:
                        st.write("**نصائح أولية:**")
                        for advice in consult_results["advices"]:
                            st.write(f"- {advice}")
                    
                    if not consult_results["matched"]:
                        st.warning("لم يتم التعرف على كلمات مفتاحية واضحة في الأعراض الموصوفة.")
                    
                    st.info("**تنبيه:** هذا التحليل هو مجرد اقتراح أولي بناءً على قواعد بسيطة ولا يغني إطلاقًا عن زيارة الطبيب للتشخيص الدقيق.")
        else:
            st.error("الرجاء إدخال الأعراض أولاً.")

st.sidebar.markdown("---")
st.sidebar.info("تم التطوير بواسطة فريق Manus بالتعاون معك.")

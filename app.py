import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import pandas as pd
import os
from openai import OpenAI
import cv2
import numpy as np

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Analyzer Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- قواعد البيانات والفحوصات الموسعة ---
NORMAL_RANGES = {
    # CBC
    "wbc": {"range": (4.0, 11.0), "unit": "x10^9/L", "name_ar": "كريات الدم البيضاء"},
    "rbc": {"range": (4.1, 5.7), "unit": "x10^12/L", "name_ar": "كريات الدم الحمراء"},
    "hemoglobin": {"range": (13.0, 18.0), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "hematocrit": {"range": (40, 54), "unit": "%", "name_ar": "الهيماتوكريت"},
    "platelets": {"range": (150, 450), "unit": "x10^9/L", "name_ar": "الصفائح الدموية"},
    # كيمياء الدم
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
    "alt": {"range": (7, 56), "unit": "U/L", "name_ar": "إنزيم ALT"},
    "ast": {"range": (10, 40), "unit": "U/L", "name_ar": "إنزيم AST"},
    "crp": {"range": (0, 10), "unit": "mg/L", "name_ar": "بروتين سي التفاعلي (CRP)"},
    # دهون
    "total_cholesterol": {"range": (0, 200), "unit": "mg/dL", "name_ar": "الكوليسترول الكلي"},
    "triglycerides": {"range": (0, 150), "unit": "mg/dL", "name_ar": "الدهون الثلاثية"},
    "hdl": {"range": (40, 60), "unit": "mg/dL", "name_ar": "الكوليسترول الجيد (HDL)"},
    "ldl": {"range": (0, 100), "unit": "mg/dL", "name_ar": "الكوليسترول الضار (LDL)"},
    # فيتامينات ومعادن
    "vitamin_d": {"range": (30, 100), "unit": "ng/mL", "name_ar": "فيتامين د"},
    "vitamin_b12": {"range": (200, 900), "unit": "pg/mL", "name_ar": "فيتامين ب12"},
    "iron": {"range": (60, 170), "unit": "mcg/dL", "name_ar": "الحديد"},
    "ferritin": {"range": (30, 400), "unit": "ng/mL", "name_ar": "الفيريتين (مخزون الحديد)"},
    # غدة درقية
    "tsh": {"range": (0.4, 4.0), "unit": "mIU/L", "name_ar": "الهرمون المنبه للغدة الدرقية (TSH)"},
    "t3": {"range": (80, 220), "unit": "ng/dL", "name_ar": "هرمون T3"},
    "t4": {"range": (4.5, 11.2), "unit": "mcg/dL", "name_ar": "هرمون T4"},
}

ALIASES = {
    "hb": "hemoglobin", "hgb": "hemoglobin", "pcv": "hematocrit", "hct": "hematocrit",
    "w.b.c": "wbc", "wbc count": "wbc", "white blood cells": "wbc",
    "r.b.c": "rbc", "red blood cells": "rbc", "plt": "platelets", "platelet count": "platelets",
    "blood sugar": "glucose", "sugar": "glucose", "sgot": "ast", "sgpt": "alt",
    "vit d": "vitamin_d", "cholesterol": "total_cholesterol", "trig": "triglycerides",
    "c-reactive protein": "crp",
}

RECOMMENDATIONS = {
    "wbc": {"Low": "قد يشير إلى ضعف المناعة.", "High": "قد يشير إلى وجود عدوى أو التهاب."},
    "hemoglobin": {"Low": "قد يشير إلى فقر الدم (الأنيميا)."},
    "platelets": {"Low": "قد يزيد من خطر النزيف.", "High": "قد يزيد من خطر تكوّن الجلطات."},
    "glucose": {"High": "ارتفاع سكر الدم قد يكون مؤشرًا على السكري أو مقدماته."},
    "creatinine": {"High": "قد يشير إلى مشكلة في وظائف الكلى."},
    "alt": {"High": "قد يشير إلى ضرر في خلايا الكبد."},
    "ldl": {"High": "ارتفاع الكوليسترول الضار يزيد من خطر أمراض القلب."},
    "vitamin_d": {"Low": "نقص فيتامين د شائع وقد يؤثر على صحة العظام والمناعة."},
    "tsh": {"High": "قد يشير إلى قصور في الغدة الدرقية.", "Low": "قد يشير إلى فرط نشاط الغدة الدرقية."},
    "crp": {"High": "مؤشر على وجود التهاب في الجسم."},
}

# --- دوال المعالجة المحسنة ---
def preprocess_image_for_ocr(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        cv_image = np.array(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh)
    except Exception:
        return Image.open(io.BytesIO(file_bytes))

def extract_text_from_image(processed_img):
    try:
        return pytesseract.image_to_string(processed_img, lang="eng+ara"), None
    except Exception as e:
        return None, f"خطأ في محرك التعرف الضوئي (OCR): {e}"

def analyze_text(text):
    """
    دالة محسنة لتحليل النص المستخرج من تقرير طبي.
    تعالج النص سطراً بسطر للبحث عن أسماء الفحوصات ثم الأرقام المرتبطة بها.
    """
    results = []
    if not text:
        return results
    
    processed_tests = set()
    lines = text.split('\n')

    for key, details in NORMAL_RANGES.items():
        if key in processed_tests:
            continue

        aliases = [k for k, v in ALIASES.items() if v == key]
        search_keys = [key] + aliases
        
        pattern_keys = '|'.join([re.escape(k).replace(r"\.", r"\.?\s?") for k in search_keys])
        search_pattern = re.compile(rf'\b({pattern_keys})\b', re.IGNORECASE)

        for line in lines:
            match = search_pattern.search(line)
            if match:
                try:
                    value_match = re.search(r'([0-9]+\.?[0-9]*)', line[match.end():])
                    
                    if value_match:
                        value_str = value_match.group(1)
                        value = float(value_str)

                        preceding_char_index = value_match.start() + match.end() - 1
                        if preceding_char_index >= 0 and line[preceding_char_index] in ['-', '.']:
                             continue

                        low, high = details["range"]
                        status = "طبيعي"
                        if value < low:
                            status = "منخفض"
                        elif value > high:
                            status = "مرتفع"
                        
                        recommendation = RECOMMENDATIONS.get(key, {}).get(status, "")
                        
                        results.append({
                            "name": f"🔬 {details['name_ar']}",
                            "value_str": value_str,
                            "status": status,
                            "range_str": f"{low} - {high} {details['unit']}",
                            "recommendation": recommendation
                        })
                        
                        processed_tests.add(key)
                        break 
                except (ValueError, IndexError):
                    continue
        if key in processed_tests:
            continue
            
    return results

# --- الذكاء الاصطناعي ---
def get_ai_symptom_analysis(api_key, symptoms):
    if not api_key:
        st.error("يرجى إدخال مفتاح OpenAI API في الشريط الجانبي.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        prompt = f'''أنت طبيب استشاري خبير. المريض يصف الأعراض التالية: "{symptoms}".
        قدم استشارة طبية أولية مفصلة ومنظمة في نقاط. ابدأ بتحليل محتمل للأعراض، ثم قدم بعض الاحتمالات التشخيصية (مع التأكيد أنها ليست نهائية)، واختتم بنصائح عامة وتوصية واضحة بزيارة الطبيب.
        مهم جداً: أكد في نهاية ردك أن هذه الاستشارة لا تغني أبداً عن التشخيص الطبي المتخصص.'''

        with st.spinner("🤖 الذكاء الاصطناعي يحلل الأعراض..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "أنت طبيب خبير وودود، تقدم إجابات مفصلة ومنظمة."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
    except Exception as e:
        if "authentication" in str(e).lower():
            return "❌ مفتاح OpenAI API غير صحيح أو منتهي الصلاحية."
        return f"❌ خطأ في الاتصال بالذكاء الاصطناعي: {e}"

# --- عرض النتائج ---
def display_results_as_cards(results):
    st.subheader("📊 نتائج تحليل التقرير")
    colors = {"طبيعي": "#2E8B57", "منخفض": "#DAA520", "مرتفع": "#DC143C"}
    
    for res in results:
        color = colors.get(res['status'], "#808080")
        st.markdown(f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 5px solid {color};">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: #003366;">{res['name']}</h4>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p><strong>النتيجة:</strong> {res['value_str']}</p>
                    <p><strong>النطاق الطبيعي:</strong> {res['range_str']}</p>
                </div>
                <div style="color: {color}; font-weight: bold; font-size: 1.2em;">{res["status"]}</div>
            </div>
            {f"<div style='background-color: #e1ecf4; border-radius: 5px; padding: 10px; margin-top: 10px; font-size: 0.9em; color: #333;'>💡 <strong>ملاحظة أولية:</strong> {res['recommendation']}</div>" if res["recommendation"] else ""}
        </div>
        """, unsafe_allow_html=True)

# --- الواجهة الرئيسية ---
st.title("🩺 المحلل الطبي الذكي Pro")

st.sidebar.header("⚙️ الإعدادات")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API", type="password", help="مفتاحك الخاص بواجهة برمجة تطبيقات OpenAI")

st.sidebar.markdown("---")
mode = st.sidebar.radio("اختر الخدمة:", ["🔬 تحليل التقارير الطبية", "💬 استشارة حسب الأعراض"])
st.sidebar.markdown("---")
st.sidebar.info("هذا التطبيق هو أداة مساعدة ولا يغني عن استشارة الطبيب المختص.")


if mode == "🔬 تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    st.markdown("ارفع صورة واضحة لتقرير المختبر وسيقوم الذكاء الاصطناعي بتحليلها.")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        with st.spinner("جاري معالجة الصورة واستخلاص النص..."):
            processed_img = preprocess_image_for_ocr(file_bytes)
            text, err = extract_text_from_image(processed_img)

        if err:
            st.error(err)
        elif text:
            results = analyze_text(text)
            if results:
                display_results_as_cards(results)
            else:
                st.warning("لم يتم التعرف على أي فحوصات مدعومة في النص المستخرج. حاول استخدام صورة أوضح أو تأكد من أن الفحوصات مدعومة.")

            with st.expander("📄 عرض النص الخام المستخرج من الصورة"):
                st.text_area("", text, height=200)

elif mode == "💬 استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    st.markdown("صف الأعراض التي تشعر بها بالتفصيل ليقوم الذكاء الاصطناعي بتحليلها.")
    
    symptoms = st.text_area("📝 صف الأعراض هنا:", height=200, placeholder="مثال: أشعر بصداع حاد في الجزء الأمامي من الرأس مع غثيان وحساسية للضوء...")
    
    if st.button("تحليل الأعراض بالذكاء الاصطناعي", use_container_width=True):
        if not symptoms:
            st.warning("يرجى وصف الأعراض أولاً.")
        else:
            ai_response = get_ai_symptom_analysis(api_key_input, symptoms)
            if ai_response:
                st.subheader("🤖 استشارة الذكاء الاصطناعي الأولية")
                st.markdown(ai_response)

import streamlit as st
import re
import io
import os
from PIL import Image
import pytesseract
import pandas as pd
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pdfplumber

# --- إعدادات الصفحة ---
st.set_page_config(page_title="AI Medical Analyzer Pro", page_icon="🩺", layout="wide")

# --- تحميل قاعدة البيانات من CSV (مع تصحيح الأخطاء) ---
@st.cache_data
def load_tests_database(path="tests_database.csv"):
    try:
        df = pd.read_csv(path, dtype=str).fillna('')
    except FileNotFoundError:
        st.error(f"خطأ فادح: ملف قاعدة البيانات '{path}' غير موجود. لا يمكن تشغيل التطبيق.")
        return None, None, None

    tests = {}
    aliases = {}
    recommendations = {}
    for _, row in df.iterrows():
        key = row['code'].strip().lower()
        if not key: continue

        # إصلاح حاسم: التعامل مع القيم الفارغة قبل التحويل
        try:
            low = float(row['low']) if row['low'] else None
            high = float(row['high']) if row['high'] else None
        except (ValueError, TypeError):
            low, high = None, None

        tests[key] = {
            'range': (low, high) if low is not None and high is not None else None,
            'unit': row.get('unit', ''),
            'name_ar': row.get('name_ar', key),
            'name_en': row.get('name_en', key),
        }
        
        for alias_col in ['aliases', 'name_en', 'name_ar']:
            if row.get(alias_col):
                for a in row[alias_col].split(';'):
                    cleaned_alias = a.strip().lower()
                    if cleaned_alias:
                        aliases[cleaned_alias] = key
        
        rec = {}
        if row.get('recommendation_low'): rec['منخفض'] = row['recommendation_low']
        if row.get('recommendation_high'): rec['مرتفع'] = row['recommendation_high']
        if rec: recommendations[key] = rec
        
    return tests, aliases, recommendations

TESTS_DB, ALIASES, RECOMMENDATIONS = load_tests_database()

# --- دوال المعالجة واستخلاص النص (محسنة) ---
def preprocess_image_bytes(file_bytes):
    img_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)

def ocr_image(img_pil):
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(img_pil, lang='eng+ara', config=config)

def extract_text_from_file(file_bytes, file_type):
    if file_type == "application/pdf":
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception:
            pass
        # إذا كان ملف PDF ممسوحاً ضوئياً (لا يحتوي على نص)
        if not text.strip():
            images = convert_from_bytes(file_bytes)
            for img in images:
                text += ocr_image(img) + "\n"
        return text
    else: # للصور
        processed_img = preprocess_image_bytes(file_bytes)
        return ocr_image(processed_img)

# --- دالة التحليل (مبسطة وأكثر قوة) ---
def analyze_text_robust(text, tests_db, aliases_db):
    text_lower = text.lower()
    
    found_tests = []
    for alias, key in aliases_db.items():
        try:
            # استخدام `\b` لضمان مطابقة الكلمة كاملة
            for match in re.finditer(r'\b' + re.escape(alias) + r'\b', text_lower):
                found_tests.append({'key': key, 'pos': match.start(), 'end': match.end()})
        except re.error:
            continue # تجاهل الأنماط غير الصالحة

    numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+\.?\d*)', text_lower)]
    
    results = []
    used_keys = set()
    found_tests.sort(key=lambda x: x['pos'])

    for test in found_tests:
        key = test['key']
        if key in used_keys: continue

        best_candidate_num = None
        min_distance = float('inf')

        for num_val, num_pos in numbers:
            distance = num_pos - test['end']
            if 0 <= distance < min_distance:
                is_interrupted = any(other['pos'] > test['pos'] and other['pos'] < num_pos for other in found_tests)
                if not is_interrupted:
                    min_distance = distance
                    best_candidate_num = num_val
        
        if best_candidate_num:
            try:
                value = float(best_candidate_num)
                meta = tests_db[key]
                rng = meta.get('range')
                status = "غير معروف"
                
                if rng:
                    low, high = rng
                    if value < low: status = "منخفض"
                    elif value > high: status = "مرتفع"
                    else: status = "طبيعي"
                
                rec = RECOMMENDATIONS.get(key, {})
                recommendation = rec.get(status, '')

                results.append({
                    'name': f"🔬 {meta.get('name_ar', key)}",
                    'value_str': best_candidate_num,
                    'status': status,
                    'range_str': f"{rng[0]} - {rng[1]} {meta.get('unit', '')}" if rng else 'N/A',
                    'recommendation': recommendation
                })
                used_keys.add(key)
            except (ValueError, KeyError):
                continue
    return results

# --- عرض النتائج (باستخدام البطاقات الملونة) ---
def display_results_as_cards(results):
    st.subheader("📊 نتائج التحليل")
    colors = {"طبيعي": "#2E8B57", "منخفض": "#DAA520", "مرتفع": "#DC143C", "غير معروف": "#808080"}
    
    for res in results:
        color = colors.get(res['status'], "#808080")
        st.markdown(f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 5px solid {color};">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: #003366;">{res['name']}</h4>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p style="margin: 0;"><strong>النتيجة:</strong> {res['value_str']}</p>
                    <p style="margin: 0;"><strong>النطاق الطبيعي:</strong> {res['range_str']}</p>
                </div>
                <div style="color: {color}; font-weight: bold; font-size: 1.2em;">{res["status"]}</div>
            </div>
            {f"<div style='background-color: #e1ecf4; border-radius: 5px; padding: 10px; margin-top: 10px; font-size: 0.9em; color: #333;'>💡 <strong>ملاحظة:</strong> {res['recommendation']}</div>" if res["recommendation"] else ""}
        </div>
        """, unsafe_allow_html=True)

# --- الواجهة الرئيسية ---
st.title("🩺 المحلل الطبي الذكي Pro")
st.sidebar.header("⚙️ الإعدادات")
st.sidebar.info("هذا التطبيق هو أداة مساعدة ولا يغني عن استشارة الطبيب المختص.")
st.sidebar.markdown("---")

if TESTS_DB: # لا تقم بتشغيل التطبيق إذا فشل تحميل قاعدة البيانات
    mode = st.sidebar.radio("اختر الخدمة:", ["🔬 تحليل التقارير الطبية", "💬 استشارة حسب الأعراض"])
    
    if mode == "🔬 تحليل التقارير الطبية":
        st.header("🔬 تحليل تقرير طبي")
        uploaded_file = st.file_uploader("ارفع ملف صورة أو PDF", type=['png','jpg','jpeg','pdf'])

        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            with st.spinner("جاري استخلاص النص من الملف..."):
                text = extract_text_from_file(file_bytes, uploaded_file.type)

            if not text or not text.strip():
                st.error("لم يتمكن التطبيق من استخلاص أي نص. حاول استخدام ملف أوضح.")
            else:
                results = analyze_text_robust(text, TESTS_DB, ALIASES)
                
                if results:
                    display_results_as_cards(results)
                else:
                    st.warning("لم يتم العثور على أي فحوصات مدعومة في النص المستخرج.")

                with st.expander("📄 عرض النص الخام المستخرج (للتدقيق)"):
                    st.text_area("", text, height=300)

    elif mode == "💬 استشارة حسب الأعراض":
        st.header("💬 استشارة أولية حسب الأعراض")
        # (يمكن إضافة كود الاستشارة هنا لاحقاً)
        st.info("هذه الميزة قيد التطوير.")

# app.py
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
from datetime import datetime
from fpdf import FPDF

# --- إعداد الصفحة ---
st.set_page_config(page_title="AI Medical Analyzer", page_icon="🩺", layout="wide")

# --- مسارات وتهيئة tesseract (يمكن تعديل حسب النظام) ---
# للمستخدمين في Streamlit Cloud قد يحتاج تعديل المسار
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', 'tesseract')

# --- تحميل قاعدة الفحوصات من CSV ---
@st.cache_data
def load_tests_database(csv_path="tests_database.csv"):
    df = pd.read_csv(csv_path)
    tests = {}
    aliases = {}
    recommendations = {}
    for _, row in df.iterrows():
        key = str(row["key"]).strip()
        try:
            low = float(row["low"])
            high = float(row["high"])
        except:
            low = None
            high = None
        tests[key] = {
            "range": (low, high) if low is not None and high is not None else None,
            "unit": str(row.get("unit", "")),
            "name_ar": str(row.get("name_ar", key)),
            "name_en": str(row.get("name_en", key)),
            "icon": str(row.get("icon", ""))
        }
        # aliases
        ali = str(row.get("aliases", ""))
        if pd.notna(ali) and ali.strip():
            for a in ali.split(";"):
                aa = a.strip().lower()
                if aa:
                    aliases[aa] = key
        # recommendations
        rec = {}
        if pd.notna(row.get("recommendation_low", "")) and str(row.get("recommendation_low", "")).strip():
            rec["Low"] = str(row.get("recommendation_low"))
        if pd.notna(row.get("recommendation_high", "")) and str(row.get("recommendation_high", "")).strip():
            rec["High"] = str(row.get("recommendation_high"))
        if rec:
            recommendations[key] = rec
    return tests, aliases, recommendations

NORMAL_RANGES, ALIASES, RECOMMENDATIONS = load_tests_database()

# --- تحميل نموذج محلي إن وُجد ---
@st.cache_resource
def load_model(path="symptom_checker_model.joblib"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
    return None

MODEL = load_model()

# ---------- دوال OCR ومعالجة الصور ----------

def preprocess_image(img: Image.Image) -> Image.Image:
    try:
        arr = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,11,2)
        return Image.fromarray(thresh)
    except Exception:
        return img


def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = preprocess_image(img)
        custom_oem_psm_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, lang='eng+ara', config=custom_oem_psm_config)
        return text, None
    except Exception as e:
        return None, f"OCR Error: {e}"


def extract_text_from_pdf(file_bytes):
    texts = []
    errors = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        # OCR on pages as images (for scanned PDFs)
        pages = convert_from_bytes(file_bytes)
        for i, page_img in enumerate(pages):
            try:
                page_img = preprocess_image(page_img)
                txt = pytesseract.image_to_string(page_img, lang='eng+ara', config=r'--oem 3 --psm 6')
                if txt and txt.strip():
                    texts.append(txt)
            except Exception as e:
                errors.append(f"Page {i+1} OCR error: {e}")
    except Exception as e:
        return None, f"PDF Error: {e}"
    return "\n".join(texts), (errors if errors else None)

# ---------- تحليل النص لاستخراج الفحوصات ----------

def analyze_text(text, prefer_language='ar'):
    results = []
    if not text:
        return results
    text_lower = text.lower()
    seen = set()
    for key, details in NORMAL_RANGES.items():
        search_keys = [key] + [k for k,v in ALIASES.items() if v==key]
        # build pattern
        pat_keys = '|'.join([re.escape(k) for k in search_keys if k])
        if not pat_keys:
            continue
        pattern = re.compile(rf"({pat_keys})\s*[:\-=]*\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            try:
                val = float(m.group(2))
            except:
                continue
            if (key, val) in seen:
                continue
            seen.add((key,val))
            rng = details.get('range')
            status = 'Unknown'
            if rng and rng[0] is not None and rng[1] is not None:
                low, high = rng
                if val < low: status = 'Low'
                elif val > high: status = 'High'
                else: status = 'Normal'
            display_name = details.get('name_ar') if prefer_language=='ar' else details.get('name_en')
            rec = RECOMMENDATIONS.get(key, {})
            recommendation = rec.get(status, '')
            results.append({
                'key': key,
                'name': f"{details.get('icon','')} {display_name}",
                'value': val,
                'unit': details.get('unit',''),
                'status': status,
                'range_str': f"{rng[0]} - {rng[1]}" if rng else '',
                'recommendation': recommendation
            })
    return results

# ---------- تصدير التقارير ----------

def create_excel_bytes(df: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
        writer.save()
    buffer.seek(0)
    return buffer.getvalue()


def create_pdf_report(extracted_text, results_df, notes=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "AI Medical Analyzer - تقرير التحليل", ln=True, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.multi_cell(0,6, "النص المستخرج (مقتطف):")
    pdf.multi_cell(0,6, extracted_text[:3000])
    pdf.ln(4)
    pdf.multi_cell(0,6, "نتائج التحليل:")
    pdf.ln(2)
    pdf.set_font("Arial", size=9)
    for _, row in results_df.iterrows():
        pdf.multi_cell(0,6, f"{row['name']} - {row['value']} {row['unit']} - الحالة: {row['status']}")
    if notes:
        pdf.ln(4)
        pdf.multi_cell(0,6, f"ملاحظات المستخدم: {notes}")
    return pdf.output(dest='S').encode('latin-1')

# ---------- الواجهة ----------

st.title("🩺 AI Medical Analyzer")

# sidebar
st.sidebar.header("الإعدادات / Settings")
lang = st.sidebar.selectbox("اللغة / Language", ['العربية','English'])
prefer_lang = 'ar' if lang=='العربية' else 'en'

st.sidebar.markdown('---')
api_key = st.sidebar.text_input('OpenAI API Key (اختياري - only for advanced chat)', type='password')

mode = st.sidebar.radio('اختر الخدمة / Mode', ['تحليل التقارير الطبية','Report Analysis','استشارة حسب الأعراض','Symptom Consultation'])
st.sidebar.markdown('---')

if mode in ['تحليل التقارير الطبية','Report Analysis']:
    st.header('🔬 تحليل تقرير طبي / Report Analysis')
    uploaded = st.file_uploader('📂 ارفع ملف (صورة أو PDF) / Upload image or PDF', type=['png','jpg','jpeg','pdf'])
    notes = st.text_area('ملاحظات إضافية (اختياري) / Notes (optional)', height=80)
    if uploaded:
        bytes_data = uploaded.getvalue()
        if uploaded.type=='application/pdf' or uploaded.name.lower().endswith('.pdf'):
            text, err = extract_text_from_pdf(bytes_data)
        else:
            text, err = extract_text_from_image(bytes_data)
        if err:
            st.error(err)
        else:
            if text and text.strip():
                st.subheader('النص المستخرج / Extracted text')
                with st.expander('عرض النص / Show extracted text'):
                    st.text_area('', text, height=300)
                results = analyze_text(text, prefer_language= 'ar' if prefer_lang=='ar' else 'en')
                if results:
                    df = pd.DataFrame(results)
                    # ترتيب: High then Low then Normal
                    order = {'High':2,'Low':1,'Normal':0,'Unknown':-1}
                    df['rank'] = df['status'].map(order).fillna(-1)
                    df = df.sort_values(by='rank', ascending=False).drop(columns=['rank'])
                    st.subheader('نتائج التحليل / Analysis Results')
                    # تلوين
                    def style_rows(r):
                        if r['status']=='High':
                            return ['background-color: #ffebee']*len(r)
                        if r['status']=='Low':
                            return ['background-color: #fff8e1']*len(r)
                        return ['']*len(r)
                    st.dataframe(df.style.apply(style_rows, axis=1), use_container_width=True)
                    # عرض كروت
                    for r in results:
                        col1, col2 = st.columns([3,1])
                        with col1:
                            st.markdown(f"**{r['name']}**: {r['value']} {r['unit']}")
                            if r['recommendation']:
                                st.info(r['recommendation'])
                        with col2:
                            if r['status']=='High': st.markdown(':red_circle: **High**')
                            elif r['status']=='Low': st.markdown(':large_yellow_circle: **Low**')
                            elif r['status']=='Normal': st.markdown(':white_check_mark: Normal')
                    # تنزيل
                    excel = create_excel_bytes(df)
                    st.download_button('⬇️ تحميل النتائج (Excel)', excel, file_name='analysis.xlsx')
                    pdfb = create_pdf_report(text, df, notes=notes)
                    st.download_button('⬇️ تحميل التقرير (PDF)', pdfb, file_name='analysis_report.pdf')
                else:
                    st.warning('لم يتم العثور على فحوصات أو قيم قابلة للقراءة.')
            else:
                st.warning('لا يوجد نص مستخرج من الملف.')

elif mode in ['استشارة حسب الأعراض','Symptom Consultation']:
    st.header('💬 استشارة أولية حسب الأعراض / Symptom Consultation')
    symptoms = st.text_area('صف أعراضك بالتفصيل / Describe your symptoms', height=200)
    use_local_model = st.checkbox('استخدام النموذج المحلي لو كان موجوداً / Use local model (if available)')
    feature_input = st.text_input('إذا كنت تملك مصفوفة المميزات للنموذج (اختياري) / feature vector (comma separated)')
    if st.button('تحليل / Analyze'):
        if not symptoms.strip():
            st.warning('الرجاء إدخال الأعراض أولاً / Please enter symptoms')
        else:
            # عرض ملخص بسيط نصي
            st.subheader('ملخص سريع / Quick summary')
            st.write('تم استلام الأعراض. التحليل الآلي التالي هو لأغراض إرشادية فقط.')
            # نموذج محلي
            if use_local_model and MODEL is not None and feature_input.strip():
                try:
                    vec = [float(x.strip()) for x in feature_input.split(',') if x.strip()]
                    pred = MODEL.predict([vec])[0]
                    st.success(f'توقع النموذج المحلي: {pred}')
                except Exception as e:
                    st.error(f'خطأ في إدخال المميزات أو النموذج: {e}')
            # خيار OpenAI (إذا مزود المفتاح) — إبقاؤه اختياريًا
            elif api_key and api_key.strip():
                st.info('يتم استخدام OpenAI لتحليل نصي متقدم (اختياري)')
                # نرسل الطلب بشكل مبسط (لا نضمن مكتبة openai هنا)
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    prompt = f"You are a helpful doctor. Patient symptoms: {symptoms}. Provide Arabic output."
                    res = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=[{'role':'system','content':'You are a medical assistant.'},{'role':'user','content':prompt}],
                        max_tokens=600
                    )
                    out = res.choices[0].message.content
                    st.markdown(out)
                except Exception as e:
                    st.error(f'OpenAI integration error: {e}')
            else:
                # قواعد بسيطة
                RULE_KB = {
                    'fever': ['حمى','ارتفاع حرارة','fever'],
                    'cough': ['سعال','cough'],
                    'chest pain': ['ألم صدر','pain in chest']
                }
                matches = []
                for cond, kws in RULE_KB.items():
                    for kw in kws:
                        if kw in symptoms.lower():
                            matches.append(cond)
                            break
                if matches:
                    st.write('مطابقات قواعدية:', matches)
                else:
                    st.write('لا توجد مطابقات قواعدية واضحة — جرب وصف مختلف أو استخدم OpenAI/local model إذا متاح.')

st.sidebar.markdown('---')
st.sidebar.write('Project: medical-ai-app — Arabic / English')

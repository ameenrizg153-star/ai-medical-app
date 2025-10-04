# app.py
import streamlit as st
import re
import io
import os
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from fpdf import FPDF
import joblib
from datetime import datetime

# --- Page config ---
st.set_page_config(page_title="AI Medical Analyzer Pro", page_icon="🩺", layout="wide")

# Tesseract command (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")

# --- Load tests database from CSV ---
@st.cache_data
def load_tests_database(path="tests_database.csv"):
    df = pd.read_csv(path, dtype=str).fillna('')
    tests = {}
    aliases = {}
    recommendations = {}
    for _, row in df.iterrows():
        key = row['code'].strip()
        try:
            low = float(row['low']) if row['low'] != '' else None
            high = float(row['high']) if row['high'] != '' else None
        except:
            low = None; high = None
        tests[key] = {
            'range': (low, high) if low is not None and high is not None else None,
            'unit': row.get('unit',''),
            'name_ar': row.get('name_ar', key),
            'name_en': row.get('name_en', key),
            'icon': row.get('icon','')
        }
        # aliases column separated by semicolon
        ali = row.get('aliases','')
        if ali:
            for a in ali.split(';'):
                aa = a.strip().lower()
                if aa:
                    aliases[aa] = key
        # recommendations
        rec_low = row.get('recommendation_low','')
        rec_high = row.get('recommendation_high','')
        rec = {}
        if rec_low:
            rec['Low'] = rec_low
        if rec_high:
            rec['High'] = rec_high
        if rec:
            recommendations[key] = rec
    return tests, aliases, recommendations

TESTS_DB, ALIASES, RECOMMENDATIONS = load_tests_database()

# --- Load optional local model ---
@st.cache_resource
def load_local_model(path="models/symptom_checker_model.joblib"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

LOCAL_MODEL = load_local_model()

# --- OCR preprocessing ---
def preprocess_image_bytes(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        return Image.fromarray(thresh)
    except Exception:
        return Image.open(io.BytesIO(file_bytes)).convert('RGB')

def ocr_image(img_pil):
    try:
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img_pil, lang='eng+ara', config=config)
        return text
    except Exception as e:
        return ""

def extract_text_from_pdf_bytes(file_bytes):
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
    except Exception:
        pass
    # OCR fallback (scanned pdfs)
    try:
        pages = convert_from_bytes(file_bytes)
        for p in pages:
            p2 = p.convert('RGB')
            t = ocr_image(p2)
            if t and t.strip():
                texts.append(t)
    except Exception:
        pass
    return "\n".join(texts)

# --- Clean OCR text: fix common OCR errors and normalize ---
COMMON_OCR_FIXES = {
    'platelats': 'platelets',
    'platelats': 'platelets',
    'platelts': 'platelets',
    'rac': 'rbc',
    'r.b.c': 'rbc',
    'h g b': 'hgb',
    'h.g.b': 'hgb',
    'h g b': 'hgb',
    'hemoglobin.': 'hemoglobin',
    'neutrophil.': 'neutrophil',
    '\ufeff': '',
    'ﬁ': 'fi'
}

def clean_ocr_text(text):
    s = text
    # Normalize some Arabic punctuation and remove weird chars
    s = s.replace('\u200f','').replace('\u200e','')
    for a,b in COMMON_OCR_FIXES.items():
        s = re.sub(re.escape(a), b, s, flags=re.IGNORECASE)
    # Replace multiple spaces and weird separators with single space
    s = re.sub(r'[ˇ˘••·••]', ' ', s)
    s = re.sub(r'[^\S\r\n]+', ' ', s)
    return s

# --- robust analyze: support name->number and number->name ---
def analyze_text_robust(text):
    text0 = clean_ocr_text(text)
    text_lower = text0.lower()
    # find numbers with positions
    num_pat = re.compile(r'([0-9]+(?:\.[0-9]+)?)')
    numbers = [(m.group(1), m.start(), m.end()) for m in num_pat.finditer(text_lower)]
    # find tests positions
    found_tests = []
    for key, meta in TESTS_DB.items():
        # build search words: key, english name, aliases
        words = [key.lower()]
        if meta.get('name_en'):
            words.append(str(meta['name_en']).lower())
        if meta.get('name_ar'):
            words.append(str(meta['name_ar']).lower())
        # add aliases mapping keys
        for a, k in ALIASES.items():
            if k == key:
                words.append(a.lower())
        # dedupe
        words = list(dict.fromkeys(words))
        for w in words:
            if not w:
                continue
            # allow fuzzy-ish matching: remove punctuation in pattern
            pat = re.escape(w).replace(r'\ ', r'\s*')
            for m in re.finditer(pat, text_lower, flags=re.IGNORECASE):
                found_tests.append({'key': key, 'name': w, 'pos': m.start(), 'end': m.end()})
    # sort tests by position
    found_tests = sorted(found_tests, key=lambda x: x['pos'])
    results = []
    used_keys = set()
    # For each found test, search nearest number after (or before) within window
    for t in found_tests:
        key = t['key']
        if key in used_keys:
            continue
        test_pos = t['end']
        candidate = None
        min_dist = 99999
        for numstr, nstart, nend in numbers:
            # number following the test
            if nstart >= test_pos and (nstart - test_pos) < min_dist and (nstart - test_pos) < 80:
                # ensure no other test in between
                interrupted = False
                for other in found_tests:
                    if other['pos'] > t['pos'] and other['pos'] < nstart:
                        interrupted = True
                        break
                if not interrupted:
                    candidate = (numstr, nstart, nend)
                    min_dist = nstart - test_pos
        # if no candidate after, try number before
        if candidate is None:
            min_dist2 = 99999
            for numstr, nstart, nend in numbers:
                if nend <= t['pos'] and (t['pos'] - nend) < min_dist2 and (t['pos'] - nend) < 50:
                    candidate = (numstr, nstart, nend)
                    min_dist2 = t['pos'] - nend
        if candidate:
            valstr = candidate[0]
            try:
                val = float(valstr)
            except:
                continue
            meta = TESTS_DB.get(key, {})
            rng = meta.get('range')
            status = 'Unknown'
            if rng:
                low, high = rng
                try:
                    if val < low:
                        status = 'منخفض'  # Low
                    elif val > high:
                        status = 'مرتفع'  # High
                    else:
                        status = 'طبيعي'  # Normal
                except Exception:
                    status = 'Unknown'
            display_name = meta.get('name_ar') or meta.get('name_en') or key
            rec = RECOMMENDATIONS.get(key, {})
            recommendation = rec.get('Low') if status == 'منخفض' else rec.get('High') if status == 'مرتفع' else ''
            results.append({
                'key': key,
                'name': f"🔬 {display_name}",
                'value_str': valstr,
                'status': status,
                'range_str': f"{rng[0]} - {rng[1]} {meta.get('unit','')}" if rng else '',
                'recommendation': recommendation
            })
            used_keys.add(key)
    return results

# --- Export helpers ---
def df_to_excel_bytes(df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
        writer.save()
    out.seek(0)
    return out.read()

def create_pdf_report(extracted_text, df, notes=''):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 8, 'AI Medical Analyzer - Report', ln=True, align='C')
    pdf.ln(4)
    pdf.set_font('Arial', size=10)
    pdf.cell(0,6, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True)
    pdf.ln(4)
    pdf.multi_cell(0,6, 'Extracted (excerpt):')
    pdf.multi_cell(0,6, extracted_text[:2000])
    pdf.ln(4)
    pdf.multi_cell(0,6, 'Results:')
    pdf.ln(2)
    for _, r in df.iterrows():
        pdf.multi_cell(0,6, f"{r['name']} - {r['value_str']} {r.get('unit','')} - Status: {r['status']}")
    if notes:
        pdf.ln(4)
        pdf.multi_cell(0,6, f"Notes: {notes}")
    return pdf.output(dest='S').encode('latin-1')

# --- UI ---
st.title("🩺 AI Medical Analyzer Pro")
st.sidebar.header("Settings / الإعدادات")
lang = st.sidebar.selectbox("Language / اللغة", ['العربية','English'])
prefer_ar = True if lang == 'العربية' else False
api_key = st.sidebar.text_input("OpenAI API Key (optional)", type='password', help="Optional: for advanced symptom analysis")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode / الوضع", ["🔬 تحليل التقارير الطبية", "💬 استشارة حسب الأعراض"])
st.sidebar.markdown("---")
st.sidebar.info("Tool provides guidance only. Not a medical diagnosis.")

if mode.startswith("🔬"):
    st.header("🔬 تحليل تقرير طبي / Report Analysis")
    uploaded = st.file_uploader("Upload image or PDF (png/jpg/jpeg/pdf) — ارفع صورة أو PDF", type=['png','jpg','jpeg','pdf'])
    notes = st.text_area("Notes / ملاحظات (optional)", height=80)
    if uploaded:
        raw = uploaded.getvalue()
        text = ""
        if uploaded.type == "application/pdf" or uploaded.name.lower().endswith('.pdf'):
            text = extract_text_from_pdf_bytes(raw)
        else:
            proc = preprocess_image_bytes(raw)
            text = ocr_image(proc)
        if not text or not text.strip():
            st.error("No text extracted. Try a clearer image or a PDF.")
        else:
            st.subheader("Extracted Text / النص المستخرج")
            with st.expander("Show extracted text / عرض النص"):
                st.text_area("", text, height=300)
            results = analyze_text_robust(text)
            if results:
                df = pd.DataFrame(results)
                st.subheader("Analysis results / نتائج التحليل")
                def style_row(r):
                    if r['status'] == 'مرتفع':
                        return ['background-color:#ffebee']*len(r)
                    if r['status'] == 'منخفض':
                        return ['background-color:#fff8e1']*len(r)
                    return ['']*len(r)
                st.dataframe(df.style.apply(style_row, axis=1), use_container_width=True)
                for r in results:
                    st.markdown(f"**{r['name']}** — {r['value_str']} — **{r['status']}**")
                    if r['recommendation']:
                        st.info(r['recommendation'])
                # export
                excel_bytes = df_to_excel_bytes(df)
                st.download_button("Download Excel / تنزيل Excel", excel_bytes, file_name="analysis.xlsx")
                pdf_bytes = create_pdf_report(text, df, notes)
                st.download_button("Download PDF / تنزيل PDF", pdf_bytes, file_name="analysis_report.pdf")
            else:
                st.warning("No supported tests found. Try cropping table only or adjust aliases in tests_database.csv.")
elif mode.startswith("💬"):
    st.header("💬 Symptom consultation / استشارة حسب الأعراض")
    symptoms = st.text_area("Describe symptoms / صف أعراضك", height=200)
    use_local = st.checkbox("Use local model if available / استخدام النموذج المحلي إن وجد")
    if st.button("Analyze / تحليل"):
        if not symptoms.strip():
            st.warning("Please enter symptoms / يرجى إدخال الأعراض")
        else:
            # local model path
            if use_local and LOCAL_MODEL is not None:
                st.info("Using local model for prediction (provide feature vector input in /models format).")
                st.warning("This app does not auto-generate feature vectors; local model usage requires preprocessed input.")
            elif api_key:
                # call OpenAI (if user provided KEY) — keep robust try/except
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    prompt = f'''أنت طبيب استشاري خبير. المريض يصف الأعراض التالية: "{symptoms}".
قدم استشارة طبية أولية مفصلة ومنظمة في نقاط. ابدأ بتحليل محتمل للأعراض، ثم قدم بعض الاحتمالات التشخيصية (مع التأكيد أنها ليست نهائية)، واختتم بنصائح عامة وتوصية واضحة بزيارة الطبيب.
مهم جدًا: أكد في نهاية ردك أن هذه الاستشارة لا تغني أبداً عن التشخيص الطبي المتخصص.'''
                    with st.spinner("AI is analyzing..."):
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a helpful medical assistant."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        out = response.choices[0].message.content
                        st.markdown(out)
                except Exception as e:
                    st.error(f"OpenAI error or not available: {e}")
            else:
                st.info("Simple rule-based suggestions (no OpenAI key provided).")
                # Very simple rule-based demo
                lower = symptoms.lower()
                suggestions = []
                if "fever" in lower or "حمى" in lower:
                    suggestions.append("Measure temperature; stay hydrated.")
                if "cough" in lower or "سعال" in lower:
                    suggestions.append("Watch for shortness of breath or bloody sputum.")
                if suggestions:
                    st.markdown("**Suggestions / اقتراحات:**")
                    for s in suggestions:
                        st.write("-", s)
                else:
                    st.write("No clear suggestions. Consider adding more details or provide OpenAI API key for advanced analysis.")

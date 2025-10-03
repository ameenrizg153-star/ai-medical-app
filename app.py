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

st.set_page_config(page_title="AI Medical Analyzer", page_icon="馃┖", layout="wide")

# tesseract path (can be adjusted via env var)
pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")

@st.cache_data
def load_tests_database(csv_path="tests_database.csv"):
    df = pd.read_csv(csv_path)
    tests = {}
    aliases = {}
    recommendations = {}
    for _, row in df.iterrows():
        key = str(row["key"]).strip()
        try:
            low = float(row["low"]) if pd.notna(row["low"]) else None
            high = float(row["high"]) if pd.notna(row["high"]) else None
        except:
            low = None; high = None
        tests[key] = {
            "range": (low, high) if low is not None and high is not None else None,
            "unit": str(row.get("unit","")),
            "name_ar": str(row.get("name_ar", key)),
            "name_en": str(row.get("name_en", key)),
            "icon": str(row.get("icon",""))
        }
        ali = str(row.get("aliases",""))
        if pd.notna(ali) and ali.strip():
            for a in ali.split(";"):
                aa = a.strip().lower()
                if aa:
                    aliases[aa] = key
        rec = {}
        if pd.notna(row.get("recommendation_low","")) and str(row.get("recommendation_low","")).strip():
            rec["Low"] = str(row.get("recommendation_low"))
        if pd.notna(row.get("recommendation_high","")) and str(row.get("recommendation_high","")).strip():
            rec["High"] = str(row.get("recommendation_high"))
        if rec:
            recommendations[key] = rec
    return tests, aliases, recommendations

NORMAL_RANGES, ALIASES, RECOMMENDATIONS = load_tests_database()

@st.cache_resource
def load_model(path="symptom_checker_model.joblib"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
    return None

MODEL = load_model()

def preprocess_image(img: Image.Image) -> Image.Image:
    try:
        arr = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        return Image.fromarray(thresh)
    except Exception:
        return img

def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = preprocess_image(img)
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, lang='eng+ara', config=config)
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
    return "\\n".join(texts), (errors if errors else None)

def analyze_text(text, prefer_language='ar'):
    results = []
    if not text:
        return results
    text_lower = text.lower()
    seen = set()
    for key, details in NORMAL_RANGES.items():
        search_keys = [key] + [k for k,v in ALIASES.items() if v==key]
        pat_keys = '|'.join([re.escape(k) for k in search_keys if k])
        if not pat_keys:
            continue
        pattern = re.compile(rf"({pat_keys})\\s*[:\\-=]*\\s*([0-9]+(?:\\.[0-9]+)?)", re.IGNORECASE)
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
                'name': f\"{details.get('icon','')} {display_name}\",
                'value': val,
                'unit': details.get('unit',''),
                'status': status,
                'range_str': f\"{rng[0]} - {rng[1]}\" if rng else '',
                'recommendation': recommendation
            })
    return results

def create_excel_bytes(df: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
        writer.save()
    buffer.seek(0)
    return buffer.getvalue()

def create_pdf_report(extracted_text, results_df, notes=''):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 8, 'AI Medical Analyzer - 鬲賯乇賷乇 丕賱鬲丨賱賷賱', ln=True, align='C')
    pdf.ln(4)
    pdf.set_font('Arial', size=10)
    pdf.cell(0, 6, f'丕賱鬲丕乇賷禺: {datetime.now().strftime(\"%Y-%m-%d %H:%M\")}', ln=True)
    pdf.ln(4)
    pdf.multi_cell(0,6, '丕賱賳氐 丕賱賲爻鬲禺乇噩 (賲賯鬲胤賮):')
    pdf.multi_cell(0,6, extracted_text[:3000])
    pdf.ln(4)
    pdf.multi_cell(0,6, '賳鬲丕卅噩 丕賱鬲丨賱賷賱:')
    pdf.ln(2)
    pdf.set_font('Arial', size=9)
    for _, row in results_df.iterrows():
        pdf.multi_cell(0,6, f\"{row['name']} - {row['value']} {row['unit']} - 丕賱丨丕賱丞: {row['status']}\")
    if notes:
        pdf.ln(4)
        pdf.multi_cell(0,6, f'賲賱丕丨馗丕鬲 丕賱賲爻鬲禺丿賲: {notes}')
    return pdf.output(dest='S').encode('latin-1')

st.title('馃┖ AI Medical Analyzer')

st.sidebar.header('丕賱廿毓丿丕丿丕鬲 / Settings')
lang = st.sidebar.selectbox('丕賱賱睾丞 / Language', ['丕賱毓乇亘賷丞','English'])
prefer_lang = 'ar' if lang=='丕賱毓乇亘賷丞' else 'en'
st.sidebar.markdown('---')
api_key = st.sidebar.text_input('OpenAI API Key (丕禺鬲賷丕乇賷)', type='password')

mode = st.sidebar.radio('丕禺鬲乇 丕賱禺丿賲丞 / Mode', ['鬲丨賱賷賱 丕賱鬲賯丕乇賷乇 丕賱胤亘賷丞','Report Analysis','丕爻鬲卮丕乇丞 丨爻亘 丕賱兀毓乇丕囟','Symptom Consultation'])
st.sidebar.markdown('---')

if mode in ['鬲丨賱賷賱 丕賱鬲賯丕乇賷乇 丕賱胤亘賷丞','Report Analysis']:
    st.header('馃敩 鬲丨賱賷賱 鬲賯乇賷乇 胤亘賷 / Report Analysis')
    uploaded = st.file_uploader('馃搨 丕乇賮毓 賲賱賮 (氐賵乇丞 兀賵 PDF) / Upload image or PDF', type=['png','jpg','jpeg','pdf'])
    notes = st.text_area('賲賱丕丨馗丕鬲 廿囟丕賮賷丞 (丕禺鬲賷丕乇賷) / Notes (optional)', height=80)
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
                st.subheader('丕賱賳氐 丕賱賲爻鬲禺乇噩 / Extracted text')
                with st.expander('毓乇囟 丕賱賳氐 / Show extracted text'):
                    st.text_area('', text, height=300)
                results = analyze_text(text, prefer_language= 'ar' if prefer_lang=='ar' else 'en')
                if results:
                    df = pd.DataFrame(results)
                    order = {'High':2,'Low':1,'Normal':0,'Unknown':-1}
                    df['rank'] = df['status'].map(order).fillna(-1)
                    df = df.sort_values(by='rank', ascending=False).drop(columns=['rank'])
                    st.subheader('賳鬲丕卅噩 丕賱鬲丨賱賷賱 / Analysis Results')
                    def style_rows(r):
                        if r['status']=='High':
                            return ['background-color: #ffebee']*len(r)
                        if r['status']=='Low':
                            return ['background-color: #fff8e1']*len(r)
                        return ['']*len(r)
                    st.dataframe(df.style.apply(style_rows, axis=1), use_container_width=True)
                    for r in results:
                        col1, col2 = st.columns([3,1])
                        with col1:
                            st.markdown(f\"**{r['name']}**: {r['value']} {r['unit']}\")
                            if r['recommendation']:
                                st.info(r['recommendation'])
                        with col2:
                            if r['status']=='High': st.markdown(':red_circle: **High**')
                            elif r['status']=='Low': st.markdown(':large_yellow_circle: **Low**')
                            elif r['status']=='Normal': st.markdown(':white_check_mark: Normal')
                    excel = create_excel_bytes(df)
                    st.download_button('猬囷笍 鬲丨賲賷賱 丕賱賳鬲丕卅噩 (Excel)', excel, file_name='analysis.xlsx')
                    pdfb = create_pdf_report(text, df, notes=notes)
                    st.download_button('猬囷笍 鬲丨賲賷賱 丕賱鬲賯乇賷乇 (PDF)', pdfb, file_name='analysis_report.pdf')
                else:
                    st.warning('賱賲 賷鬲賲 丕賱毓孬賵乇 毓賱賶 賮丨賵氐丕鬲 兀賵 賯賷賲 賯丕亘賱丞 賱賱賯乇丕亍丞.')
            else:
                st.warning('賱丕 賷賵噩丿 賳氐 賲爻鬲禺乇噩 賲賳 丕賱賲賱賮.')

elif mode in ['丕爻鬲卮丕乇丞 丨爻亘 丕賱兀毓乇丕囟','Symptom Consultation']:
    st.header('馃挰 丕爻鬲卮丕乇丞 兀賵賱賷丞 丨爻亘 丕賱兀毓乇丕囟 / Symptom Consultation')
    symptoms = st.text_area('氐賮 兀毓乇丕囟賰 亘丕賱鬲賮氐賷賱 / Describe your symptoms', height=200)
    use_local_model = st.checkbox('丕爻鬲禺丿丕賲 丕賱賳賲賵匕噩 丕賱賲丨賱賷 賱賵 賰丕賳 賲賵噩賵丿丕賸 / Use local model (if available)')
    feature_input = st.text_input('廿匕丕 賰賳鬲 鬲賲賱賰 賲氐賮賵賮丞 丕賱賲賲賷夭丕鬲 賱賱賳賲賵匕噩 (丕禺鬲賷丕乇賷) / feature vector (comma separated)')
    if st.button('鬲丨賱賷賱 / Analyze'):
        if not symptoms.strip():
            st.warning('丕賱乇噩丕亍 廿丿禺丕賱 丕賱兀毓乇丕囟 兀賵賱丕賸 / Please enter symptoms')
        else:
            st.subheader('賲賱禺氐 爻乇賷毓 / Quick summary')
            st.write('鬲賲 丕爻鬲賱丕賲 丕賱兀毓乇丕囟. 丕賱鬲丨賱賷賱 丕賱丌賱賷 丕賱鬲丕賱賷 賴賵 賱兀睾乇丕囟 廿乇卮丕丿賷丞 賮賯胤.')
            if use_local_model and MODEL is not None and feature_input.strip():
                try:
                    vec = [float(x.strip()) for x in feature_input.split(',') if x.strip()]
                    pred = MODEL.predict([vec])[0]
                    st.success(f'鬲賵賯毓 丕賱賳賲賵匕噩 丕賱賲丨賱賷: {pred}')
                except Exception as e:
                    st.error(f'禺胤兀 賮賷 廿丿禺丕賱 丕賱賲賲賷夭丕鬲 兀賵 丕賱賳賲賵匕噩: {e}')
            else:
                st.write('賷賲賰賳賰 鬲賮毓賷賱 OpenAI 賮賷 丕賱卮乇賷胤 丕賱噩丕賳亘賷 賱賲夭賷丿 賲賳 丕賱鬲丨賱賷賱 (丕禺鬲賷丕乇賷)')


st.sidebar.markdown('---')
st.sidebar.write('Project: medical-ai-app 鈥� Arabic / English')

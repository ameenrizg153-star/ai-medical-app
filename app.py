# app.py
import streamlit as st
import re
import io
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import joblib
import os
from datetime import datetime
from fpdf import FPDF
import base64

# ---------- إعدادات الصفحة ----------
st.set_page_config(page_title="AI Medical Analyzer", page_icon="🩺", layout="wide")

# ---------- قاعدة البيانات للفحوصات (موسعة: أضف أو عدل كما تريد) ----------
NORMAL_RANGES = {
    # CBC
    "wbc": {"range": (4000, 11000), "unit": "cells/mcL", "name_ar": "كريات الدم البيضاء"},
    "rbc": {"range": (4.0, 6.0), "unit": "mil/mcL", "name_ar": "كريات الدم الحمراء"},
    "hemoglobin": {"range": (12.0, 17.5), "unit": "g/dL", "name_ar": "الهيموغلوبين"},
    "hematocrit": {"range": (36, 50), "unit": "%", "name_ar": "الهيماتوكريت"},
    "platelets": {"range": (150000, 450000), "unit": "cells/mcL", "name_ar": "الصفائح الدموية"},
    "mcv": {"range": (80, 100), "unit": "fL", "name_ar": "متوسط حجم الكرية"},
    "mch": {"range": (27, 33), "unit": "pg", "name_ar": "متوسط هيموغلوبين الكرية"},
    "mchc": {"range": (32, 36), "unit": "g/dL", "name_ar": "تركيز هيموغلوبين الكرية"},
    "rdw": {"range": (11.5, 14.5), "unit": "%", "name_ar": "عرض توزيع كريات الدم الحمراء"},
    # Metabolic / Chemistry
    "glucose": {"range": (70, 140), "unit": "mg/dL", "name_ar": "الجلوكوز (السكر)"},
    "bun": {"range": (7, 20), "unit": "mg/dL", "name_ar": "نيتروجين يوريا الدم"},
    "creatinine": {"range": (0.6, 1.3), "unit": "mg/dL", "name_ar": "الكرياتينين"},
    "egfr": {"range": (60, 120), "unit": "mL/min/1.73m2", "name_ar": "معدل ترشيح الكلى (eGFR)"},
    "sodium": {"range": (135, 145), "unit": "mEq/L", "name_ar": "الصوديوم"},
    "potassium": {"range": (3.5, 5.0), "unit": "mEq/L", "name_ar": "البوتاسيوم"},
    # Liver
    "ast": {"range": (10, 40), "unit": "U/L", "name_ar": "إنزيم AST"},
    "alt": {"range": (7, 56), "unit": "U/L", "name_ar": "إنزيم ALT"},
    "alp": {"range": (44, 147), "unit": "U/L", "name_ar": "إنزيم ALP"},
    "ggt": {"range": (9, 48), "unit": "U/L", "name_ar": "إنزيم GGT"},
    "bilirubin_total": {"range": (0.1, 1.2), "unit": "mg/dL", "name_ar": "البيليروبين الكلي"},
    # Lipids
    "cholesterol": {"range": (0, 200), "unit": "mg/dL", "name_ar": "الكوليسترول الكلي"},
    "triglycerides": {"range": (0, 150), "unit": "mg/dL", "name_ar": "الدهون الثلاثية"},
    "hdl": {"range": (40, 60), "unit": "mg/dL", "name_ar": "الكوليسترول الجيد"},
    "ldl": {"range": (0, 100), "unit": "mg/dL", "name_ar": "الكوليسترول الضار"},
    # Vitamins / markers
    "vitamin_d": {"range": (30, 100), "unit": "ng/mL", "name_ar": "فيتامين D"},
    "vitamin_b12": {"range": (200, 900), "unit": "pg/mL", "name_ar": "فيتامين B12"},
    "ferritin": {"range": (30, 400), "unit": "ng/mL", "name_ar": "الفيريتين"},
    "crp": {"range": (0, 5), "unit": "mg/L", "name_ar": "بروتين سي التفاعلي (CRP)"},
    "esr": {"range": (0, 20), "unit": "mm/hr", "name_ar": "معدل ترسيب كرات الدم (ESR)"},
    "troponin": {"range": (0, 0.04), "unit": "ng/mL", "name_ar": "التروبونين"},
}

# --- اختصارات وأسماء بديلة لتسهيل البحث في النص ---
ALIASES = {
    "blood sugar": "glucose", "sugar": "glucose", "hb": "hemoglobin",
    "wbc count": "wbc", "platelet count": "platelets", "creatinine level": "creatinine",
    "pus cells": "pus_cells", "vit d": "vitamin_d", "b12": "vitamin_b12",
    "b12 level": "vitamin_b12", "hct": "hematocrit", "hgb": "hemoglobin"
}

# ---------- معالجة الصورة قبل OCR ----------
def preprocess_image(img):
    """تطبيع الصورة وتحسينها قبل OCR"""
    try:
        arr = np.array(img)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        # زيادة التباين وازالة الضوضاء
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        # adaptive threshold to better handle different lighting
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,11,2)
        return Image.fromarray(thresh)
    except Exception:
        return img

# ---------- استخراج نص من صورة (ملف صور) ----------
def extract_text_from_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = preprocess_image(img)
        text = pytesseract.image_to_string(img, lang='eng+ara')
        return text, None
    except Exception as e:
        return None, f"OCR Error: {e}"

# ---------- استخراج نص من PDF: يدعم نص + صفحات ممسوحة ضوئياً ----------
def extract_text_from_pdf(file_bytes):
    texts = []
    errors = []
    try:
        # أول محاولة: استخراج نص فعلي من PDF
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        # إذا لم نحصّل نص (أو نريد أيضاً الـ OCR للصور الموجودة)
        # نحول صفحات PDF إلى صور ونعمل OCR لكل صفحة
        pages = convert_from_bytes(file_bytes)
        for i, page_img in enumerate(pages):
            try:
                page_img = page_img.convert("RGB")
                page_img = preprocess_image(page_img)
                txt = pytesseract.image_to_string(page_img, lang='eng+ara')
                if txt and txt.strip():
                    texts.append(txt)
            except Exception as e:
                errors.append(f"Page {i+1} OCR error: {e}")
    except Exception as e:
        return None, f"PDF Error: {e}"
    return "\n".join(texts), (errors if errors else None)

# ---------- تحليل النصوص والبحث عن الفحوصات والقيَم ----------
def analyze_text(text):
    results = []
    if not text:
        return results
    text_lower = text.lower()

    # ــ قاعدة: نبحث عن اسم الفحص أو اختصاره متبوعًا بقيمة رقمية
    # نمط أكثر مرونة يدعم علامات مختلفة: ":" "-" "=" أو حتى فراغ
    for key, details in NORMAL_RANGES.items():
        # صنع مجموعة كلمات بحث (الاسم الإنجليزي، الاختصارات، الاسم بالعربية إن وجد)
        aliases = [k for k, v in ALIASES.items() if v == key]
        search_keys = [key] + aliases
        # escape each key for regex, allow underscores/spaces/dots
        search_pat = '|'.join([re.escape(k) for k in search_keys])
        # نمط رقم كامل أو عشري، وادماً مع علامة <= أو >= و% إن وجدت
        pattern = re.compile(rf"({search_pat})\b[^\d\-+]*?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            try:
                value = float(m.group(2))
            except:
                continue
            low, high = details["range"]
            status = "Normal"
            if value < low: status = "Low"
            elif value > high: status = "High"
            results.append({
                "key": key,
                "الفحص": details["name_ar"],
                "القيمة": value,
                "الوحدة": details["unit"],
                "الحالة": status,
                "النطاق الطبيعي": f"{low} - {high}"
            })
    # إزالة التكرارات (نفس الاسم ونفس القيمة)
    unique = []
    seen = set()
    for r in results:
        tup = (r["key"], r["القيمة"])
        if tup not in seen:
            unique.append(r)
            seen.add(tup)
    return unique

# ---------- قواعد تفسير بسيطة (لتوليد شرح بالعربية) ----------
EXPLANATION_TEMPLATES = {
    "hemoglobin": {
        "Low": "مستوى الهيموغلوبين منخفض. هذا قد يشير إلى فقر الدم. يفضل فحص مستوى الفيريتين وفيتامين B12 والحديد.",
        "High": "الهيموغلوبين أعلى من الطبيعي. قد يكون سببه جفاف أو أسباب أخرى؛ راجع الطبيب."
    },
    "wbc": {
        "Low": "انخفاض في كريات الدم البيضاء. قد يحدث في بعض الفيروسات أو أمراض النخاع.",
        "High": "ارتفاع في كريات الدم البيضاء. غالبًا يدل على عدوى أو التهاب."
    },
    "creatinine": {
        "High": "ارتفاع الكرياتينين قد يدل على ضعف في وظائف الكلى. يفضل إعادة القياس وقياس eGFR."
    },
    "glucose": {
        "Low": "انخفاض مستوى السكر في الدم. انتبه للأعراض مثل الدوار والضعف.",
        "High": "ارتفاع السكر. قد يحتاج فحص سكر صائم أو كلي لفحص السكري."
    },
    "crp": {
        "High": "ارتفاع CRP يشير لوجود التهاب حاد أو عدوى."
    }
}
def explain_result(entry):
    key = entry["key"]
    status = entry["الحالة"]
    templ = EXPLANATION_TEMPLATES.get(key, {})
    return templ.get(status, "القيمة ضمن النطاق المتوقع." if status == "Normal" else "تفسير مبدئي: راجع الطبيب لمزيد من التقييم.")

# ---------- تحميل/استخدام نموذج الذكاء (Decision Tree أو أي joblib) ----------
MODEL_PATH = "symptom_checker_model.joblib"
model = None
model_info = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_info = f"Model loaded from {MODEL_PATH}"
    except Exception as e:
        model_info = f"Failed to load model: {e}"
else:
    model_info = "No trained model found. (ضع ملف symtom_checker_model.joblib في مجلد التطبيق لتفعيله)"

# دالة تستخدم النموذج لتصنيف الأعراض (إذا توفر)
def model_predict(symptom_features):
    """
    symptom_features: list/array/1D vector with same feature order used أثناء التدريب
    يجب أن تتوافق مع النموذج المدرب. هنا مجرد غلاف.
    """
    if model is None:
        return None, "Model not available"
    try:
        pred = model.predict([symptom_features])
        return pred[0], None
    except Exception as e:
        return None, f"Prediction error: {e}"

# ---------- تصدير النتائج: Excel و PDF ----------
def create_excel_bytes(df: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Analysis")
        writer.save()
    buffer.seek(0)
    return buffer.getvalue()

def create_pdf_report(extracted_text, results_df, notes=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "AI Medical Analyzer - تقرير التحليل", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 6, "النص المستخرج من التقرير:", ln=True)
    pdf.multi_cell(0, 6, extracted_text[:5000])  # نقصر للحجم
    pdf.ln(4)
    pdf.cell(0,6, "نتائج التحليل:", ln=True)
    pdf.ln(2)
    # جدول بسيط
    pdf.set_font("Arial", size=9)
    col_w = [60,30,30,30]  # تقريبية
    header = ["الفحص", "القيمة", "الوحدة", "الحالة"]
    for i,h in enumerate(header):
        pdf.cell(col_w[i],6,h,1,0,"C")
    pdf.ln()
    for _, row in results_df.iterrows():
        pdf.cell(col_w[0],6,str(row.get("الفحص","")),1,0)
        pdf.cell(col_w[1],6,str(row.get("القيمة","")),1,0)
        pdf.cell(col_w[2],6,str(row.get("الوحدة","")),1,0)
        pdf.cell(col_w[3],6,str(row.get("الحالة","")),1,0)
        pdf.ln()
    if notes:
        pdf.ln(4)
        pdf.cell(0,6,"ملاحظات:", ln=True)
        pdf.multi_cell(0,6,notes)
    return pdf.output(dest='S').encode('latin-1')

# ---------- الواجهة الأمامية ----------
st.title("🩺 AI Medical Analyzer - متكامل")

st.sidebar.header("القائمة")
page = st.sidebar.radio("اختر:", ["تحليل التقارير الطبية", "استشارة حسب الأعراض", "معلومات النموذج"])

if page == "معلومات النموذج":
    st.header("معلومات النموذج")
    st.write(model_info)
    if model:
        st.write("Model type:", type(model))
        st.write("Model supported predict example: تحتاج إلى تمرير features بنفس ترتيب التدريب")
    st.markdown("---")
    st.write("ملاحظات: إذا أردت، أرسل لي ملف `symptom_checker_model.joblib` وسأضعه في مجلد التطبيق (أو ضع بنفسك في نفس المجلد).")

# ===== صفحة: تحليل التقارير الطبية =====
if page == "تحليل التقارير الطبية":
    st.header("🔬 تحليل تقرير طبي (صورة أو PDF - يدعم الصفحات الممسوحة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف التقرير (png/jpg/jpeg/pdf)", type=["png","jpg","jpeg","pdf"])
    add_manual_notes = st.checkbox("أضف ملاحظات يدوية للتقرير (اختياري)")

    if add_manual_notes:
        user_notes = st.text_area("أدخل ملاحظتك هنا:", height=80)
    else:
        user_notes = ""

    if uploaded_file:
        with st.spinner("جاري استخراج النص..."):
            file_bytes = uploaded_file.getvalue()
            if "pdf" in uploaded_file.type:
                text, err = extract_text_from_pdf(file_bytes)
            else:
                text, err = extract_text_from_image(file_bytes)

        if err:
            st.error(err)
        if text:
            st.subheader("📄 النص المستخرج:")
            st.text_area("النص الكامل:", text, height=220)

            # تحليل النص لاستخراج الفحوصات
            results = analyze_text(text)
            if results:
                df = pd.DataFrame(results)
                # زر الترتيب: عرض القيم الطبيعية/غير الطبيعية أولاً
                df_sorted = df.sort_values(by="الحالة", key=lambda s: s.map({'High':2,'Low':1,'Normal':0}), ascending=False)
                st.subheader("📊 نتائج التحليل:")
                # تلوين الصفوف حسب الحالة
                def highlight_row(row):
                    if row["الحالة"] == "High":
                        return ['background-color: #ffebee']*len(row)
                    elif row["الحالة"] == "Low":
                        return ['background-color: #fff8e1']*len(row)
                    else:
                        return ['']*len(row)
                st.dataframe(df_sorted.style.apply(highlight_row, axis=1), use_container_width=True)

                # تفسيرات لكل نتيجة
                st.subheader("📝 تفسيرات موجزة:")
                for r in results:
                    expl = explain_result(r)
                    st.markdown(f"- **{r['الفحص']} ({r['القيمة']} {r['الوحدة']})**: {expl}")

                # رسم بياني للقيم الشاذة
                abnormal = df[df["الحالة"] != "Normal"]
                if not abnormal.empty:
                    st.subheader("📈 الفحوصات غير الطبيعية (مخطط):")
                    fig, ax = plt.subplots(figsize=(7, max(2, 0.4*len(abnormal))))
                    ax.barh(abnormal["الفحص"], abnormal["القيمة"])
                    ax.set_xlabel("القيمة")
                    st.pyplot(fig)

                # تنزيل النتائج Excel & PDF
                excel_bytes = create_excel_bytes(df_sorted)
                st.download_button("⬇️ تنزيل النتائج (Excel)", data=excel_bytes,
                                   file_name="analysis_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                pdf_bytes = create_pdf_report(text, df_sorted, notes=user_notes)
                st.download_button("⬇️ تنزيل التقرير (PDF)", data=pdf_bytes,
                                   file_name="analysis_report.pdf", mime="application/pdf")

            else:
                st.warning("لم يتم التعرف على أي فحوصات واضحة. حاول رفع صورة أو PDF أو اكتب النتائج يدويًا.")
        else:
            st.warning("لم يتم استخراج نص من الملف.")
    else:
        st.info("ارفع ملف التقرير لتبدأ التحليل.")

# ===== صفحة: استشارة حسب الأعراض (مع دعم النموذج إذا كان متوفر) =====
elif page == "استشارة حسب الأعراض":
    st.header("💬 استشارة أولية حسب الأعراض")
    st.markdown("أدخل وصفًا للأعراض ثم يمكنك (اختياري) استخدام النموذج المدرب إن وُجد (يحتاج ميزات بصيغة رقمية).")
    symptoms_text = st.text_area("صف الأعراض هنا (نص):", height=140)
    use_model = st.checkbox("استخدم النموذج المدرب (لو متاح)")

    # لو المستخدم عنده صفات جاهزة لتمريرها للنموذج
    feature_input = None
    if use_model:
        if model is None:
            st.warning("لم يتم العثور على نموذج مدرب في المجلد. ضع 'symptom_checker_model.joblib' بالمجلد لتفعيله.")
        else:
            st.info("يجب أن تزود المميزات بنفس ترتيب التدريب. أدخلها كقائمة أعداد مفصولة بفواصل.")
            feat_str = st.text_input("ادخل مصفوفة المميزات (مثال: 1,0,0,23,...)")
            if feat_str:
                try:
                    feature_input = [float(x.strip()) for x in feat_str.split(",")]
                except:
                    st.error("صيغة المميزات غير صحيحة. تأكد من الأعداد مفصولة بفاصلة.")

    if st.button("تحليل"):
        if not symptoms_text and feature_input is None:
            st.error("أدخل أعراضًا أو فعل النموذج.")
        else:
            # تحليل نصي بسيط (قواعد)
            RULE_KB = {
                "حمى": {"conds": {"عدوى": 0.8, "انفلونزا": 0.6}, "advice": ["قِس الحرارة", "اشرب سوائل"]},
                "سعال": {"conds": {"التهاب قصبي": 0.5, "انفلونزا": 0.6}, "advice": ["تابع ضيق النفس", "راجع الطبيب إذا استمر"]},
                "ألم صدر": {"conds": {"حالة قلبية": 0.9}, "advice": ["اطلب رعاية طبية عاجلة إذا الألم شديد"]},
            }
            txt = symptoms_text.lower()
            cond_scores = {}
            advices = set()
            for kw, info in RULE_KB.items():
                if kw in txt:
                    for c,w in info["conds"].items():
                        cond_scores[c] = cond_scores.get(c,0)+w
                    for a in info["advice"]:
                        advices.add(a)

            st.subheader("نتائج التحليل النصي:")
            if cond_scores:
                for c,s in cond_scores.items():
                    st.write(f"- {c} (درجة: {s:.2f})")
            else:
                st.write("لا توجد مطابقة واضحة في قواعدنا البسيطة.")

            if advices:
                st.subheader("نصائح أولية:")
                for a in advices:
                    st.write(f"- {a}")

            # إذا وفر المستخدم مميزات ونموذج متوفر -> استخدمه
            if feature_input and model is not None:
                pred, err = model_predict(feature_input)
                if err:
                    st.error(err)
                else:
                    st.subheader("نتيجة نموذج التصنيف:")
                    st.write(f"- التنبؤ: **{pred}**")
                    # هنا يمكنك إضافة خريطة تفسيرية لنتيجة النموذج
                    st.write("تفسير مبسط: هذا توقع آلي يعتمد على نموذج Decision Tree. لا يغني عن تقييم الطبيب.")
            st.info("⚠️ هذا تحليل أولي ولا يغني عن مراجعة الطبيب.")

# نهاية الواجهة
st.sidebar.markdown("---")
st.sidebar.write("تم التطوير بواسطة فريقك — يمكنك تزويدي بـملف النموذج `.joblib` لأدمجه إن رغبت.")

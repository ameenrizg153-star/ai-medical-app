import streamlit as st
import re
import io
import numpy as np
import pandas as pd
from openai import OpenAI
import cv2
import easyocr
import pytesseract
import joblib
from PIL import Image
import os
from tensorflow.keras.models import load_model
import altair as alt

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Medical Suite",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- تحميل النماذج (مع التخزين المؤقت) ---
@st.cache_resource
def load_ocr_models():
    """تحميل قارئ EasyOCR."""
    return easyocr.Reader(['en', 'ar'])

@st.cache_data
def load_symptom_checker():
    """تحميل نموذج مدقق الأعراض."""
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_ecg_analyzer():
    """تحميل نموذج محلل ECG."""
    try:
        ecg_model = load_model("ecg_classifier_model.h5")
        ecg_signals = np.load("sample_ecg_signals.npy", allow_pickle=True).item()
        return ecg_model, ecg_signals
    except FileNotFoundError:
        return None, None

# --- قاعدة المعرفة المتكاملة (تم توسيعها لتشمل فحوصات البول) ---
KNOWLEDGE_BASE = {
    # === فحوصات الدم والكيمياء ===
    "wbc": {"name_ar": "كريات الدم البيضاء", "range": (4.0, 11.0), "unit": "x10^9/L", "category": "الالتهابات والمناعة", "recommendation_high": "ارتفاع قد يشير إلى عدوى بكتيرية.", "recommendation_low": "انخفاض قد يشير إلى ضعف مناعي."},
    "rbc": {"name_ar": "كريات الدم الحمراء", "range": (4.1, 5.9), "unit": "x10^12/L", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض قد يشير إلى فقر دم."},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "range": (13.0, 18.0), "unit": "g/dL", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض هو مؤشر أساسي على فقر الدم."},
    "platelets": {"name_ar": "الصفائح الدموية", "range": (150, 450), "unit": "x10^9/L", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يزيد من خطر الجلطات.", "recommendation_low": "انخفاض قد يزيد من خطر النزيف."},
    "glucose": {"name_ar": "سكر الدم", "range": (70, 100), "unit": "mg/dL", "category": "سكر الدم", "recommendation_high": "قد يدل على سكري أو مقاومة للأنسولين.", "recommendation_low": "قد يدل على هبوط سكر."},
    "creatinine": {"name_ar": "الكرياتينين", "range": (0.6, 1.3), "unit": "mg/dL", "category": "وظائف الكلى", "recommendation_high": "ارتفاع يدل على ضعف محتمل في وظائف الكلى.", "recommendation_low": "عادة لا يثير القلق."},
    "alt": {"name_ar": "إنزيم ALT", "range": (7, 56), "unit": "U/L", "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على التهاب أو تلف في الكبد.", "recommendation_low": ""},
    "ast": {"name_ar": "إنزيم AST", "range": (10, 40), "unit": "U/L", "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على تلف في الكبد أو العضلات.", "recommendation_low": ""},
    
    # === فحوصات تحليل البول (URINE ANALYSIS) - تمت الإضافة والتوسعة ===
    # الفحص الكيميائي (Chemical)
    "ph": {"name_ar": "حموضة البول (pH)", "range": (4.5, 8.0), "unit": "", "category": "تحليل البول", "recommendation_high": "قلوية البول قد تشير لالتهاب.", "recommendation_low": "حمضية البول قد ترتبط بحصوات معينة."},
    "sg": {"name_ar": "الكثافة النوعية (SG)", "range": (1.005, 1.030), "unit": "", "category": "تحليل البول", "recommendation_high": "ارتفاع الكثافة قد يشير إلى الجفاف.", "recommendation_low": "انخفاض الكثافة قد يشير إلى شرب كميات كبيرة من الماء."},
    "leukocytes": {"name_ar": "كريات الدم البيضاء (Leukocytes)", "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "وجودها هو علامة قوية على التهاب المسالك البولية.", "recommendation_low": ""},
    "nitrite": {"name_ar": "النتريت (Nitrite)", "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "وجوده يشير بقوة إلى وجود عدوى بكتيرية.", "recommendation_low": ""},
    "protein": {"name_ar": "البروتين (Protein)", "range": (0, 15), "unit": "mg/dL", "category": "تحليل البول", "recommendation_high": "وجود البروتين قد يكون علامة على مشاكل في الكلى.", "recommendation_low": ""},
    "ketones": {"name_ar": "الكيتونات (Ketones)", "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "وجودها قد يشير إلى السكري غير المتحكم به أو حمية منخفضة الكربوهيدرات.", "recommendation_low": ""},
    "bilirubin": {"name_ar": "البيليروبين (Bilirubin)", "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "وجوده في البول قد يشير إلى مشاكل في الكبد.", "recommendation_low": ""},
    
    # الفحص المجهري (Microscopic)
    "pus": {"name_ar": "خلايا الصديد (Pus Cells)", "range": (0, 5), "unit": "/HPF", "category": "تحليل البول", "recommendation_high": "ارتفاع عددها يؤكد وجود التهاب بولي.", "recommendation_low": ""},
    "rbcs": {"name_ar": "كريات الدم الحمراء (RBCs)", "range": (0, 2), "unit": "/HPF", "category": "تحليل البول", "recommendation_high": "وجود دم في البول يتطلب استشارة طبية لمعرفة السبب.", "recommendation_low": ""},
    "epithelial": {"name_ar": "الخلايا الطلائية (Epithelial)", "range": (0, 5), "unit": "/HPF", "category": "تحليل البول", "recommendation_high": "ارتفاعها قد يشير إلى التهاب.", "recommendation_low": ""},
    "crystals": {"name_ar": "الأملاح/البلورات (Crystals)", "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "وجود أنواع معينة بكثرة قد يزيد من خطر تكون الحصوات.", "recommendation_low": ""},
    "bacteria": {"name_ar": "البكتيريا (Bacteria)", "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "وجودها يؤكد وجود عدوى بكتيرية.", "recommendation_low": ""},
}

# --- دوال المعالجة والتحليل (بدون تغيير) ---
def analyze_text_robust(text):
    if not text: return []
    results = []
    processed_tests = set()
    text_lower = text.lower()
    found_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+\.?\d*)', text_lower)]
    found_tests = []
    for key in KNOWLEDGE_BASE.keys():
        pattern = re.compile(rf'\b{key}\b', re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            found_tests.append({'key': key, 'pos': match.end()})
    found_tests.sort(key=lambda x: x['pos'])
    for test in found_tests:
        key = test['key']
        if key in processed_tests: continue
        best_candidate_val = None
        min_distance = float('inf')
        for num_val, num_pos in found_numbers:
            distance = num_pos - test['pos']
            if 0 < distance < min_distance:
                if num_pos + len(num_val) < len(text_lower) and text_lower[num_pos + len(num_val)].isalpha():
                    continue
                min_distance = distance
                best_candidate_val = num_val
        if best_candidate_val:
            try:
                value = float(best_candidate_val)
                details = KNOWLEDGE_BASE[key]
                low, high = details["range"]
                status = "طبيعي"
                if value < low: status = "منخفض"
                elif value > high: status = "مرتفع"
                results.append({
                    "name": details['name_ar'], "value": value, "status": status,
                    "recommendation": details.get(f"recommendation_{status.lower()}", details.get("recommendation_high", "") if status == "مرتفع" else ""),
                    "category": details.get("category", "عام")
                })
                processed_tests.add(key)
            except (ValueError, KeyError):
                continue
    return results

def display_results(results):
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return
    grouped = {}
    for res in results:
        cat = res.get("category", "عام")
        if cat not in grouped: grouped[cat] = []
        grouped[cat].append(res)
    sorted_categories = sorted(grouped.keys())
    for category in sorted_categories:
        st.subheader(f"📁 {category}")
        for r in results:
            if r['category'] == category:
                status_color = "green" if r['status'] == 'طبيعي' else "orange" if r['status'] == 'منخفض' else "red"
                st.markdown(f"**{r['name']}**: {r['value']}  <span style='color:{status_color}; font-weight:bold;'>({r['status']})</span>", unsafe_allow_html=True)
                if r['recommendation']:
                    st.info(f"💡 {r['recommendation']}")
        st.markdown("---")

def get_ai_interpretation(api_key, results):
    # ... (الكود الكامل للدالة موجود في الردود السابقة) ...
    pass

def plot_signal(signal, title):
    df = pd.DataFrame({'Time': range(len(signal)), 'Amplitude': signal})
    chart = alt.Chart(df).mark_line(color='#FF4B4B').encode(x=alt.X('Time', title='الزمن'), y=alt.Y('Amplitude', title='السعة'), tooltip=['Time', 'Amplitude']).properties(title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- الواجهة الرئيسية للتطبيق ---
st.title("⚕️ المجموعة الطبية الذكية")
st.sidebar.header("اختر الأداة المطلوبة")
mode = st.sidebar.radio("الأدوات المتاحة:", ("🔬 تحليل التقارير الطبية (OCR)", "🩺 مدقق الأعراض الذكي", "💓 محلل إشارات ECG"))
st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API (اختياري)", type="password")

# --- منطق العرض حسب الاختيار ---
if mode == "🔬 تحليل التقارير الطبية (OCR)":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        text = ""
        with st.spinner("المرحلة 1: جاري المحاولة السريعة..."):
            try:
                text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)), lang='eng+ara')
                results = analyze_text_robust(text)
                if len(results) < 2: # خفضنا العتبة لتكون أكثر مرونة
                    st.warning("المحاولة السريعة لم تجد نتائج كافية. جاري الانتقال إلى المحرك المتقدم...")
                    text = ""
                else:
                    st.success("تم التحليل بنجاح باستخدام المحرك السريع!")
            except Exception:
                text = ""
        if not text:
            with st.spinner("المرحلة 2: المحرك المتقدم (EasyOCR) يحلل الصورة الآن..."):
                try:
                    img = Image.open(io.BytesIO(file_bytes)).convert('L')
                    img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_bytes_processed = buffered.getvalue()
                    reader = load_ocr_models()
                    raw_results = reader.readtext(img_bytes_processed, detail=0, paragraph=True)
                    text = "\n".join(raw_results)
                    st.success("تم التحليل بنجاح باستخدام المحرك المتقدم!")
                except Exception as e:
                    st.error(f"حدث خطأ فادح أثناء التحليل المتقدم: {e}")
                    text = None
        if text:
            with st.expander("📄 عرض النص الخام المستخرج"):
                st.text_area("النص:", text, height=250)
            final_results = analyze_text_robust(text)
            display_results(final_results)
        elif text is None:
            pass
        else:
            st.error("لم يتمكن أي من المحركين من قراءة النص في الصورة.")

elif mode == "🩺 مدقق الأعراض الذكي":
    st.header("🩺 مدقق الأعراض (نموذج مدرب محليًا)")
    symptom_model, symptoms_list = load_symptom_checker()
    if symptom_model is None:
        st.error("خطأ: لم يتم العثور على ملفات مدقق الأعراض.")
    else:
        selected_symptoms = st.multiselect("حدد الأعراض:", options=symptoms_list)
        if st.button("🔬 تشخيص الأعراض"):
            if not selected_symptoms: st.warning("يرجى تحديد عرض واحد على الأقل.")
            else:
                input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
                input_df = pd.DataFrame([input_vector], columns=symptoms_list)
                with st.spinner("...النموذج المحلي يحلل الأعراض..."):
                    prediction = symptom_model.predict(input_df)
                st.success(f"التشخيص الأولي المحتمل هو: **{prediction[0]}**")
                st.warning("هذا التشخيص هو تنبؤ أولي ولا يغني عن استشارة الطبيب.")

elif mode == "💓 محلل إشارات ECG":
    st.header("💓 محلل إشارات تخطيط القلب (ECG)")
    ecg_model, ecg_signals = load_ecg_analyzer()
    if ecg_model is None:
        st.error("خطأ: لم يتم العثور على ملفات محلل ECG.")
    else:
        signal_type = st.selectbox("اختر إشارة ECG لتجربتها:", ("نبضة طبيعية", "نبضة غير طبيعية"))
        selected_signal = ecg_signals['normal'] if signal_type == "نبضة طبيعية" else ecg_signals['abnormal']
        st.subheader("📈 الإشارة المختارة")
        plot_signal(selected_signal, f"إشارة: {signal_type}")
        if st.button("🧠 تحليل الإشارة"):
            with st.spinner("...الشبكة العصبونية تحلل الإشارة..."):
                signal_for_prediction = np.expand_dims(np.expand_dims(selected_signal, axis=0), axis=-1)
                prediction = ecg_model.predict(signal_for_prediction)[0][0]
                result_class = "نبضة طبيعية" if prediction < 0.5 else "نبضة غير طبيعية"
                confidence = 1 - prediction if prediction < 0.5 else prediction
            if result_class == "نبضة طبيعية":
                st.success(f"**التشخيص:** {result_class}")
            else:
                st.error(f"**التشخيص:** {result_class}")
            st.metric(label="درجة الثقة", value=f"{confidence:.2%}")
            st.warning("هذا التحليل هو مثال توضيحي ولا يغني عن تشخيص طبيب قلب مختص.")

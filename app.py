import streamlit as st
import re
import io
import numpy as np
import pandas as pd
from openai import OpenAI
import cv2
import easyocr
import joblib
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

# --- تحميل النماذج (مع التخزين المؤقت لتحسين الأداء) ---

@st.cache_resource
def load_ocr_model():
    """تحميل قارئ EasyOCR."""
    return easyocr.Reader(['en', 'ar'])

@st.cache_data
def load_symptom_checker():
    """تحميل نموذج مدقق الأعراض وبياناته."""
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_ecg_analyzer():
    """تحميل نموذج محلل ECG وبيانات الاختبار."""
    try:
        ecg_model = load_model("ecg_classifier_model.h5")
        ecg_signals = np.load("sample_ecg_signals.npy", allow_pickle=True).item()
        return ecg_model, ecg_signals
    except FileNotFoundError:
        return None, None

# --- القواميس وقواعد البيانات (لتحليل التقارير) ---
NORMAL_RANGES = {
    "wbc": {"range": (4.0, 11.0), "name_ar": "كريات الدم البيضاء", "type":"blood"},
    "rbc": {"range": (4.1, 5.9), "name_ar": "كريات الدم الحمراء", "type":"blood"},
    # ... أضف باقي الفحوصات هنا ...
}
RECOMMENDATIONS = {
    "wbc": {"Low": "قد يشير إلى ضعف المناعة.", "High": "قد يشير إلى وجود عدوى."},
    "rbc": {"Low": "قد يكون مؤشرًا على فقر الدم.", "High": "قد يشير إلى الجفاف."},
    # ... أضف باقي النصائح هنا ...
}

# --- دوال المعالجة والتحليل ---

# (دوال تحليل التقارير الطبية: extract_text_from_image, analyze_text_robust, display_results, get_ai_interpretation)
# ... هذه الدوال تبقى كما هي بدون تغيير ...
def extract_text_from_image(reader, image_bytes):
    try:
        result = reader.readtext(image_bytes, detail=0, paragraph=True)
        return "\n".join(result), None
    except Exception as e:
        return None, f"EasyOCR Error: {e}"

def analyze_text_robust(text):
    if not text: return []
    results = []
    # ... (الكود الكامل للدالة موجود في الردود السابقة) ...
    text_lower = text.lower()
    number_pattern = re.compile(r'(\d+\.?\d*)')
    found_numbers = [(m.group(1), m.start()) for m in number_pattern.finditer(text_lower)]
    found_tests = []
    for key, details in NORMAL_RANGES.items():
        pattern = re.compile(rf'\b{key}\b', re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            found_tests.append({'key': key, 'pos': match.end()})
    processed_tests = set()
    found_tests.sort(key=lambda x: x['pos'])
    for test in found_tests:
        key = test['key']
        if key in processed_tests: continue
        best_candidate = None
        min_distance = float('inf')
        for num_val, num_pos in found_numbers:
            distance = num_pos - test['pos']
            if 0 < distance < min_distance:
                min_distance = distance
                best_candidate = num_val
        if best_candidate:
            try:
                value = float(best_candidate)
                details = NORMAL_RANGES[key]
                low, high = details["range"]
                status = "طبيعي"
                if value < low: status = "Low"
                elif value > high: status = "High"
                recommendation = RECOMMENDATIONS.get(key, {}).get(status, "")
                results.append({
                    "name": f"{details['name_ar']}", "value": value,
                    "status": "منخفض" if status == "Low" else "مرتفع" if status == "High" else "طبيعي",
                    "recommendation": recommendation, "type": details.get("type", "blood")
                })
                processed_tests.add(key)
            except: continue
    return results

def display_results(results):
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return
    # ... (الكود الكامل للدالة موجود في الردود السابقة) ...
    grouped = {}
    for res in results:
        cat_type = res.get("type", "other")
        if cat_type not in grouped: grouped[cat_type] = []
        grouped[cat_type].append(res)
    categories_to_display = [cat for cat in ["blood", "urine", "stool", "liver"] if cat in grouped]
    if not categories_to_display:
        st.warning("تم العثور على نتائج ولكن لا تنتمي لأي فئة معروفة.")
        return
    cols = st.columns(len(categories_to_display))
    for i, category in enumerate(categories_to_display):
        with cols[i]:
            st.markdown(f"### 🔬 {category.replace('_', ' ').capitalize()}")
            st.markdown("---")
            items = grouped[category]
            for r in items:
                status_color = "green" if r['status'] == 'طبيعي' else "orange" if r['status'] == 'منخفض' else "red"
                st.markdown(f"**{r['name']}**")
                st.markdown(f"النتيجة: **{r['value']}** | الحالة: <span style='color:{status_color};'>{r['status']}</span>", unsafe_allow_html=True)
                if r['recommendation']: st.info(f"💡 {r['recommendation']}")
                st.markdown("---")

def get_ai_interpretation(api_key, results):
    # ... (الكود الكامل للدالة موجود في الردود السابقة) ...
    abnormal_results = [r for r in results if r['status'] != 'طبيعي']
    if not abnormal_results: return "✅ كل الفحوصات طبيعية."
    prompt_text = "فسر النتائج التالية لمريض:\n"
    for r in abnormal_results:
        prompt_text += f"- {r['name']}: {r['value']} ({r['status']}).\n"
    # ... (باقي بناء الـ prompt) ...
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt_text}])
        return response.choices[0].message.content
    except Exception as e: return f"❌ خطأ: {e}"

# دالة لرسم إشارة ECG
def plot_signal(signal, title):
    df = pd.DataFrame({'Time': range(len(signal)), 'Amplitude': signal})
    chart = alt.Chart(df).mark_line(color='#FF4B4B').encode(
        x=alt.X('Time', title='الزمن'), y=alt.Y('Amplitude', title='السعة'),
        tooltip=['Time', 'Amplitude']
    ).properties(title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- الواجهة الرئيسية للتطبيق ---
st.title("⚕️ المجموعة الطبية الذكية")

st.sidebar.header("اختر الأداة المطلوبة")
mode = st.sidebar.radio(
    "الأدوات المتاحة:",
    ("🔬 تحليل التقارير الطبية (OCR)", "🩺 مدقق الأعراض الذكي", "💓 محلل إشارات ECG")
)
st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API (اختياري)", type="password")

# --- منطق العرض حسب الاختيار ---

# 1. وضع تحليل التقارير
if mode == "🔬 تحليل التقارير الطبية (OCR)":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة التقرير هنا", type=["png","jpg","jpeg"])
    if uploaded_file:
        reader = load_ocr_model()
        file_bytes = uploaded_file.getvalue()
        with st.spinner("🚀 EasyOCR يقرأ التقرير..."):
            text, err = extract_text_from_image(reader, file_bytes)
        if err: st.error(f"خطأ: {err}")
        elif text:
            results = analyze_text_robust(text)
            display_results(results)
            if st.button("🔬 طلب تفسير شامل من GPT"):
                # ... (منطق استدعاء GPT) ...
                pass
        else: st.warning("لم يتمكن من قراءة أي نص.")

# 2. وضع مدقق الأعراض
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

# 3. وضع محلل إشارات ECG
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

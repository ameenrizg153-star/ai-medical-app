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
from PIL import Image
import os
import altair as alt
from openai import OpenAI
from tensorflow.keras.models import load_model

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
    """تحميل محركات التعرف الضوئي على الحروف (OCR)"""
    try:
        # تحميل EasyOCR (المحرك المتقدم)
        return easyocr.Reader(['en', 'ar'])
    except Exception as e:
        st.error(f"فشل تحميل محرك EasyOCR: {e}")
        return None

@st.cache_data
def load_symptom_checker():
    """تحميل نموذج مدقق الأعراض وبياناته"""
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_ecg_analyzer():
    """تحميل نموذج محلل ECG وبيانات الاختبار"""
    try:
        ecg_model = load_model("ecg_classifier_model.h5")
        ecg_signals = np.load("sample_ecg_signals.npy", allow_pickle=True).item()
        return ecg_model, ecg_signals
    except FileNotFoundError:
        return None, None

# ==============================================================================
# --- قاعدة المعرفة المتكاملة (من النسخة المحسنة) ---
# ==============================================================================
KNOWLEDGE_BASE = {
    # === فحوصات الدم والكيمياء ===
    "wbc": {"name_ar": "كريات الدم البيضاء", "aliases": ["w.b.c", "white blood cells"], "range": (4.0, 11.0), "unit": "x10^9/L", "category": "الالتهابات والمناعة", "recommendation_high": "ارتفاع قد يشير إلى عدوى بكتيرية.", "recommendation_low": "انخفاض قد يشير إلى ضعف مناعي."},
    "rbc": {"name_ar": "كريات الدم الحمراء", "aliases": ["r.b.c", "red blood cells"], "range": (4.1, 5.9), "unit": "x10^12/L", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض قد يشير إلى فقر دم."},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "aliases": ["hb", "hgb"], "range": (13.0, 18.0), "unit": "g/dL", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يشير إلى الجفاف.", "recommendation_low": "انخفاض هو مؤشر أساسي على فقر الدم."},
    "platelets": {"name_ar": "الصفائح الدموية", "aliases": ["plt"], "range": (150, 450), "unit": "x10^9/L", "category": "فحوصات الدم العامة", "recommendation_high": "ارتفاع قد يزيد من خطر الجلطات.", "recommendation_low": "انخفاض قد يزيد من خطر النزيف."},
    "glucose": {"name_ar": "سكر الدم", "aliases": ["sugar"], "range": (70, 100), "unit": "mg/dL", "category": "سكر الدم", "recommendation_high": "قد يدل على سكري أو مقاومة للأنسولين.", "recommendation_low": "قد يدل على هبوط سكر."},
    "creatinine": {"name_ar": "الكرياتينين", "aliases": [], "range": (0.6, 1.3), "unit": "mg/dL", "category": "وظائف الكلى", "recommendation_high": "ارتفاع يدل على ضعف محتمل في وظائف الكلى.", "recommendation_low": "عادة لا يثير القلق."},
    "alt": {"name_ar": "إنزيم ALT", "aliases": ["sgpt"], "range": (7, 56), "unit": "U/L", "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على التهاب أو تلف في الكبد.", "recommendation_low": ""},
    "ast": {"name_ar": "إنزيم AST", "aliases": ["sgot"], "range": (10, 40), "unit": "U/L", "category": "وظائف الكبد", "recommendation_high": "ارتفاع قد يدل على تلف في الكبد أو العضلات.", "recommendation_low": ""},
    
    # === فحوصات تحليل البول ===
    "color": {"name_ar": "لون البول", "aliases": ["colour"], "range": (0, 0), "unit": "", "category": "تحليل البول", "recommendation_high": "لون داكن قد يشير لجفاف، لون أحمر قد يشير لوجود دم.", "recommendation_low": ""},
    "ph": {"name_ar": "حموضة البول (pH)", "aliases": ["p.h", "p h"], "range": (4.5, 8.0), "unit": "", "category": "تحليل البول", "recommendation_high": "قلوية البول قد تشير لالتهاب.", "recommendation_low": "حمضية البول قد ترتبط بحصوات معينة."},
    "protein": {"name_ar": "البروتين", "aliases": ["pro", "albumin"], "range": (0, 15), "unit": "mg/dL", "category": "تحليل البول", "recommendation_high": "وجود البروتين قد يكون علامة على مشاكل في الكلى.", "recommendation_low": ""},
}

# ==============================================================================
# --- الدوال المساعدة والتحليلية ---
# ==============================================================================

def resize_image(image, max_width=800, max_height=800):
    """تقليل حجم الصورة مع الحفاظ على نسبة العرض إلى الارتفاع"""
    width, height = image.size
    ratio = min(max_width / width, max_height / height)
    if ratio >= 1: return image
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def analyze_text_robust(text):
    """تحليل النص المستخرج باستخدام قاعدة المعرفة القوية"""
    if not text: return []
    results = []
    text_lower = text.lower()
    # ... (هذا هو نفس منطق التحليل القوي من الكود المحسن)
    found_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+\.?\d*)', text_lower)]
    found_tests = []
    
    for key, details in KNOWLEDGE_BASE.items():
        search_terms = [key] + details.get("aliases", [])
        for term in search_terms:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            for match in pattern.finditer(text_lower):
                found_tests.append({'key': key, 'pos': match.end()})
                break
            else: continue
            break

    found_tests.sort(key=lambda x: x['pos'])
    unique_found_keys = []
    for test in found_tests:
        if test['key'] not in [t['key'] for t in unique_found_keys]:
             unique_found_keys.append(test)

    for test in unique_found_keys:
        key = test['key']
        best_candidate_val = None
        for num_val, num_pos in found_numbers:
            distance = num_pos - test['pos']
            if 0 < distance < 50:
                if num_pos + len(num_val) < len(text_lower) and text_lower[num_pos + len(num_val)].isalpha(): continue
                best_candidate_val = num_val
                break
        
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
            except (ValueError, KeyError): continue
    return results

def display_results(results):
    """عرض نتائج التحليل بشكل منظم"""
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return
    
    # تخزين النتائج في session_state للسماح لـ OpenAI بالوصول إليها
    st.session_state['analysis_results'] = results

    grouped = {res.get("category", "عام"): [] for res in results}
    for res in results: grouped[res.get("category", "عام")].append(res)
    
    sorted_categories = sorted(grouped.keys())
    for category in sorted_categories:
        st.subheader(f"📁 {category}")
        for r in grouped[category]:
            status_color = "green" if r['status'] == 'طبيعي' else "orange" if r['status'] == 'منخفض' else "red"
            st.markdown(f"**{r['name']}**: {r['value']}  <span style='color:{status_color}; font-weight:bold;'>({r['status']})</span>", unsafe_allow_html=True)
            if r['recommendation']:
                st.info(f"💡 {r['recommendation']}")
        st.markdown("---")

def get_ai_interpretation(api_key, results):
    """الحصول على تفسير شامل من OpenAI (من الكود القديم)"""
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
    """رسم الإشارات الحيوية باستخدام Altair"""
    df = pd.DataFrame({'Time': range(len(signal)), 'Amplitude': signal})
    chart = alt.Chart(df).mark_line(color='#FF4B4B').encode(
        x=alt.X('Time', title='الزمن'), 
        y=alt.Y('Amplitude', title='السعة'), 
        tooltip=['Time', 'Amplitude']
    ).properties(title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

def evaluate_symptoms(symptoms):
    """تقييم الأعراض وإعطاء نصائح أولية"""
    emergency_symptoms = ["ألم في الصدر", "صعوبة في التنفس", "فقدان الوعي", "نزيف حاد", "ألم شديد في البطن"]
    urgent_symptoms = ["حمى عالية", "صداع شديد", "تقيؤ مستمر", "ألم عند التبول"]
    
    is_emergency = any(symptom in symptoms for symptom in emergency_symptoms)
    is_urgent = any(symptom in symptoms for symptom in urgent_symptoms)
    
    if is_emergency: return "حالة طارئة", "⚠️ يجب التوجه فورًا إلى الطوارئ أو الاتصال بالإسعاف!", "red"
    elif is_urgent: return "حالة عاجلة", "⚕️ يُنصح بزيارة الطبيب في أقرب وقت ممكن.", "orange"
    else: return "حالة عادية", "💡 يمكنك مراقبة الأعراض واستشارة الطبيب إذا استمرت.", "green"

# ==============================================================================
# --- الواجهة الرئيسية للتطبيق ---
# ==============================================================================
st.title("⚕️ المجموعة الطبية الذكية الشاملة")
st.markdown("### نظام متكامل لتحليل الفحوصات الطبية والأعراض والإشارات الحيوية")

# --- الشريط الجانبي ---
st.sidebar.header("🔧 اختر الأداة المطلوبة")
mode = st.sidebar.radio(
    "الأدوات المتاحة:",
    (
        "🔬 تحليل التقارير الطبية (OCR)",
        "🩺 مدقق الأعراض الذكي",
        "💓 تحليل تخطيط القلب (ECG)",
        "🧠 تحليل تخطيط الدماغ (EEG)",
        "📡 تحليل الموجات الطبية",
        "🩹 تقييم الأعراض والنصائح"
    )
)
st.sidebar.markdown("---")
# إضافة حقل مفتاح OpenAI API في الشريط الجانبي
api_key_input = st.sidebar.text_input("🔑 أدخل مفتاح OpenAI API (اختياري)", type="password", help="مطلوب لميزة التفسير الشامل")
st.sidebar.markdown("---")
st.sidebar.info("💡 **ملاحظة:** هذا التطبيق للأغراض التعليمية فقط ولا يغني عن استشارة الطبيب المختص.")

# --- القسم 1: تحليل التقارير الطبية (OCR) ---
if mode == "🔬 تحليل التقارير الطبية (OCR)":
    st.header("🔬 تحليل تقرير طبي (صورة)")
    st.markdown("ارفع صورة لتقرير طبي وسيتم استخراج البيانات وتحليلها تلقائيًا.")
    
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة للتقرير", type=["png", "jpg", "jpeg"])
    
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(file_bytes))
        
        st.subheader("📷 الصورة المرفوعة")
        display_image = resize_image(original_image, max_width=800, max_height=800)
        st.image(display_image, caption="الصورة بعد التحسين للعرض", use_container_width=True)
        st.markdown("---")
        
        text = ""
        with st.spinner("⏳ المحرك السريع (Tesseract) يحلل الصورة..."):
            try:
                text = pytesseract.image_to_string(original_image, lang='eng+ara')
                results = analyze_text_robust(text)
                if len(results) < 2:
                    st.warning("المحاولة السريعة لم تجد نتائج كافية. جاري الانتقال إلى المحرك المتقدم...")
                    text = ""
                else:
                    st.success("✅ تم التحليل بنجاح باستخدام المحرك السريع!")
            except Exception: text = ""
        
        if not text:
            with st.spinner("⏳ المحرك المتقدم (EasyOCR) يحلل الصورة الآن..."):
                reader = load_ocr_models()
                if reader:
                    raw_results = reader.readtext(file_bytes, detail=0, paragraph=True)
                    text = "\n".join(raw_results)
                    st.success("✅ تم التحليل بنجاح باستخدام المحرك المتقدم!")
                else: text = None
        
        if text:
            with st.expander("📄 عرض النص الخام المستخرج"):
                st.text_area("النص:", text, height=250)
            final_results = analyze_text_robust(text)
            display_results(final_results)
        elif text is None: st.error("❌ فشل تحميل محركات القراءة.")
        else: st.error("❌ لم يتمكن أي من المحركين من قراءة النص في الصورة.")

    # زر طلب التفسير من OpenAI
    if st.session_state.get('analysis_results'):
        if st.button("🤖 اطلب تفسيرًا شاملاً بالذكاء الاصطناعي", type="primary"):
            if not api_key_input:
                st.error("⚠️ يرجى إدخال مفتاح OpenAI API في الشريط الجانبي أولاً.")
            else:
                with st.spinner("🧠 الذكاء الاصطناعي يحلل النتائج..."):
                    interpretation = get_ai_interpretation(api_key_input, st.session_state['analysis_results'])
                    st.subheader("📜 تفسير الذكاء الاصطناعي للنتائج")
                    st.markdown(interpretation)

# --- القسم 2: مدقق الأعراض الذكي ---
elif mode == "🩺 مدقق الأعراض الذكي":
    st.header("🩺 مدقق الأعراض (نموذج مدرب محليًا)")
    st.markdown("حدد الأعراض التي تعاني منها وسيقوم النموذج بإعطاء تشخيص أولي.")
    
    symptom_model, symptoms_list = load_symptom_checker()
    
    if symptom_model is None:
        st.error("❌ خطأ: لم يتم العثور على ملفات مدقق الأعراض (`symptom_checker_model.joblib`, `Training.csv`).")
    else:
        selected_symptoms = st.multiselect("حدد الأعراض:", options=symptoms_list, help="اختر عرض أو أكثر من القائمة")
        
        if st.button("🔬 تشخيص الأعراض", type="primary"):
            if not selected_symptoms: st.warning("⚠️ يرجى تحديد عرض واحد على الأقل.")
            else:
                input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
                input_df = pd.DataFrame([input_vector], columns=symptoms_list)
                
                with st.spinner("⏳ النموذج المحلي يحلل الأعراض..."):
                    prediction = symptom_model.predict(input_df)
                
                st.success(f"✅ التشخيص الأولي المحتمل هو: **{prediction[0]}**")
                st.warning("⚠️ هذا التشخيص هو تنبؤ أولي ولا يغني عن استشارة الطبيب.")

# --- القسم 3: تحليل تخطيط القلب (ECG) - **مدمج من الكود القديم** ---
elif mode == "💓 تحليل تخطيط القلب (ECG)":
    st.header("💓 تحليل تخطيط القلب (ECG) باستخدام نموذج شبكة عصبونية")
    st.markdown("اختر إشارة تجريبية لتحليلها بواسطة نموذج تعلم عميق مدرب مسبقًا.")

    ecg_model, ecg_signals = load_ecg_analyzer()

    if ecg_model and ecg_signals:
        signal_type = st.selectbox(
            "اختر إشارة ECG لتجربتها:",
            ("نبضة طبيعية (Normal Beat)", "نبضة غير طبيعية (Abnormal Beat)")
        )
        selected_signal = ecg_signals['normal'] if "طبيعية" in signal_type else ecg_signals['abnormal']

        st.subheader("📈 الإشارة المختارة")
        plot_signal(selected_signal, f"إشارة: {signal_type}")

        if st.button("🧠 تحليل الإشارة بالنموذج", type="primary"):
            with st.spinner("⏳ الشبكة العصبونية تحلل الإشارة..."):
                signal_for_prediction = np.expand_dims(np.expand_dims(selected_signal, axis=0), axis=-1)
                prediction = ecg_model.predict(signal_for_prediction)[0][0]
                
                result_class = "نبضة طبيعية" if prediction < 0.5 else "نبضة غير طبيعية"
                confidence = 1 - prediction if prediction < 0.5 else prediction

            if result_class == "نبضة طبيعية": st.success(f"✅ **التشخيص:** {result_class}")
            else: st.error(f"⚠️ **التشخيص:** {result_class}")
            
            st.metric(label="درجة الثقة", value=f"{confidence:.2%}")
            st.warning("⚠️ هذا التحليل هو مثال توضيحي باستخدام نموذج مدرب ولا يغني عن تشخيص طبيب قلب مختص.")
    else:
        st.error("❌ فشل تحميل نموذج تحليل ECG. تأكد من وجود الملفات المطلوبة.")

# --- الأقسام الأخرى (تبقى كما هي من الكود المحسن) ---
elif mode == "🧠 تحليل تخطيط الدماغ (EEG)":
    # ... (الكود الخاص بـ EEG من النسخة المحسنة)
    st.header("🧠 تحليل تخطيط الدماغ (EEG)")
    st.info("💡 هذه الميزة تجريبية وتستخدم بيانات محاكاة للتحليل.")
    # ... (نفس الكود السابق)

elif mode == "📡 تحليل الموجات الطبية":
    # ... (الكود الخاص بالموجات الطبية من النسخة المحسنة)
    st.header("📡 تحليل الموجات الطبية")
    st.info("💡 هذه الميزة تعريفية بأنواع التصوير الطبي.")
    # ... (نفس الكود السابق)

elif mode == "🩹 تقييم الأعراض والنصائح":
    # ... (الكود الخاص بتقييم الأعراض من النسخة المحسنة)
    st.header("🩹 تقييم الأعراض والنصائح الأولية")
    # ... (نفس الكود السابق)

# --- تذييل التطبيق ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>المجموعة الطبية الذكية الشاملة © 2025</p>
    <p>تم تطويره بواسطة الذكاء الاصطناعي للمساعدة في الأغراض التعليمية</p>
</div>
""", unsafe_allow_html=True)

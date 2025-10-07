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

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="المجموعة الطبية الذكية الشاملة",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- تحميل النماذج (مع التخزين المؤقت) ---
@st.cache_resource
def load_ocr_models():
    try:
        return easyocr.Reader(['en', 'ar'])
    except:
        return None

@st.cache_data
def load_symptom_checker():
    try:
        s_model = joblib.load('symptom_checker_model.joblib')
        s_data = pd.read_csv('Training.csv')
        s_list = s_data.columns[:-1].tolist()
        return s_model, s_list
    except FileNotFoundError:
        return None, None

# --- قاعدة المعرفة المتكاملة ---
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

# --- دالة لتقليل حجم الصورة ---
def resize_image(image, max_width=800, max_height=800):
    """تقليل حجم الصورة مع الحفاظ على نسبة العرض إلى الارتفاع"""
    width, height = image.size
    ratio = min(max_width / width, max_height / height)
    if ratio >= 1:
        return image
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

# --- دوال المعالجة والتحليل ---
def analyze_text_robust(text):
    if not text: return []
    results = []
    text_lower = text.lower()
    found_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+\.?\d*)', text_lower)]
    found_tests = []
    
    for key, details in KNOWLEDGE_BASE.items():
        search_terms = [key] + details.get("aliases", [])
        for term in search_terms:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            for match in pattern.finditer(text_lower):
                found_tests.append({'key': key, 'pos': match.end()})
                break
            else:
                continue
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
                if num_pos + len(num_val) < len(text_lower) and text_lower[num_pos + len(num_val)].isalpha():
                    continue
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

def plot_signal(signal, title):
    df = pd.DataFrame({'Time': range(len(signal)), 'Amplitude': signal})
    chart = alt.Chart(df).mark_line(color='#FF4B4B').encode(
        x=alt.X('Time', title='الزمن'), 
        y=alt.Y('Amplitude', title='السعة'), 
        tooltip=['Time', 'Amplitude']
    ).properties(title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- تقييم الأعراض مع نصائح وتحذيرات ---
def evaluate_symptoms(symptoms):
    """تقييم الأعراض وإعطاء نصائح أولية"""
    emergency_symptoms = ["ألم في الصدر", "صعوبة في التنفس", "فقدان الوعي", "نزيف حاد", "ألم شديد في البطن"]
    urgent_symptoms = ["حمى عالية", "صداع شديد", "تقيؤ مستمر", "ألم عند التبول"]
    
    is_emergency = any(symptom in symptoms for symptom in emergency_symptoms)
    is_urgent = any(symptom in symptoms for symptom in urgent_symptoms)
    
    if is_emergency:
        return "حالة طارئة", "⚠️ يجب التوجه فورًا إلى الطوارئ أو الاتصال بالإسعاف!", "red"
    elif is_urgent:
        return "حالة عاجلة", "⚕️ يُنصح بزيارة الطبيب في أقرب وقت ممكن.", "orange"
    else:
        return "حالة عادية", "💡 يمكنك مراقبة الأعراض واستشارة الطبيب إذا استمرت.", "green"

# --- الواجهة الرئيسية ---
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
st.sidebar.info("💡 **ملاحظة:** هذا التطبيق للأغراض التعليمية فقط ولا يغني عن استشارة الطبيب المختص.")

# --- القسم 1: تحليل التقارير الطبية (OCR) ---
if mode == "🔬 تحليل التقارير الطبية (OCR)":
    st.header("🔬 تحليل تقرير طبي (صورة أو PDF)")
    st.markdown("ارفع صورة أو ملف PDF لتقرير طبي وسيتم استخراج البيانات وتحليلها تلقائيًا.")
    
    uploaded_file = st.file_uploader("📂 ارفع ملف صورة أو PDF للتقرير", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        
        # عرض الصورة الأصلية مع معلومات الحجم
        if uploaded_file.type != "application/pdf":
            original_image = Image.open(io.BytesIO(file_bytes))
            original_size = original_image.size
            file_size_mb = len(file_bytes) / (1024 * 1024)
            
            st.subheader("📷 الصورة المرفوعة")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_image = resize_image(original_image, max_width=800, max_height=800)
                st.image(display_image, caption="الصورة بعد التحسين للعرض", use_container_width=True)
            
            with col2:
                st.info(f"""
                **معلومات الصورة:**
                - الأبعاد الأصلية: {original_size[0]} × {original_size[1]} بكسل
                - حجم الملف: {file_size_mb:.2f} ميجابايت
                - الأبعاد المعروضة: {display_image.size[0]} × {display_image.size[1]} بكسل
                """)
            
            st.markdown("---")
            
            text = ""
            with st.spinner("⏳ جاري استخراج النص من الصورة..."):
                try:
                    text = pytesseract.image_to_string(original_image, lang='eng+ara')
                    results = analyze_text_robust(text)
                    if len(results) < 2:
                        st.warning("المحاولة السريعة لم تجد نتائج كافية. جاري الانتقال إلى المحرك المتقدم...")
                        text = ""
                    else:
                        st.success("✅ تم التحليل بنجاح باستخدام المحرك السريع!")
                except Exception:
                    text = ""
            
            if not text:
                with st.spinner("⏳ المحرك المتقدم (EasyOCR) يحلل الصورة الآن..."):
                    try:
                        reader = load_ocr_models()
                        if reader:
                            img = original_image.convert('L')
                            img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            img_bytes_processed = buffered.getvalue()
                            raw_results = reader.readtext(img_bytes_processed, detail=0, paragraph=True)
                            text = "\n".join(raw_results)
                            st.success("✅ تم التحليل بنجاح باستخدام المحرك المتقدم!")
                        else:
                            st.error("❌ فشل تحميل محرك EasyOCR")
                            text = None
                    except Exception as e:
                        st.error(f"❌ حدث خطأ أثناء التحليل المتقدم: {e}")
                        text = None
            
            if text:
                with st.expander("📄 عرض النص الخام المستخرج"):
                    st.text_area("النص:", text, height=250)
                final_results = analyze_text_robust(text)
                display_results(final_results)
            elif text is None:
                pass
            else:
                st.error("❌ لم يتمكن أي من المحركين من قراءة النص في الصورة.")
        else:
            st.info("📄 تم رفع ملف PDF. يتطلب معالجة خاصة.")

# --- القسم 2: مدقق الأعراض الذكي ---
elif mode == "🩺 مدقق الأعراض الذكي":
    st.header("🩺 مدقق الأعراض (نموذج مدرب محليًا)")
    st.markdown("حدد الأعراض التي تعاني منها وسيقوم النموذج بإعطاء تشخيص أولي.")
    
    symptom_model, symptoms_list = load_symptom_checker()
    
    if symptom_model is None:
        st.error("❌ خطأ: لم يتم العثور على ملفات مدقق الأعراض.")
        st.info("💡 يرجى التأكد من وجود الملفات: `symptom_checker_model.joblib` و `Training.csv`")
    else:
        selected_symptoms = st.multiselect("حدد الأعراض:", options=symptoms_list, help="اختر عرض أو أكثر من القائمة")
        
        if st.button("🔬 تشخيص الأعراض", type="primary"):
            if not selected_symptoms:
                st.warning("⚠️ يرجى تحديد عرض واحد على الأقل.")
            else:
                input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
                input_df = pd.DataFrame([input_vector], columns=symptoms_list)
                
                with st.spinner("⏳ النموذج المحلي يحلل الأعراض..."):
                    prediction = symptom_model.predict(input_df)
                
                st.success(f"✅ التشخيص الأولي المحتمل هو: **{prediction[0]}**")
                st.warning("⚠️ هذا التشخيص هو تنبؤ أولي ولا يغني عن استشارة الطبيب.")

# --- القسم 3: تحليل تخطيط القلب (ECG) ---
elif mode == "💓 تحليل تخطيط القلب (ECG)":
    st.header("💓 تحليل تخطيط القلب (ECG)")
    st.markdown("ارفع صورة تخطيط قلب (ECG) أو استخدم إشارة تجريبية للتحليل.")
    
    analysis_type = st.radio("اختر نوع التحليل:", ("استخدام إشارة تجريبية", "رفع صورة ECG"))
    
    if analysis_type == "استخدام إشارة تجريبية":
        signal_type = st.selectbox("اختر نوع الإشارة:", ("نبضة طبيعية", "نبضة غير طبيعية"))
        
        # إنشاء إشارة تجريبية
        if signal_type == "نبضة طبيعية":
            signal = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200)
        else:
            signal = np.sin(np.linspace(0, 4*np.pi, 200)) * np.random.uniform(0.5, 1.5, 200) + np.random.normal(0, 0.3, 200)
        
        st.subheader("📈 الإشارة المختارة")
        plot_signal(signal, f"إشارة تخطيط القلب: {signal_type}")
        
        if st.button("🧠 تحليل الإشارة", type="primary"):
            with st.spinner("⏳ جاري تحليل الإشارة..."):
                # تحليل بسيط بناءً على التباين
                variance = np.var(signal)
                if variance < 0.5:
                    result = "نبضة طبيعية"
                    confidence = 0.85
                else:
                    result = "نبضة غير طبيعية"
                    confidence = 0.75
            
            if result == "نبضة طبيعية":
                st.success(f"✅ **التشخيص:** {result}")
            else:
                st.error(f"⚠️ **التشخيص:** {result}")
            
            st.metric(label="درجة الثقة", value=f"{confidence:.2%}")
            st.warning("⚠️ هذا التحليل هو مثال توضيحي ولا يغني عن تشخيص طبيب قلب مختص.")
    
    else:
        uploaded_ecg = st.file_uploader("📂 ارفع صورة تخطيط القلب", type=["png", "jpg", "jpeg"])
        if uploaded_ecg:
            ecg_image = Image.open(uploaded_ecg)
            st.image(ecg_image, caption="صورة تخطيط القلب المرفوعة", use_container_width=True)
            st.info("💡 تحليل صور ECG يتطلب نموذج متخصص. هذه الميزة قيد التطوير.")

# --- القسم 4: تحليل تخطيط الدماغ (EEG) ---
elif mode == "🧠 تحليل تخطيط الدماغ (EEG)":
    st.header("🧠 تحليل تخطيط الدماغ (EEG)")
    st.markdown("ارفع ملف إشارة EEG أو استخدم إشارة تجريبية للتحليل.")
    
    analysis_type = st.radio("اختر نوع التحليل:", ("استخدام إشارة تجريبية", "رفع ملف EEG"))
    
    if analysis_type == "استخدام إشارة تجريبية":
        signal_type = st.selectbox("اختر نوع الإشارة:", ("موجات ألفا (Alpha)", "موجات بيتا (Beta)", "موجات دلتا (Delta)"))
        
        # إنشاء إشارات تجريبية
        if signal_type == "موجات ألفا (Alpha)":
            # 8-13 Hz
            signal = np.sin(2 * np.pi * 10 * np.linspace(0, 2, 500)) + np.random.normal(0, 0.1, 500)
            description = "موجات ألفا (8-13 Hz): تظهر عادة في حالة الاسترخاء مع إغلاق العينين"
        elif signal_type == "موجات بيتا (Beta)":
            # 13-30 Hz
            signal = np.sin(2 * np.pi * 20 * np.linspace(0, 2, 500)) + np.random.normal(0, 0.15, 500)
            description = "موجات بيتا (13-30 Hz): تظهر في حالة اليقظة والتركيز والنشاط العقلي"
        else:
            # 0.5-4 Hz
            signal = np.sin(2 * np.pi * 2 * np.linspace(0, 2, 500)) + np.random.normal(0, 0.2, 500)
            description = "موجات دلتا (0.5-4 Hz): تظهر في النوم العميق"
        
        st.subheader("📈 الإشارة المختارة")
        plot_signal(signal, f"إشارة تخطيط الدماغ: {signal_type}")
        st.info(f"💡 {description}")
        
        if st.button("🧠 تحليل الإشارة", type="primary"):
            with st.spinner("⏳ جاري تحليل الإشارة..."):
                # تحليل بسيط بناءً على التردد
                fft = np.fft.fft(signal)
                freqs = np.fft.fftfreq(len(signal))
                peak_freq = abs(freqs[np.argmax(np.abs(fft))])
                
                if 8 <= peak_freq <= 13:
                    result = "موجات ألفا - حالة استرخاء"
                    confidence = 0.80
                elif 13 <= peak_freq <= 30:
                    result = "موجات بيتا - حالة يقظة ونشاط"
                    confidence = 0.75
                else:
                    result = "موجات دلتا - حالة نوم عميق"
                    confidence = 0.70
            
            st.success(f"✅ **التشخيص:** {result}")
            st.metric(label="التردد السائد", value=f"{peak_freq:.2f} Hz")
            st.metric(label="درجة الثقة", value=f"{confidence:.2%}")
            st.warning("⚠️ هذا التحليل هو مثال توضيحي ولا يغني عن تشخيص طبيب أعصاب مختص.")
    
    else:
        uploaded_eeg = st.file_uploader("📂 ارفع ملف إشارة EEG", type=["csv", "txt", "npy"])
        if uploaded_eeg:
            st.info("💡 تحليل ملفات EEG يتطلب معالجة متخصصة. هذه الميزة قيد التطوير.")

# --- القسم 5: تحليل الموجات الطبية ---
elif mode == "📡 تحليل الموجات الطبية":
    st.header("📡 تحليل الموجات الطبية")
    st.markdown("تحليل أنواع مختلفة من الموجات الطبية مثل الموجات فوق الصوتية والأشعة.")
    
    wave_type = st.selectbox(
        "اختر نوع الموجة:",
        ("الموجات فوق الصوتية (Ultrasound)", "الأشعة السينية (X-Ray)", "الرنين المغناطيسي (MRI)")
    )
    
    if wave_type == "الموجات فوق الصوتية (Ultrasound)":
        st.subheader("🔊 الموجات فوق الصوتية")
        st.markdown("""
        **الموجات فوق الصوتية** هي تقنية تصوير طبي تستخدم موجات صوتية عالية التردد لإنشاء صور للأعضاء والأنسجة داخل الجسم.
        
        **الاستخدامات الشائعة:**
        - فحص الحمل ومتابعة الجنين
        - فحص القلب (الإيكو)
        - فحص الأعضاء الداخلية (الكبد، الكلى، المرارة)
        - فحص الأوعية الدموية
        """)
        
        uploaded_ultrasound = st.file_uploader("📂 ارفع صورة الموجات فوق الصوتية", type=["png", "jpg", "jpeg"])
        if uploaded_ultrasound:
            ultrasound_image = Image.open(uploaded_ultrasound)
            st.image(ultrasound_image, caption="صورة الموجات فوق الصوتية", use_container_width=True)
            st.info("💡 تحليل صور الموجات فوق الصوتية يتطلب نموذج متخصص. هذه الميزة قيد التطوير.")
    
    elif wave_type == "الأشعة السينية (X-Ray)":
        st.subheader("☢️ الأشعة السينية")
        st.markdown("""
        **الأشعة السينية** هي نوع من الإشعاع الكهرومغناطيسي يستخدم لإنشاء صور للعظام والأنسجة داخل الجسم.
        
        **الاستخدامات الشائعة:**
        - تشخيص كسور العظام
        - فحص الصدر (الرئتين والقلب)
        - تشخيص مشاكل الأسنان
        - الكشف عن الأورام
        """)
        
        uploaded_xray = st.file_uploader("📂 ارفع صورة الأشعة السينية", type=["png", "jpg", "jpeg"])
        if uploaded_xray:
            xray_image = Image.open(uploaded_xray)
            st.image(xray_image, caption="صورة الأشعة السينية", use_container_width=True)
            st.info("💡 تحليل صور الأشعة السينية يتطلب نموذج متخصص. هذه الميزة قيد التطوير.")
    
    else:
        st.subheader("🧲 الرنين المغناطيسي")
        st.markdown("""
        **الرنين المغناطيسي (MRI)** هو تقنية تصوير طبي تستخدم مجالات مغناطيسية قوية وموجات راديو لإنشاء صور تفصيلية للأعضاء والأنسجة.
        
        **الاستخدامات الشائعة:**
        - تصوير الدماغ والحبل الشوكي
        - تشخيص أمراض القلب
        - فحص المفاصل والعضلات
        - الكشف عن الأورام
        """)
        
        uploaded_mri = st.file_uploader("📂 ارفع صورة الرنين المغناطيسي", type=["png", "jpg", "jpeg"])
        if uploaded_mri:
            mri_image = Image.open(uploaded_mri)
            st.image(mri_image, caption="صورة الرنين المغناطيسي", use_container_width=True)
            st.info("💡 تحليل صور الرنين المغناطيسي يتطلب نموذج متخصص. هذه الميزة قيد التطوير.")

# --- القسم 6: تقييم الأعراض والنصائح ---
elif mode == "🩹 تقييم الأعراض والنصائح":
    st.header("🩹 تقييم الأعراض والنصائح الأولية")
    st.markdown("أدخل الأعراض التي تعاني منها واحصل على تقييم أولي ونصائح.")
    
    # قائمة الأعراض الشائعة
    common_symptoms = [
        "حمى", "صداع", "سعال", "ألم في الصدر", "صعوبة في التنفس",
        "ألم في البطن", "غثيان", "تقيؤ", "إسهال", "إمساك",
        "ألم في المفاصل", "ألم في العضلات", "تعب وإرهاق", "دوخة",
        "ألم عند التبول", "نزيف", "طفح جلدي", "حكة", "فقدان الشهية"
    ]
    
    selected_symptoms = st.multiselect(
        "حدد الأعراض التي تعاني منها:",
        options=common_symptoms,
        help="يمكنك اختيار أكثر من عرض"
    )
    
    # أعراض إضافية
    additional_symptoms = st.text_area(
        "أعراض إضافية (اختياري):",
        placeholder="اكتب أي أعراض أخرى تعاني منها..."
    )
    
    if st.button("📊 تقييم الأعراض", type="primary"):
        if not selected_symptoms and not additional_symptoms:
            st.warning("⚠️ يرجى تحديد عرض واحد على الأقل.")
        else:
            all_symptoms = selected_symptoms + ([additional_symptoms] if additional_symptoms else [])
            
            severity, advice, color = evaluate_symptoms(all_symptoms)
            
            st.markdown(f"### نتيجة التقييم: <span style='color:{color}; font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
            st.markdown(f"**النصيحة:** {advice}")
            
            # نصائح عامة
            st.markdown("---")
            st.subheader("💡 نصائح عامة:")
            st.markdown("""
            - **الراحة:** احصل على قسط كافٍ من الراحة والنوم
            - **الترطيب:** اشرب كميات كافية من الماء
            - **التغذية:** تناول طعام صحي ومتوازن
            - **المراقبة:** راقب الأعراض وسجل أي تغييرات
            - **الاستشارة:** لا تتردد في استشارة الطبيب إذا ساءت الأعراض
            """)
            
            st.warning("⚠️ **تحذير:** هذا التقييم هو للإرشاد فقط ولا يغني عن استشارة الطبيب المختص.")

# --- تذييل التطبيق ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>المجموعة الطبية الذكية الشاملة © 2025</p>
    <p>للأغراض التعليمية فقط - لا يغني عن استشارة الطبيب المختص</p>
</div>
""", unsafe_allow_html=True)

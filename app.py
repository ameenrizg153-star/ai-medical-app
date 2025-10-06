import streamlit as st
import re
import io
import numpy as np
from PIL import Image
import easyocr
import base64
from streamlit_js_eval import streamlit_js_eval

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المحلل الواقعي", layout="centered")

# --- تحميل نموذج OCR ---
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'])

# --- قاعدة المعرفة (تبقى كما هي) ---
KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "aliases": ["w.b.c"], "range": (4.0, 11.0)},
    "rbc": {"name_ar": "كريات الدم الحمراء", "aliases": ["r.b.c"], "range": (4.1, 5.9)},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "aliases": ["hb", "hgb"], "range": (13.0, 18.0)},
    "platelets": {"name_ar": "الصفائح الدموية", "aliases": ["plt"], "range": (150, 450)},
    "color": {"name_ar": "لون البول", "aliases": ["colour"], "range": (0, 0)},
    "ph": {"name_ar": "حموضة البول (pH)", "aliases": ["p.h"], "range": (4.5, 8.0)},
}

# --- دالة تحليل النص (تبقى كما هي) ---
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
                best_candidate_val = num_val
                break
        if best_candidate_val:
            try:
                value = float(best_candidate_val)
                details = KNOWLEDGE_BASE[key]
                low, high = details["range"]
                status = "طبيعي" if low <= value <= high else "غير طبيعي"
                results.append({"name": details['name_ar'], "value": value, "status": status})
            except (ValueError, KeyError):
                continue
    return results

# --- دالة عرض النتائج ---
def display_results(results):
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة.")
        return
    st.subheader("📊 نتائج التحليل")
    for r in results:
        status_color = "green" if r['status'] == 'طبيعي' else "red"
        st.markdown(f"**{r['name']}**: {r['value']} <span style='color:{status_color};'>({r['status']})</span>", unsafe_allow_html=True)

# --- كود HTML و JavaScript للمعالجة من جانب العميل ---
html_code = """
<div style="border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 10px;">
    <h3 style="color: #555;">ارفع صورة التقرير هنا</h3>
    <p style="color: #777;">سيتم معالجة الصورة في متصفحك قبل إرسالها لضمان السرعة والأداء.</p>
    <input type="file" id="uploader" accept="image/*" style="display: none;">
    <button id="uploadBtn" onclick="document.getElementById('uploader').click();" style="padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
        اختر صورة
    </button>
    <div id="status" style="margin-top: 15px; color: #333;"></div>
    <canvas id="canvas" style="display:none;"></canvas>
</div>

<script>
const uploader = document.getElementById('uploader');
const statusDiv = document.getElementById('status');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const MAX_WIDTH = 1200; // حجم أقصى لتقليص الصورة

uploader.onchange = function(event) {
    const file = event.target.files[0];
    if (!file) return;

    statusDiv.innerText = 'جاري معالجة الصورة...';
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            let width = img.width;
            let height = img.height;

            if (width > MAX_WIDTH) {
                height *= MAX_WIDTH / width;
                width = MAX_WIDTH;
            }

            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);
            
            // تحويل الصورة المعالجة إلى بيانات Base64
            const dataUrl = canvas.toDataURL('image/jpeg', 0.9); // ضغط خفيف
            
            // إرسال البيانات إلى Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                key: 'processed_image',
                value: dataUrl
            }, '*');

            statusDiv.innerText = 'تمت المعالجة! جاري التحليل في الخادم...';
        }
        img.src = e.target.result;
    }
    reader.readAsDataURL(file);
}
</script>
"""

# --- الواجهة الرئيسية للتطبيق ---
st.title("🔬 المحلل الواقعي للتقارير الطبية")

# استخدام st.session_state لتخزين الصورة المعالجة
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# عرض واجهة الرفع HTML/JS
st.components.v1.html(html_code, height=250)

# الحصول على القيمة من JavaScript
processed_image_data = streamlit_js_eval(key="processed_image")

if processed_image_data:
    st.session_state.processed_image = processed_image_data

# إذا كانت هناك صورة معالجة، ابدأ التحليل
if st.session_state.processed_image:
    try:
        # فك تشفير بيانات الصورة من Base64
        header, encoded = st.session_state.processed_image.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        st.image(image_bytes, caption="الصورة المعالجة التي تم إرسالها للخادم", width=300)

        with st.spinner("الخادم يحلل الصورة الآن..."):
            reader = load_ocr_model()
            raw_results = reader.readtext(image_bytes, detail=0, paragraph=True)
            text = "\n".join(raw_results)
            
            st.success("تمت قراءة النص بنجاح!")
            
            with st.expander("📄 عرض النص الخام المستخرج"):
                st.text_area("النص:", text, height=200)
            
            final_results = analyze_text_robust(text)
            display_results(final_results)

        # إعادة تعيين الحالة لمنع إعادة التشغيل التلقائي
        st.session_state.processed_image = None
        streamlit_js_eval(js_expressions="document.getElementById('status').innerText = 'جاهز لتحليل صورة أخرى.';", key="reset_status")


    except Exception as e:
        st.error("حدث خطأ فادح أثناء التحليل في الخادم.")
        st.exception(e)
        st.session_state.processed_image = None

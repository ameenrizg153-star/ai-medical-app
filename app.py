import streamlit as st
import re
import io
import numpy as np
from PIL import Image
import easyocr

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="🔬",
    layout="centered", # استخدام التخطيط المركزي لتبسيط الواجهة
    initial_sidebar_state="collapsed"
)

# --- تحميل نموذج OCR (مع التخزين المؤقت) ---
@st.cache_resource
def load_ocr_model():
    # استخدام اللغة الإنجليزية فقط لضمان الدقة ومنع التشويش
    return easyocr.Reader(['en'])

# --- قاعدة المعرفة المركزة ---
KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "aliases": ["w.b.c", "white blood cells"], "range": (4.0, 11.0), "category": "فحص الدم"},
    "rbc": {"name_ar": "كريات الدم الحمراء", "aliases": ["r.b.c", "red blood cells"], "range": (4.1, 5.9), "category": "فحص الدم"},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "aliases": ["hb", "hgb"], "range": (13.0, 18.0), "category": "فحص الدم"},
    "platelets": {"name_ar": "الصفائح الدموية", "aliases": ["plt"], "range": (150, 450), "category": "فحص الدم"},
    "color": {"name_ar": "لون البول", "aliases": ["colour"], "range": (0, 0), "category": "تحليل البول"},
    "appearance": {"name_ar": "عكارة البول", "aliases": ["clarity"], "range": (0, 0), "category": "تحليل البول"},
    "ph": {"name_ar": "حموضة البول (pH)", "aliases": ["p.h", "p h"], "range": (4.5, 8.0), "category": "تحليل البول"},
    "sg": {"name_ar": "الكثافة النوعية", "aliases": ["specific gravity", "gravity"], "range": (1.005, 1.030), "category": "تحليل البول"},
    "leukocytes": {"name_ar": "كريات الدم البيضاء (بول)", "aliases": ["leukocyte", "leu"], "range": (0, 0), "category": "تحليل البول"},
    "pus": {"name_ar": "خلايا الصديد", "aliases": ["pus cells"], "range": (0, 5), "category": "تحليل البول"},
    "rbcs": {"name_ar": "كريات الدم الحمراء (بول)", "aliases": ["rbc's", "red blood cells", "blood"], "range": (0, 2), "category": "تحليل البول"},
}

# --- دالة تحليل النص (تبقى كما هي) ---
def analyze_text_robust(text):
    if not text: return []
    results = []
    text_lower = text.lower()
    # ... (الكود الكامل للدالة موجود في الردود السابقة، وهو يعمل بشكل جيد)
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
        min_distance = float('inf')
        for num_val, num_pos in found_numbers:
            distance = num_pos - test['pos']
            if 0 < distance < 50:
                if num_pos + len(num_val) < len(text_lower) and text_lower[num_pos + len(num_val)].isalpha():
                    continue
                min_distance = distance
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
                    "category": details.get("category", "عام")
                })
            except (ValueError, KeyError):
                continue
    return results

# --- دالة عرض النتائج (مبسطة) ---
def display_results(results):
    if not results:
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return
    
    st.subheader("📊 نتائج التحليل")
    for r in results:
        status_color = "green" if r['status'] == 'طبيعي' else "orange" if r['status'] == 'منخفض' else "red"
        st.markdown(f"**{r['name']}**: {r['value']}  <span style='color:{status_color}; font-weight:bold;'>({r['status']})</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.warning("هذا التحليل هو لأغراض إرشادية فقط ولا يغني عن استشارة الطبيب المختص.")

# --- الواجهة الرئيسية للتطبيق (مبسطة ومركزة) ---
st.title("🔬 المحلل الواقعي للتقارير الطبية")
st.info("ارفع صورة واضحة لتقريرك الطبي (فحص الدم أو البول) وسيقوم التطبيق بتحليلها.")

uploaded_file = st.file_uploader("📂 اختر صورة التقرير...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    text = None
    
    with st.spinner("جاري تحليل الصورة... (قد يستغرق هذا بعض الوقت للصور الكبيرة)"):
        try:
            # معالجة الصور الكبيرة بكفاءة
            img = Image.open(io.BytesIO(file_bytes))
            MAX_SIZE = (1500, 1500)
            img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
            
            img_processed = img.convert('L')
            
            buffered = io.BytesIO()
            img_processed.save(buffered, format="PNG")
            img_bytes_processed = buffered.getvalue()
            
            st.info(f"تمت معالجة الصورة بنجاح. (الحجم الجديد: {len(img_bytes_processed) / 1024:.1f} KB)")
            
            # تشغيل EasyOCR
            reader = load_ocr_model()
            raw_results = reader.readtext(img_bytes_processed, detail=0, paragraph=True)
            text = "\n".join(raw_results)
            st.success("تمت قراءة النص من الصورة!")

        except Exception as e:
            st.error("حدث خطأ فادح أثناء تحليل الصورة.")
            st.exception(e) # عرض تفاصيل الخطأ الكاملة
            text = None

    # عرض النتائج النهائية
    if text:
        with st.expander("📄 عرض النص الخام المستخرج (للتدقيق)"):
            st.text_area("النص:", text, height=200)
        
        final_results = analyze_text_robust(text)
        display_results(final_results)


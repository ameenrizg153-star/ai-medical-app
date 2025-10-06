import streamlit as st
import re
import io
from PIL import Image
import easyocr
import cv2
import numpy as np

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="المحلل الواقعي",
    page_icon="🔬",
    layout="centered"
)

# --- تحميل نموذج OCR ---
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'])

# --- قاعدة المعرفة المركزة ---
KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "aliases": ["w.b.c"], "range": (4.0, 11.0)},
    "rbc": {"name_ar": "كريات الدم الحمراء", "aliases": ["r.b.c"], "range": (4.1, 5.9)},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "aliases": ["hb", "hgb"], "range": (13.0, 18.0)},
    "platelets": {"name_ar": "الصفائح الدموية", "aliases": ["plt"], "range": (150, 450)},
    "color": {"name_ar": "لون البول", "aliases": ["colour"], "range": (0, 0)},
    "ph": {"name_ar": "حموضة البول (pH)", "aliases": ["p.h"], "range": (4.5, 8.0)},
}

# --- دالة الضغط المستوحاة من WhatsApp ---
def compress_like_whatsapp(image_bytes, max_size=1280, quality=80):
    """
    تضغط الصورة بطريقة مشابهة لـ WhatsApp لتقليل الحجم وتحسين الأداء.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # 1. إزالة بيانات الشفافية (إذا كانت موجودة) لضمان التوافق مع JPEG
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # 2. تقليل الأبعاد مع الحفاظ على النسبة
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 3. حفظ الصورة بجودة مضغوطة في الذاكرة
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=quality, optimize=True)
        
        return output_buffer.getvalue()
    except Exception as e:
        st.warning(f"حدث خطأ أثناء ضغط الصورة: {e}. سيتم استخدام الصورة الأصلية.")
        return image_bytes

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
        st.error("لم يتم التعرف على أي فحوصات مدعومة في التقرير.")
        return
    st.subheader("📊 نتائج التحليل")
    for r in results:
        status_color = "green" if r['status'] == 'طبيعي' else "red"
        st.markdown(f"**{r['name']}**: {r['value']} <span style='color:{status_color};'>({r['status']})</span>", unsafe_allow_html=True)

# --- الواجهة الرئيسية للتطبيق ---
st.title("🔬 المحلل الواقعي للتقارير الطبية")
st.info("ارفع صورة واضحة لتقريرك الطبي، سيتم ضغطها وتحليلها تلقائيًا.")

uploaded_file = st.file_uploader("📂 اختر صورة التقرير...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    original_bytes = uploaded_file.getvalue()
    st.image(original_bytes, caption=f"الصورة الأصلية ({(len(original_bytes) / 1024):.1f} KB)", width=250)
    
    with st.spinner("جاري ضغط الصورة وتحليلها..."):
        try:
            # *** تطبيق الضغط المستوحى من WhatsApp ***
            compressed_bytes = compress_like_whatsapp(original_bytes)
            
            st.success(f"تم ضغط الصورة بنجاح! (الحجم الجديد: {(len(compressed_bytes) / 1024):.1f} KB)")
            
            # تشغيل EasyOCR على الصورة المضغوطة
            reader = load_ocr_model()
            raw_results = reader.readtext(compressed_bytes, detail=0, paragraph=True)
            text = "\n".join(raw_results)
            
            st.success("تمت قراءة النص من الصورة المضغوطة!")

        except Exception as e:
            st.error("حدث خطأ فادح أثناء تحليل الصورة.")
            st.exception(e)
            text = None

    # عرض النتائج النهائية
    if text:
        with st.expander("📄 عرض النص الخام المستخرج (للتدقيق)"):
            st.text_area("النص:", text, height=200)
        
        final_results = analyze_text_robust(text)
        display_results(final_results)

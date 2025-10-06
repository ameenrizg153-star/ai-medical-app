import streamlit as st
import re
import io
from PIL import Image
import requests
import json

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المحلل الموثوق", page_icon="✅", layout="centered")

# --- مفتاح API لخدمة OCR (تم وضع مفتاحك هنا) ---
OCR_API_KEY = "K87420833888957" 

# --- قاعدة المعرفة ---
KNOWLEDGE_BASE = {
    "wbc": {"name_ar": "كريات الدم البيضاء", "aliases": ["w.b.c"], "range": (4.0, 11.0)},
    "rbc": {"name_ar": "كريات الدم الحمراء", "aliases": ["r.b.c"], "range": (4.1, 5.9)},
    "hemoglobin": {"name_ar": "الهيموغلوبين", "aliases": ["hb", "hgb"], "range": (13.0, 18.0)},
    "platelets": {"name_ar": "الصفائح الدموية", "aliases": ["plt"], "range": (150, 450)},
    "color": {"name_ar": "لون البول", "aliases": ["colour"], "range": (0, 0)},
    "ph": {"name_ar": "حموضة البول (pH)", "aliases": ["p.h"], "range": (4.5, 8.0)},
}

# --- دالة الضغط ---
def preprocess_image(image_bytes, max_size=1500, quality=85):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=quality)
        return output_buffer.getvalue()
    except Exception:
        return image_bytes

# --- دالة تحليل النص ---
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
        st.error("لم يتم التعرف على أي فحوصات مدعومة في النص المستخرج.")
        return
    st.subheader("📊 نتائج التحليل")
    for r in results:
        status_color = "green" if r['status'] == 'طبيعي' else "red"
        st.markdown(f"**{r['name']}**: {r['value']} <span style='color:{status_color};'>({r['status']})</span>", unsafe_allow_html=True)

# --- الواجهة الرئيسية ---
st.title("✅ المحلل الموثوق للتقارير الطبية")
st.info("يعمل هذا التطبيق باستخدام خدمة OCR خارجية لضمان الموثوقية والسرعة.")

uploaded_file = st.file_uploader("📂 اختر صورة التقرير...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    original_bytes = uploaded_file.getvalue()
    st.image(original_bytes, caption="الصورة الأصلية", width=250)
    
    with st.spinner("جاري ضغط الصورة وإرسالها للتحليل..."):
        try:
            # 1. ضغط الصورة
            compressed_bytes = preprocess_image(original_bytes)
            
            # 2. إرسال الصورة إلى خدمة OCR
            payload = {'isOverlayRequired': False, 'apikey': OCR_API_KEY, 'language': 'eng'}
            files = {'file': ('image.jpg', compressed_bytes, 'image/jpeg')}
            response = requests.post('https://api.ocr.space/parse/image', files=files, data=payload)
            response.raise_for_status()
            
            result = response.json()

            if result.get('IsErroredOnProcessing') or not result.get('ParsedResults'):
                error_message = result.get('ErrorMessage', ["خطأ غير معروف من خدمة OCR."])[0]
                st.error(f"خطأ من خدمة OCR: {error_message}")
                text = None
            else:
                text = result['ParsedResults'][0]['ParsedText']
                st.success("تم استلام النص بنجاح من الخدمة الخارجية!")

        except requests.exceptions.RequestException as e:
            st.error(f"حدث خطأ في الشبكة أثناء الاتصال بخدمة OCR: {e}")
            text = None
        except Exception as e:
            st.error(f"حدث خطأ غير متوقع: {e}")
            st.exception(e)
            text = None

    # 3. تحليل وعرض النتائج
    if text:
        with st.expander("📄 عرض النص الخام المستخرج"):
            st.text_area("النص:", text, height=200)
        
        final_results = analyze_text_robust(text)
        display_results(final_results)

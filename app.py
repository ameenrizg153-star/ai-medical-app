import os
import re
import io
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
import requests

# Optional: for tflite usage
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- Authentication ---
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "test1234":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("Please enter the password `test1234` to proceed.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# ---------------- Config ----------------
st.set_page_config(page_title="AI Medical Assistant", layout="wide")

# --- Main App Logic ---
if check_password():

    # The rest of your app goes here
    st.title("🩺 AI Medical Assistant — Demo")
    st.markdown("**Important:** This tool provides *preliminary* guidance only. Not a substitute for professional medical advice.")

    st.sidebar.header("Settings")
    use_local_ocr = st.sidebar.checkbox("Use Tesseract OCR", value=True, disabled=True)
    
    # We assume Tesseract is installed on Streamlit Community Cloud
    # No need for tesseract_available() check here as it's provided.

    tab1, tab2, tab3 = st.tabs(["Lab Tests (OCR)", "ECG Image (Placeholder)", "Symptom Checker (Placeholder)"])

    # ---------------- Helpers ----------------
    @st.cache_data
    def extract_text_from_image_pytesseract(img_bytes):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img, lang='eng+ara')
        except Exception as e:
            return f"Error during OCR: {e}"

    @st.cache_data
    def extract_text_from_pdf_bytes(file_bytes):
        try:
            import pdfplumber
            fp = io.BytesIO(file_bytes)
            text = ""
            with pdfplumber.open(fp) as pdf:
                for page in pdf.pages:
                    p = page.extract_text(x_tolerance=2) or ""
                    text += p + "\n"
            return text
        except ImportError:
            st.error("pdfplumber is not installed.")
            return ""
        except Exception as e:
            return f"Error reading PDF: {e}"

    def analyze_text_simple(text):
        normal_ranges = {
            "glucose": (70, 140), "hemoglobin": (12, 17.5), "wbc": (4000, 11000),
            "creatinine": (0.6, 1.3), "cholesterol": (125, 200),
        }
        out = []
        t = text.lower()
        for key in normal_ranges:
            pattern = rf"{key}[:\s]*([0-9]+(?:\.[0-9]+)?)"
            m = re.search(pattern, t)
            if m:
                try:
                    val = float(m.group(1))
                    low, high = normal_ranges[key]
                    if val < low: status = ("Low", "منخفض")
                    elif val > high: status = ("High", "مرتفع")
                    else: status = ("Normal", "طبيعي")
                    out.append({
                        "test": key, "value": val, "status_en": status[0],
                        "status_ar": status[1], "normal_range": f"{low} - {high}"
                    })
                except ValueError:
                    pass
        if "infection" in t or "عدوى" in t:
            out.append({"note": "Possible infection / قد تكون هناك عدوى"})
        return out

    # ------------- UI for Tabs -------------
    with tab1:
        st.header("Upload medical test (image or PDF)")
        uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "pdf"], key="lab")

        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            
            if uploaded_file.type.startswith("image/"):
                st.image(file_bytes, caption="Uploaded Image", use_column_width=True)
            else:
                st.info(f"PDF file uploaded: {uploaded_file.name}")

            if st.button("Start analysis", key="start_lab"):
                extracted_text = ""
                with st.spinner("Processing file..."):
                    if uploaded_file.type.startswith("image/"):
                        extracted_text = extract_text_from_image_pytesseract(file_bytes)
                    elif uploaded_file.type == "application/pdf":
                        extracted_text = extract_text_from_pdf_bytes(file_bytes)

                if extracted_text:
                    st.subheader("Extracted Text:")
                    st.text_area("Text", extracted_text, height=250)
                    
                    with st.spinner("Analyzing text..."):
                        st.subheader("Quick Analysis:")
                        results = analyze_text_simple(extracted_text)
                        if results:
                            for r in results:
                                if "test" in r:
                                    st.markdown(f"**{r['test'].capitalize()}**: {r['value']} → **{r['status_ar']} ({r['status_en']})** (Normal: {r['normal_range']})")
                                else:
                                    st.info(r.get("note"))
                        else:
                            st.warning("No clear lab values found with the simple analyzer.")
                else:
                    st.error("Could not extract any text from the file.")

    with tab2:
        st.header("ECG Image Analysis")
        st.info("This feature is a placeholder and not yet implemented.")

    with tab3:
        st.header("Symptom Checker")
        st.info("This feature is a placeholder and not yet implemented.")

    st.markdown("---")
    st.markdown("**Disclaimer:** This application provides preliminary information and suggestions only.")

import os
import re
import io
import streamlit as st
from PIL import Image
import pytesseract
import requests
import pandas as pd # Added for better display of results

# --- Authentication ---
def check_password():
    """Returns `True` if the user had a correct password."""
    def password_entered():
        if st.session_state["password"] == "test1234":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.write("Please enter the password `test1234` to proceed.")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# ---------------- Config ----------------
st.set_page_config(page_title="AI Medical Assistant", layout="wide")

# --- Main App Logic ---
if check_password():
    
    # --- Extensive Normal Ranges Database ---
    # This dictionary is now much larger to recognize more tests.
    # Normal ranges can vary by lab, age, and sex. These are general estimates.
    normal_ranges = {
        # Complete Blood Count (CBC)
        "wbc": (4.0, 11.0), "white blood cell": (4.0, 11.0),
        "rbc": (4.2, 5.9), "red blood cell": (4.2, 5.9),
        "hemoglobin": (13.5, 17.5), "hgb": (13.5, 17.5),
        "hematocrit": (40, 52), "hct": (40, 52),
        "mcv": (80, 100),
        "mch": (27, 33),
        "mchc": (32, 36),
        "rdw": (11.5, 14.5),
        "platelet": (150, 450), "plt": (150, 450),
        "neutrophils": (40, 75),
        "lymphocytes": (20, 45),
        "monocytes": (2, 10),
        "eosinophils": (1, 6),
        "basophils": (0, 2),
        "pus cells": (0, 5),

        # Kidney Function
        "creatinine": (0.6, 1.3),
        "bun": (7, 20), "urea": (15, 45),
        "uric acid": (3.5, 7.2),

        # Liver Function
        "ast": (10, 40), "sgot": (10, 40),
        "alt": (7, 55), "sgpt": (7, 55),
        "alp": (40, 130),
        "total bilirubin": (0.1, 1.2),
        "direct bilirubin": (0.0, 0.3),
        "total protein": (6.0, 8.3),
        "albumin": (3.5, 5.5),

        # Lipid Profile
        "total cholesterol": (125, 200), "cholesterol": (125, 200),
        "triglycerides": (0, 150), "tg": (0, 150),
        "hdl": (40, 60),
        "ldl": (0, 100),

        # Diabetes
        "glucose": (70, 100), # Fasting glucose
        "hba1c": (4.0, 5.6),

        # Urine Analysis (Example values, often qualitative)
        "specific gravity": (1.005, 1.030),
        "ph": (4.5, 8.0),
    }

    # --- AI Model Loading (Placeholder) ---
    AI_MODEL_URL = "https://huggingface.co/your-username/your-repo/resolve/main/your_model.pkl?download=true" # IMPORTANT: Replace with your actual model URL

    @st.cache_resource
    def load_ai_model(url):
        if url.startswith("https://huggingface.co"):
            try:
                # This is a placeholder for model loading logic
                # For a real implementation, you'd use joblib, pickle, or tensorflow to load the model from the downloaded file
                st.sidebar.success("AI Model placeholder loaded.")
                # For example:
                # from urllib.request import urlopen
                # import joblib
                # model_file = urlopen(url)
                # model = joblib.load(model_file)
                # return model
                return "AI Model Ready" # Placeholder return
            except Exception as e:
                st.sidebar.error(f"AI Model Error: {e}")
                return None
        return None

    # --- Helper Functions ---
    @st.cache_data
    def extract_text_from_image(img_bytes):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img, lang='eng+ara')
        except Exception as e:
            return f"Error during OCR: {e}"

    def analyze_text_simple(text):
        # ... (The analysis function is now more powerful due to the larger dictionary)
        # We will also make it more flexible
        results = []
        text_lower = text.lower()
        
        for key, (low, high) in normal_ranges.items():
            # Flexible pattern to find the key and then a number
            # It handles spaces, colons, and different cases
            pattern = rf"\b{key.replace('.', r'\.')}\b\s*[:\-]*\s*([0-9]+\.?[0-9]*)"
            match = re.search(pattern, text_lower)
            
            if match:
                try:
                    value = float(match.group(1))
                    status = "Normal"
                    if value < low: status = "Low"
                    elif value > high: status = "High"
                    
                    results.append({
                        "Test": key.replace("_", " ").title(),
                        "Result": value,
                        "Status": status,
                        "Normal Range": f"{low} - {high}"
                    })
                except (ValueError, IndexError):
                    continue
        return results

    def predict_with_model(data):
        # This is a placeholder function.
        # In a real scenario, you would preprocess the 'data' to match your model's input
        # and then call model.predict(processed_data)
        if ai_model:
            st.subheader("🤖 AI Model Insights (Placeholder)")
            st.info("Based on the analyzed values, the AI model suggests monitoring kidney and liver functions.")
            st.warning("This is a simulated AI prediction.")

    # ------------- UI -------------
    st.title("🩺 AI Medical Assistant v2.0")
    
    # --- Sidebar ---
    st.sidebar.header("Settings")
    st.sidebar.info("This app uses OCR to extract lab results and provides a simple analysis.")
    hide_image = st.sidebar.checkbox("Hide image after analysis", value=True)
    use_ai_model = st.sidebar.checkbox("Enable AI Model Analysis")
    
    ai_model = None
    if use_ai_model:
        ai_model = load_ai_model(AI_MODEL_URL)

    # --- Main Uploader Tab ---
    st.header("Upload Medical Test (Image)")
    uploaded_file = st.file_uploader("Upload an image of your lab report", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        
        # We use a session state to "remember" the analysis result
        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False

        if not st.session_state.analysis_done:
            st.image(file_bytes, caption="Uploaded Image", use_column_width=True)

        if st.button("Start Analysis", key="start_lab"):
            st.session_state.analysis_done = True
            with st.spinner("Reading text from image..."):
                extracted_text = extract_text_from_image(file_bytes)

            if "Error" in extracted_text:
                st.error(extracted_text)
            else:
                if not hide_image:
                    st.image(file_bytes, caption="Uploaded Image", use_column_width=True)
                
                st.subheader("📄 Extracted Text")
                st.text_area("Full text from report", extracted_text, height=200)
                
                with st.spinner("Analyzing values..."):
                    analysis_results = analyze_text_simple(extracted_text)
                    
                    if analysis_results:
                        st.subheader("📊 Analysis Results")
                        df = pd.DataFrame(analysis_results)
                        
                        # Color the status column for better readability
                        def color_status(val):
                            color = 'green' if val == 'Normal' else 'red' if val in ['High', 'Low'] else 'black'
                            return f'color: {color}'
                        
                        st.dataframe(df.style.applymap(color_status, subset=['Status']), use_container_width=True)

                        # AI Model Prediction (if enabled)
                        if use_ai_model and ai_model:
                            predict_with_model(df)

                    else:
                        st.warning("No recognizable lab values were found. The OCR might have misread the text or the tests are not in the database.")

    st.markdown("---")
    st.markdown("**Disclaimer:** This is a demonstration tool and not a substitute for professional medical advice.")

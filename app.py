import os
import re
import io
import streamlit as st
from PIL import Image
import pytesseract
import requests
import pandas as pd
import numpy as np # Required for mean calculation

# --- Authentication ---
def check_password():
    def password_entered():
        if st.session_state.get("password") == "test1234":
            st.session_state["password_correct"] = True
            if "password" in st.session_state: del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 Password incorrect")
    st.write("Please enter the password `test1234` to proceed.")
    return False

# ---------------- Config ----------------
st.set_page_config(page_title="AI Medical Assistant", layout="wide")

# --- Main App Logic ---
if check_password():
    
    # --- Extensive Normal Ranges Database ---
    normal_ranges = {
        # CBC
        "wbc": (4.0, 11.0), "white blood cell": (4.0, 11.0), "pus cells": (0, 5),
        "rbc": (4.2, 5.9), "red blood cell": (4.2, 5.9),
        "hemoglobin": (13.5, 17.5), "hgb": (13.5, 17.5),
        "hematocrit": (40, 52), "hct": (40, 52),
        "platelet": (150, 450), "plt": (150, 450),
        # Urine
        "specific gravity": (1.005, 1.030),
        "ph": (4.5, 8.0),
        "protein": (0, 0), # Should be 0 or Negative
        "glucose": (0, 0), # Should be 0 or Negative
        "ketone": (0, 0), "acetone": (0, 0),
        "bilirubin": (0, 0),
        "urobilinogen": (0.2, 1.0),
        "nitrite": (0, 0),
        "hb (blood)": (0, 0),
    }

    # --- Helper Functions ---
    @st.cache_data
    def extract_text_from_image(img_bytes):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img, lang='eng+ara')
        except Exception as e:
            return f"Error during OCR: {e}"

    def analyze_text_smart(text):
        """
        A smarter analyzer that can handle ranges (e.g., "1-2"), 
        qualitative results (e.g., "Negative"), and variations.
        """
        results = []
        text_lower = text.lower()
        lines = text_lower.split('\n')

        for key, (low, high) in normal_ranges.items():
            # Search line by line for better context
            for line in lines:
                # Use flexible pattern to find the key
                # This handles cases like (w.8.c) for wbc
                search_key = key.replace("wbc", "w.b.c").replace("r.b.c", "r.b.c")
                if re.search(r'\b' + re.escape(search_key) + r'\b', line):
                    
                    # --- Case 1: Find numerical values (including ranges) ---
                    # Pattern to find numbers, ranges (1-2), or decimals (1.025)
                    num_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)', line) # For ranges like 1-2
                    if not num_match:
                        num_match = re.search(r'([0-9]+\.?[0-9]*)', line) # For single numbers

                    if num_match:
                        try:
                            # If it's a range, take the average
                            if len(num_match.groups()) == 2 and num_match.group(2):
                                val1 = float(num_match.group(1))
                                val2 = float(num_match.group(2))
                                value = (val1 + val2) / 2
                                result_str = f"{val1}-{val2}"
                            else: # It's a single number
                                value = float(num_match.group(1))
                                result_str = str(value)

                            status = "Normal"
                            if value < low: status = "Low"
                            elif value > high: status = "High"
                            
                            results.append({
                                "Test": key.title(), "Result": result_str,
                                "Status": status, "Normal Range": f"{low} - {high}"
                            })
                            break # Move to the next key once found
                        except (ValueError, IndexError):
                            continue
                    
                    # --- Case 2: Find qualitative results ---
                    qual_match = re.search(r'(negative|nil|normal|clear|acidic|yellow)', line)
                    if qual_match:
                        result_str = qual_match.group(1).title()
                        status = "Normal" # Assume these are normal
                        # For PH, acidic is normal
                        if key == 'ph' and result_str == 'Acidic':
                            status = "Normal"
                        
                        results.append({
                            "Test": key.title(), "Result": result_str,
                            "Status": status, "Normal Range": "Qualitative"
                        })
                        break # Move to the next key
            
        # Remove duplicates (if any)
        unique_results = [dict(t) for t in {tuple(d.items()) for d in results}]
        return unique_results


    # ------------- UI -------------
    st.title("🩺 AI Medical Assistant v3.0 (Smart Analyzer)")
    st.sidebar.header("Settings")
    hide_image = st.sidebar.checkbox("Hide image after analysis", value=True)

    st.header("Upload Medical Test (Image)")
    uploaded_file = st.file_uploader("Upload an image of your lab report", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False

        if not st.session_state.analysis_done:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Start Analysis", key="start_lab"):
            st.session_state.analysis_done = True
            with st.spinner("Reading text from image..."):
                extracted_text = extract_text_from_image(uploaded_file.getvalue())

            if "Error" in extracted_text:
                st.error(extracted_text)
            else:
                if not hide_image:
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                st.subheader("📄 Extracted Text")
                st.text_area("Full text from report", extracted_text, height=200)
                
                with st.spinner("Analyzing values with Smart Analyzer..."):
                    analysis_results = analyze_text_smart(extracted_text)
                    
                    if analysis_results:
                        st.subheader("📊 Smart Analysis Results")
                        df = pd.DataFrame(analysis_results)
                        def color_status(val):
                            color = 'green' if val == 'Normal' else 'red' if val in ['High', 'Low'] else 'black'
                            return f'color: {color}'
                        st.dataframe(df.style.applymap(color_status, subset=['Status']), use_container_width=True)
                    else:
                        st.warning("No recognizable lab values were found.")

    st.markdown("---")
    st.markdown("**Disclaimer:** This is a demonstration tool.")

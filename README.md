# 🩺 Medical AI App / المحلل الطبي الذكي

نسخة: v1

---
## ENGLISH — Overview

**Medical AI App** is a local Streamlit application to:
- Extract text from scanned lab reports (images and scanned PDFs) using OCR.
- Automatically detect and analyze lab tests (blood, biochemistry, urine, stool, hormones).
- Provide simple rule-based recommendations and optional AI-assisted symptom analysis (OpenAI) — optional and requires API key.
- Export results to Excel / PDF.
- Support Arabic and English UI.

**Important:** This tool provides *preliminary* suggestions only. It does **NOT** replace medical consultation.

### Quick start (local)
1. Install system packages (Linux / Termux):
   - `tesseract-ocr` and `poppler-utils` (see `packages.txt`)
2. Create and activate Python venv (recommended)
3. Install Python packages:
```bash
pip install -r requirements.txt

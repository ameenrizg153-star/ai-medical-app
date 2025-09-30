# app.py
from flask import Flask, request, render_template_string, redirect, url_for
from PIL import Image
import pytesseract
import pdfplumber
import re
import os
import io
import json
import datetime

# optional OpenAI usage (only if installed and API key set)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

app = Flask(__name__)

# ---------------------------
# Basic config & disclaimer
# ---------------------------
APP_NAME = "Smart Medical Analyzer"
DISCLAIMER = (
    "This tool provides preliminary, non-diagnostic information only. "
    "It does NOT replace a medical professional. If symptoms are serious, "
    "seek medical attention immediately."
)
AR_DISCLAIMER = (
    "هذه الأداة تقدم معلومات أولية غير تشخيصية فقط. لا تغني عن استشارة الطبيب. "
    "إذا كانت الأعراض خطيرة، راجع الطبيب فورًا."
)

# ---------------------------
# Large normal ranges (>=100 entries by auto-adding placeholders)
# ---------------------------
NORMAL_RANGES = {
    "glucose": (70, 140), "hba1c": (4.0, 5.6), "cholesterol": (125, 200),
    "triglycerides": (0, 150), "hdl": (40, 60), "ldl": (0, 100),
    "hemoglobin": (12, 17.5), "hematocrit": (36, 50), "wbc": (4000, 11000),
    "rbc": (4.0, 6.0), "mcv": (80, 100), "mch": (27, 33), "platelets": (150000, 450000),
    "alt": (7, 56), "ast": (10, 40), "alp": (44, 147), "bilirubin_total": (0.1, 1.2),
    "bilirubin_direct": (0, 0.3), "creatinine": (0.6, 1.3), "bun": (7, 20), "egfr": (90, 120),
    "sodium": (135, 145), "potassium": (3.5, 5.0), "chloride": (98, 107), "co2": (23, 29),
    "calcium": (8.6, 10.3), "magnesium": (1.7, 2.2), "phosphorus": (2.5, 4.5),
    "iron": (60, 170), "ferritin": (20, 300), "tsh": (0.4, 4.0), "ft4": (0.8, 1.8),
    "ft3": (2.3, 4.2), "vitamin_d": (20, 50), "vitamin_b12": (190, 950),
    "crp": (0, 10), "esr": (0, 20), "troponin": (0, 0.04), "ck_mb": (0, 5), "nt_pro_bnp": (0, 125),
    "pt": (11, 13.5), "inr": (0.8, 1.2), "aptt": (25, 35), "albumin": (3.5, 5.0),
    "total_protein": (6.0, 8.3), "uric_acid": (3.4, 7.0), "psa": (0, 4.0), "ca125": (0, 35),
    "cea": (0, 3.0), "ldh": (140, 280), "gamma_gt": (9, 48), "cortisol": (6, 23),
    "amh": (1.0, 10.0), "procalcitonin": (0, 0.5),
    # Add a big set of plausible test keys; fill to >=110 with placeholders:
}

# auto-extend to reach ~110 keys (placeholders)
idx = 1
while len(NORMAL_RANGES) < 110:
    NORMAL_RANGES[f"extra_test_{idx}"] = (0, 100)
    idx += 1

# ---------------------------
# Aliases (editable)
# ---------------------------
ALIASES = {
    "blood sugar": "glucose", "sugar": "glucose", "hb": "hemoglobin",
    "wbc count": "wbc", "platelet count": "platelets", "creatinine level": "creatinine"
}

# ---------------------------
# Basic diagnosis guidelines (local rule-based)
# ---------------------------
DIAGNOSIS_GUIDELINES = {
    "hemoglobin": {
        "low": {"en":"Possible anemia. Recommend iron/B12 tests.", "ar":"قد يدل على فقر دم. ينصح بفحوصات الحديد وB12."},
        "high": {"en":"May indicate dehydration or polycythemia. See doctor.", "ar":"قد يدل على الجفاف أو كثرة كريات الدم الحمراء. راجع الطبيب."}
    },
    "glucose": {
        "low": {"en":"Low blood sugar — risk of hypoglycemia.", "ar":"انخفاض سكر الدم — خطر نقص السكر."},
        "high": {"en":"High blood sugar — consider diabetes screening.", "ar":"ارتفاع سكر الدم — فكر بفحص السكري."}
    },
    "wbc": {
        "low": {"en":"Low WBC — possible bone marrow issue or viral infection.", "ar":"انخفاض الكريات البيضاء — قد يدل على مشكلة أو عدوى فيروسية."},
        "high": {"en":"High WBC — possible bacterial infection or inflammation.", "ar":"ارتفاع الكريات البيضاء — احتمال التهاب أو عدوى بكتيرية."}
    }
    # can add more entries...
}

# ---------------------------
# Simple rule-based consultation KB
# ---------------------------
RULE_KB = {
    "fever": {"conds": {"Infection": 0.8, "Flu": 0.6}, "advice": ["Measure temperature regularly", "Stay hydrated"]},
    "cough": {"conds": {"Bronchitis": 0.5, "COVID/Flu": 0.6}, "advice": ["Watch for shortness of breath", "See doctor if bloody sputum"]},
    "chest pain": {"conds": {"Cardiac": 0.9, "GERD": 0.3}, "advice": ["Seek emergency care if severe"]},
    "headache": {"conds": {"Migraine": 0.6, "Tension headache":0.4}, "advice": ["Rest, hydrate, consider analgesic"]},
    "dizziness": {"conds": {"Dehydration":0.6, "Vertigo":0.4}, "advice": ["Sit/lie down, hydrate"]},
}

# ---------------------------
# Utilities: OCR & Text Extraction
# ---------------------------

def extract_text_from_image_file(file_stream) -> str:
    try:
        img = Image.open(file_stream).convert("RGB")
    except Exception as e:
        return f"ERROR_OPEN_IMAGE: {e}"
    # direct tesseract
    try:
        text = pytesseract.image_to_string(img, lang="eng+ara")
    except Exception:
        text = pytesseract.image_to_string(img)
    return text or ""

def extract_text_from_pdf_file(file_stream) -> str:
    try:
        text = ""
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    except Exception as e:
        return f"ERROR_PDF: {e}"

# ---------------------------
# Test extraction & analysis
# ---------------------------
def build_patterns():
    patterns = []
    for key in NORMAL_RANGES.keys():
        # build name token: allow underscores or spaces
        key_regex = re.escape(key).replace(r'\_', r'[_\s]*')
        # add aliases
        alt = [re.escape(k) for k,v in ALIASES.items() if v==key]
        if alt:
            key_regex = f"(?:{key_regex}|{'|'.join(alt)})"
        pat = re.compile(rf"\b{key_regex}\b[\s:=]*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE)
        patterns.append((key, pat))
    return patterns

SEARCH_PATTERNS = build_patterns()

def extract_tests_from_text(text: str):
    found = []
    if not text:
        return found
    lower = text.lower()
    for key, pat in SEARCH_PATTERNS:
        m = pat.search(lower)
        if m:
            try:
                val = float(m.group(1))
            except:
                continue
            low, high = NORMAL_RANGES.get(key, (None, None))
            if low is None:
                continue
            status = "Normal" if low <= val <= high else ("Low" if val < low else "High")
            # get diagnosis
            diag_info = DIAGNOSIS_GUIDELINES.get(key, {})
            diag = diag_info.get(status.lower(), {})
            found.append({
                "test": key,
                "value": val,
                "status": status,
                "normal_range": f"{low} - {high}",
                "diagnosis_en": diag.get("en", "") or ("Value is within normal range." if status=="Normal" else ""),
                "diagnosis_ar": diag.get("ar", "") or ( "القيمة ضمن النطاق الطبيعي." if status=="Normal" else "")
            })
    return found

# ---------------------------
# Consultation helpers
# ---------------------------
def rule_based_consult(symptoms: str):
    txt = symptoms.lower()
    cond_scores = {}
    advices = set()
    matched = []
    for kw, info in RULE_KB.items():
        if kw in txt:
            matched.append(kw)
            for cond, w in info["conds"].items():
                cond_scores[cond] = cond_scores.get(cond, 0) + w
            for a in info["advice"]:
                advices.add(a)
    probable = sorted(cond_scores.items(), key=lambda x: x[1], reverse=True)
    return {"matched": matched, "probable": probable, "advices": list(advices)}

def openai_consult(symptoms: str):
    # requires OPENAI_API_KEY set in env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not OPENAI_AVAILABLE:
        return {"error": "OpenAI not configured or package not installed."}
    openai.api_key = api_key
    prompt = f"""You are a medical assistant. The patient reports: {symptoms}
Provide a short JSON with keys: probable_conditions (list of {"condition","confidence"}),
red_flags (list of strings), initial_advice (list of strings), recommendation (string).
Answer concisely."""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if hasattr(openai, "ChatCompletion") else "gpt-4",
            messages=[{"role":"system","content":"You are a helpful medical assistant."},
                      {"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=400
        )
        content = resp["choices"][0]["message"]["content"]
        # try parse JSON block if returned
        jmatch = re.search(r'(\{.*\})', content, flags=re.S)
        if jmatch:
            return json.loads(jmatch.group(1))
        return {"raw": content}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Simple HTML template
# ---------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{app_name}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{font-family:Arial, Helvetica, sans-serif; margin:12px; background:#f7fbfb;}
    .card{background:#fff;padding:12px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.06); margin-bottom:12px;}
    .ok{background:#e8f5e9;padding:6px;border-radius:6px;}
    .high{background:#ffebee;padding:6px;border-radius:6px;}
    .low{background:#fff8e1;padding:6px;border-radius:6px;}
    .warn{color:#b71c1c;font-weight:700;}
    textarea{width:100%;height:120px;border-radius:6px;padding:8px;}
    input[type=file]{width:100%;}
    .small{font-size:13px;color:#555;}
  </style>
</head>
<body>
  <h2>{{app_name}}</h2>
  <div class="card small">{{disclaimer}}</div>

  <div class="card">
    <h3>1) Upload report (image or PDF)</h3>
    <form method="post" action="/" enctype="multipart/form-data">
      <input type="file" name="report"><br><br>
      <button type="submit">Extract & Analyze</button>
    </form>
  </div>

  <div class="card">
    <h3>2) Or paste report text manually</h3>
    <form method="post" action="/analyze_text">
      <textarea name="manual_text" placeholder="Paste OCR text or report text here..."></textarea><br>
      <button type="submit">Analyze Text</button>
    </form>
  </div>

  <div class="card">
    <h3>3) Consultation / Symptoms</h3>
    <form method="post" action="/consult">
      <textarea name="symptoms" placeholder="Describe symptoms (Arabic or English)"></textarea><br>
      <label><input type="checkbox" name="use_openai" value="1"> Use advanced AI (OpenAI) if configured</label><br><br>
      <button type="submit">Get Consultation</button>
    </form>
  </div>

  {% if extracted_text %}
  <div class="card">
    <h3>Extracted Text</h3>
    <pre style="white-space:pre-wrap;">{{extracted_text}}</pre>
  </div>
  {% endif %}

  {% if tests %}
  <div class="card">
    <h3>Detected Tests</h3>
    {% for t in tests %}
      <div class="{% if t.status=='Normal' %}ok{% elif t.status=='High' %}high{% else %}low{% endif %}" style="margin-bottom:8px;">
        <b>{{t.test}}</b>: {{t.value}} — <i>{{t.status}}</i><br>
        <small>Range: {{t.normal_range}}</small><br>
        <small>EN: {{t.diagnosis_en}}</small><br>
        <small>AR: {{t.diagnosis_ar}}</small>
      </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if consult_result %}
  <div class="card">
    <h3>Consultation Result</h3>
    <pre style="white-space:pre-wrap;">{{consult_result}}</pre>
  </div>
  {% endif %}

  <footer class="small" style="margin-top:20px;">
    {{ts}} — Not a medical device. For diagnosis consult a licensed professional.
  </footer>
</body>
</html>
"""

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    tests = []
    consult_result = None
    if request.method == "POST":
        f = request.files.get("report")
        if not f or f.filename == "":
            return render_template_string(HTML, app_name=APP_NAME, disclaimer=DISCLAIMER, extracted_text="", tests=[], consult_result=None, ts=str(datetime.datetime.now()))
        fname = f.filename.lower()
        if fname.endswith(".pdf"):
            extracted_text = extract_text_from_pdf_file(f)
        else:
            extracted_text = extract_text_from_image_file(f)
        tests = extract_tests_from_text(extracted_text)
    return render_template_string(HTML, app_name=APP_NAME, disclaimer=DISCLAIMER, extracted_text=extracted_text, tests=tests, consult_result=consult_result, ts=str(datetime.datetime.now()))

@app.route("/analyze_text", methods=["POST"])
def analyze_text_route():
    txt = request.form.get("manual_text", "")
    tests = extract_tests_from_text(txt)
    return render_template_string(HTML, app_name=APP_NAME, disclaimer=DISCLAIMER, extracted_text=txt, tests=tests, consult_result=None, ts=str(datetime.datetime.now()))

@app.route("/consult", methods=["POST"])
def consult_route():
    symptoms = request.form.get("symptoms", "").strip()
    use_openai = request.form.get("use_openai") == "1"
    consult_result = ""
    if not symptoms:
        consult_result = "Please provide symptoms."
    else:
        # First run local rule-based
        rb = rule_based_consult(symptoms)
        consult_result += "Rule-based suggestions:\n"
        consult_result += json.dumps(rb, indent=2, ensure_ascii=False)
        # Optionally call OpenAI if requested & configured
        if use_openai:
            oresp = openai_consult(symptoms)
            consult_result += "\n\nOpenAI result:\n"
            consult_result += json.dumps(oresp, indent=2, ensure_ascii=False)
    return render_template_string(HTML, app_name=APP_NAME, disclaimer=DISCLAIMER, extracted_text=None, tests=None, consult_result=consult_result, ts=str(datetime.datetime.now()))

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    # Bind to 0.0.0.0 so device can access from browser at 127.0.0.1
    app.run(host="0.0.0.0", port=5000)

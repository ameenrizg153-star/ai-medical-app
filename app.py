
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd, os, io
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(__file__)
TESTS_CSV = os.path.join(BASE_DIR, "lab_tests_full.csv")

app = Flask(__name__)
app.secret_key = "change-me"
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_tests_db():
    return pd.read_csv(TESTS_CSV)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download-csv")
def download_csv():
    return send_file(TESTS_CSV, as_attachment=True, download_name="lab_tests_full.csv")

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    try:
        if filename.lower().endswith('.csv'):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)
    except Exception as e:
        flash(f"Error reading uploaded file: {e}", "error")
        return redirect(url_for('index'))

    cols = [c.lower().strip() for c in data.columns]
    if 'code' not in cols or 'result' not in cols:
        flash("CSV must contain 'Code' and 'Result' columns (case-insensitive).", "error")
        return redirect(url_for('index'))
    data.columns = cols
    tests_db = load_tests_db()
    analyses = []
    for _, row in data.iterrows():
        code = str(row['code']).strip().lower()
        value = row['result']
        match = tests_db[tests_db['code'].str.lower() == code]
        if match.shape[0] == 0:
            analyses.append({'code':code, 'name_ar': code, 'value': value, 'status': 'Unknown', 'advice': ''})
            continue
        t = match.iloc[0]
        low = t['low'] if pd.notna(t['low']) else None
        high = t['high'] if pd.notna(t['high']) else None
        name_ar = t['name_ar']
        advice = ""
        status = "طبيعي"
        try:
            valf = float(value)
            if low is not None and valf < float(low):
                status = "منخفض"
                advice = t.get('recommendation_low','') or ''
            elif high is not None and valf > float(high):
                status = "مرتفع"
                advice = t.get('recommendation_high','') or ''
        except:
            status = "غير قابل للمقارنة"
        analyses.append({'code':code, 'name_ar': name_ar, 'value': value, 'status': status, 'advice': advice})
    return render_template("result.html", analyses=analyses)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)

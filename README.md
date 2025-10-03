# medical-ai-app

AI Medical Analyzer — تطبيق Streamlit لقراءة تقارير المختبر (صور/PDF)، استخراج القيم، تفسير أولي، وتصدير تقارير.
الواجهة ثنائية اللغة: العربية وEnglish.

## محتويات
- `app.py`: التطبيق الرئيسي
- `tests_database.csv`: قاعدة بيانات الفحوصات (قابلة للتعديل)
- `train_model.py`: سكربت تدريب نموذج بسيط
- `requirements.txt`: المكتبات المطلوبة
- `database/Training.csv`: (اختياري) بيانات التدريب لنموذج الأعراض

## تشغيل محليًا
1. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```
2. تأكد من تثبيت Tesseract و Poppler (لمستخدمي PDF)
3. تشغيل التطبيق:
```bash
streamlit run app.py
```

## إضافة / تعديل فحص
افتح `tests_database.csv` وأضف سطرًا جديدًا بالصيغة المناسبة ثم أعد تشغيل التطبيق.

## رفع إلى GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## ملاحظات أمنية
- البيانات طبية وحساسة: لا ترفع ملفات المرضى إلى سيرفرات عامة بدون موافقة.
- لحماية OpenAI API Key استخدم `st.secrets` عند النشر على Streamlit Cloud.

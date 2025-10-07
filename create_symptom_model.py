import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. إنشاء بيانات تدريب وهمية
training_data = {
    'fever': [1, 0, 1, 0, 1, 1, 0],
    'cough': [1, 1, 0, 1, 1, 0, 1],
    'headache': [0, 1, 1, 1, 0, 1, 0],
    'sore_throat': [1, 1, 0, 0, 1, 0, 0],
    'fatigue': [0, 1, 1, 1, 1, 1, 0],
    'prognosis': [
        'Flu', 'Migraine', 'Flu', 'Migraine', 
        'Flu', 'Common Cold', 'Common Cold'
    ]
}
df = pd.DataFrame(training_data)

# 2. حفظ بيانات التدريب في ملف CSV (مهم جدًا للتطبيق الرئيسي)
df.to_csv('Training.csv', index=False)
print("✅ ملف 'Training.csv' تم إنشاؤه بنجاح.")

# 3. فصل الميزات عن التشخيص
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# 4. تدريب نموذج شجرة القرار
model = DecisionTreeClassifier()
model.fit(X, y)
print("🧠 تم تدريب نموذج مدقق الأعراض.")

# 5. حفظ النموذج المدرب في ملف joblib
joblib.dump(model, 'symptom_checker_model.joblib')
print("✅ ملف 'symptom_checker_model.joblib' تم إنشاؤه بنجاح.")

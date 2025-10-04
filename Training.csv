import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- تحميل قاعدة البيانات ---
data_path = "database/Training.csv"
data = pd.read_csv(data_path)

# فصل الخصائص (الأعراض) عن الهدف (المرض)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- تدريب نموذج شجرة القرار ---
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# --- اختبار النموذج ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ تم تدريب النموذج بدقة: {acc*100:.2f}%")

# --- حفظ النموذج ---
os.makedirs("models", exist_ok=True)
model_path = "models/symptom_checker_model.joblib"
joblib.dump(model, model_path)
print(f"💾 تم حفظ النموذج في: {model_path}")

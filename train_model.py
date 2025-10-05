import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# تحميل البيانات
data = pd.read_csv('Training.csv')

# تقسيم الميزات والهدف
X = data.iloc[:, 1:-1]  # الميزات (باستثناء symptom وdiagnosis)
y = data['diagnosis']   # الهدف

# تقسيم التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء نموذج Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# اختبار النموذج
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# حفظ النموذج
joblib.dump(model, 'symptom_checker_model.joblib')

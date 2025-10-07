import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("=" * 60)
print("تدريب نموذج Decision Tree للتشخيص الطبي")
print("=" * 60)

# تحميل البيانات
print("\n1. تحميل البيانات...")
data = pd.read_csv("Training.csv")
print(f"   ✓ تم تحميل {len(data)} عينة")
print(f"   ✓ عدد الأعراض: {len(data.columns) - 2}")

# تقسيم الميزات والهدف
X = data.iloc[:, 1:-1]  # الميزات (باستثناء ID و diagnosis)
y = data["diagnosis"]   # الهدف

print(f"\n2. تقسيم البيانات...")
# تقسيم التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   ✓ عينات التدريب: {len(X_train)}")
print(f"   ✓ عينات الاختبار: {len(X_test)}")

# إنشاء نموذج Decision Tree مع معلمات محسّنة
print(f"\n3. تدريب النموذج...")
model = DecisionTreeClassifier(
    max_depth=10,           # الحد الأقصى لعمق الشجرة
    min_samples_split=2,    # الحد الأدنى لعدد العينات لتقسيم العقدة
    min_samples_leaf=1,     # الحد الأدنى لعدد العينات في الورقة
    random_state=42
)

model.fit(X_train, y_train)
print(f"   ✓ تم تدريب النموذج بنجاح")

# اختبار النموذج
print(f"\n4. تقييم النموذج...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"   ✓ دقة النموذج: {accuracy:.2%}")

# عرض تقرير التصنيف
print(f"\n5. تقرير التصنيف التفصيلي:")
print("-" * 60)
print(classification_report(y_test, y_pred, zero_division=0))

# حفظ النموذج
print(f"\n6. حفظ النموذج...")
joblib.dump(model, "symptom_checker_model.joblib")
print(f"   ✓ تم حفظ النموذج في: symptom_checker_model.joblib")

print("\n" + "=" * 60)
print("اكتمل التدريب بنجاح!")
print("=" * 60)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # مكتبة أفضل لحفظ نماذج scikit-learn

# --- الخطوة 1: تحميل وتنظيف البيانات ---
try:
    # قراءة بيانات التدريب
    df_train = pd.read_csv('Training.csv')

    # إزالة أي مسافات زائدة من أسماء الأعمدة
    df_train.columns = df_train.columns.str.strip()

    # فصل الميزات (الأعراض) عن الهدف (التشخيص)
    X = df_train.drop('prognosis', axis=1)
    y = df_train['prognosis']

    print("✅ تم تحميل البيانات بنجاح.")
    print(f"عدد العينات: {len(df_train)}, عدد الميزات: {len(X.columns)}")

except FileNotFoundError:
    print("❌ خطأ: ملف 'Training.csv' غير موجود. يرجى التأكد من وجوده في نفس المجلد.")
    exit()
except KeyError:
    print("❌ خطأ: لم يتم العثور على عمود 'prognosis' في ملف التدريب. يرجى التأكد من وجوده.")
    exit()


# --- الخطوة 2: تقسيم البيانات ---
# تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار للتحقق من دقة النموذج
# 80% للتدريب، 20% للاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ تم تقسيم البيانات إلى مجموعات تدريب واختبار.")


# --- الخطوة 3: تدريب النموذج ---
# استخدام نموذج "الغابات العشوائية" (Random Forest) لأنه قوي ومتعدد الاستخدامات
print("⏳ جاري تدريب نموذج Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ تم الانتهاء من تدريب النموذج.")


# --- الخطوة 4: تقييم أداء النموذج ---
# اختبار دقة النموذج على بيانات الاختبار التي لم يرها من قبل
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 دقة النموذج على بيانات الاختبار: {accuracy * 100:.2f}%")


# --- الخطوة 5: حفظ النموذج المدرب ---
# حفظ النموذج في ملف لاستخدامه لاحقاً في تطبيق الويب
model_filename = 'trained_medical_model.joblib'
joblib.dump(model, model_filename)
print(f"💾 تم حفظ النموذج المدرب في ملف باسم: '{model_filename}'")


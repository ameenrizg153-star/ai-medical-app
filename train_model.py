# train_model.py  (مثال بسيط لتدريب DecisionTree وحفظه joblib)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# تأكد وجود ملف Training.csv داخل مجلد database
path = os.path.join('database','Training.csv')
if not os.path.exists(path):
    print('Put your Training.csv in database/Training.csv then re-run this script')
else:
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    joblib.dump(clf, 'symptom_checker_model.joblib')
    print('Model saved to symptom_checker_model.joblib')

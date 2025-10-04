# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

db_path = "database/Training.csv"
if not os.path.exists(db_path):
    print("Training dataset not found at 'database/Training.csv'. Please provide training data.")
else:
    data = pd.read_csv(db_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc*100:.2f}%")
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/symptom_checker_model.joblib")
    print("Saved model to models/symptom_checker_model.joblib")

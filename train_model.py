
import pandas as pd, os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

CSV_PATH = "lab_tests_full.csv"
if not os.path.exists(CSV_PATH):
    print("CSV missing:", CSV_PATH)
else:
    df = pd.read_csv(CSV_PATH)
    print("Loaded tests:", len(df))
    # Placeholder: no labeled training data included
    model = DecisionTreeClassifier()
    joblib.dump(model, "medical_model_placeholder.joblib")
    print("Saved placeholder model")

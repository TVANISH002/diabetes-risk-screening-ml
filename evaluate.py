import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv("diabetes.csv")
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = joblib.load("models/diabetes_pipeline.joblib")
    proba = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, proba)

    target_recall = 0.85
    idx = np.where(recall >= target_recall)[0]

    if len(idx) == 0 or len(thresholds) == 0:
        chosen_threshold = 0.5
    else:
        t_index = max(idx[-1] - 1, 0)
        chosen_threshold = float(thresholds[min(t_index, len(thresholds)-1)])

    preds = (proba >= chosen_threshold).astype(int)
    cm = confusion_matrix(y, preds)

    print("Chosen threshold:", chosen_threshold)
    print("Confusion Matrix:\n", cm)
    print("\nReport:\n", classification_report(y, preds, zero_division=0))

    with open("models/threshold.json", "w") as f:
        json.dump(
            {"threshold": chosen_threshold, "target_recall": target_recall},
            f,
            indent=2
        )

    print("\nSaved -> models/threshold.json")

if __name__ == "__main__":
    main()

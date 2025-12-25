import os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df

def load_threshold(path="models/threshold.json", default=0.5) -> float:
    try:
        with open(path) as f:
            return float(json.load(f).get("threshold", default))
    except Exception:
        return float(default)

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_data("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    base = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True))
    ])

    param_grid = {
        "svm__kernel": ["rbf", "linear"],
        "svm__C": [0.3, 1, 3, 10, 30],
        "svm__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
        "svm__class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_

    calibrated = CalibratedClassifierCV(best_pipe, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    proba = calibrated.predict_proba(X_test)[:, 1]

    # IMPORTANT: use your screening threshold (from threshold.json if present)
    THRESHOLD = load_threshold(default=0.5)
    preds = (proba >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print("Best Params:", grid.best_params_)
    print(f"\nThreshold used: {THRESHOLD:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(calibrated, "models/diabetes_pipeline.joblib")

    # Save metrics so Streamlit can show them
    with open("reports/metrics.json", "w") as f:
        json.dump({
            "best_params": grid.best_params_,
            "threshold_used": float(THRESHOLD),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    print("\nSaved -> models/diabetes_pipeline.joblib")
    print("Saved -> reports/metrics.json")

if __name__ == "__main__":
    main()

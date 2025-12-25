import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


APP_TITLE = "Diabetes Risk Screening API"
MODEL_PATH = Path("models/diabetes_pipeline.joblib")
THRESH_PATH = Path("models/threshold.json")

app = FastAPI(title=APP_TITLE)

# Load artifacts once on startup
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.5
if THRESH_PATH.exists():
    try:
        THRESHOLD = float(json.loads(THRESH_PATH.read_text()).get("threshold", 0.5))
    except Exception:
        THRESHOLD = 0.5


class Patient(BaseModel):
    Pregnancies: int = Field(ge=0, le=20)
    Glucose: float = Field(ge=0, le=300)
    BloodPressure: float = Field(ge=0, le=200)
    SkinThickness: float = Field(ge=0, le=100)
    Insulin: float = Field(ge=0, le=900)
    BMI: float = Field(ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: int = Field(ge=1, le=120)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/meta")
def meta():
    return {"threshold": THRESHOLD, "model_artifact": str(MODEL_PATH)}


@app.post("/predict")
def predict(p: Patient):
    df = pd.DataFrame([p.model_dump()])
    risk = float(model.predict_proba(df)[0][1])

    label = "higher_risk" if risk >= THRESHOLD else "lower_risk"

    # Optional: risk banding (nice for UI & recruiter story)
    if risk < THRESHOLD:
        band = "low"
    elif risk < max(THRESHOLD + 0.15, 0.35):
        band = "moderate"
    else:
        band = "high"

    return {
        "risk": risk,
        "threshold": THRESHOLD,
        "screening_label": label,
        "risk_band": band
    }

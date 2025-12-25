import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Diabetes Risk Screening", layout="centered")
st.title("Diabetes Risk Screening (Machine Learning)")
st.caption("Educational screening tool — not medical advice.")

#Load model 
model = joblib.load("models/diabetes_pipeline.joblib")

#Load threshold
THRESHOLD = 0.5
try:
    with open("models/threshold.json") as f:
        THRESHOLD = float(json.load(f).get("threshold", 0.5))
except Exception:
    THRESHOLD = 0.5

#Load metrics for sidebar(if it is available)
metrics_path = Path("reports/metrics.json")
metrics = None
if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception:
        metrics = None

#Sidebar (Premium touch)
with st.sidebar:
    st.subheader("Model Info")
    st.write("Model: Calibrated SVM (screening-oriented)")
    st.write(f"Threshold: **{THRESHOLD:.2f}**")

    if metrics:
        st.markdown("**Test Metrics**")
        st.write(f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
        st.write(f"PR-AUC: {metrics.get('pr_auc', 0):.3f}")
        st.write(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
        st.write(f"Recall: {metrics.get('recall', 0):.3f}")
        st.write(f"Precision: {metrics.get('precision', 0):.3f}")

    st.divider()
    st.caption("Disclaimer: For learning/demo only. Not medical advice.")

st.subheader("Patient Inputs")

c1, c2 = st.columns(2)
with c1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
with c2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 30.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

# Input validation
warnings = []
if glucose == 0:
    warnings.append("Glucose is 0 — in real screening this is typically missing/invalid.")
if bmi == 0:
    warnings.append("BMI is 0 — in real screening this is typically missing/invalid.")
if bp == 0:
    warnings.append("Blood Pressure is 0 — in real screening this is typically missing/invalid.")

if warnings:
    for w in warnings:
        st.warning(w)

if st.button("Run Screening"):
    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    risk = float(model.predict_proba(input_df)[0][1])

    st.metric("Estimated Diabetes Risk (probability)", f"{risk:.2f}")

    # Risk bands look more healthcare-like than binary
    if risk < THRESHOLD:
        st.success(f"Screening Result: Lower Risk (threshold={THRESHOLD:.2f})")
    elif risk < max(THRESHOLD + 0.15, 0.35):
        st.warning("Screening Result: Moderate Risk")
    else:
        st.error("Screening Result: Higher Risk")

st.markdown(
    "Disclaimer: This app is for learning and demonstration only and is not a substitute for professional medical advice."
)

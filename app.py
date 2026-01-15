import json
from pathlib import Path

import requests
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Diabetes Risk Screening", layout="centered")
st.title("Diabetes Risk Screening (Machine Learning)")
st.caption("Educational screening tool — not medical advice.")

API_URL = "http://127.0.0.1:8000"
METRICS_PATH = Path("reports/metrics.json")

# ----------------------------
# Helpers
# ----------------------------
def api_get(path: str, timeout: int = 5):
    r = requests.get(f"{API_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict, timeout: int = 10):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def load_local_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except Exception:
            return None
    return None

def show_run_commands():
    st.code(
        "Terminal 1 (FastAPI):\n"
        "uvicorn API:app --reload --host 127.0.0.1 --port 8000\n\n"
        "Terminal 2 (Streamlit):\n"
        "streamlit run app.py"
    )

# ----------------------------
# Sidebar
# ----------------------------
metrics = load_local_metrics()
threshold = 0.5
api_ok = False

with st.sidebar:
    st.subheader("API Status")

    # Quick connect test
    try:
        health = api_get("/health")
        api_ok = health.get("status") == "ok"
    except Exception as e:
        st.error("FastAPI is not reachable ❌")
        show_run_commands()
        st.caption(f"Error: {e}")

    if api_ok:
        st.success("FastAPI is running ✅")

        # Meta info
        try:
            meta = api_get("/meta")
            threshold = float(meta.get("threshold", 0.5))
            st.markdown("**Model Info (from FastAPI /meta)**")
            st.write("Model: Calibrated SVM (screening-oriented)")
            st.write(f"Threshold: **{threshold:.2f}**")
        except Exception:
            st.warning("Could not fetch /meta. Using default threshold 0.50")

        # Local metrics (optional)
        if metrics:
            st.divider()
            st.markdown("**Test Metrics (local reports/metrics.json)**")
            st.write(f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
            st.write(f"PR-AUC: {metrics.get('pr_auc', 0):.3f}")
            st.write(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
            st.write(f"Recall: {metrics.get('recall', 0):.3f}")
            st.write(f"Precision: {metrics.get('precision', 0):.3f}")

    st.divider()
    st.caption("Disclaimer: For learning/demo only. Not medical advice.")

# ----------------------------
# Connection check (main page)
# ----------------------------
st.subheader("Connection Check")
cc1, cc2 = st.columns(2)

with cc1:
    if st.button("Check /health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.success(f"Connected ✅ ({r.status_code})")
            st.json(r.json())
        except Exception as e:
            st.error("Not connected ❌")
            st.write(e)

with cc2:
    if st.button("Open Swagger link"):
        st.write(f"{API_URL}/docs")

st.divider()

# ----------------------------
# Inputs
# ----------------------------
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

warnings = []
if glucose == 0:
    warnings.append("Glucose is 0 — in real screening this is typically missing/invalid.")
if bmi == 0:
    warnings.append("BMI is 0 — in real screening this is typically missing/invalid.")
if bp == 0:
    warnings.append("Blood Pressure is 0 — in real screening this is typically missing/invalid.")
for w in warnings:
    st.warning(w)

# ----------------------------
# Prediction via FastAPI
# ----------------------------
st.subheader("Prediction")

# Optional quick test button to force a POST /predict and show proof
if st.button("TEST /predict (sample payload)"):
    sample_payload = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }
    try:
        st.write("Calling API /predict...")
        out = api_post("/predict", sample_payload)
        st.success("Got response from API ✅ (check FastAPI terminal for POST /predict 200 OK)")
        st.json(out)
    except Exception as e:
        st.error("Predict test failed ❌")
        st.write(e)

# Normal UI flow
run = st.button("Run Screening")

if run:
    if not api_ok:
        st.error("FastAPI is not reachable. Start the API first.")
        show_run_commands()
    else:
        payload = {
            "Pregnancies": int(pregnancies),
            "Glucose": float(glucose),
            "BloodPressure": float(bp),
            "SkinThickness": float(skin),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age": int(age),
        }

        try:
            st.write("Calling API /predict...")  # <-- proof on UI
            out = api_post("/predict", payload)
            st.success("Response received ✅ (check FastAPI terminal for POST /predict 200 OK)")

            risk = float(out.get("risk", 0.0))
            thr = float(out.get("threshold", threshold))
            label = out.get("screening_label", "unknown")
            band = out.get("risk_band", "unknown")

            st.metric("Estimated Diabetes Risk (probability)", f"{risk:.2f}")

            if label == "lower_risk" or band == "low":
                st.success(f"Screening Result: Lower Risk (threshold={thr:.2f})")
            elif band == "moderate":
                st.warning("Screening Result: Moderate Risk")
            else:
                st.error("Screening Result: Higher Risk")

            with st.expander("View API response (raw JSON)"):
                st.json(out)

        except requests.exceptions.HTTPError as e:
            st.error("API returned an error response (validation/server).")
            try:
                st.json(e.response.json())
            except Exception:
                st.write(str(e))
        except requests.exceptions.RequestException as e:
            st.error("FastAPI is not reachable. Start the API first.")
            show_run_commands()
            st.caption(f"Error: {e}")

st.markdown(
    "Disclaimer: This app is for learning and demonstration only and is not a substitute for professional medical advice."
)

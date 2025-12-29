# =========================================
# 💳 FRAUD DETECTION APP — STREAMLIT
# =========================================
import streamlit as st
st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.cache_data.clear()
st.cache_resource.clear()

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
THRESHOLD = 0.636
MODEL_PATH = "model/fraud_rf_model.pkl"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    artifacts = joblib.load(MODEL_PATH)
    return artifacts

artifacts = load_model()
pipeline = artifacts["pipeline"]
features = artifacts["features"]

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
**Machine Learning model for real-time fraud risk evaluation.**  
This system uses a trained Random Forest model and a cost-optimized decision threshold.
""")

st.divider()

# --------------------------------------------------
# MODEL INFO
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Recall (Fraud)", "0.85")

with col3:
    st.metric("Precision (Fraud)", "0.67")

st.caption(f"⚙️ Decision Threshold: {THRESHOLD}")

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("🧾 Transaction Input")

with st.form("fraud_form"):
    amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=120.0)

    time = st.number_input(
        "⏱️ Time (seconds since start of day)",
        min_value=0,
        max_value=86400,
        value=36000
    )

    st.markdown("### 🔢 PCA Variables")
    cols = st.columns(4)
    v_inputs = {}

    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            v_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    submitted = st.form_submit_button("🔍 Analyze Transaction")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:
    data = {
        "Time": time,
        "Amount": amount,
        **v_inputs
    }

    df = pd.DataFrame([data])

    # Feature engineering
    df["Hour"] = (df["Time"] % 86400) // 3600
    df["Amount_scaled"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    X = df[features]

    proba = pipeline.predict_proba(X)[0][1]
    decision = int(proba >= THRESHOLD)

    st.divider()
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Fraud Probability", f"{proba:.2%}")

    with col2:
        if decision:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ LEGIT TRANSACTION")

    st.caption(f"Threshold used: {THRESHOLD}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("📌 Demo app — ML Fraud Detection | Built with Streamlit")


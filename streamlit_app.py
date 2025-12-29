# =========================================
# 💳 FRAUD DETECTION APP — STREAMLIT
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "fraud_rf_model.pkl"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

model_artifacts = load_model()

pipeline = model_artifacts["pipeline"]
features = model_artifacts["features"]
THRESHOLD = model_artifacts["threshold"]

# --------------------------------------------------
# RELOAD MODEL BUTTON
# --------------------------------------------------
st.sidebar.markdown("### 🔄 Model Control")

if st.sidebar.button("Reload Model"):
    st.cache_resource.clear()
    st.success("Model cache cleared. Reloading...")
    st.rerun()

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
**Machine Learning system for real-time fraud detection.**  
Random Forest model optimized with cost-based threshold.
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
# RANDOM TRANSACTION GENERATOR
# --------------------------------------------------
def generate_random_transaction():
    return {
        "amount": round(np.random.uniform(1, 2500), 2),
        "time": np.random.randint(0, 86400),
        "pca": np.random.normal(0, 1, 28).round(4)
    }

# Init session state
if "random_data" not in st.session_state:
    st.session_state.random_data = generate_random_transaction()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("🧾 Transaction Input")

with st.form("fraud_form"):

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.form_submit_button("🎲 Generate Random"):
            st.session_state.random_data = generate_random_transaction()

    with col1:
        amount = st.number_input(
            "💰 Transaction Amount",
            min_value=0.0,
            value=st.session_state.random_data["amount"]
        )

        time = st.number_input(
            "⏱️ Time (seconds since start of day)",
            min_value=0,
            max_value=86400,
            value=st.session_state.random_data["time"]
        )

    st.markdown("### 🔢 PCA Variables (V1–V28)")

    pca_input = st.text_area(
        "Enter 28 PCA values separated by commas:",
        value=",".join(map(str, st.session_state.random_data["pca"])),
        height=120
    )

    submitted = st.form_submit_button("🔍 Analyze Transaction")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:
    try:
        pca_values = [float(x.strip()) for x in pca_input.split(",")]

        if len(pca_values) != 28:
            st.error("❌ You must enter exactly 28 PCA values.")
            st.stop()

        # Build input dataframe
        data = {
            "Time": time,
            "Amount": amount
        }

        for i in range(28):
            data[f"V{i+1}"] = pca_values[i]

        df = pd.DataFrame([data])

        # Feature engineering (must match training)
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

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("📌 Demo app — ML Fraud Detection | Built with Streamlit")


# =========================================
# 💳 FRAUD DETECTION APP — STREAMLIT
# =========================================

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
THRESHOLD = 0.636
MODEL_PATH = "fraud_rf_model.pkl"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model_artifacts = load_model()

pipeline = model_artifacts["pipeline"]
features = model_artifacts["features"]
THRESHOLD = model_artifacts["threshold"]


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

# -------- RANDOM GENERATOR ----------
def generate_random_transaction():
    return {
        "amount": round(np.random.uniform(1, 2500), 2),
        "time": np.random.randint(0, 86400),
        "pca": np.random.normal(0, 1, 28).round(4)
    }


# -------- SESSION STATE INIT --------
if "random_data" not in st.session_state:
    st.session_state.random_data = generate_random_transaction()


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


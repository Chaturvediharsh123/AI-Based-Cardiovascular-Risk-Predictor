import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Cardio Risk Analyzer",
    layout="wide"
)

st.title("🫀 Cardiovascular Risk Prediction System")
st.write("AI-powered multi-model heart disease analysis")

# -------------------- LOAD MODELS (CACHED) --------------------
@st.cache_resource
def load_models():
    ecg_model = joblib.load("models/ecg_model.pkl")
    heart_model = joblib.load("models/heart_attack_model.pkl")
    diag_model = joblib.load("models/diagnosis_model.pkl")
    return ecg_model, heart_model, diag_model

ecg_model, heart_model, diag_model = load_models()

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 120, 45)
bp = st.sidebar.number_input("Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 180)
hr = st.sidebar.number_input("Heart Rate", 40, 180, 75)

# -------------------- ECG INPUT --------------------
st.header("📈 ECG Signal Input")

ecg_file = st.file_uploader("Upload ECG CSV file", type=["csv"])

ecg_data = None
if ecg_file is not None:
    ecg_df = pd.read_csv(ecg_file)
    st.subheader("ECG Signal Preview")
    st.line_chart(ecg_df)
    ecg_data = ecg_df.values.flatten()

# -------------------- PREDICTION BUTTON --------------------
if st.button("🔍 Analyze Heart Risk"):

    if ecg_data is None:
        st.error("Please upload ECG data")
    else:
        try:
            # ---------- 1️⃣ ECG MODEL ----------
            ecg_input = ecg_data.reshape(1, -1)
            ecg_risk = ecg_model.predict(ecg_input)[0]

            # ---------- 2️⃣ HEART ATTACK MODEL ----------
            clinical_input = np.array([[age, bp, chol, hr]])
            heart_prob = heart_model.predict_proba(clinical_input)[0][1]

            # ---------- 3️⃣ DIAGNOSIS MODEL ----------
            diag_input = np.array([[ecg_risk, heart_prob]])
            diagnosis = diag_model.predict(diag_input)[0]

            # ---------- FINAL RISK LOGIC ----------
            final_score = (0.5 * ecg_risk) + (0.5 * heart_prob)

            if final_score < 0.3:
                risk_label = "LOW RISK"
                color = "green"
            elif final_score < 0.7:
                risk_label = "MEDIUM RISK"
                color = "orange"
            else:
                risk_label = "HIGH RISK"
                color = "red"

            # -------------------- OUTPUT --------------------
            st.header("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("ECG Risk Score", f"{ecg_risk:.2f}")
            col2.metric("Heart Attack Probability", f"{heart_prob*100:.2f}%")
            col3.metric("Final Risk Score", f"{final_score*100:.2f}%")

            st.subheader("🧠 Diagnosis Result")

            if color == "green":
                st.success(f"✅ {risk_label} – Healthy condition")
            elif color == "orange":
                st.warning(f"⚠️ {risk_label} – Medical monitoring advised")
            else:
                st.error(f"🚨 {risk_label} – Immediate medical attention required")

            st.write(f"Diagnosis Model Output: **{diagnosis}**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

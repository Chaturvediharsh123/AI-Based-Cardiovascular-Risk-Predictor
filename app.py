try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    import subprocess
    subprocess.check_call(["pip", "install", "gradio"])
    import gradio as gr

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ---------------- LOAD MODELS ----------------
ecg_model = tf.keras.models.load_model(
    "timeseriesmodel.keras",
    compile=False
)

heart_model = joblib.load("model_xgb.pkl")
diag_model = joblib.load("model_cardio.pkl")

# ---------------- PREDICTION FUNCTION ----------------
def predict_risk(ecg_file, age, bp, chol, hr):
    try:
        # Read ECG
        ecg_df = pd.read_csv(ecg_file.name)
        ecg_data = ecg_df.values.flatten()
        ecg_input = ecg_data.reshape(1, -1, 1)

        # ECG prediction
        ecg_prob = ecg_model.predict(ecg_input)[0][0]

        # Clinical prediction
        clinical = np.array([[age, bp, chol, hr]])
        heart_prob = heart_model.predict_proba(clinical)[0][1]

        # Final risk
        final_risk = (ecg_prob + heart_prob) / 2

        # Diagnosis model
        diagnosis = diag_model.predict([[ecg_prob, heart_prob]])[0]

        # Risk label
        if final_risk < 0.3:
            label = "✅ LOW RISK"
        elif final_risk < 0.7:
            label = "⚠️ MEDIUM RISK"
        else:
            label = "🚨 HIGH RISK"

        return (
            round(ecg_prob, 3),
            round(heart_prob, 3),
            round(final_risk, 3),
            label,
            diagnosis
        )

    except Exception as e:
        return "Error", "Error", "Error", str(e), "Error"


# ---------------- GRADIO UI ----------------
interface = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.File(label="Upload ECG CSV"),
        gr.Number(label="Age", value=45),
        gr.Number(label="Blood Pressure", value=120),
        gr.Number(label="Cholesterol", value=180),
        gr.Number(label="Heart Rate", value=75),
    ],
    outputs=[
        gr.Textbox(label="ECG Risk Probability"),
        gr.Textbox(label="Heart Attack Probability"),
        gr.Textbox(label="Final Risk Score"),
        gr.Textbox(label="Risk Level"),
        gr.Textbox(label="Diagnosis Output"),
    ],
    title="🫀 Heart Risk Prediction (Gradio)",
    description="Educational purpose only. Not a medical diagnosis."
)

interface.launch()

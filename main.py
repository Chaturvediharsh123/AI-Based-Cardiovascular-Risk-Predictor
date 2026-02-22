
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

##
ecg_model = tf.keras.models.load_model(
    "timeseriesmodel.keras",
    compile=False
)

heart_model = joblib.load("model_xgb.pkl")
diag_model = joblib.load("model_cardio.pkl")

# ---------------- PREDICTION FUNCTION ----------------
def predict_risk(ecg_file, age, bp, chol, hr):

    ecg_df = pd.read_csv(ecg_file.name)
    ecg_data = ecg_df.values.flatten()
    ecg_input = ecg_data.reshape(1, -1, 1)


    ecg_prob = float(ecg_model.predict(ecg_input)[0][0])

    
    clinical = np.array([[age, bp, chol, hr]])
    heart_prob = float(heart_model.predict_proba(clinical)[0][1])


    final_risk = (ecg_prob + heart_prob) / 2


    diagnosis = diag_model.predict([[ecg_prob, heart_prob]])[0]

    if final_risk < 0.3:
        risk_label = "LOW RISK"
    elif final_risk < 0.7:
        risk_label = "MEDIUM RISK"
    else:
        risk_label = "HIGH RISK"

    return (
        round(ecg_prob, 3),
        round(heart_prob, 3),
        round(final_risk, 3),
        risk_label,
        diagnosis
    )


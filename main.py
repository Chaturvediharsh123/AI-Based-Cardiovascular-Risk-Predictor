import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ---------------- MODEL LOADING ----------------
def load_models():
    """
    Load all required ML models.
    Returns: (ecg_model, heart_model, diag_model, success_flag)
    """
    try:
        ecg_model = tf.keras.models.load_model("timeseriesmodel.keras", compile=False)
        heart_model = joblib.load("model_xgb.pkl")
        diag_model = joblib.load("model_cardio.pkl")
        return ecg_model, heart_model, diag_model, True
    except Exception as e:
        # Returns None for models but flags that we are in "Demo Mode"
        return None, None, None, False

def get_model_info(ecg_model):
    """
    Get expected input dimensions from ECG model.
    Returns: (expected_timesteps, expected_features, expected_size)
    """
    if ecg_model is not None:
        expected_timesteps = ecg_model.input_shape[1]
        expected_features = ecg_model.input_shape[2]
    else:
        expected_timesteps = 140  # Fallback default
        expected_features = 1
    
    expected_size = expected_timesteps * expected_features
    return expected_timesteps, expected_features, expected_size

# ---------------- DATA PROCESSING ----------------
def process_ecg(file, expected_timesteps, expected_features, expected_size):
    """
    Process ECG data from uploaded file or generate demo data.
    
    Args:
        file: Uploaded CSV file or None
        expected_timesteps: Expected number of timesteps
        expected_features: Expected number of features
        expected_size: Expected total size
    
    Returns:
        processed_input: Reshaped and normalized ECG data for model input
        raw_df: Raw ECG dataframe for visualization
    """
    if file is not None:
        df = pd.read_csv(file)
        data = df.values.flatten().astype(np.float32)
    else:
        # Generate realistic looking noisy sine wave for demo
        t = np.linspace(0, 4*np.pi, expected_size)
        data = np.sin(t) + 0.5 * np.sin(3*t) + np.random.normal(0, 0.1, expected_size)
        df = pd.DataFrame(data, columns=["Amplitude"])

    # Handle size mismatches
    if len(data) < expected_size:
        data = np.pad(data, (0, expected_size - len(data)))
    else:
        data = data[:expected_size]

    # Normalize
    mean, std = np.mean(data), np.std(data)
    if std == 0:
        std = 1
    data = (data - mean) / std
    
    # Reshape for model input
    data = data.reshape(expected_timesteps, expected_features)
    processed_input = np.expand_dims(data, axis=0)
    
    return processed_input, df

# ---------------- PREDICTION FUNCTIONS ----------------
def predict_ecg_risk(ecg_model, ecg_input):
    """Predict risk from ECG data"""
    if ecg_model is not None:
        ecg_prob = float(ecg_model(ecg_input, training=False).numpy()[0][0])
        if np.isnan(ecg_prob):
            ecg_prob = 0.5
    else:
        ecg_prob = None
    return ecg_prob

def predict_clinical_risk(heart_model, age, bp, chol, hr):
    """Predict risk from clinical parameters"""
    if heart_model is not None:
        clinical_input = np.array([[age, 1, 0, bp, chol, 0, 1, hr, 0, 1, 1, 0, 2, 0, 0]], dtype=np.float32)
        heart_prob = float(heart_model.predict_proba(clinical_input)[0][1])
    else:
        heart_prob = None
    return heart_prob

def simulate_clinical_risk(age, bp, chol, hr):
    """Simulate clinical risk when model is not available"""
    return np.clip((age/100)*0.4 + (bp/200)*0.3 + (chol/350)*0.3, 0, 1)

def simulate_ecg_risk(hr):
    """Simulate ECG risk when model is not available"""
    return np.clip(np.random.normal(0.4, 0.1) + (hr/200)*0.2, 0, 1)

def calculate_ensemble_risk(ecg_prob, heart_prob):
    """Calculate ensemble risk from both predictions"""
    if ecg_prob is not None and heart_prob is not None:
        return (ecg_prob + heart_prob) / 2
    elif ecg_prob is not None:
        return ecg_prob
    elif heart_prob is not None:
        return heart_prob
    else:
        return 0.5

def predict_diagnosis(diag_model, age, heart_prob):
    """Predict diagnosis code"""
    if diag_model is not None:
        diag_input = np.array([[age, 0, 250, 0, 35, 1, 250000, 1.0, 137, 1, 0, 100, heart_prob]], dtype=np.float32)
        diagnosis = diag_model.predict(diag_input)[0]
    else:
        diagnosis = None
    return diagnosis

def simulate_diagnosis(final_risk):
    """Simulate diagnosis when model is not available"""
    return "I48" if final_risk > 0.6 else "Z00"

def get_risk_level(final_risk):
    """
    Determine risk level based on final risk score.
    Returns: (level_string, color_code, message)
    """
    if final_risk < 0.3:
        return "LOW", "🟢", "Patient parameters are within normal ranges."
    elif final_risk < 0.7:
        return "MEDIUM", "🟡", "Elevated biomarkers detected. Observation recommended."
    else:
        return "CRITICAL", "🔴", "Immediate clinical evaluation required!"

# ---------------- MAIN PREDICTION PIPELINE ----------------
def run_prediction_pipeline(uploaded_file, age, bp, chol, hr, models_loaded, ecg_model, heart_model, diag_model):
    """
    Run the complete prediction pipeline.
    
    Returns:
        Dictionary containing all prediction results
    """
    # Get model info
    expected_timesteps, expected_features, expected_size = get_model_info(ecg_model)
    
    # Process ECG data
    ecg_input, ecg_df = process_ecg(uploaded_file, expected_timesteps, expected_features, expected_size)
    
    # Make predictions based on model availability
    if models_loaded:
        # Real predictions
        ecg_prob = predict_ecg_risk(ecg_model, ecg_input)
        heart_prob = predict_clinical_risk(heart_model, age, bp, chol, hr)
        final_risk = calculate_ensemble_risk(ecg_prob, heart_prob)
        diagnosis = predict_diagnosis(diag_model, age, heart_prob)
    else:
        # Simulated predictions
        ecg_prob = simulate_ecg_risk(hr)
        heart_prob = simulate_clinical_risk(age, bp, chol, hr)
        final_risk = calculate_ensemble_risk(ecg_prob, heart_prob)
        diagnosis = simulate_diagnosis(final_risk)
    
    # Get risk level
    risk_level, risk_icon, risk_message = get_risk_level(final_risk)
    
    return {
        'ecg_prob': ecg_prob,
        'heart_prob': heart_prob,
        'final_risk': final_risk,
        'diagnosis': diagnosis,
        'ecg_df': ecg_df,
        'risk_level': risk_level,
        'risk_icon': risk_icon,
        'risk_message': risk_message,
        'models_loaded': models_loaded
    }
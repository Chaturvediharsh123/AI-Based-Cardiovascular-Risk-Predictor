# AI-Based-Cardiovascular-Risk-Predictor

##  Overview
This project is a Machine Learning-based Cardiovascular Disease Prediction System developed as part of my BTech Artificial Intelligence coursework.

The system uses three trained models to analyze patient data and provide:

- ECG-based prediction
- Heart risk prediction
- Final cardiovascular diagnosis

---

##  Features

✔ Loads three trained ML models  
✔ Predicts heart-related risks  
✔ Clean and modular Python structure  
✔ Easy to convert into FastAPI or Streamlit app  
✔ Designed for academic and research purposes  

---

##  Models Used

The system uses three trained models:

1. **ECG Model** – Analyzes ECG-related features  
2. **Heart Risk Model** – Predicts heart disease risk  
3. **Diagnosis Model** – Provides final cardiovascular diagnosis  

Models are stored as:
cardio_project/
│
├── main.py
├── app.py
├── ecg_model.pkl
├── heart_model.pkl
├── diagnosis_model.pkl
└── README.md




## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/cardio-project.git

cd cardio_project

pip install joblib scikit-learn xgboost
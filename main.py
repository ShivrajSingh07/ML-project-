import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Diabetes Risk & Glucose Predictor",
    page_icon="🩺",
    layout="wide"
)

# ==============================================================================
# Load or Train Models (Cached for Fast Loading)
# ==============================================================================
@st.cache_resource
def load_models():
    artifact_path = "diabetes_model_artifacts.pkl"
    if os.path.exists(artifact_path):
        try:
            return joblib.load(artifact_path)
        except Exception:
            pass

    # Fallback: train on the dataset if pkl is missing
    csv_path = "diabetes_case_study_5000rows.csv"
    if not os.path.exists(csv_path):
        csv_path = "/content/diabetes_case_study_5000rows.csv"

    df = pd.read_csv(csv_path).drop_duplicates()

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'string']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    le_diet = LabelEncoder()

    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_status'] = le_smoking.fit_transform(df['smoking_status'])
    df['diet_quality'] = le_diet.fit_transform(df['diet_quality'])

    scale_cols = [
        'age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'cholesterol_level', 'blood_glucose_fasting', 'hba1c',
        'physical_activity_days', 'alcohol_units_week', 'stress_level', 'sleep_hours'
    ]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Regression: glucose_90day_avg (Ridge & Lasso)
    features_reg = ['blood_glucose_fasting', 'hba1c', 'bmi', 'cholesterol_level', 'age']
    X_reg = df[features_reg]
    y_reg = df['glucose_90day_avg']
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)

    ridge_reg = Ridge(alpha=1.0).fit(Xr_tr, yr_tr)
    lasso_reg = Lasso(alpha=0.1).fit(Xr_tr, yr_tr)

    # Classification: diabetes_risk (Gradient Boosting, Ridge, Lasso)
    X_clf = df.drop(['patient_id', 'diabetes_risk', 'glucose_90day_avg'], axis=1)
    y_clf = df['diabetes_risk']
    features_clf = list(X_clf.columns)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_clf, y_clf, test_size=0.20, random_state=42)

    gb_clf = GradientBoostingClassifier(random_state=42).fit(Xc_tr, yc_tr)
    ridge_clf = RidgeClassifier(alpha=1.0).fit(Xc_tr, yc_tr)
    lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=42).fit(Xc_tr, yc_tr)

    return {
        'ridge_reg': ridge_reg,
        'lasso_reg': lasso_reg,
        'ridge_clf': ridge_clf,
        'lasso_clf': lasso_clf,
        'gb_clf': gb_clf,
        'scaler': scaler,
        'scale_cols': scale_cols,
        'le_gender': le_gender,
        'le_smoking': le_smoking,
        'le_diet': le_diet,
        'features_reg': features_reg,
        'features_clf': features_clf
    }

artifacts = load_models()
ridge_reg = artifacts['ridge_reg']
lasso_reg = artifacts.get('lasso_reg', ridge_reg)
gb_clf = artifacts['gb_clf']
ridge_clf = artifacts['ridge_clf']
lasso_clf = artifacts.get('lasso_clf')
scaler = artifacts['scaler']
scale_cols = artifacts['scale_cols']
le_gender = artifacts['le_gender']
le_smoking = artifacts['le_smoking']
le_diet = artifacts['le_diet']
features_reg = artifacts['features_reg']
features_clf = artifacts['features_clf']


# ==============================================================================
# Header
# ==============================================================================
st.title("🩺 Diabetes Risk & Blood Glucose Predictor")
st.write(
    "Enter patient measurements below to predict **Diabetes Risk** and estimated **90-Day Blood Glucose Level**. "
    "Powered by trained **Ridge**, **Lasso**, and **Gradient Boosting** models."
)
st.write("---")

# ==============================================================================
# Input Form - Normal, Spacious 2-Column Layout
# ==============================================================================
st.subheader("Patient Health Information")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("**Basic Details & Vitals**")
    age = st.number_input("Age (Years)", min_value=18, max_value=95, value=45, step=1)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=12.0, max_value=50.0, value=26.5, step=0.1)
    bp_systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80.0, max_value=220.0, value=125.0, step=1.0)
    bp_diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=50.0, max_value=130.0, value=82.0, step=1.0)
    smoking = st.selectbox("Smoking Status", options=["Never", "Former", "Current"])
    diet = st.selectbox("Diet Quality", options=["Good", "Average", "Poor"])

with col2:
    st.markdown("**Biomarkers & Lifestyle**")
    fasting_glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=110.0, step=1.0)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=15.0, value=6.0, step=0.1)
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100.0, max_value=400.0, value=205.0, step=1.0)
    family_history = st.selectbox("Family History of Diabetes?", options=["No", "Yes"])
    activity_days = st.slider("Physical Activity (Days per week)", min_value=0, max_value=7, value=3)
    sleep_hours = st.slider("Sleep Hours (per night)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    stress_level = st.slider("Stress Level (1 = Low, 10 = High)", min_value=1, max_value=10, value=5)
    alcohol_units = st.number_input("Alcohol Intake (Units per week)", min_value=0.0, max_value=35.0, value=2.0, step=0.5)

st.write("")
predict_button = st.button("🔍 Predict Diabetes Risk & Glucose Level", type="primary", use_container_width=True)

# ==============================================================================
# Predictions & Results Section
# ==============================================================================
if predict_button:
    # 1. Format raw inputs
    raw_dict = {
        'age': float(age),
        'gender': gender,
        'bmi': float(bmi),
        'blood_pressure_systolic': float(bp_systolic),
        'blood_pressure_diastolic': float(bp_diastolic),
        'cholesterol_level': float(cholesterol),
        'blood_glucose_fasting': float(fasting_glucose),
        'hba1c': float(hba1c),
        'smoking_status': smoking,
        'physical_activity_days': float(activity_days),
        'family_history_diabetes': 1.0 if family_history == "Yes" else 0.0,
        'alcohol_units_week': float(alcohol_units),
        'stress_level': float(stress_level),
        'sleep_hours': float(sleep_hours),
        'diet_quality': diet
    }

    input_df = pd.DataFrame([raw_dict])

    # 2. Encode and Scale
    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['smoking_status'] = le_smoking.transform(input_df['smoking_status'])
    input_df['diet_quality'] = le_diet.transform(input_df['diet_quality'])
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # 3. Predict Classification (Diabetes Risk)
    X_c = input_df[features_clf]
    risk_pred = gb_clf.predict(X_c)[0]
    risk_prob = gb_clf.predict_proba(X_c)[0][1]

    # Predict with Ridge and Lasso
    ridge_c_pred = ridge_clf.predict(X_c)[0]
    lasso_c_pred = lasso_clf.predict(X_c)[0] if lasso_clf else risk_pred

    # 4. Predict Regression (90-Day Glucose)
    X_r = input_df[features_reg]
    glucose_ridge = ridge_reg.predict(X_r)[0]
    glucose_lasso = lasso_reg.predict(X_r)[0] if lasso_reg else glucose_ridge

    st.write("---")
    st.subheader("Prediction Results")

    res_col1, res_col2 = st.columns(2, gap="large")

    with res_col1:
        if risk_pred == 1:
            st.error("### ⚠️ Result: High Diabetes Risk")
            st.write(f"**Estimated Risk Probability:** `{risk_prob * 100:.1f}%`")
            st.write("The model identified elevated indicators for diabetes. A medical checkup is recommended.")
        else:
            st.success("### ✅ Result: Low Diabetes Risk")
            st.write(f"**Estimated Risk Probability:** `{risk_prob * 100:.1f}%`")
            st.write("The patient's current measurements are within a lower risk range.")

        st.caption(f"Model Consensus: Ridge Classifier = **{'High Risk' if ridge_c_pred == 1 else 'Low Risk'}** | Lasso Logistic = **{'High Risk' if lasso_c_pred == 1 else 'Low Risk'}**")

    with res_col2:
        st.info("### 🩸 Estimated 90-Day Glucose")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Ridge Regression", f"{glucose_ridge:.1f} mg/dL")
        metric_col2.metric("Lasso Regression", f"{glucose_lasso:.1f} mg/dL")

        if glucose_ridge < 140:
            st.write("Status: **Normal Glucose Range** (< 140 mg/dL)")
        elif glucose_ridge <= 180:
            st.write("Status: **Borderline / Pre-diabetic Range** (140 - 180 mg/dL)")
        else:
            st.write("Status: **Elevated Glucose Range** (> 180 mg/dL)")

    # Actionable Advice
    st.markdown("**Actionable Recommendations:**")
    recs = []
    if fasting_glucose >= 126 or hba1c >= 6.5:
        recs.append("Consult a doctor for an oral glucose tolerance test (OGTT).")
    if bmi >= 25.0:
        recs.append("Incorporate regular physical activity to help lower body mass index.")
    if activity_days < 3:
        recs.append("Aim for at least 30 minutes of moderate exercise 3 to 5 days a week.")
    if sleep_hours < 6.5:
        recs.append("Improve sleep consistency; aim for 7 to 8 hours of sleep per night.")
    if not recs:
        recs.append("Keep up the healthy lifestyle habits and continue routine health checks.")

    for r in recs:
        st.write(f"- {r}")

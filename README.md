# 🩺 Diabetes Risk Prediction & Health Analytics (CSM-355)

A clean Machine Learning project and interactive web application designed to assess **Diabetes Risk** and estimate **90-Day Blood Glucose Levels** using clinical biomarkers.

---

## 📌 Project Overview

This project implements and compares linear models, regularized models (**Ridge & Lasso**), and ensemble methods on a 5,000-patient dataset:
1. **Regression Task**: Predicting **90-Day Blood Glucose Average** (`glucose_90day_avg`) using **Linear Regression**, **Ridge Regression**, and **Lasso Regression**.
2. **Classification Task**: Predicting **Diabetes Risk** (`diabetes_risk`: 0 = Low Risk, 1 = High Risk) using **Logistic Regression**, **Ridge Classifier**, **Lasso (L1 Logistic)**, and **Gradient Boosting Classifier**.
3. **Frontend Application**: A clean, normal, and user-friendly web interface (`main.py`) built with **Streamlit** for real-time patient assessment.

---

## 📁 Project Structure

```text
Diabetes/
├── diabetes_case_study_5000rows.csv    # Patient health records dataset (5,000 rows)
├── Diabetes_ML_Project_CSM355.ipynb    # Clean & pre-executed ML Notebook (Q1 to Q5)
├── diabetes_model_artifacts.pkl        # Serialized trained models & preprocessing pipeline
├── main.py                             # Clean & normal Streamlit web frontend
└── README.md                           # Project documentation
```

---

## 📊 Dataset Summary

The dataset contains **5,000 patient records** with 15 input features:

| Category | Features |
| :--- | :--- |
| **Biomarkers** | Fasting Blood Glucose, HbA1c, Cholesterol Level |
| **Vitals & Physical** | Age, Gender, BMI, Systolic & Diastolic Blood Pressure |
| **Lifestyle & Habits** | Diet Quality, Smoking Status, Physical Activity Days, Alcohol Units/Week, Sleep Hours, Stress Level |
| **Medical History** | Family History of Diabetes |
| **Target 1 (Continuous)** | `glucose_90day_avg` (mg/dL) |
| **Target 2 (Binary)** | `diabetes_risk` (0: Low Risk, 1: High Risk) |

---

## 🧠 Model Comparisons (Ridge, Lasso & Best Models)

### 1. Regression Models (`glucose_90day_avg`)
Predicting sustained 3-month average blood glucose levels:

| Model | $R^2$ Score | MAE (mg/dL) | RMSE (mg/dL) |
| :--- | :---: | :---: | :---: |
| **Ridge Regression ($\alpha=1.0$)** | **0.7190** | **8.84** | **11.17** |
| **Linear Regression** | 0.7190 | 8.84 | 11.17 |
| **Lasso Regression ($\alpha=0.1$)** | 0.7188 | 8.85 | 11.17 |

> **Ridge vs. Lasso in Regression:**
> - **Ridge ($L_2$ Regularization)** adds a penalty term ($\alpha \sum w_i^2$) that shrinks coefficients smoothly. This prevents weight inflation caused by strong correlation between Fasting Glucose and HbA1c.
> - **Lasso ($L_1$ Regularization)** adds a penalty term ($\alpha \sum |w_i|$) that can drive non-essential coefficients to zero, performing automatic feature selection.

---

### 2. Classification Models (`diabetes_risk`)
Predicting whether a patient is at high or low risk for diabetes:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Best Model)** | **81.40%** | **77.62%** | 72.80% | **0.7513** |
| **Lasso (L1 Logistic)** | **81.30%** | 76.68% | **74.09%** | **0.7536** |
| **Ridge Classifier ($\alpha=1.0$)** | 81.20% | 76.76% | 73.58% | 0.7513 |
| **Logistic Regression** | 81.10% | 76.41% | 73.83% | 0.7510 |

---

## 💻 Frontend Web App (`main.py`)

The frontend has a normal, clean, and spacious layout that is easy to read:
- **Clean 2-Column Input Form**: Basic vitals on the left, biomarkers and lifestyle on the right with plenty of breathing room.
- **Clear Predictions**:
  - `✅ Low Diabetes Risk` or `⚠️ High Diabetes Risk` banner with probability score.
  - Side-by-side **Ridge** and **Lasso** 90-day glucose predictions.
  - Actionable clinical tips based on patient values.

---

## 🚀 How to Run the Project

### Prerequisites
Install dependencies:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

### 1. Run the Frontend Web Application
```bash
streamlit run main.py
```
Open your browser at `http://localhost:8501`.

### 2. Run or View the Notebook
Open [`Diabetes_ML_Project_CSM355.ipynb`](file:///d:/Desktop/Diabetes/Diabetes_ML_Project_CSM355.ipynb) in **VS Code**, **Jupyter Lab**, or **Google Colab**. All cells are pre-executed with outputs, tables, and comparison charts.

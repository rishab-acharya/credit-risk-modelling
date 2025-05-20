# 🏦 Credit Risk Modelling & Scoring

This project simulates a real-world credit scoring pipeline for a financial institution. It includes data cleaning, model development (GLM, Logistic Regression, XGBoost), customer risk segmentation, and deployment via a Streamlit app.

---

## 📦 Features

- Cleaned and transformed German Credit Data
- Exploratory Data Analysis (EDA) with key insights
- Machine learning models for classification:
  - Logistic Regression
  - Generalized Linear Model (GLM) with Stepwise AIC
  - XGBoost
- Model comparison using AUC and confusion matrix
- Risk segmentation using quantile bands
- Streamlit app for real-time scoring and band assignment

---

## 📊 Model Performance

| Model      | AUC    | Accuracy |
|------------|--------|----------|
| GLM        | 0.81   | 77.5%    |
| Logistic   | 0.79   | 76%      |
| XGBoost    | 0.79   | 78%      |

---

## 🧪 Tech Stack

- Python, pandas, NumPy
- scikit-learn, statsmodels, XGBoost
- Streamlit (app deployment)
- Git for version control

---

## 🚀 Run Locally

```bash
# Activate your environment first
streamlit run app.py

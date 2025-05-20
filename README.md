# 💳 Credit Risk Modelling & Scoring Pipeline

This project is an end-to-end **credit risk scoring system** built using real-world financial data, SQL databases, machine learning (Logistic Regression + XGBoost), and an interactive Streamlit app. 

It simulates how banks or fintech companies assess default risk, assign customer risk bands, and apply pricing strategies based on predicted risk.

---

## 🧠 Project Features

✅ Cleaned and encoded UCI credit data  
✅ Stored data in a SQLite database  
✅ Built and compared **Logistic Regression** and **XGBoost** models  
✅ Evaluated with **AUC, precision, recall**  
✅ Created **risk bands** using probability quantiles  
✅ Assigned **interest rates** (5%, 10%, 20%) by risk  
✅ Developed a **Streamlit app** for live scoring  

---



---

## 🚀 How to Run Locally

### 📦 1. Install requirements

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install -r requirements.txt
python scripts/01_load_and_clean.py
python scripts/02_sql_setup.py


## 📊 3. Train Models
python scripts/03_logistic_model.py
python scripts/04_xgboost_model.py


##🎯 4. Segment Customers by Risk
python scripts/05_risk_segmentation.py

##  5. Launch the App
streamlit run app.py



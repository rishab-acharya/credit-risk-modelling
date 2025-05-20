import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
from sklearn.preprocessing import StandardScaler

# Load data from SQL to get column names
conn = sqlite3.connect(r"data/credit.db")
df = pd.read_sql("SELECT * FROM credit_data;", conn)
conn.close()

X = df.drop("default", axis=1)
y = df["default"]

# Train scaler and model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dtrain = xgb.DMatrix(X_scaled, label=y)
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "eta": 0.1
}
model = xgb.train(params, dtrain, num_boost_round=100)

# Streamlit UI
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.markdown("Enter customer information to predict default risk, risk band, and interest rate.")

# Build input form dynamically
user_input = {}
for col in X.columns:
    if df[col].dtype == "int64":
        val = st.number_input(f"{col}", value=int(df[col].mean()), step=1)
    else:
        val = st.selectbox(f"{col}", sorted(df[col].unique()))
    user_input[col] = val

# Predict button
if st.button("Predict Risk"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    dinput = xgb.DMatrix(input_scaled)
    prob = model.predict(dinput)[0]

    # Risk band + rate
    if prob < 0.33:
        risk = "Low"
        rate = 5.0
    elif prob < 0.66:
        risk = "Medium"
        rate = 10.0
    else:
        risk = "High"
        rate = 20.0

    st.success(f"🧠 Predicted default probability: **{prob:.2f}**")
    st.info(f"💡 Risk Band: **{risk}**, Suggested Interest Rate: **{rate:.1f}%**")

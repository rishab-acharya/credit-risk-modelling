import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# Load model and feature list
with open("outputs/glm_stepwise.pkl", "rb") as f:
    glm_model = pickle.load(f)

with open("outputs/glm_selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# App Title
st.title("📊 Credit Risk Score & Risk Band Prediction")
st.markdown("This tool uses a stepwise-selected Generalized Linear Model (GLM) to predict default risk and assign a risk segment.")

# Sidebar Input Form
st.sidebar.header("📥 Input Customer Details")
input_data = {}

for feature in selected_features:
    if feature == "CreditAmount":
        input_data[feature] = st.sidebar.slider(feature, 250, 20000, 3000)
    elif feature == "Duration":
        input_data[feature] = st.sidebar.slider(feature, 4, 72, 24)
    elif feature == "Age":
        input_data[feature] = st.sidebar.slider(feature, 18, 75, 35)
    else:
        input_data[feature] = st.sidebar.number_input(feature, min_value=0, max_value=10, step=1, value=1)

# Predict Button
if st.sidebar.button("🔍 Predict Risk"):
    user_df = pd.DataFrame([input_data])
    user_df = sm.add_constant(user_df, has_constant='add')

    # Predict default probability
    pred_prob = glm_model.predict(user_df)[0]
    # Load saved bin edges
    bins = np.load("outputs/glm_risk_bins.npy")
    risk_band = pd.cut([pred_prob], bins=bins, labels=["Low", "Medium", "High"], include_lowest=True)[0]


    st.success("✅ Prediction Complete!")
    st.metric("💡 Default Probability", f"{pred_prob:.2%}")
    st.metric("📉 Risk Segment", risk_band)

# Footer
st.markdown("---")
st.caption("Built with 💻 by Rishab Acharya · GLM Stepwise Model · 2025")

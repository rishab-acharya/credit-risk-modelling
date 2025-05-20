import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# Load cleaned data
df = pd.read_csv("data/credit_data_cleaned.csv")

# Load retrained GLM model and selected features
with open("outputs/glm_stepwise.pkl", "rb") as f:
    glm_model = pickle.load(f)

with open("outputs/glm_selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# Prepare features
X = sm.add_constant(df[selected_features], has_constant='add')

# Predict probabilities and segment
df["default_proba_glm"] = glm_model.predict(X)
df["risk_band_glm"] = pd.qcut(df["default_proba_glm"], q=3, labels=["Low", "Medium", "High"])

# Save
df.to_csv("data/credit_data_segmented.csv", index=False)
print("✅ Risk segmentation completed and saved to data/credit_data_segmented.csv")

# Save quantile edges for Streamlit use
proba = df["default_proba_glm"]
bins = pd.qcut(proba, q=3, retbins=True, duplicates='drop')[1]  # returns bin edges
np.save("outputs/glm_risk_bins.npy", bins)

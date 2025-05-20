import pandas as pd
import sqlite3
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# STEP 1: Load data from database
conn = sqlite3.connect(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit.db")
df = pd.read_sql("SELECT * FROM credit_data;", conn)
conn.close()

# STEP 2: Split features + label
X = df.drop("default", axis=1)
y = df["default"]

# STEP 3: Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 4: Split and format for XGBoost
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# STEP 5: Re-train XGBoost model (or load from earlier)
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "eta": 0.1
}
model = xgb.train(params, dtrain, num_boost_round=100)

# STEP 6: Predict default probability
y_probs = model.predict(dtest)

# STEP 7: Create risk bands using quantiles
risk_band = pd.qcut(y_probs, q=3, labels=["Low", "Medium", "High"])

# STEP 8: Define pricing logic
def assign_rate(risk):
    return {"Low": 5.0, "Medium": 10.0, "High": 20.0}[risk]

rates = risk_band.map(assign_rate)

# STEP 9: Output CSV with segmentation
output_df = pd.DataFrame({
    "prob_default": y_probs,
    "risk_band": risk_band,
    "interest_rate": rates
})
output_df.index.name = "customer_id"
output_df.to_csv(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\risk_segments.csv")

print("✅ Risk segmentation and pricing saved to 'risk_segments.csv'")
print(output_df.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Plot risk distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=output_df, x="risk_band", order=["Low", "Medium", "High"], palette="Set2")
plt.title("Customer Distribution by Risk Band")
plt.xlabel("Risk Band")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\risk_distribution.png")
plt.show()

import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load from SQL
conn = sqlite3.connect(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit.db")
df = pd.read_sql("SELECT * FROM credit_data;", conn)
conn.close()


X = df.drop("default", axis=1)
y = df["default"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "eta": 0.1
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

#  Predict
y_probs = model.predict(dtest)
y_preds = [1 if p > 0.5 else 0 for p in y_probs]

auc = roc_auc_score(y_test, y_probs)
print("🌲 XGBoost ROC AUC Score:", round(auc, 4))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_preds))

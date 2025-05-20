import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# Connect to SQLite 
conn = sqlite3.connect(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit.db")
query = "SELECT * FROM credit_data;"
df = pd.read_sql(query, conn)
conn.close()


X = df.drop("default", axis=1)
y = df["default"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)


y_probs = model.predict_proba(X_test)[:, 1]
y_preds = model.predict(X_test)

auc = roc_auc_score(y_test, y_probs)
print("🔍 ROC AUC Score:", round(auc, 4))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_preds))

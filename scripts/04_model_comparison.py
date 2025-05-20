import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, confusion_matrix
)
from xgboost import XGBClassifier
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/credit_data_cleaned.csv")

# Define target and features
target = 'default'
X = df.drop(columns=target)
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 1. GLM with same variables ===
glm_features = X.columns.tolist()
glm_formula = f"{target} ~ {' + '.join(glm_features)}"
glm_model = glm(formula=glm_formula, data=df, family=Binomial()).fit()
glm_preds = glm_model.predict(X_test)
glm_auc = roc_auc_score(y_test, glm_preds)

# === 2. Logistic Regression (scikit-learn) ===
logit_model = LogisticRegression(max_iter=1000)
logit_model.fit(X_train, y_train)
logit_probs = logit_model.predict_proba(X_test)[:, 1]
logit_auc = roc_auc_score(y_test, logit_probs)

# === 3. XGBoost ===
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)

# === Evaluation Metrics ===
def evaluate_model(name, probs, true_labels):
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(true_labels, preds)
    acc = accuracy_score(true_labels, preds)
    auc = roc_auc_score(true_labels, probs)
    print(f"📊 {name} Results:")
    print("AUC:", round(auc, 4))
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", cm)
    print("-" * 30)

evaluate_model("GLM", glm_preds, y_test)
evaluate_model("Logistic Regression", logit_probs, y_test)
evaluate_model("XGBoost", xgb_probs, y_test)

# === ROC Curve Plot ===
fpr_glm, tpr_glm, _ = roc_curve(y_test, glm_preds)
fpr_logit, tpr_logit, _ = roc_curve(y_test, logit_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_glm, tpr_glm, label=f"GLM (AUC = {glm_auc:.2f})")
plt.plot(fpr_logit, tpr_logit, label=f"Logistic (AUC = {logit_auc:.2f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {xgb_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/roc_model_comparison.png")
plt.show()

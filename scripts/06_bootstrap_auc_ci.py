import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import joblib

# Load predictions and true labels from earlier saved models
y_test = joblib.load("outputs/y_test.pkl")
glm_preds = joblib.load("outputs/glm_preds.pkl")
logistic_preds = joblib.load("outputs/logistic_preds.pkl")
xgb_preds = joblib.load("outputs/xgb_preds.pkl")

def bootstrap_auc(y_true, y_pred, n_iterations=1000):
    auc_scores = []
    np.random.seed(42)
    for _ in range(n_iterations):
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc = roc_auc_score(y_true[indices], y_pred[indices])
        auc_scores.append(auc)
    return np.percentile(auc_scores, [2.5, 97.5]), np.mean(auc_scores)

# Run bootstrapping
models = {'GLM': glm_preds, 'Logistic': logistic_preds, 'XGBoost': xgb_preds}

for name, preds in models.items():
    ci, mean_auc = bootstrap_auc(y_test, preds)
    print(f"{name} - Mean AUC: {mean_auc:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

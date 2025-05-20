import pandas as pd
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
import statsmodels.api as sm
import numpy as np
import os
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix

# Load cleaned dataset
df = pd.read_csv("data/credit_data_cleaned.csv")

target = 'default'
features = df.columns.drop(target)

def stepwise_selection(data, target, candidate_features):
    included = []
    while True:
        changed = False
        # forward step
        excluded = list(set(candidate_features) - set(included))
        new_pvals = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            formula = f"{target} ~ {' + '.join(included + [new_column])}"
            model = glm(formula=formula, data=data, family=Binomial()).fit()
            new_pvals[new_column] = model.aic
        
        if not new_pvals.empty:
            best_feature = new_pvals.idxmin()
            if best_feature not in included:
                included.append(best_feature)
                changed = True
        
        # backward step
        if len(included) > 1:
            aic_with = pd.Series(index=included, dtype=float)
            for col in included:
                test_features = [f for f in included if f != col]
                formula = f"{target} ~ {' + '.join(test_features)}"
                model = glm(formula=formula, data=data, family=Binomial()).fit()
                aic_with[col] = model.aic
            worst_feature = aic_with.idxmin()
            if aic_with[worst_feature] < model.aic:
                included.remove(worst_feature)
                changed = True

        if not changed:
            break

    return included

# Stepwise feature selection
selected_features = stepwise_selection(df, target, features)
print("✅ Selected features via stepwise AIC:", selected_features)

# Final model
formula = f"{target} ~ {' + '.join(selected_features)}"
final_model = glm(formula=formula, data=df, family=Binomial()).fit()
print(final_model.summary())


df['pred_prob'] = final_model.predict(df[selected_features])
df['pred_label'] = (df['pred_prob'] > 0.5).astype(int)

auc = roc_auc_score(df[target], df['pred_prob'])
cm = confusion_matrix(df[target], df['pred_label'])

print("\n🔍 AUC Score:", round(auc, 4))
print("\n📊 Confusion Matrix:\n", cm)


X = sm.add_constant(df[selected_features])
y = df['default']
glm_matrix_model = sm.GLM(y, X, family=sm.families.Binomial()).fit()


with open("outputs/glm_stepwise.pkl", "wb") as f:
    pickle.dump(glm_matrix_model, f)


with open("outputs/glm_selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)


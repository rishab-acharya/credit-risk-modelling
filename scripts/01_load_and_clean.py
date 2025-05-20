import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load raw data
df = pd.read_csv(
    r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit_data.csv",  # Ensure this is your renamed 'german.data'
    header=None,
    sep='\s+'
)

# 2. Assign column names (based on german.doc)
df.columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "EmploymentSince",
    "InstallmentRate", "PersonalStatusSex", "OtherDebtors", "ResidenceSince", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "NumberCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker", "default"
]

# 3. Convert target to binary: 1 = good (no default), 2 = bad (default)
df["default"] = df["default"].map({1: 0, 2: 1})

# 4. Label encode categorical features
label_enc = LabelEncoder()

# Categorical columns (excluding target and already numerical)
cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    df[col] = label_enc.fit_transform(df[col])

# 5. Final check
print("✅ Cleaned Data Preview:")
print(df.head())
print("\n📊 Data Types:\n", df.dtypes)

# 6. Save cleaned dataset for later use (EDA, modelling, etc.)
df.to_csv(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit_data_cleaned.csv", index=False)
print("\n📁 Cleaned dataset saved as 'credit_data_cleaned.csv'")

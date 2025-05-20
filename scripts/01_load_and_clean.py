import pandas as pd
from sklearn.preprocessing import LabelEncoder  

# Load the raw dataset
df = pd.read_csv(
    r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit_data.csv",
    header=None,
    sep=r'\s+'  
)



df.columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "EmploymentSince",
    "InstallmentRate", "PersonalStatusSex", "OtherDebtors", "ResidenceSince", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "NumberCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker", "default"
]

# Convert target variable: 1 = Good → 0, 2 = Bad → 1
df["default"] = df["default"].map({1: 0, 2: 1})


label_enc = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    df[col] = label_enc.fit_transform(df[col])


print("✅ Cleaned Data Preview:")
print(df.head())
print("\n📊 Data Types:\n", df.dtypes)

# Save cleaned dataset
df.to_csv(
    r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit_data_cleaned.csv",
    index=False
)
print("\n📁 Cleaned dataset saved as 'credit_data_cleaned.csv'")

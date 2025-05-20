import pandas as pd

# Load file using whitespace as delimiter
df = pd.read_csv(
    r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit_data.csv",  # your renamed german.data
    header=None,
    sep='\s+'
)

# Confirm shape and preview
print("Shape:", df.shape)
print(df.head())

# Column names based on german.doc
df.columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "EmploymentSince",
    "InstallmentRate", "PersonalStatusSex", "OtherDebtors", "ResidenceSince", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "NumberCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker", "default"
]

from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove("Status")  # We'll handle it separately to preserve order

# Label Encode categorical features
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Map target values: 1 = Good, 2 = Bad → convert to binary
df['default'] = df['default'].map({1: 0, 2: 1})  # 1 = default

# Final check
print(df.dtypes)
print(df.head())

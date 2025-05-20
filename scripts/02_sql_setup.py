import pandas as pd
import sqlite3

# STEP 1: Load the cleaned data
df = pd.read_csv(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit_data.csv", sep='\s+', header=None)

# Assign proper column names
df.columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "EmploymentSince",
    "InstallmentRate", "PersonalStatusSex", "OtherDebtors", "ResidenceSince", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "NumberCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker", "default"
]

# STEP 2: Optional — encode text columns
from sklearn.preprocessing import LabelEncoder
cat_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
df['default'] = df['default'].map({1: 0, 2: 1})

# STEP 3: Connect to SQLite and save table
conn = sqlite3.connect(r"C:\Users\racharya\OneDrive - University of Edinburgh\credit-risk-modelling\data\credit.db")
df.to_sql("credit_data", conn, if_exists="replace", index=False)
print("✅ Data stored in credit.db successfully.")

# STEP 4: Run a test query
query = "SELECT * FROM credit_data WHERE CreditAmount > 5000 LIMIT 5;"
results = pd.read_sql(query, conn)
print("\nSample Query Result:\n", results)

conn.close()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ✅ Make sure 'outputs/' exists
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/credit_data_cleaned.csv")

# EDA logic continues...


# --- Basic Overview ---
print("📌 Shape:", df.shape)
print("🧾 Column Types:\n", df.dtypes)
print("🧼 Null Values:\n", df.isnull().sum())
print("🎯 Target Distribution:\n", df['default'].value_counts(normalize=True))

# --- Summary Stats ---
print("\n📈 Descriptive Statistics:")
print(df.describe())

# --- Target Distribution Plot ---
plt.figure(figsize=(5, 4))
sns.countplot(x='default', data=df)
plt.title("Target Variable Distribution (Default = 1)")
plt.xlabel("Default")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/eda_target_distribution.png")
plt.close()

# --- Correlation Heatmap ---
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/eda_correlation_heatmap.png")
plt.close()

# --- Boxplots: Numerical vs Target ---
numeric_cols = df.select_dtypes(include='number').drop(columns=['default']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='default', y=col, data=df)
    plt.title(f"{col} by Default")
    plt.tight_layout()
    plt.savefig(f"outputs/eda_boxplot_{col}.png")
    plt.close()

print("✅ EDA plots saved in /outputs/")

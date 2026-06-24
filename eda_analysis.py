import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset.csv")

# Dataset Information
print("\n===== DATASET INFO =====")
print(df.info())

# First 5 Rows
print("\n===== FIRST 5 ROWS =====")
print(df.head())

# Summary Statistics
print("\n===== SUMMARY STATISTICS =====")
print(df.describe())

# Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Select Numerical Columns
numeric_df = df.select_dtypes(include=['number'])

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("images/correlation_matrix.png")
plt.close()

# Histograms
numeric_df.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig("images/histograms.png")
plt.close()

print("\n===== EDA INSIGHTS =====")

# Basic Insights
if "Income" in df.columns:
    print(f"Average Income: {df['Income'].mean():.2f}")

if "Credit_Score" in df.columns:
    print(f"Average Credit Score: {df['Credit_Score'].mean():.2f}")

if "Loan_Amount" in df.columns:
    print(f"Average Loan Amount: {df['Loan_Amount'].mean():.2f}")

print("\nCorrelation Matrix:")
print(numeric_df.corr())

print("\nEDA Completed Successfully!")
print("Images saved in 'images' folder:")
print("- correlation_matrix.png")
print("- histograms.png")
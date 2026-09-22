import pandas as pd
from pathlib import Path

# Read output file
file = Path(r"C:\Users\maury\Downloads\InsightAI_Output\final_output.csv")
df = pd.read_csv(file)

print("========== VALIDATION ==========")

# Required columns
required = [
    "order_id", "order_date", "customer_name",
    "city", "state", "region", "country",
    "category", "sub_category", "product_name",
    "quantity", "unit_price", "revenue", "profit"
]

missing_columns = [col for col in required if col not in df.columns]

print("\nMissing Columns:", missing_columns)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicates
print("\nDuplicates:", df.duplicated().sum())

# Negative values
print("\nNegative Quantity:", (df["quantity"] < 0).sum())
print("Negative Unit Price:", (df["unit_price"] < 0).sum())
print("Negative Revenue:", (df["revenue"] < 0).sum())
print("Negative Profit:", (df["profit"] < 0).sum())

# Revenue validation
expected_revenue = df["quantity"] * df["unit_price"]

wrong_revenue = (
    (df["revenue"] - expected_revenue).abs() > 0.01
).sum()

print("\nIncorrect Revenue:", wrong_revenue)

# Date validation
df["order_date"] = pd.to_datetime(
    df["order_date"],
    format="%m-%d-%y",
    errors="coerce"
)

print("Invalid Dates:", df["order_date"].isna().sum())

print("\n========== VALIDATION COMPLETED ==========")


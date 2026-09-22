import pandas as pd

# Read data
df = pd.read_csv(
    r"C:\Users\maury\Downloads\InsightAI_Output\final_output.csv"
)

print("========== FEATURE ENGINEERING ==========")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Date features
df["order_date"] = pd.to_datetime(
    df["order_date"],
    format="mixed",
    errors="coerce"
)

df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["quarter"] = df["order_date"].dt.quarter
df["day_of_week"] = df["order_date"].dt.dayofweek

# Business features
df["profit_margin"] = df["profit"] / df["revenue"] * 100
df["revenue_per_unit"] = df["revenue"] / df["quantity"]

print("\nNew Features:")
print([
    "year", "month", "quarter",
    "day_of_week", "profit_margin",
    "revenue_per_unit"
])

print("\n========== SAMPLE ==========")
print(df.head())

# Save
output = r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"
df.to_csv(output, index=False)

print("\n========== FEATURE ENGINEERING COMPLETED ==========")
print("Saved to:", output)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


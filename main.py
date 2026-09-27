import pandas as pd
import numpy as np

raw_file = r"C:\Users\maury\Downloads\archive\product_sales_dataset_final.csv"
etl_file = r"C:\Users\maury\Downloads\InsightAI_Output\final_output.csv"

raw = pd.read_csv(raw_file)
etl = pd.read_csv(etl_file)

# Clean column names for comparison
raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")
etl.columns = etl.columns.str.strip().str.lower().str.replace(" ", "_")

print("\n==============================")
print("RAW vs ETL COMPARISON")
print("==============================")

print("RAW rows:", len(raw))
print("ETL rows:", len(etl))

print("\nRAW columns:")
print(raw.columns.tolist())

print("\nETL columns:")
print(etl.columns.tolist())

# Compare important numeric columns
for col in ["quantity", "unit_price", "revenue", "profit"]:

    if col not in raw.columns or col not in etl.columns:
        print(f"\n{col}: NOT FOUND")
        continue

    raw_values = pd.to_numeric(raw[col], errors="coerce")
    etl_values = pd.to_numeric(etl[col], errors="coerce")

    # Compare same row positions
    n = min(len(raw_values), len(etl_values))

    difference = raw_values.iloc[:n].values - etl_values.iloc[:n].values

    print(f"\n--- {col.upper()} ---")
    print("Raw mean :", raw_values.mean())
    print("ETL mean  :", etl_values.mean())
    print("Max diff  :", np.nanmax(np.abs(difference)))
    print("Changed   :", np.sum(np.abs(difference) > 1e-10))

print("\n==============================")
print("PROFIT FORMULA")
print("==============================")

if all(c in etl.columns for c in ["profit", "revenue"]):

    margin = etl["profit"] / etl["revenue"]

    print("Mean margin  :", margin.mean())
    print("Median margin:", margin.median())
    print("Min margin   :", margin.min())
    print("Max margin   :", margin.max())

print("\n==============================")
print("DONE")
print("==============================")

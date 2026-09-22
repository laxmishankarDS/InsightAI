import pandas as pd

# Read feature data
df = pd.read_csv(
    r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"
)

df.columns = df.columns.str.strip().str.lower()

print("========== EDA ==========")

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nDescription:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nUnique Values:")
print(df.nunique())

print("\nCorrelation:")
print(df.select_dtypes("number").corr())

print("\nProfit Analysis:")
print("Maximum:", df["profit"].max())
print("Minimum:", df["profit"].min())
print("Average:", df["profit"].mean())
print("Std:", df["profit"].std())

df.describe().to_csv("summary.csv")

print("\n========== EDA COMPLETED ==========")
print("Summary saved to: summary.csv")


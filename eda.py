import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use("ggplot")

df=pd.read_csv("c:/Users/maury/Downloads/InsightAI_Output/final_output.csv")
df.columns = df.columns.str.strip()
pd.set_option("display.max_column",None)
print(df.columns)
print(df.head())
print("\n")
print(df.tail())
print("\n")
print(df.shape)
print("\n")
print(df.columns)
print("\n")
print(df.describe())
print("\n")
print(df.isnull().sum())
print("\n")
print(df.duplicated().sum())
print("\n")
print(df.dtypes)
print("\n")
print(df.nunique())
print("\n")
numeric_df = df.select_dtypes(include=['number'])
print(numeric_df.corr())
print("\n")
numeric_df = df.select_dtypes(include=['number'])
print(numeric_df.corr())
print(df.columns.tolist())
max_profit = max(df['Profit'])
print(max_profit)

min_pro=df['Profit'].min();print(min_pro)
std_pro=df['Profit'].std();print(std_pro)
fre_pro=df['Profit'].value_counts();print(fre_pro)
avg_pro=df['Profit'].mean();print(avg_pro)
print(df[['Profit']])

print(df.columns.tolist())
print(df.head())


summary=df.describe()
summary.to_csv("summary.csv")
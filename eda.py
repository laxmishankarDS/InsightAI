import os
import pandas as pd
import numpy as np


# ============================================================
# INSIGHTAI - DYNAMIC EXPLORATORY DATA ANALYSIS
# ============================================================

print("========== EDA ==========")


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"
)

OUTPUT_FOLDER = (
    r"C:\Users\maury\Downloads\InsightAI_Output"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_FOLDER,
    "summary.csv"
)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError(
        "Dataset is empty."
    )


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace("/", "_", regex=False)
)

# Remove duplicate column names
df = df.loc[
    :,
    ~df.columns.duplicated()
]


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("\nColumns:")
print(
    df.columns.tolist()
)

print("\nHead:")
print(
    df.head()
)

print("\nShape:")
print(
    df.shape
)


# ============================================================
# DATA TYPES
# ============================================================

print("\n========== DATA TYPES ==========")

print(
    df.dtypes
)


# ============================================================
# NUMERIC / CATEGORICAL COLUMNS
# ============================================================

numeric_columns = (
    df.select_dtypes(
        include=np.number
    ).columns.tolist()
)

categorical_columns = (
    df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
)

datetime_columns = (
    df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()
)


print("\nNumeric Columns:")
print(
    numeric_columns
)

print("\nCategorical Columns:")
print(
    categorical_columns
)

print("\nDatetime Columns:")
print(
    datetime_columns
)


# ============================================================
# DESCRIPTION
# ============================================================

print("\n========== DESCRIPTION ==========")

if numeric_columns:

    print(
        df[numeric_columns].describe()
    )

else:

    print(
        "No numeric columns available."
    )


# ============================================================
# MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
)

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage
})

missing_report = (
    missing_report
    .sort_values(
        "missing_count",
        ascending=False
    )
)

print(
    missing_report
)


# ============================================================
# DUPLICATES
# ============================================================

print("\n========== DUPLICATES ==========")

duplicate_count = (
    df.duplicated().sum()
)

print(
    "Duplicate Rows:",
    duplicate_count
)


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n========== UNIQUE VALUES ==========")

unique_values = (
    df.nunique(dropna=False)
    .sort_values(
        ascending=False
    )
)

print(
    unique_values
)


# ============================================================
# CORRELATION
# ============================================================

print("\n========== CORRELATION ==========")

if len(numeric_columns) >= 2:

    correlation = (
        df[numeric_columns]
        .corr()
    )

    print(
        correlation
    )

else:

    correlation = pd.DataFrame()

    print(
        "Not enough numeric columns for correlation."
    )


# ============================================================
# NUMERIC COLUMN ANALYSIS
# ============================================================

print("\n========== NUMERIC ANALYSIS ==========")

if numeric_columns:

    numeric_analysis = pd.DataFrame({
        "column": numeric_columns,
        "min": [
            df[column].min()
            for column in numeric_columns
        ],
        "max": [
            df[column].max()
            for column in numeric_columns
        ],
        "mean": [
            df[column].mean()
            for column in numeric_columns
        ],
        "median": [
            df[column].median()
            for column in numeric_columns
        ],
        "std": [
            df[column].std()
            for column in numeric_columns
        ],
        "missing": [
            df[column].isnull().sum()
            for column in numeric_columns
        ]
    })

    print(
        numeric_analysis
    )

else:

    numeric_analysis = pd.DataFrame()

    print(
        "No numeric columns available."
    )


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

print("\n========== CATEGORICAL ANALYSIS ==========")

if categorical_columns:

    for column in categorical_columns:

        print(
            f"\nColumn: {column}"
        )

        print(
            "Unique:",
            df[column].nunique(
                dropna=False
            )
        )

        print(
            "Top Values:"
        )

        print(
            df[column]
            .value_counts(
                dropna=False
            )
            .head(10)
        )

else:

    print(
        "No categorical columns available."
    )


# ============================================================
# PROFIT ANALYSIS
# ============================================================
# This section runs only when a profit column exists.
# Therefore EDA remains dynamic.

print("\n========== PROFIT ANALYSIS ==========")

if "profit" in df.columns:

    profit = pd.to_numeric(
        df["profit"],
        errors="coerce"
    )

    print(
        "Maximum:",
        profit.max()
    )

    print(
        "Minimum:",
        profit.min()
    )

    print(
        "Average:",
        profit.mean()
    )

    print(
        "Median:",
        profit.median()
    )

    print(
        "Std:",
        profit.std()
    )

else:

    print(
        "Profit column not available."
    )

    print(
        "Profit-specific analysis skipped."
    )


# ============================================================
# TARGET-LIKE COLUMN INFORMATION
# ============================================================

print(
    "\n========== NUMERIC COLUMN SUMMARY =========="
)

if numeric_columns:

    for column in numeric_columns:

        print(
            f"{column}: "
            f"unique={df[column].nunique()}, "
            f"missing={df[column].isnull().sum()}"
        )

else:

    print(
        "No numeric columns available."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

summary = df.describe(
    include="all"
).transpose()

summary.to_csv(
    SUMMARY_FILE
)


# ============================================================
# SAVE MISSING VALUE REPORT
# ============================================================

missing_file = os.path.join(
    OUTPUT_FOLDER,
    "missing_values.csv"
)

missing_report.to_csv(
    missing_file
)


# ============================================================
# SAVE CORRELATION
# ============================================================

if not correlation.empty:

    correlation_file = os.path.join(
        OUTPUT_FOLDER,
        "correlation.csv"
    )

    correlation.to_csv(
        correlation_file
    )


# ============================================================
# SAVE NUMERIC ANALYSIS
# ============================================================

if not numeric_analysis.empty:

    numeric_file = os.path.join(
        OUTPUT_FOLDER,
        "numeric_analysis.csv"
    )

    numeric_analysis.to_csv(
        numeric_file,
        index=False
    )


# ============================================================
# COMPLETION
# ============================================================

print(
    "\n========== EDA COMPLETED =========="
)

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)

print(
    "Summary saved to:",
    SUMMARY_FILE
)

print(
    "Missing values saved to:",
    missing_file
)

if not correlation.empty:

    print(
        "Correlation saved to:",
        correlation_file
    )

if not numeric_analysis.empty:

    print(
        "Numeric analysis saved to:",
        numeric_file
    )


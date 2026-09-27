import os
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# INSIGHTAI - DYNAMIC DATA VALIDATION
# ============================================================

print("========== VALIDATION ==========")


# ============================================================
# PATH
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\maury\Downloads\InsightAI_Output\final_output.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError(
        "Validation failed: dataset is empty."
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


print("\nFile:")
print(INPUT_FILE)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# COLUMN INFORMATION
# ============================================================

print("\n========== COLUMN CHECK ==========")

print(
    "Columns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
)

missing_report = pd.DataFrame({
    "missing_count": missing_values,
    "missing_percentage": missing_percentage
})

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
# DATA TYPES
# ============================================================

print("\n========== DATA TYPES ==========")

print(
    df.dtypes
)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = (
    df.select_dtypes(
        include=np.number
    ).columns.tolist()
)

print("\nNumeric Columns:")
print(
    numeric_columns
)


# ============================================================
# NUMERIC VALIDATION
# ============================================================

print("\n========== NUMERIC VALIDATION ==========")

numeric_validation = []

for column in numeric_columns:

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    numeric_validation.append({
        "column": column,
        "missing": int(series.isna().sum()),
        "negative": int((series < 0).sum()),
        "zero": int((series == 0).sum()),
        "infinite": int(
            np.isinf(series).sum()
        ),
        "minimum": series.min(),
        "maximum": series.max()
    })


numeric_validation_df = pd.DataFrame(
    numeric_validation
)

if not numeric_validation_df.empty:
    print(
        numeric_validation_df
    )
else:
    print(
        "No numeric columns found."
    )


# ============================================================
# DATE DETECTION AND VALIDATION
# ============================================================

print("\n========== DATE VALIDATION ==========")

date_columns = []

for column in df.columns:

    column_name = str(column).lower()

    date_keywords = [
        "date",
        "datetime",
        "timestamp",
        "time_stamp"
    ]

    name_looks_like_date = any(
        keyword in column_name
        for keyword in date_keywords
    )

    if not name_looks_like_date:
        continue

    converted = pd.to_datetime(
        df[column],
        format="mixed",
        errors="coerce"
    )

    non_empty = df[column].notna().sum()

    if non_empty == 0:
        continue

    valid_ratio = (
        converted.notna().sum()
        / non_empty
    )

    if valid_ratio >= 0.70:

        date_columns.append(column)

        invalid_dates = (
            df[column].notna()
            & converted.isna()
        ).sum()

        print(
            f"{column}: "
            f"Invalid Dates = {invalid_dates}"
        )


if not date_columns:

    print(
        "No date columns detected."
    )


# ============================================================
# SALES-SPECIFIC VALIDATION
# ============================================================
# These checks run only when the relevant columns exist.
# Therefore unrelated datasets remain supported.

print("\n========== BUSINESS VALIDATION ==========")


# ------------------------------------------------------------
# QUANTITY
# ------------------------------------------------------------

if "quantity" in df.columns:

    quantity = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    print(
        "Negative Quantity:",
        int((quantity < 0).sum())
    )


# ------------------------------------------------------------
# UNIT PRICE
# ------------------------------------------------------------

if "unit_price" in df.columns:

    unit_price = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    )

    print(
        "Negative Unit Price:",
        int((unit_price < 0).sum())
    )


# ------------------------------------------------------------
# REVENUE
# ------------------------------------------------------------

if "revenue" in df.columns:

    revenue = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    print(
        "Negative Revenue:",
        int((revenue < 0).sum())
    )


# ------------------------------------------------------------
# PROFIT
# ------------------------------------------------------------

if "profit" in df.columns:

    profit = pd.to_numeric(
        df["profit"],
        errors="coerce"
    )

    print(
        "Negative Profit:",
        int((profit < 0).sum())
    )


# ============================================================
# REVENUE CONSISTENCY
# ============================================================

print("\n========== REVENUE CONSISTENCY ==========")

if (
    "quantity" in df.columns
    and "unit_price" in df.columns
    and "revenue" in df.columns
):

    quantity = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    unit_price = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    )

    revenue = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    expected_revenue = (
        quantity * unit_price
    )

    valid_rows = (
        expected_revenue.notna()
        & revenue.notna()
    )

    wrong_revenue = (
        (
            revenue[valid_rows]
            - expected_revenue[valid_rows]
        ).abs() > 0.01
    ).sum()

    print(
        "Incorrect Revenue:",
        int(wrong_revenue)
    )

else:

    print(
        "Revenue consistency check skipped."
    )


# ============================================================
# PROFIT MARGIN CONSISTENCY
# ============================================================

print("\n========== PROFIT MARGIN CONSISTENCY ==========")

if (
    "profit" in df.columns
    and "revenue" in df.columns
):

    profit = pd.to_numeric(
        df["profit"],
        errors="coerce"
    )

    revenue = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    expected_margin = np.where(
        revenue.notna()
        & revenue.ne(0),
        (
            profit
            / revenue
        ) * 100,
        np.nan
    )

    expected_margin = pd.Series(
        expected_margin,
        index=df.index
    )

    if "profit_margin" in df.columns:

        actual_margin = pd.to_numeric(
            df["profit_margin"],
            errors="coerce"
        )

        valid_rows = (
            expected_margin.notna()
            & actual_margin.notna()
        )

        wrong_margin = (
            (
                actual_margin[valid_rows]
                - expected_margin[valid_rows]
            ).abs() > 0.01
        ).sum()

        print(
            "Incorrect Profit Margin:",
            int(wrong_margin)
        )

    else:

        print(
            "profit_margin column not present."
        )

else:

    print(
        "Profit margin consistency check skipped."
    )


# ============================================================
# INFINITE VALUES
# ============================================================

print("\n========== INFINITE VALUES ==========")

infinite_report = {}

for column in numeric_columns:

    infinite_count = int(
        np.isinf(
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
        ).sum()
    )

    infinite_report[column] = infinite_count


print(
    infinite_report
)


# ============================================================
# SAVE VALIDATION REPORT
# ============================================================

OUTPUT_FOLDER = INPUT_FILE.parent

validation_file = (
    OUTPUT_FOLDER
    / "validation_report.csv"
)


validation_summary = {
    "rows": df.shape[0],
    "columns": df.shape[1],
    "duplicate_rows": duplicate_count,
    "total_missing_values": int(
        df.isnull().sum().sum()
    ),
    "numeric_columns": len(
        numeric_columns
    ),
    "date_columns": len(
        date_columns
    )
}


validation_report = pd.DataFrame(
    [validation_summary]
)

validation_report.to_csv(
    validation_file,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print(
    "\n========== VALIDATION COMPLETED =========="
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
    "Validation Report:",
    validation_file
)


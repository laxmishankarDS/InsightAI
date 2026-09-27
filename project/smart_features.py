import os
import pandas as pd
import numpy as np


# ============================================================
# INSIGHTAI - DYNAMIC FEATURE ENGINEERING
# ============================================================

print("========== FEATURE ENGINEERING ==========")


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"C:\Users\maury\Downloads\InsightAI_Output\final_output.csv"
)

OUTPUT_FILE = (
    r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"
)


# ============================================================
# READ DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("Input dataset is empty.")


print("\nInput File:")
print(INPUT_FILE)

print("\nOriginal Shape:")
print(df.shape)


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


# Remove duplicate column names if any
df = df.loc[:, ~df.columns.duplicated()]


print("\nCleaned Columns:")
print(df.columns.tolist())


# ============================================================
# TRACK NEW FEATURES
# ============================================================

new_features = []


# ============================================================
# DATE DETECTION
# ============================================================

print("\n========== DATE DETECTION ==========")

date_columns = []

for column in df.columns:

    column_name = str(column).lower()

    # Strong date/datetime column names
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

    # Existing datetime dtype
    already_datetime = pd.api.types.is_datetime64_any_dtype(
        df[column]
    )

    if already_datetime or name_looks_like_date:

        converted = pd.to_datetime(
            df[column],
            format="mixed",
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        # Only accept column as date when
        # at least 70% of non-null values convert successfully
        if valid_ratio >= 0.70:

            df[column] = converted

            date_columns.append(column)


if date_columns:
    print("Date Columns Detected:")
    print(date_columns)
else:
    print("No date columns detected.")


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

for column in date_columns:

    prefix = str(column).lower()

    # Avoid creating duplicate features
    year_feature = f"{prefix}_year"
    month_feature = f"{prefix}_month"
    quarter_feature = f"{prefix}_quarter"
    day_feature = f"{prefix}_day_of_week"

    if year_feature not in df.columns:
        df[year_feature] = df[column].dt.year
        new_features.append(year_feature)

    if month_feature not in df.columns:
        df[month_feature] = df[column].dt.month
        new_features.append(month_feature)

    if quarter_feature not in df.columns:
        df[quarter_feature] = df[column].dt.quarter
        new_features.append(quarter_feature)

    if day_feature not in df.columns:
        df[day_feature] = df[column].dt.dayofweek
        new_features.append(day_feature)


# ============================================================
# NUMERIC DETECTION
# ============================================================

print("\n========== NUMERIC DETECTION ==========")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("Numeric Columns:")
print(numeric_columns)


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

for column in df.columns:

    # Skip date columns
    if column in date_columns:
        continue

    # Skip already numeric columns
    if pd.api.types.is_numeric_dtype(df[column]):
        continue

    converted = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    valid_ratio = converted.notna().mean()

    # Convert only when strongly numeric
    if valid_ratio >= 0.95:
        df[column] = converted


# Refresh numeric columns
numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# ============================================================
# PROFIT MARGIN
# ============================================================

print("\n========== PROFIT MARGIN ==========")

if (
    "profit_margin" in df.columns
):

    # transform.py may have already created it.
    # Do not unnecessarily recreate it.

    df["profit_margin"] = pd.to_numeric(
        df["profit_margin"],
        errors="coerce"
    )

    df["profit_margin"] = (
        df["profit_margin"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    print(
        "profit_margin already exists."
    )

elif (
    "profit" in df.columns
    and "revenue" in df.columns
):

    df["profit"] = pd.to_numeric(
        df["profit"],
        errors="coerce"
    )

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    df["profit_margin"] = np.where(
        df["revenue"].notna()
        & df["revenue"].ne(0),
        (
            df["profit"]
            / df["revenue"]
        ) * 100,
        np.nan
    )

    df["profit_margin"] = (
        df["profit_margin"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    new_features.append(
        "profit_margin"
    )

    print(
        "profit_margin created."
    )

else:

    print(
        "profit_margin not applicable for this dataset."
    )


# ============================================================
# REVENUE PER UNIT
# ============================================================

print("\n========== REVENUE PER UNIT ==========")

if "revenue_per_unit" in df.columns:

    df["revenue_per_unit"] = pd.to_numeric(
        df["revenue_per_unit"],
        errors="coerce"
    )

    df["revenue_per_unit"] = (
        df["revenue_per_unit"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    print(
        "revenue_per_unit already exists."
    )

elif (
    "revenue" in df.columns
    and "quantity" in df.columns
):

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df["revenue_per_unit"] = np.where(
        df["quantity"].notna()
        & df["quantity"].ne(0),
        (
            df["revenue"]
            / df["quantity"]
        ),
        np.nan
    )

    df["revenue_per_unit"] = (
        df["revenue_per_unit"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    new_features.append(
        "revenue_per_unit"
    )

    print(
        "revenue_per_unit created."
    )

else:

    print(
        "revenue_per_unit not applicable for this dataset."
    )


# ============================================================
# REMOVE DUPLICATE NEW FEATURE NAMES
# ============================================================

new_features = list(
    dict.fromkeys(new_features)
)


# ============================================================
# FEATURE VERIFICATION
# ============================================================

print("\n========== FEATURE VERIFICATION ==========")

print(
    "New Features Created:"
)

if new_features:
    for feature in new_features:
        print(
            f"  + {feature}"
        )
else:
    print(
        "  No new features were required."
    )


print(
    "\nprofit_margin present:",
    "profit_margin" in df.columns
)

print(
    "revenue_per_unit present:",
    "revenue_per_unit" in df.columns
)

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)


# ============================================================
# REPLACE INFINITE VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SAMPLE
# ============================================================

print("\n========== SAMPLE ==========")

print(
    df.head()
)


# ============================================================
# SAVE FEATURE DATA
# ============================================================

output_folder = os.path.dirname(
    OUTPUT_FILE
)

os.makedirs(
    output_folder,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

if not os.path.exists(OUTPUT_FILE):
    raise RuntimeError(
        "feature_data.csv was not created."
    )


# ============================================================
# COMPLETION
# ============================================================

print(
    "\n========== FEATURE ENGINEERING COMPLETED =========="
)

print(
    "Saved to:",
    OUTPUT_FILE
)

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)


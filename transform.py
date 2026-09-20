import pandas as pd


def transform_data(df):

    print("\n========== TRANSFORM STAGE ==========")

    # Remove duplicate rows
    duplicate_rows = df.duplicated().sum()

    df.drop_duplicates(inplace=True)

    print(f"Duplicates Removed : {duplicate_rows}")

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:

        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

        df[col] = df[col].astype(str).str.strip().str.title()

    # Convert date column if present
    date_columns = [
        "Order Date",
        "Date",
        "Sales Date",
        "order_date"
    ]

    for col in date_columns:

        if col in df.columns:

            df[col] = pd.to_datetime(df[col], errors="coerce")

            df["Year"] = df[col].dt.year
            df["Month"] = df[col].dt.month
            df["Month_Name"] = df[col].dt.month_name()
            df["Quarter"] = df[col].dt.quarter

            break

    # Remove negative sales
    if "Sales" in df.columns:
        df = df[df["Sales"] >= 0]

    # Remove negative quantity
    if "Quantity" in df.columns:
        df = df[df["Quantity"] >= 0]

    print("Transformation Completed")

    print("Final Shape :", df.shape)

    return df
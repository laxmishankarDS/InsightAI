import pandas as pd


def transform_data(df):
    print("\n========== TRANSFORM STAGE ==========")

    if df is None or df.empty:
        raise ValueError("No data available for transformation.")

    df = df.copy()

    # Remove empty and duplicate rows
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Standardize common column names
    rename = {
        "date": "order_date",
        "orderdate": "order_date",
        "qty": "quantity",
        "units": "quantity",
        "sales": "revenue",
        "total_sales": "revenue",
        "net_profit": "profit"
    }

    df = df.rename(columns=rename)

    print("Transformation Successful")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    return df



import pandas as pd


def transform_data(df):
    """
    Clean and transform the extracted DataFrame.
    """

    print("\n========== TRANSFORM STAGE ==========")

    if df is None or df.empty:
        raise ValueError("No data available for transformation.")

    # Make a copy so the original DataFrame is not modified
    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("Transformation Successful")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    return df


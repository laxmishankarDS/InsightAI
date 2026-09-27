
# ============================================================
# INSIGHTAI
# DYNAMIC DATA TRANSFORMATION
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """
    Standardize DataFrame column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


# ============================================================
# STANDARDIZE COMMON COLUMN NAMES
# ============================================================

def standardize_column_names(df):
    """
    Convert common alternative column names into
    standardized names used by InsightAI.
    """

    df = df.copy()

    rename_map = {

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        "orderid": "order_id",
        "order_number": "order_id",
        "order_no": "order_id",

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        "date": "order_date",
        "orderdate": "order_date",
        "order_dt": "order_date",
        "transaction_date": "order_date",
        "transactiondate": "order_date",
        "purchase_date": "order_date",

        # ----------------------------------------------------
        # QUANTITY
        # ----------------------------------------------------

        "qty": "quantity",
        "units": "quantity",
        "unit": "quantity",
        "number_of_units": "quantity",
        "units_sold": "quantity",

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        "price": "unit_price",
        "unitprice": "unit_price",
        "selling_price": "unit_price",
        "unit_cost": "unit_price",

        # ----------------------------------------------------
        # REVENUE / SALES
        # ----------------------------------------------------

        "sales": "revenue",
        "total_sales": "revenue",
        "total_sale": "revenue",
        "sales_amount": "revenue",
        "sales_value": "revenue",
        "total_revenue": "revenue",
        "revenue_amount": "revenue",

        # ----------------------------------------------------
        # PROFIT
        # ----------------------------------------------------

        "net_profit": "profit",
        "gross_profit": "profit",
        "profit_amount": "profit",
        "profit_value": "profit",

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        "total_cost": "cost",
        "cost_amount": "cost",
    }

    rename_operations = {}

    for old_name, new_name in rename_map.items():

        if (
            old_name in df.columns
            and new_name not in df.columns
        ):

            rename_operations[old_name] = new_name

    if rename_operations:

        df = df.rename(
            columns=rename_operations
        )

    return df


# ============================================================
# DATE DETECTION AND CONVERSION
# ============================================================

def detect_and_convert_dates(df):
    """
    Detect likely date columns and safely convert them.
    """

    df = df.copy()

    for column in df.columns:

        column_name = str(column).lower()

        is_date_column = (
            "date" in column_name
            or "datetime" in column_name
            or column_name == "timestamp"
            or column_name.endswith("_dt")
        )

        if not is_date_column:
            continue

        try:

            converted = pd.to_datetime(
                df[column],
                format="mixed",
                errors="coerce"
            )

            original_non_null = (
                df[column].notna().sum()
            )

            converted_non_null = (
                converted.notna().sum()
            )

            if original_non_null == 0:
                continue

            conversion_ratio = (
                converted_non_null
                / original_non_null
            )

            # Convert only when most values are valid dates.
            if conversion_ratio >= 0.70:

                df[column] = converted

        except Exception:

            continue

    return df


# ============================================================
# NUMERIC COLUMN DETECTION
# ============================================================

def convert_numeric_columns(df):
    """
    Safely convert columns that strongly appear numeric.

    Text and categorical columns remain unchanged.
    """

    df = df.copy()

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        column_name = str(column).lower()

        # Do not convert date/time columns here.
        if (
            "date" in column_name
            or "time" in column_name
        ):
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        original_non_null = (
            df[column].notna().sum()
        )

        converted_non_null = (
            converted.notna().sum()
        )

        if original_non_null == 0:
            continue

        conversion_ratio = (
            converted_non_null
            / original_non_null
        )

        # Convert only when at least 95% of
        # non-null values are numeric.
        if conversion_ratio >= 0.95:

            df[column] = converted

    return df


# ============================================================
# CREATE PROFIT MARGIN
# ============================================================

def create_profit_margin(df):
    """
    Create profit_margin only when both revenue and profit
    columns are available.

    Formula:

        Profit Margin = (Profit / Revenue) * 100
    """

    df = df.copy()

    if (
        "revenue" not in df.columns
        or "profit" not in df.columns
    ):

        print(
            "Profit Margin: Not created "
            "(revenue and/or profit missing)."
        )

        return df

    # --------------------------------------------------------
    # Convert required columns to numeric
    # --------------------------------------------------------

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    df["profit"] = pd.to_numeric(
        df["profit"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Calculate profit margin
    # --------------------------------------------------------

    df["profit_margin"] = np.where(
        (
            df["revenue"].notna()
            & (df["revenue"] != 0)
        ),

        (
            df["profit"]
            / df["revenue"]
        ) * 100,

        np.nan
    )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    df["profit_margin"] = (
        df["profit_margin"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    print(
        "Profit Margin: Created successfully."
    )

    return df


# ============================================================
# MAIN TRANSFORMATION FUNCTION
# ============================================================

def transform_data(df):

    print(
        "\n========== TRANSFORM STAGE =========="
    )

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if df is None:

        raise ValueError(
            "No data available for transformation."
        )

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "Cannot transform an empty DataFrame."
        )

    # --------------------------------------------------------
    # COPY DATA
    # --------------------------------------------------------

    df = df.copy()

    original_rows = len(df)
    original_columns = len(df.columns)

    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY ROWS
    # --------------------------------------------------------

    df = df.dropna(
        how="all"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE ROWS
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        df = df.drop_duplicates()

    # --------------------------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------------------------

    df = clean_column_names(
        df
    )

    # --------------------------------------------------------
    # STANDARDIZE COMMON COLUMN NAMES
    # --------------------------------------------------------

    df = standardize_column_names(
        df
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE COLUMN NAMES
    # --------------------------------------------------------

    if df.columns.duplicated().any():

        print(
            "WARNING: Duplicate column names detected."
        )

        df = df.loc[
            :,
            ~df.columns.duplicated()
        ]

    # --------------------------------------------------------
    # DATE CONVERSION
    # --------------------------------------------------------

    df = detect_and_convert_dates(
        df
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    df = convert_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # CREATE PROFIT MARGIN
    # --------------------------------------------------------

    df = create_profit_margin(
        df
    )

    # --------------------------------------------------------
    # REMOVE INFINITE VALUES
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # TRANSFORMATION SUMMARY
    # ========================================================

    print(
        "\n========== TRANSFORMATION SUMMARY =========="
    )

    print(
        "Original rows:",
        original_rows
    )

    print(
        "Final rows:",
        len(df)
    )

    print(
        "Removed rows:",
        original_rows - len(df)
    )

    print(
        "Original columns:",
        original_columns
    )

    print(
        "Final columns:",
        len(df.columns)
    )

    print(
        "Removed duplicates:",
        duplicate_count
    )

    # --------------------------------------------------------
    # FINAL COLUMNS
    # --------------------------------------------------------

    print(
        "\nFinal columns:"
    )

    for column in df.columns:

        print(
            " -",
            column
        )

    # --------------------------------------------------------
    # PROFIT MARGIN SUMMARY
    # --------------------------------------------------------

    if "profit_margin" in df.columns:

        valid_margin = (
            df["profit_margin"]
            .dropna()
        )

        if not valid_margin.empty:

            print(
                "\nProfit Margin:"
            )

            print(
                "Available: Yes"
            )

            print(
                "Min:",
                round(
                    valid_margin.min(),
                    4
                )
            )

            print(
                "Max:",
                round(
                    valid_margin.max(),
                    4
                )
            )

            print(
                "Mean:",
                round(
                    valid_margin.mean(),
                    4
                )
            )

        else:

            print(
                "\nProfit Margin column exists "
                "but contains no valid values."
            )

    else:

        print(
            "\nProfit Margin: Not available "
            "for this dataset."
        )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print(
        "\nTransformation Successful"
    )

    return df


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    from extract import extract_data

    INPUT_FOLDER = (
        r"C:\Users\maury\Downloads\archive"
    )

    try:

        # ----------------------------------------------------
        # EXTRACT
        # ----------------------------------------------------

        raw_data = extract_data(
            INPUT_FOLDER
        )

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        transformed_data = transform_data(
            raw_data
        )

        # ----------------------------------------------------
        # TEST RESULT
        # ----------------------------------------------------

        print(
            "\n========== TRANSFORM TEST RESULT =========="
        )

        print(
            "Rows:",
            transformed_data.shape[0]
        )

        print(
            "Columns:",
            transformed_data.shape[1]
        )

        print(
            "\nData types:"
        )

        print(
            transformed_data.dtypes
        )

        print(
            "\nFirst 5 rows:"
        )

        print(
            transformed_data.head()
        )

    except Exception as error:

        print(
            "\n========== TRANSFORMATION ERROR =========="
        )

        print(
            type(error).__name__,
            ":",
            error
        )


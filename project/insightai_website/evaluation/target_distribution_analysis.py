
# ============================================================
# INSIGHTAI
# TARGET DISTRIBUTION ANALYSIS
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\maury\OneDrive\Documents\insightAI"

DATA_PATH = r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "insightai_website",
    "evaluation"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "target_distribution_analysis.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "target_distribution_report.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "profit_margin"

CATEGORY_COLUMN = "category"


# ============================================================
# HELPER
# ============================================================

def safe_float(value):
    """
    Convert numpy values safely to Python float.
    """

    if pd.isna(value):
        return None

    return float(value)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INSIGHTAI - TARGET DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/6] Loading dataset...")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # VALIDATE COLUMNS
    # --------------------------------------------------------

    print("\n[2/6] Validating columns...")

    required_columns = [
        TARGET_COLUMN,
        CATEGORY_COLUMN
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(
                missing_columns
            )
        )

    print("Required columns: PASS")

    # --------------------------------------------------------
    # CLEAN TARGET
    # --------------------------------------------------------

    print("\n[3/6] Preparing target data...")

    analysis_df = df[
        [
            TARGET_COLUMN,
            CATEGORY_COLUMN
        ]
    ].copy()

    analysis_df[TARGET_COLUMN] = pd.to_numeric(
        analysis_df[TARGET_COLUMN],
        errors="coerce"
    )

    before_rows = len(
        analysis_df
    )

    analysis_df = analysis_df.dropna(
        subset=[
            TARGET_COLUMN,
            CATEGORY_COLUMN
        ]
    )

    after_rows = len(
        analysis_df
    )

    print(
        f"Rows before cleaning: {before_rows:,}"
    )

    print(
        f"Rows after cleaning: {after_rows:,}"
    )

    # --------------------------------------------------------
    # OVERALL TARGET STATISTICS
    # --------------------------------------------------------

    print("\n[4/6] Calculating overall target statistics...")

    target = analysis_df[
        TARGET_COLUMN
    ]

    overall_statistics = {

        "count": int(
            target.count()
        ),

        "mean": safe_float(
            target.mean()
        ),

        "median": safe_float(
            target.median()
        ),

        "std": safe_float(
            target.std()
        ),

        "variance": safe_float(
            target.var()
        ),

        "min": safe_float(
            target.min()
        ),

        "q1": safe_float(
            target.quantile(0.25)
        ),

        "q3": safe_float(
            target.quantile(0.75)
        ),

        "max": safe_float(
            target.max()
        ),

        "iqr": safe_float(
            target.quantile(0.75)
            - target.quantile(0.25)
        ),

        "p01": safe_float(
            target.quantile(0.01)
        ),

        "p05": safe_float(
            target.quantile(0.05)
        ),

        "p10": safe_float(
            target.quantile(0.10)
        ),

        "p90": safe_float(
            target.quantile(0.90)
        ),

        "p95": safe_float(
            target.quantile(0.95)
        ),

        "p99": safe_float(
            target.quantile(0.99)
        )
    }

    # --------------------------------------------------------
    # CATEGORY-WISE DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\n[5/6] Calculating category-wise distributions..."
    )

    results = []

    for category, group in analysis_df.groupby(
        CATEGORY_COLUMN,
        dropna=False
    ):

        values = group[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        category_name = (
            "Unknown"
            if pd.isna(category)
            else str(category)
        )

        q1 = np.percentile(
            values,
            25
        )

        q3 = np.percentile(
            values,
            75
        )

        iqr = q3 - q1

        mean_value = np.mean(
            values
        )

        median_value = np.median(
            values
        )

        std_value = np.std(
            values,
            ddof=1
        ) if len(values) > 1 else 0.0

        # ----------------------------------------------------
        # Distance between mean and median
        # ----------------------------------------------------

        mean_median_difference = (
            mean_value - median_value
        )

        # ----------------------------------------------------
        # Coefficient of variation
        # ----------------------------------------------------

        if np.isclose(mean_value, 0):

            coefficient_of_variation = None

        else:

            coefficient_of_variation = (
                std_value / abs(mean_value)
            )

        result = {

            "category": category_name,

            "sample_count": int(
                len(values)
            ),

            "mean": safe_float(
                mean_value
            ),

            "median": safe_float(
                median_value
            ),

            "std": safe_float(
                std_value
            ),

            "variance": safe_float(
                np.var(
                    values,
                    ddof=1
                ) if len(values) > 1 else 0.0
            ),

            "min": safe_float(
                np.min(values)
            ),

            "q1": safe_float(
                q1
            ),

            "q3": safe_float(
                q3
            ),

            "iqr": safe_float(
                iqr
            ),

            "max": safe_float(
                np.max(values)
            ),

            "range": safe_float(
                np.max(values)
                - np.min(values)
            ),

            "p01": safe_float(
                np.percentile(
                    values,
                    1
                )
            ),

            "p05": safe_float(
                np.percentile(
                    values,
                    5
                )
            ),

            "p10": safe_float(
                np.percentile(
                    values,
                    10
                )
            ),

            "p90": safe_float(
                np.percentile(
                    values,
                    90
                )
            ),

            "p95": safe_float(
                np.percentile(
                    values,
                    95
                )
            ),

            "p99": safe_float(
                np.percentile(
                    values,
                    99
                )
            ),

            "mean_median_difference":
                safe_float(
                    mean_median_difference
                ),

            "coefficient_of_variation":
                (
                    None
                    if coefficient_of_variation
                    is None
                    else float(
                        coefficient_of_variation
                    )
                )
        }

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="category"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # ADD TARGET SPREAD RELATIVE TO OVERALL
    # --------------------------------------------------------

    overall_mean = overall_statistics[
        "mean"
    ]

    overall_std = overall_statistics[
        "std"
    ]

    results_df[
        "difference_from_overall_mean"
    ] = (
        results_df["mean"]
        - overall_mean
    )

    if overall_std and not np.isclose(
        overall_std,
        0
    ):

        results_df[
            "mean_difference_in_overall_std"
        ] = (
            results_df["difference_from_overall_mean"]
            / overall_std
        )

    else:

        results_df[
            "mean_difference_in_overall_std"
        ] = np.nan

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    print(
        "\n[6/6] Saving reports..."
    )

    results_df.to_csv(
        CSV_PATH,
        index=False
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    category_means = {
        str(row["category"]):
            float(row["mean"])
        for _, row in results_df.iterrows()
    }

    category_stds = {
        str(row["category"]):
            float(row["std"])
        for _, row in results_df.iterrows()
    }

    category_ranges = {
        str(row["category"]):
            float(row["range"])
        for _, row in results_df.iterrows()
    }

    highest_mean_category = None
    lowest_mean_category = None
    highest_std_category = None
    lowest_std_category = None

    if len(results_df) > 0:

        highest_mean_category = str(
            results_df.loc[
                results_df["mean"].idxmax(),
                "category"
            ]
        )

        lowest_mean_category = str(
            results_df.loc[
                results_df["mean"].idxmin(),
                "category"
            ]
        )

        highest_std_category = str(
            results_df.loc[
                results_df["std"].idxmax(),
                "category"
            ]
        )

        lowest_std_category = str(
            results_df.loc[
                results_df["std"].idxmin(),
                "category"
            ]
        )

    report = {

        "analysis":
            "target_distribution_analysis",

        "target":
            TARGET_COLUMN,

        "category_column":
            CATEGORY_COLUMN,

        "dataset_rows":
            int(len(df)),

        "analysis_rows":
            int(len(analysis_df)),

        "number_of_categories":
            int(len(results_df)),

        "overall_target_statistics":
            overall_statistics,

        "category_means":
            category_means,

        "category_standard_deviations":
            category_stds,

        "category_ranges":
            category_ranges,

        "highest_mean_category":
            highest_mean_category,

        "lowest_mean_category":
            lowest_mean_category,

        "highest_std_category":
            highest_std_category,

        "lowest_std_category":
            lowest_std_category,

        "category_results":
            results
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # PRINT OVERALL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL TARGET DISTRIBUTION")
    print("=" * 70)

    for key, value in overall_statistics.items():

        if value is None:

            print(
                f"{key}: None"
            )

        else:

            print(
                f"{key}: {value:.6f}"
            )

    # --------------------------------------------------------
    # PRINT CATEGORY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CATEGORY-WISE TARGET DISTRIBUTION")
    print("=" * 70)

    display_columns = [

        "category",

        "sample_count",

        "mean",

        "median",

        "std",

        "min",

        "q1",

        "q3",

        "max",

        "iqr",

        "range"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # INTERPRETATION SUPPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DISTRIBUTION SUMMARY")
    print("=" * 70)

    print(
        f"Number of categories: "
        f"{len(results_df)}"
    )

    if highest_mean_category is not None:

        print(
            f"Highest target mean category: "
            f"{highest_mean_category}"
        )

    if lowest_mean_category is not None:

        print(
            f"Lowest target mean category: "
            f"{lowest_mean_category}"
        )

    if highest_std_category is not None:

        print(
            f"Highest target standard deviation: "
            f"{highest_std_category}"
        )

    if lowest_std_category is not None:

        print(
            f"Lowest target standard deviation: "
            f"{lowest_std_category}"
        )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"CSV report:\n{CSV_PATH}"
    )

    print(
        f"JSON report:\n{REPORT_PATH}"
    )

    print(
        "\nTARGET DISTRIBUTION ANALYSIS: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


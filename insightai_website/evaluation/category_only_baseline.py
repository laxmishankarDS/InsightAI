
# ============================================================
# INSIGHTAI
# CATEGORY-ONLY BASELINE ANALYSIS
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

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
    "category_only_baseline.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "category_only_baseline_report.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "profit_margin"

CATEGORY_COLUMN = "category"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INSIGHTAI - CATEGORY-ONLY BASELINE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATA
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
    # PREPARE DATA
    # --------------------------------------------------------

    print("\n[3/6] Preparing data...")

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

    analysis_df = analysis_df.dropna(
        subset=[
            TARGET_COLUMN,
            CATEGORY_COLUMN
        ]
    )

    X = analysis_df[
        [CATEGORY_COLUMN]
    ].copy()

    y = analysis_df[
        TARGET_COLUMN
    ].copy()

    print(
        f"Usable rows: {len(analysis_df):,}"
    )

    print(
        f"Categories: {X[CATEGORY_COLUMN].nunique():,}"
    )

    # --------------------------------------------------------
    # EXACT SAME TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\n[4/6] Creating exact train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Test rows: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # CATEGORY-ONLY PIPELINE
    # --------------------------------------------------------

    category_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),

            (
                "model",
                LinearRegression()
            )
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\n[5/6] Training category-only baseline..."
    )

    category_pipeline.fit(
        X_train,
        y_train
    )

    predictions = category_pipeline.predict(
        X_test
    )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    y_test_array = np.asarray(
        y_test,
        dtype=float
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    r2 = r2_score(
        y_test_array,
        predictions
    )

    mae = mean_absolute_error(
        y_test_array,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_array,
            predictions
        )
    )

    mean_error = np.mean(
        predictions - y_test_array
    )

    median_error = np.median(
        predictions - y_test_array
    )

    # --------------------------------------------------------
    # CATEGORY-WISE BASELINE
    # --------------------------------------------------------

    comparison_df = pd.DataFrame({

        "category":
            X_test[CATEGORY_COLUMN].values,

        "actual":
            y_test_array,

        "predicted":
            predictions,

        "error":
            predictions - y_test_array,

        "absolute_error":
            np.abs(
                predictions - y_test_array
            )
    })

    category_results = []

    for category, group in comparison_df.groupby(
        "category"
    ):

        actual = group["actual"].to_numpy(
            dtype=float
        )

        predicted = group["predicted"].to_numpy(
            dtype=float
        )

        category_results.append({

            "category": str(category),

            "sample_count": int(
                len(group)
            ),

            "actual_mean": float(
                np.mean(actual)
            ),

            "predicted_mean": float(
                np.mean(predicted)
            ),

            "mean_error": float(
                np.mean(
                    predicted - actual
                )
            ),

            "mae": float(
                mean_absolute_error(
                    actual,
                    predicted
                )
            ),

            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted
                    )
                )
            )
        })

    category_results_df = pd.DataFrame(
        category_results
    )

    category_results_df = (
        category_results_df
        .sort_values("category")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # SAVE CATEGORY RESULTS
    # --------------------------------------------------------

    category_results_df.to_csv(
        CSV_PATH,
        index=False
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "analysis":
            "category_only_baseline",

        "target":
            TARGET_COLUMN,

        "feature":
            CATEGORY_COLUMN,

        "model":
            "Linear Regression",

        "dataset_rows":
            int(len(df)),

        "usable_rows":
            int(len(analysis_df)),

        "training_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "number_of_categories":
            int(
                X[CATEGORY_COLUMN].nunique()
            ),

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "metrics": {

            "r2":
                float(r2),

            "mae":
                float(mae),

            "rmse":
                float(rmse),

            "mean_error":
                float(mean_error),

            "median_error":
                float(median_error)
        },

        "category_results":
            category_results
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
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n[6/6] Results")

    print("\n" + "=" * 70)
    print("CATEGORY-ONLY BASELINE METRICS")
    print("=" * 70)

    print(
        f"R²:          {r2:.6f}"
    )

    print(
        f"MAE:         {mae:.6f}"
    )

    print(
        f"RMSE:        {rmse:.6f}"
    )

    print(
        f"Mean Error:  {mean_error:.6f}"
    )

    print(
        f"Median Error:{median_error:.6f}"
    )

    print("\n" + "=" * 70)
    print("CATEGORY-WISE BASELINE")
    print("=" * 70)

    print(
        category_results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # COMPARISON WITH CURRENT MODEL
    # --------------------------------------------------------

    current_model_r2 = 0.578510
    current_model_mae = 5.614508
    current_model_rmse = 6.777454

    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT GRADIENT BOOSTING MODEL")
    print("=" * 70)

    print(
        f"{'Metric':<10}"
        f"{'Category Only':>20}"
        f"{'Gradient Boosting':>20}"
    )

    print("-" * 55)

    print(
        f"{'R²':<10}"
        f"{r2:>20.6f}"
        f"{current_model_r2:>20.6f}"
    )

    print(
        f"{'MAE':<10}"
        f"{mae:>20.6f}"
        f"{current_model_mae:>20.6f}"
    )

    print(
        f"{'RMSE':<10}"
        f"{rmse:>20.6f}"
        f"{current_model_rmse:>20.6f}"
    )

    # --------------------------------------------------------
    # ADDITIONAL INFORMATION
    # --------------------------------------------------------

    r2_improvement = (
        current_model_r2 - r2
    )

    mae_reduction = (
        mae - current_model_mae
    )

    rmse_reduction = (
        rmse - current_model_rmse
    )

    print("\n" + "=" * 70)
    print("ADDITIONAL FEATURE CONTRIBUTION")
    print("=" * 70)

    print(
        f"R² difference: "
        f"{r2_improvement:.6f}"
    )

    print(
        f"MAE difference: "
        f"{mae_reduction:.6f}"
    )

    print(
        f"RMSE difference: "
        f"{rmse_reduction:.6f}"
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
        "\nCATEGORY-ONLY BASELINE ANALYSIS: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


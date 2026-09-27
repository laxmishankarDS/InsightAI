
# ============================================================
# INSIGHTAI
# CATEGORY-WISE ERROR ANALYSIS
# ============================================================

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\maury\OneDrive\Documents\insightAI"

DATA_PATH = r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"

MODEL_PATH = r"C:\Users\maury\Downloads\InsightAI_Output\best_model.joblib"

METADATA_PATH = r"C:\Users\maury\Downloads\InsightAI_Output\model_metadata.json"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "insightai_website",
    "evaluation"
)

CATEGORY_ERROR_CSV = os.path.join(
    OUTPUT_DIR,
    "category_error_analysis.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "category_error_analysis_report.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

CATEGORY_COLUMN = "category"


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_r2(y_true, y_pred):
    """
    Calculate R2 safely.

    R2 is undefined when:
    - There are fewer than 2 samples
    - Actual target values have zero variance
    """

    if len(y_true) < 2:
        return np.nan

    if np.isclose(np.var(y_true), 0):
        return np.nan

    return r2_score(y_true, y_pred)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INSIGHTAI - CATEGORY-WISE ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/7] Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    print("\n[2/7] Loading model metadata...")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    target = metadata.get("target")

    selected_features = metadata.get(
        "selected_features",
        []
    )

    numeric_features = metadata.get(
        "numeric_features",
        []
    )

    categorical_features = metadata.get(
        "categorical_features",
        []
    )

    best_model_name = metadata.get(
        "best_model",
        "Unknown"
    )

    print(f"Target: {target}")
    print(f"Best model: {best_model_name}")

    # --------------------------------------------------------
    # METADATA VALIDATION
    # --------------------------------------------------------

    if not target:
        raise ValueError(
            "Target not found in model_metadata.json"
        )

    if not selected_features:
        raise ValueError(
            "selected_features not found in model_metadata.json"
        )

    # --------------------------------------------------------
    # VALIDATE DATASET COLUMNS
    # --------------------------------------------------------

    print("\n[3/7] Validating required columns...")

    required_columns = set(selected_features)

    required_columns.add(target)
    required_columns.add(CATEGORY_COLUMN)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns missing from dataset:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    print("Required columns: PASS")

    # --------------------------------------------------------
    # PREPARE FEATURES AND TARGET
    # --------------------------------------------------------

    print("\n[4/7] Preparing exact test split...")

    X = df[selected_features].copy()

    y = df[target].copy()

    # --------------------------------------------------------
    # EXACT SAME TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # GET CATEGORY VALUES USING ORIGINAL INDEX
    # --------------------------------------------------------

    category_test = df.loc[
        X_test.index,
        CATEGORY_COLUMN
    ].copy()

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Test rows: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n[5/7] Loading trained model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    print("\n[6/7] Generating predictions...")

    predictions = model.predict(X_test)

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    y_test_array = np.asarray(
        y_test,
        dtype=float
    )

    print(
        f"Predictions generated: {len(predictions):,}"
    )

    # --------------------------------------------------------
    # BUILD ANALYSIS DATAFRAME
    # --------------------------------------------------------

    print(
        "\n[7/7] Calculating category-wise metrics..."
    )

    analysis_df = pd.DataFrame({

        "actual": y_test_array,

        "predicted": predictions,

        "error": (
            predictions - y_test_array
        ),

        "absolute_error": np.abs(
            predictions - y_test_array
        ),

        "category": category_test.values
    })

    # --------------------------------------------------------
    # CATEGORY-WISE METRICS
    # --------------------------------------------------------

    results = []

    for category, group in analysis_df.groupby(
        CATEGORY_COLUMN,
        dropna=False
    ):

        actual = group["actual"].to_numpy(
            dtype=float
        )

        predicted = group["predicted"].to_numpy(
            dtype=float
        )

        errors = group["error"].to_numpy(
            dtype=float
        )

        absolute_errors = group[
            "absolute_error"
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        # ----------------------------------------------------
        # SAFE R2
        # ----------------------------------------------------

        category_r2 = safe_r2(
            actual,
            predicted
        )

        # ----------------------------------------------------
        # CATEGORY NAME
        # ----------------------------------------------------

        category_name = (
            "Unknown"
            if pd.isna(category)
            else str(category)
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        category_result = {

            "category": category_name,

            "sample_count": int(
                len(group)
            ),

            # Actual distribution
            "actual_mean": float(
                np.mean(actual)
            ),

            "actual_median": float(
                np.median(actual)
            ),

            "actual_std": float(
                np.std(actual)
            ),

            "actual_min": float(
                np.min(actual)
            ),

            "actual_max": float(
                np.max(actual)
            ),

            # Prediction distribution
            "predicted_mean": float(
                np.mean(predicted)
            ),

            "predicted_median": float(
                np.median(predicted)
            ),

            # Error metrics
            "mean_error": float(
                np.mean(errors)
            ),

            "median_error": float(
                np.median(errors)
            ),

            "mae": float(
                mean_absolute_error(
                    actual,
                    predicted
                )
            ),

            "rmse": float(
                rmse
            ),

            # R2
            "r2": (
                None
                if pd.isna(category_r2)
                else float(category_r2)
            ),

            # Error percentiles
            "p50_absolute_error": float(
                np.percentile(
                    absolute_errors,
                    50
                )
            ),

            "p90_absolute_error": float(
                np.percentile(
                    absolute_errors,
                    90
                )
            ),

            "p95_absolute_error": float(
                np.percentile(
                    absolute_errors,
                    95
                )
            ),

            "max_absolute_error": float(
                np.max(absolute_errors)
            )
        }

        results.append(
            category_result
        )

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="category"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    results_df.to_csv(
        CATEGORY_ERROR_CSV,
        index=False
    )

    # --------------------------------------------------------
    # CATEGORY SUMMARY
    # --------------------------------------------------------

    category_count = len(
        results_df
    )

    largest_category = None
    highest_mae_category = None
    highest_rmse_category = None

    if category_count > 0:

        largest_category = (
            results_df
            .sort_values(
                by="sample_count",
                ascending=False
            )
            .iloc[0]["category"]
        )

        highest_mae_category = (
            results_df
            .sort_values(
                by="mae",
                ascending=False
            )
            .iloc[0]["category"]
        )

        highest_rmse_category = (
            results_df
            .sort_values(
                by="rmse",
                ascending=False
            )
            .iloc[0]["category"]
        )

    # --------------------------------------------------------
    # CREATE JSON REPORT
    # --------------------------------------------------------

    report = {

        "analysis":
            "category_wise_error_analysis",

        "model":
            best_model_name,

        "target":
            target,

        "category_column":
            CATEGORY_COLUMN,

        "dataset_rows":
            int(len(df)),

        "training_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "number_of_categories":
            int(category_count),

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "selected_features":
            selected_features,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "largest_category_by_test_samples":
            (
                str(largest_category)
                if largest_category is not None
                else None
            ),

        "highest_mae_category":
            (
                str(highest_mae_category)
                if highest_mae_category is not None
                else None
            ),

        "highest_rmse_category":
            (
                str(highest_rmse_category)
                if highest_rmse_category is not None
                else None
            ),

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
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CATEGORY-WISE RESULTS")
    print("=" * 70)

    display_columns = [

        "category",

        "sample_count",

        "actual_mean",

        "predicted_mean",

        "mean_error",

        "mae",

        "rmse",

        "r2"
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
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Number of categories: "
        f"{category_count}"
    )

    if largest_category is not None:

        print(
            f"Largest category by test samples: "
            f"{largest_category}"
        )

    if highest_mae_category is not None:

        print(
            f"Highest MAE category: "
            f"{highest_mae_category}"
        )

    if highest_rmse_category is not None:

        print(
            f"Highest RMSE category: "
            f"{highest_rmse_category}"
        )

    # --------------------------------------------------------
    # FILES SAVED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"CSV report:\n"
        f"{CATEGORY_ERROR_CSV}"
    )

    print(
        f"JSON report:\n"
        f"{REPORT_PATH}"
    )

    print("\nCATEGORY ERROR ANALYSIS: PASS")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


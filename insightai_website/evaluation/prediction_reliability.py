
# ============================================================
# INSIGHTAI
# PREDICTION RELIABILITY / CALIBRATION ANALYSIS
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

TARGET_COLUMN = "profit_margin"

LOW_THRESHOLD = 20

HIGH_THRESHOLD = 35

CALIBRATION_BINS = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_r2(y_true, y_pred):
    """
    Safely calculate R².

    Returns NaN when the group is too small
    or when the actual target has no variance.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return np.nan

    if np.isclose(np.var(y_true), 0):
        return np.nan

    return r2_score(y_true, y_pred)


def get_target_range(value):
    """
    Assign actual target value to a target range.
    """

    if value < LOW_THRESHOLD:
        return "Low (<20)"

    if value < HIGH_THRESHOLD:
        return "Medium (20-35)"

    return "High (>=35)"


def calculate_rmse(errors):
    """
    Calculate RMSE directly from residuals.
    """

    errors = np.asarray(errors)

    return float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("INSIGHTAI - PREDICTION RELIABILITY ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading data...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )


# ============================================================
# 2. LOAD MODEL METADATA
# ============================================================

print("\n[2] Loading model metadata...")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_PATH}"
    )

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


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


print("Target:", TARGET_COLUMN)

print(
    "Selected features:",
    selected_features
)

print(
    "Numeric features:",
    numeric_features
)

print(
    "Categorical features:",
    categorical_features
)

print(
    "Best model:",
    best_model_name
)


# ============================================================
# 3. VALIDATE MODEL FEATURES
# ============================================================

print("\n[3] Validating model features...")

if not selected_features:
    raise ValueError(
        "No selected features found in model metadata."
    )


missing_features = [
    column
    for column in selected_features
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        "The following model features are missing "
        f"from dataset: {missing_features}"
    )

print("All model features are available.")


# ============================================================
# 4. PREPARE X AND y
# ============================================================

print("\n[4] Preparing X and y...")

X = df[
    selected_features
].copy()

y = df[
    TARGET_COLUMN
].copy()

if X.empty:
    raise ValueError(
        "Feature dataset is empty."
    )

if y.empty:
    raise ValueError(
        "Target dataset is empty."
    )


# ============================================================
# 5. RECREATE EXACT TRAIN / TEST SPLIT
# ============================================================

print(
    "\n[5] Recreating exact train/test split..."
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# 6. LOAD SAVED MODEL
# ============================================================

print("\n[6] Loading saved model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\n[7] Generating predictions...")

y_true = np.asarray(
    y_test,
    dtype=float
)

y_pred = np.asarray(
    model.predict(X_test),
    dtype=float
)

if len(y_true) != len(y_pred):
    raise ValueError(
        "Actual and predicted values have different lengths."
    )

errors = (
    y_pred -
    y_true
)

absolute_errors = np.abs(
    errors
)


# ============================================================
# 8. OVERALL METRICS
# ============================================================

print("\n[8] Calculating overall metrics...")

r2 = r2_score(
    y_true,
    y_pred
)

mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

mean_error = np.mean(
    errors
)

median_error = np.median(
    errors
)

mean_absolute_error_value = np.mean(
    absolute_errors
)

median_absolute_error = np.median(
    absolute_errors
)


# ============================================================
# 9. TARGET RANGE ANALYSIS
# ============================================================

print("\n[9] Target range analysis...")

min_target = float(
    np.min(y_true)
)

max_target = float(
    np.max(y_true)
)

range_labels = np.array([
    get_target_range(value)
    for value in y_true
])


range_results = []

target_ranges = [
    "Low (<20)",
    "Medium (20-35)",
    "High (>=35)"
]


for group in target_ranges:

    mask = (
        range_labels == group
    )

    group_true = y_true[mask]

    group_pred = y_pred[mask]

    group_errors = errors[mask]

    group_abs_errors = absolute_errors[mask]

    if len(group_true) == 0:
        continue

    group_r2 = safe_r2(
        group_true,
        group_pred
    )

    range_results.append({

        "target_range": group,

        "sample_count": int(
            len(group_true)
        ),

        "actual_mean": float(
            np.mean(group_true)
        ),

        "predicted_mean": float(
            np.mean(group_pred)
        ),

        "mean_error": float(
            np.mean(group_errors)
        ),

        "median_error": float(
            np.median(group_errors)
        ),

        "mae": float(
            np.mean(group_abs_errors)
        ),

        "rmse": calculate_rmse(
            group_errors
        ),

        "r2": (
            float(group_r2)
            if not np.isnan(group_r2)
            else None
        ),

        "p50_absolute_error": float(
            np.percentile(
                group_abs_errors,
                50
            )
        ),

        "p90_absolute_error": float(
            np.percentile(
                group_abs_errors,
                90
            )
        ),

        "p95_absolute_error": float(
            np.percentile(
                group_abs_errors,
                95
            )
        ),

        "max_absolute_error": float(
            np.max(
                group_abs_errors
            )
        )
    })


range_df = pd.DataFrame(
    range_results
)


# ============================================================
# 10. PREDICTION BIAS ANALYSIS
# ============================================================

print("\n[10] Bias analysis...")

over_prediction_count = int(
    np.sum(
        errors > 0
    )
)

under_prediction_count = int(
    np.sum(
        errors < 0
    )
)

exact_prediction_count = int(
    np.sum(
        errors == 0
    )
)

total_predictions = len(
    errors
)

over_prediction_percentage = (
    over_prediction_count /
    total_predictions *
    100
)

under_prediction_percentage = (
    under_prediction_count /
    total_predictions *
    100
)

exact_prediction_percentage = (
    exact_prediction_count /
    total_predictions *
    100
)


# ============================================================
# 11. RESIDUAL DISTRIBUTION
# ============================================================

print(
    "\n[11] Calculating residual distribution..."
)

residual_percentiles = {

    "p01": float(
        np.percentile(
            errors,
            1
        )
    ),

    "p05": float(
        np.percentile(
            errors,
            5
        )
    ),

    "p10": float(
        np.percentile(
            errors,
            10
        )
    ),

    "p25": float(
        np.percentile(
            errors,
            25
        )
    ),

    "p50": float(
        np.percentile(
            errors,
            50
        )
    ),

    "p75": float(
        np.percentile(
            errors,
            75
        )
    ),

    "p90": float(
        np.percentile(
            errors,
            90
        )
    ),

    "p95": float(
        np.percentile(
            errors,
            95
        )
    ),

    "p99": float(
        np.percentile(
            errors,
            99
        )
    )
}


# ============================================================
# 12. ABSOLUTE ERROR DISTRIBUTION
# ============================================================

print(
    "\n[12] Calculating absolute error distribution..."
)

absolute_error_percentiles = {

    "p50": float(
        np.percentile(
            absolute_errors,
            50
        )
    ),

    "p75": float(
        np.percentile(
            absolute_errors,
            75
        )
    ),

    "p90": float(
        np.percentile(
            absolute_errors,
            90
        )
    ),

    "p95": float(
        np.percentile(
            absolute_errors,
            95
        )
    ),

    "p99": float(
        np.percentile(
            absolute_errors,
            99
        )
    )
}


# ============================================================
# 13. PREDICTION ERROR BANDS
# ============================================================

print(
    "\n[13] Calculating prediction error bands..."
)

band_results = []

bands = [

    (
        "Very Low Error (<=2)",
        absolute_errors <= 2
    ),

    (
        "Low Error (2-5)",
        (
            (absolute_errors > 2) &
            (absolute_errors <= 5)
        )
    ),

    (
        "Moderate Error (5-10)",
        (
            (absolute_errors > 5) &
            (absolute_errors <= 10)
        )
    ),

    (
        "High Error (10-15)",
        (
            (absolute_errors > 10) &
            (absolute_errors <= 15)
        )
    ),

    (
        "Very High Error (>15)",
        absolute_errors > 15
    )
]


for band_name, mask in bands:

    count = int(
        np.sum(mask)
    )

    percentage = (
        count /
        total_predictions *
        100
    )

    band_results.append({

        "error_band": band_name,

        "sample_count": count,

        "percentage": float(
            percentage
        )
    })


band_df = pd.DataFrame(
    band_results
)


# ============================================================
# 14. EXTREME ERROR ANALYSIS
# ============================================================

print(
    "\n[14] Analyzing extreme errors..."
)

top_n = min(
    20,
    total_predictions
)

extreme_indices = np.argsort(
    absolute_errors
)[-top_n:][::-1]


extreme_rows = []


for index in extreme_indices:

    extreme_rows.append({

        "actual_profit_margin": float(
            y_true[index]
        ),

        "predicted_profit_margin": float(
            y_pred[index]
        ),

        "error": float(
            errors[index]
        ),

        "absolute_error": float(
            absolute_errors[index]
        ),

        "target_range": str(
            range_labels[index]
        )
    })


extreme_df = pd.DataFrame(
    extreme_rows
)


# ============================================================
# 15. CALIBRATION ANALYSIS
# ============================================================

print(
    "\n[15] Calibration analysis..."
)

calibration_summary = pd.DataFrame()

try:

    calibration_bins = pd.qcut(
        y_true,
        q=CALIBRATION_BINS,
        duplicates="drop"
    )

    calibration_df = pd.DataFrame({

        "actual_bin": calibration_bins,

        "actual": y_true,

        "predicted": y_pred,

        "error": errors,

        "absolute_error": absolute_errors

    })


    # IMPORTANT:
    # The previous version had an invalid
    # aggregation:
    #
    # mean_error=("lambda x: np.nan")
    #
    # That syntax is invalid for pandas GroupBy.
    #
    # Here we calculate all required values
    # using valid named aggregations.

    calibration_summary = (
        calibration_df
        .groupby(
            "actual_bin",
            observed=True
        )
        .agg(
            sample_count=(
                "actual",
                "size"
            ),

            actual_mean=(
                "actual",
                "mean"
            ),

            predicted_mean=(
                "predicted",
                "mean"
            ),

            actual_median=(
                "actual",
                "median"
            ),

            predicted_median=(
                "predicted",
                "median"
            ),

            mean_error=(
                "error",
                "mean"
            ),

            median_error=(
                "error",
                "median"
            ),

            mae=(
                "absolute_error",
                "mean"
            ),

            p90_absolute_error=(
                "absolute_error",
                lambda x: np.percentile(
                    x,
                    90
                )
            ),

            max_absolute_error=(
                "absolute_error",
                "max"
            )
        )
        .reset_index()
    )


    calibration_summary[
        "absolute_bias"
    ] = (
        calibration_summary[
            "mean_error"
        ].abs()
    )


    calibration_summary[
        "actual_bin"
    ] = calibration_summary[
        "actual_bin"
    ].astype(str)


    calibration_summary[
        "calibration_gap"
    ] = (
        calibration_summary[
            "predicted_mean"
        ]
        -
        calibration_summary[
            "actual_mean"
        ]
    )


    calibration_summary[
        "absolute_calibration_gap"
    ] = (
        calibration_summary[
            "calibration_gap"
        ].abs()
    )


    calibration_summary = calibration_summary[
        [
            "actual_bin",
            "sample_count",
            "actual_mean",
            "predicted_mean",
            "actual_median",
            "predicted_median",
            "mean_error",
            "median_error",
            "mae",
            "p90_absolute_error",
            "max_absolute_error",
            "absolute_bias",
            "calibration_gap",
            "absolute_calibration_gap"
        ]
    ]


except Exception as exc:

    print(
        "Calibration analysis failed:",
        exc
    )

    calibration_summary = pd.DataFrame()


# ============================================================
# 16. ACTUAL VS PREDICTED DATA
# ============================================================

print(
    "\n[16] Preparing actual vs predicted data..."
)

actual_predicted_df = pd.DataFrame({

    "actual_profit_margin": y_true,

    "predicted_profit_margin": y_pred,

    "error": errors,

    "absolute_error": absolute_errors,

    "target_range": range_labels

})


# ============================================================
# 17. SAVE OUTPUT PATHS
# ============================================================

print(
    "\n[17] Preparing output files..."
)

actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "prediction_reliability_actual_vs_predicted.csv"
)

range_path = os.path.join(
    OUTPUT_DIR,
    "prediction_reliability_by_range.csv"
)

band_path = os.path.join(
    OUTPUT_DIR,
    "prediction_error_bands.csv"
)

extreme_path = os.path.join(
    OUTPUT_DIR,
    "prediction_extreme_errors.csv"
)

calibration_path = os.path.join(
    OUTPUT_DIR,
    "prediction_calibration.csv"
)

report_path = os.path.join(
    OUTPUT_DIR,
    "prediction_reliability_report.json"
)


# ============================================================
# 18. SAVE CSV FILES
# ============================================================

print(
    "\n[18] Saving CSV results..."
)

actual_predicted_df.to_csv(
    actual_predicted_path,
    index=False
)

range_df.to_csv(
    range_path,
    index=False
)

band_df.to_csv(
    band_path,
    index=False
)

extreme_df.to_csv(
    extreme_path,
    index=False
)

calibration_summary.to_csv(
    calibration_path,
    index=False
)


# ============================================================
# 19. CREATE REPORT
# ============================================================

print(
    "\n[19] Creating JSON report..."
)

report = {

    "analysis":
        "Prediction Reliability / Calibration Analysis",

    "model":
        best_model_name,

    "target":
        TARGET_COLUMN,

    "test_rows":
        int(total_predictions),

    "test_size":
        TEST_SIZE,

    "random_state":
        RANDOM_STATE,

    "target_min":
        min_target,

    "target_max":
        max_target,

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
            float(median_error),

        "mean_absolute_error":
            float(
                mean_absolute_error_value
            ),

        "median_absolute_error":
            float(
                median_absolute_error
            )
    },

    "bias": {

        "over_prediction_count":
            over_prediction_count,

        "under_prediction_count":
            under_prediction_count,

        "exact_prediction_count":
            exact_prediction_count,

        "over_prediction_percentage":
            float(
                over_prediction_percentage
            ),

        "under_prediction_percentage":
            float(
                under_prediction_percentage
            ),

        "exact_prediction_percentage":
            float(
                exact_prediction_percentage
            )
    },

    "target_range_thresholds": {

        "low":
            LOW_THRESHOLD,

        "high":
            HIGH_THRESHOLD
    },

    "residual_percentiles":
        residual_percentiles,

    "absolute_error_percentiles":
        absolute_error_percentiles,

    "error_bands":
        band_results,

    "calibration": {

        "number_of_bins":
            int(
                len(calibration_summary)
            ),

        "available":
            not calibration_summary.empty
    },

    "files": {

        "actual_vs_predicted":
            actual_predicted_path,

        "by_range":
            range_path,

        "error_bands":
            band_path,

        "extreme_errors":
            extreme_path,

        "calibration":
            calibration_path
    }
}


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# 20. PRINT FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PREDICTION RELIABILITY RESULTS"
)

print(
    "=" * 70
)


print(
    f"\nModel: {best_model_name}"
)

print(
    f"Target: {TARGET_COLUMN}"
)

print(
    f"Test Rows: {total_predictions:,}"
)


# ------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------

print(
    "\nOverall Metrics:"
)

print(
    f"R²:              {r2:.6f}"
)

print(
    f"MAE:             {mae:.6f}"
)

print(
    f"RMSE:            {rmse:.6f}"
)

print(
    f"Mean Error:      {mean_error:.6f}"
)

print(
    f"Median Error:    {median_error:.6f}"
)


# ------------------------------------------------------------
# Prediction bias
# ------------------------------------------------------------

print(
    "\nPrediction Bias:"
)

print(
    f"Over Prediction: "
    f"{over_prediction_count:,} "
    f"({over_prediction_percentage:.2f}%)"
)

print(
    f"Under Prediction: "
    f"{under_prediction_count:,} "
    f"({under_prediction_percentage:.2f}%)"
)

print(
    f"Exact Prediction: "
    f"{exact_prediction_count:,} "
    f"({exact_prediction_percentage:.2f}%)"
)


# ------------------------------------------------------------
# Target range
# ------------------------------------------------------------

print(
    "\nTarget Range Analysis:"
)

if not range_df.empty:

    print(
        range_df.to_string(
            index=False
        )
    )

else:

    print(
        "Target range analysis unavailable."
    )


# ------------------------------------------------------------
# Error bands
# ------------------------------------------------------------

print(
    "\nPrediction Error Bands:"
)

if not band_df.empty:

    print(
        band_df.to_string(
            index=False
        )
    )

else:

    print(
        "Error band analysis unavailable."
    )


# ------------------------------------------------------------
# Residual percentiles
# ------------------------------------------------------------

print(
    "\nResidual Percentiles:"
)

for key, value in residual_percentiles.items():

    print(
        f"{key.upper():>4}: {value:.6f}"
    )


# ------------------------------------------------------------
# Absolute error percentiles
# ------------------------------------------------------------

print(
    "\nAbsolute Error Percentiles:"
)

for key, value in absolute_error_percentiles.items():

    print(
        f"{key.upper():>4}: {value:.6f}"
    )


# ------------------------------------------------------------
# Extreme errors
# ------------------------------------------------------------

print(
    "\nTop 20 Extreme Errors:"
)

if not extreme_df.empty:

    print(
        extreme_df.to_string(
            index=False
        )
    )

else:

    print(
        "Extreme error analysis unavailable."
    )


# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------

print(
    "\nCalibration Summary:"
)

if not calibration_summary.empty:

    print(
        calibration_summary.to_string(
            index=False
        )
    )

else:

    print(
        "Calibration summary unavailable."
    )


# ------------------------------------------------------------
# Saved files
# ------------------------------------------------------------

print(
    "\nSaved files:"
)

print(
    actual_predicted_path
)

print(
    range_path
)

print(
    band_path
)

print(
    extreme_path
)

print(
    calibration_path
)

print(
    report_path
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 70
)

if calibration_summary.empty:

    print(
        "PREDICTION RELIABILITY ANALYSIS COMPLETED "
        "WITH CALIBRATION WARNING"
    )

else:

    print(
        "PREDICTION RELIABILITY ANALYSIS PASS"
    )

print(
    "=" * 70
)


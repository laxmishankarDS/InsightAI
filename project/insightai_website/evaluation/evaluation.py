# ============================================================
# INSIGHTAI
# MODEL EVALUATION
#
# Purpose:
# - Evaluate saved best model
# - Recreate training/test split
# - Calculate R2, MAE, RMSE
# - Generate actual vs predicted data
# - Perform residual/error analysis
# - Compare evaluation metrics with training metadata
# - Generate evaluation report
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

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

FEATURE_FILE = (
    r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"
)

OUTPUT_FOLDER = (
    r"C:\Users\maury\Downloads\InsightAI_Output"
)

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "best_model.joblib"
)

METADATA_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_metadata.json"
)

MODEL_RESULTS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_results.csv"
)

ACTUAL_PREDICTED_FILE = os.path.join(
    OUTPUT_FOLDER,
    "actual_vs_predicted.csv"
)

ERROR_ANALYSIS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "error_analysis.csv"
)

EVALUATION_REPORT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "evaluation_report.json"
)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

MIN_NUMERIC_CONVERSION_RATIO = 0.95


# ============================================================
# HELPER
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):

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

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    return df


# ============================================================
# HELPER
# DETECT NUMERIC-LIKE COLUMNS
# ============================================================

def detect_numeric_like_columns(df):

    numeric_like = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_empty = int(
            series.notna().sum()
        )

        if non_empty == 0:
            continue

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

        valid_ratio = (
            converted.notna().sum()
            / non_empty
        )

        if (
            valid_ratio
            >= MIN_NUMERIC_CONVERSION_RATIO
        ):
            numeric_like.append(
                column
            )

    return numeric_like


# ============================================================
# HELPER
# CONVERT NUMERIC-LIKE COLUMNS
# ============================================================

def convert_numeric_like_columns(df):

    df = df.copy()

    numeric_like_columns = (
        detect_numeric_like_columns(df)
    )

    for column in numeric_like_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# START
# ============================================================

print(
    "\n============================================================"
)

print(
    "INSIGHTAI - MODEL EVALUATION"
)

print(
    "============================================================"
)


# ============================================================
# CHECK OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Feature dataset": FEATURE_FILE,
    "Best model": MODEL_FILE,
    "Model metadata": METADATA_FILE
}

missing_files = []

for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):

        missing_files.append(
            f"{file_name}: {file_path}"
        )


if missing_files:

    raise FileNotFoundError(
        "\nRequired evaluation files are missing:\n\n"
        + "\n".join(missing_files)
        + "\n\nRun model_training.py first."
    )


# ============================================================
# LOAD METADATA
# ============================================================

print(
    "\nLoading model metadata..."
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


# ============================================================
# READ MODEL INFORMATION
# ============================================================

TARGET = metadata.get(
    "target"
)

BEST_MODEL_NAME = metadata.get(
    "best_model"
)

SELECTED_FEATURES = metadata.get(
    "selected_features",
    []
)

NUMERIC_FEATURES = metadata.get(
    "numeric_features",
    []
)

CATEGORICAL_FEATURES = metadata.get(
    "categorical_features",
    []
)

METADATA_TEST_SIZE = metadata.get(
    "test_size",
    TEST_SIZE
)

METADATA_RANDOM_STATE = metadata.get(
    "random_state",
    RANDOM_STATE
)


# ============================================================
# METADATA VALIDATION
# ============================================================

if not TARGET:

    raise ValueError(
        "Target information is missing "
        "from model_metadata.json."
    )


if not BEST_MODEL_NAME:

    raise ValueError(
        "Best model information is missing "
        "from model_metadata.json."
    )


if not SELECTED_FEATURES:

    raise ValueError(
        "Selected feature information is missing "
        "from model_metadata.json."
    )


print(
    "\nModel information:"
)

print(
    "Target:",
    TARGET
)

print(
    "Best Model:",
    BEST_MODEL_NAME
)

print(
    "Selected Features:",
    len(SELECTED_FEATURES)
)

print(
    "Test Size:",
    METADATA_TEST_SIZE
)

print(
    "Random State:",
    METADATA_RANDOM_STATE
)


# ============================================================
# LOAD DATASET
# ============================================================

print(
    "\nLoading feature dataset..."
)

df = pd.read_csv(
    FEATURE_FILE
)

if df.empty:

    raise ValueError(
        "Feature dataset is empty."
    )


print(
    "Original dataset shape:",
    df.shape
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df = clean_column_names(
    df
)


# ============================================================
# REPLACE INFINITE VALUES
# ============================================================

df = df.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


# ============================================================
# CONVERT NUMERIC-LIKE COLUMNS
# ============================================================

df = convert_numeric_like_columns(
    df
)


# ============================================================
# VALIDATE TARGET
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"\nTarget column '{TARGET}' "
        "was not found in feature dataset.\n\n"
        "Available columns:\n"
        f"{df.columns.tolist()}"
    )


df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    column
    for column in SELECTED_FEATURES
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "\nThe following model features "
        "are missing from the dataset:\n\n"
        + "\n".join(
            missing_features
        )
    )


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[
    SELECTED_FEATURES
].copy()

y = df[
    TARGET
].copy()


# ============================================================
# REMOVE MISSING TARGET ROWS
# ============================================================

valid_target_rows = y.notna()

X = X.loc[
    valid_target_rows
].copy()

y = y.loc[
    valid_target_rows
].copy()


if X.empty or y.empty:

    raise ValueError(
        "No valid rows remain after "
        "removing missing target values."
    )


# ============================================================
# VALIDATE NUMERIC FEATURES
# ============================================================

for column in NUMERIC_FEATURES:

    if column not in X.columns:
        continue

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# VALIDATE CATEGORICAL FEATURES
# ============================================================

for column in CATEGORICAL_FEATURES:

    if column not in X.columns:
        continue

    X[column] = X[column].astype(
        "object"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

TOTAL_ROWS = len(X)

TOTAL_FEATURES = len(
    SELECTED_FEATURES
)

print(
    "\nEvaluation dataset:"
)

print(
    "Rows:",
    TOTAL_ROWS
)

print(
    "Features:",
    TOTAL_FEATURES
)


# ============================================================
# RECREATE TRAIN / TEST SPLIT
#
# IMPORTANT:
# This matches model_training.py
# ============================================================

print(
    "\nRecreating train/test split..."
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=METADATA_TEST_SIZE,
        random_state=METADATA_RANDOM_STATE
    )
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
# LOAD SAVED MODEL
# ============================================================

print(
    "\nLoading saved best model..."
)

best_model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)

print(
    "Model:",
    BEST_MODEL_NAME
)


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print(
    "\nGenerating test predictions..."
)

predictions = best_model.predict(
    X_test
)


predictions = np.asarray(
    predictions,
    dtype=float
)

actual_values = np.asarray(
    y_test,
    dtype=float
)


# ============================================================
# BASIC PREDICTION VALIDATION
# ============================================================

if len(predictions) != len(actual_values):

    raise RuntimeError(
        "Prediction count does not match "
        "test target count."
    )


if not np.isfinite(
    predictions
).all():

    raise ValueError(
        "Model generated non-finite predictions."
    )


# ============================================================
# EVALUATION METRICS
# ============================================================

evaluation_r2 = r2_score(
    actual_values,
    predictions
)

evaluation_mae = mean_absolute_error(
    actual_values,
    predictions
)

evaluation_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        predictions
    )
)


# ============================================================
# ERROR CALCULATION
# ============================================================

errors = (
    actual_values
    - predictions
)

absolute_errors = np.abs(
    errors
)

squared_errors = (
    errors ** 2
)


# ============================================================
# ERROR STATISTICS
# ============================================================

mean_error = float(
    np.mean(errors)
)

median_error = float(
    np.median(errors)
)

mean_absolute_error_value = float(
    np.mean(absolute_errors)
)

median_absolute_error_value = float(
    np.median(absolute_errors)
)

maximum_absolute_error = float(
    np.max(absolute_errors)
)

minimum_absolute_error = float(
    np.min(absolute_errors)
)


# ============================================================
# PREDICTION STATISTICS
# ============================================================

actual_mean = float(
    np.mean(actual_values)
)

actual_median = float(
    np.median(actual_values)
)

actual_min = float(
    np.min(actual_values)
)

actual_max = float(
    np.max(actual_values)
)

prediction_mean = float(
    np.mean(predictions)
)

prediction_median = float(
    np.median(predictions)
)

prediction_min = float(
    np.min(predictions)
)

prediction_max = float(
    np.max(predictions)
)


# ============================================================
# ERROR PERCENTILES
# ============================================================

error_percentiles = {
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
# ACTUAL VS PREDICTED DATAFRAME
# ============================================================

actual_vs_predicted = pd.DataFrame(
    {
        "actual": actual_values,
        "predicted": predictions,
        "error": errors,
        "absolute_error": absolute_errors,
        "squared_error": squared_errors
    }
)


# ============================================================
# SAVE ACTUAL VS PREDICTED
# ============================================================

actual_vs_predicted.to_csv(
    ACTUAL_PREDICTED_FILE,
    index=False
)


# ============================================================
# ERROR ANALYSIS DATAFRAME
# ============================================================

error_analysis = pd.DataFrame(
    {
        "metric": [
            "test_rows",
            "r2",
            "mae",
            "rmse",
            "mean_error",
            "median_error",
            "mean_absolute_error",
            "median_absolute_error",
            "minimum_absolute_error",
            "maximum_absolute_error",
            "actual_mean",
            "actual_median",
            "actual_min",
            "actual_max",
            "prediction_mean",
            "prediction_median",
            "prediction_min",
            "prediction_max",
            "absolute_error_p50",
            "absolute_error_p75",
            "absolute_error_p90",
            "absolute_error_p95",
            "absolute_error_p99"
        ],
        "value": [
            len(actual_values),
            float(evaluation_r2),
            float(evaluation_mae),
            float(evaluation_rmse),
            mean_error,
            median_error,
            mean_absolute_error_value,
            median_absolute_error_value,
            minimum_absolute_error,
            maximum_absolute_error,
            actual_mean,
            actual_median,
            actual_min,
            actual_max,
            prediction_mean,
            prediction_median,
            prediction_min,
            prediction_max,
            error_percentiles["p50"],
            error_percentiles["p75"],
            error_percentiles["p90"],
            error_percentiles["p95"],
            error_percentiles["p99"]
        ]
    }
)


# ============================================================
# SAVE ERROR ANALYSIS
# ============================================================

error_analysis.to_csv(
    ERROR_ANALYSIS_FILE,
    index=False
)


# ============================================================
# TRAINING METRICS FROM METADATA
# ============================================================

training_metrics = metadata.get(
    "metrics",
    {}
)

training_r2 = training_metrics.get(
    "r2"
)

training_mae = training_metrics.get(
    "mae"
)

training_rmse = training_metrics.get(
    "rmse"
)


# ============================================================
# METRIC DIFFERENCE
# ============================================================

if training_r2 is not None:

    r2_difference = (
        float(evaluation_r2)
        - float(training_r2)
    )

else:

    r2_difference = None


if training_mae is not None:

    mae_difference = (
        float(evaluation_mae)
        - float(training_mae)
    )

else:

    mae_difference = None


if training_rmse is not None:

    rmse_difference = (
        float(evaluation_rmse)
        - float(training_rmse)
    )

else:

    rmse_difference = None


# ============================================================
# MODEL RESULT COMPARISON
# ============================================================

model_comparison = []

if os.path.exists(
    MODEL_RESULTS_FILE
):

    try:

        model_results_df = pd.read_csv(
            MODEL_RESULTS_FILE
        )

        if not model_results_df.empty:

            model_comparison = (
                model_results_df
                .to_dict(
                    orient="records"
                )
            )

    except Exception as error:

        print(
            "\nWarning: Could not read "
            "model_results.csv."
        )

        print(
            "Reason:",
            str(error)
        )


# ============================================================
# EVALUATION STATUS
# ============================================================

evaluation_status = "PASS"

status_reasons = []


if not np.isfinite(
    evaluation_r2
):

    evaluation_status = "REVIEW"

    status_reasons.append(
        "R2 is not finite."
    )


if not np.isfinite(
    evaluation_mae
):

    evaluation_status = "REVIEW"

    status_reasons.append(
        "MAE is not finite."
    )


if not np.isfinite(
    evaluation_rmse
):

    evaluation_status = "REVIEW"

    status_reasons.append(
        "RMSE is not finite."
    )


if evaluation_mae < 0:

    evaluation_status = "REVIEW"

    status_reasons.append(
        "MAE cannot be negative."
    )


if evaluation_rmse < 0:

    evaluation_status = "REVIEW"

    status_reasons.append(
        "RMSE cannot be negative."
    )


if not status_reasons:

    status_reasons.append(
        "Saved model generated valid predictions "
        "and evaluation metrics."
    )


# ============================================================
# EVALUATION REPORT
# ============================================================

evaluation_report = {

    "project":
        "InsightAI",

    "evaluation_type":
        "holdout_test_evaluation",

    "evaluation_status":
        evaluation_status,

    "status_reasons":
        status_reasons,

    "dataset": {

        "feature_file":
            FEATURE_FILE,

        "total_rows":
            int(TOTAL_ROWS),

        "test_rows":
            int(len(X_test)),

        "training_rows":
            int(len(X_train)),

        "feature_count":
            int(TOTAL_FEATURES)
    },

    "model": {

        "model_name":
            BEST_MODEL_NAME,

        "model_file":
            MODEL_FILE,

        "target":
            TARGET,

        "selected_features":
            SELECTED_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES
    },

    "evaluation_metrics": {

        "r2":
            float(evaluation_r2),

        "mae":
            float(evaluation_mae),

        "rmse":
            float(evaluation_rmse)
    },

    "training_metrics": {

        "r2":
            training_r2,

        "mae":
            training_mae,

        "rmse":
            training_rmse
    },

    "metric_difference": {

        "r2_difference":
            r2_difference,

        "mae_difference":
            mae_difference,

        "rmse_difference":
            rmse_difference
    },

    "prediction_statistics": {

        "actual_mean":
            actual_mean,

        "actual_median":
            actual_median,

        "actual_min":
            actual_min,

        "actual_max":
            actual_max,

        "prediction_mean":
            prediction_mean,

        "prediction_median":
            prediction_median,

        "prediction_min":
            prediction_min,

        "prediction_max":
            prediction_max
    },

    "error_statistics": {

        "mean_error":
            mean_error,

        "median_error":
            median_error,

        "mean_absolute_error":
            mean_absolute_error_value,

        "median_absolute_error":
            median_absolute_error_value,

        "minimum_absolute_error":
            minimum_absolute_error,

        "maximum_absolute_error":
            maximum_absolute_error,

        "percentiles":
            error_percentiles
    },

    "model_comparison":
        model_comparison,

    "evaluation_settings": {

        "test_size":
            METADATA_TEST_SIZE,

        "random_state":
            METADATA_RANDOM_STATE
    },

    "output_files": {

        "actual_vs_predicted":
            ACTUAL_PREDICTED_FILE,

        "error_analysis":
            ERROR_ANALYSIS_FILE,

        "evaluation_report":
            EVALUATION_REPORT_FILE
    }
}


# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

with open(
    EVALUATION_REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        evaluation_report,
        file,
        indent=4
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n============================================================"
)

print(
    "EVALUATION RESULTS"
)

print(
    "============================================================"
)

print(
    "Model:",
    BEST_MODEL_NAME
)

print(
    "Target:",
    TARGET
)

print(
    "Test Rows:",
    len(X_test)
)

print(
    "\nR2:",
    f"{evaluation_r2:.6f}"
)

print(
    "MAE:",
    f"{evaluation_mae:.6f}"
)

print(
    "RMSE:",
    f"{evaluation_rmse:.6f}"
)


# ============================================================
# PRINT ERROR ANALYSIS
# ============================================================

print(
    "\n============================================================"
)

print(
    "ERROR ANALYSIS"
)

print(
    "============================================================"
)

print(
    "Mean Error:",
    f"{mean_error:.6f}"
)

print(
    "Median Error:",
    f"{median_error:.6f}"
)

print(
    "Mean Absolute Error:",
    f"{mean_absolute_error_value:.6f}"
)

print(
    "Maximum Absolute Error:",
    f"{maximum_absolute_error:.6f}"
)

print(
    "\nAbsolute Error Percentiles:"
)

print(
    "P50:",
    f"{error_percentiles['p50']:.6f}"
)

print(
    "P75:",
    f"{error_percentiles['p75']:.6f}"
)

print(
    "P90:",
    f"{error_percentiles['p90']:.6f}"
)

print(
    "P95:",
    f"{error_percentiles['p95']:.6f}"
)

print(
    "P99:",
    f"{error_percentiles['p99']:.6f}"
)


# ============================================================
# TRAINING VS EVALUATION
# ============================================================

print(
    "\n============================================================"
)

print(
    "TRAINING VS EVALUATION"
)

print(
    "============================================================"
)

if training_r2 is not None:

    print(
        "Training R2:",
        f"{float(training_r2):.6f}"
    )

    print(
        "Evaluation R2:",
        f"{float(evaluation_r2):.6f}"
    )

    print(
        "R2 Difference:",
        f"{float(r2_difference):.6f}"
    )


if training_mae is not None:

    print(
        "\nTraining MAE:",
        f"{float(training_mae):.6f}"
    )

    print(
        "Evaluation MAE:",
        f"{float(evaluation_mae):.6f}"
    )

    print(
        "MAE Difference:",
        f"{float(mae_difference):.6f}"
    )


if training_rmse is not None:

    print(
        "\nTraining RMSE:",
        f"{float(training_rmse):.6f}"
    )

    print(
        "Evaluation RMSE:",
        f"{float(evaluation_rmse):.6f}"
    )

    print(
        "RMSE Difference:",
        f"{float(rmse_difference):.6f}"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print(
    "\n============================================================"
)

print(
    "EVALUATION FILES CREATED"
)

print(
    "============================================================"
)

print(
    "1.",
    ACTUAL_PREDICTED_FILE
)

print(
    "2.",
    ERROR_ANALYSIS_FILE
)

print(
    "3.",
    EVALUATION_REPORT_FILE
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n============================================================"
)

print(
    "MODEL EVALUATION COMPLETED"
)

print(
    "============================================================"
)

print(
    "Evaluation Status:",
    evaluation_status
)

print(
    "Best Model:",
    BEST_MODEL_NAME
)

print(
    "Target:",
    TARGET
)

print(
    "\nReady for further evaluation and error analysis."
)
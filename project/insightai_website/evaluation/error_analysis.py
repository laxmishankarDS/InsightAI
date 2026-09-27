# ============================================================
# INSIGHTAI
# ERROR / RESIDUAL ANALYSIS
# ============================================================

import os
import json
import warnings
import joblib
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

FEATURE_FILE = r"C:\Users\maury\Downloads\InsightAI_Output\feature_data.csv"

OUTPUT_FOLDER = r"C:\Users\maury\Downloads\InsightAI_Output"

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "best_model.joblib"
)

METADATA_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_metadata.json"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "detailed_error_analysis.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_FOLDER,
    "error_analysis_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
MIN_NUMERIC_CONVERSION_RATIO = 0.95

TOP_ERROR_ROWS = 100


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("INSIGHTAI - ERROR / RESIDUAL ANALYSIS")
print("=" * 60)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    """
    Standardize dataframe column names.
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


def detect_numeric_like_columns(df):
    """
    Detect object columns that contain mostly numeric values.
    """

    numeric_like = []

    for col in df.columns:

        if df[col].dtype == "object":

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            non_null = df[col].notna().sum()

            if non_null == 0:
                continue

            conversion_ratio = (
                converted.notna().sum()
                / non_null
            )

            if conversion_ratio >= MIN_NUMERIC_CONVERSION_RATIO:
                numeric_like.append(col)

    return numeric_like


def convert_numeric_like_columns(df):
    """
    Convert numeric-like columns.
    """

    df = df.copy()

    numeric_like = detect_numeric_like_columns(df)

    for col in numeric_like:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading model metadata...")

if not os.path.exists(METADATA_FILE):

    raise FileNotFoundError(
        f"Model metadata not found:\n{METADATA_FILE}"
    )


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


target = metadata["target"]

best_model_name = metadata["best_model"]

selected_features = metadata["selected_features"]

numeric_features = metadata.get(
    "numeric_features",
    []
)

categorical_features = metadata.get(
    "categorical_features",
    []
)

test_size = metadata.get(
    "test_size",
    DEFAULT_TEST_SIZE
)

random_state = metadata.get(
    "random_state",
    DEFAULT_RANDOM_STATE
)


print("\nModel information:")
print("Target:", target)
print("Best Model:", best_model_name)
print("Selected Features:", len(selected_features))
print("Test Size:", test_size)
print("Random State:", random_state)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature dataset...")

if not os.path.exists(FEATURE_FILE):

    raise FileNotFoundError(
        f"Feature dataset not found:\n{FEATURE_FILE}"
    )


df = pd.read_csv(FEATURE_FILE)

print(
    "Original dataset shape:",
    df.shape
)


# ============================================================
# CLEAN DATA
# ============================================================

df = clean_column_names(df)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = convert_numeric_like_columns(df)


# ============================================================
# VALIDATE TARGET
# ============================================================

if target not in df.columns:

    raise ValueError(
        f"Target column '{target}' not found."
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    col
    for col in selected_features
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "Selected features missing:\n"
        + str(missing_features)
    )


# ============================================================
# PREPARE X / Y
# ============================================================

X = df[selected_features].copy()

y = pd.to_numeric(
    df[target],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

valid_rows = y.notna()

X = X.loc[valid_rows].copy()

y = y.loc[valid_rows].copy()


# ============================================================
# RESTORE FEATURE TYPES
# ============================================================

for col in numeric_features:

    if col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )


for col in categorical_features:

    if col in X.columns:

        X[col] = X[col].astype("object")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nRecreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


print("Training rows:", len(X_train))

print("Testing rows:", len(X_test))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading saved model...")

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )


model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

print("Model:", best_model_name)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)


# ============================================================
# CREATE ERROR DATAFRAME
# ============================================================

error_df = X_test.copy()

error_df["actual"] = y_test.values

error_df["predicted"] = predictions

error_df["error"] = (
    error_df["predicted"]
    - error_df["actual"]
)

error_df["absolute_error"] = (
    error_df["error"]
    .abs()
)

error_df["squared_error"] = (
    error_df["error"]
    ** 2
)


# ============================================================
# PERCENT ERROR
# ============================================================

error_df["absolute_percentage_error"] = np.where(
    error_df["actual"].abs() > 1e-10,
    (
        error_df["absolute_error"]
        / error_df["actual"].abs()
    ) * 100,
    np.nan
)


# ============================================================
# ERROR DIRECTION
# ============================================================

error_df["error_direction"] = np.where(
    error_df["error"] > 0,
    "over_prediction",
    np.where(
        error_df["error"] < 0,
        "under_prediction",
        "exact"
    )
)


# ============================================================
# BASIC METRICS
# ============================================================

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# ============================================================
# ERROR STATISTICS
# ============================================================

mean_error = float(
    error_df["error"].mean()
)

median_error = float(
    error_df["error"].median()
)

mean_absolute_error_value = float(
    error_df["absolute_error"].mean()
)

median_absolute_error = float(
    error_df["absolute_error"].median()
)

maximum_absolute_error = float(
    error_df["absolute_error"].max()
)

minimum_absolute_error = float(
    error_df["absolute_error"].min()
)


# ============================================================
# ERROR PERCENTILES
# ============================================================

percentiles = {
    "p50": float(
        error_df["absolute_error"].quantile(0.50)
    ),

    "p75": float(
        error_df["absolute_error"].quantile(0.75)
    ),

    "p90": float(
        error_df["absolute_error"].quantile(0.90)
    ),

    "p95": float(
        error_df["absolute_error"].quantile(0.95)
    ),

    "p99": float(
        error_df["absolute_error"].quantile(0.99)
    )
}


# ============================================================
# ERROR DIRECTION COUNTS
# ============================================================

over_predictions = int(
    (
        error_df["error_direction"]
        == "over_prediction"
    ).sum()
)

under_predictions = int(
    (
        error_df["error_direction"]
        == "under_prediction"
    ).sum()
)

exact_predictions = int(
    (
        error_df["error_direction"]
        == "exact"
    ).sum()
)


total_predictions = len(error_df)


over_percentage = (
    over_predictions
    / total_predictions
) * 100

under_percentage = (
    under_predictions
    / total_predictions
) * 100


# ============================================================
# TOP ERROR RECORDS
# ============================================================

top_errors = (
    error_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(TOP_ERROR_ROWS)
    .copy()
)


# ============================================================
# SAVE DETAILED ANALYSIS
# ============================================================

top_errors.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FEATURE-WISE NUMERIC ERROR ANALYSIS
# ============================================================

feature_error_analysis = {}


for feature in numeric_features:

    if feature not in error_df.columns:
        continue

    temp = error_df[
        [feature, "absolute_error"]
    ].dropna()

    if len(temp) < 2:
        continue

    correlation = temp[
        feature
    ].corr(
        temp["absolute_error"]
    )

    if pd.isna(correlation):
        correlation = 0.0

    feature_error_analysis[feature] = {
        "correlation_with_absolute_error":
            float(correlation),

        "mean_value":
            float(temp[feature].mean()),

        "mean_absolute_error":
            float(temp["absolute_error"].mean())
    }


# ============================================================
# CATEGORICAL ERROR ANALYSIS
# ============================================================

categorical_error_analysis = {}


for feature in categorical_features:

    if feature not in error_df.columns:
        continue

    grouped = (
        error_df
        .groupby(feature, dropna=False)
        ["absolute_error"]
        .agg(
            count="count",
            mean_absolute_error="mean",
            median_absolute_error="median"
        )
        .sort_values(
            "mean_absolute_error",
            ascending=False
        )
        .head(20)
    )

    records = []

    for index, row in grouped.iterrows():

        records.append({
            "value": str(index),
            "count": int(row["count"]),
            "mean_absolute_error":
                float(row["mean_absolute_error"]),
            "median_absolute_error":
                float(row["median_absolute_error"])
        })

    categorical_error_analysis[feature] = records


# ============================================================
# SUMMARY REPORT
# ============================================================

summary = {

    "model": best_model_name,

    "target": target,

    "test_rows": int(len(y_test)),

    "features": selected_features,

    "metrics": {

        "r2": float(r2),

        "mae": float(mae),

        "rmse": float(rmse)
    },

    "error_statistics": {

        "mean_error":
            mean_error,

        "median_error":
            median_error,

        "mean_absolute_error":
            mean_absolute_error_value,

        "median_absolute_error":
            median_absolute_error,

        "minimum_absolute_error":
            minimum_absolute_error,

        "maximum_absolute_error":
            maximum_absolute_error
    },

    "absolute_error_percentiles": percentiles,

    "error_direction": {

        "over_prediction_count":
            over_predictions,

        "under_prediction_count":
            under_predictions,

        "exact_prediction_count":
            exact_predictions,

        "over_prediction_percentage":
            float(over_percentage),

        "under_prediction_percentage":
            float(under_percentage)
    },

    "numeric_feature_error_analysis":
        feature_error_analysis,

    "categorical_feature_error_analysis":
        categorical_error_analysis,

    "top_error_rows_saved":
        TOP_ERROR_ROWS
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS RESULTS")
print("=" * 60)

print(
    "\nModel:",
    best_model_name
)

print(
    "Target:",
    target
)

print(
    "Test Rows:",
    len(y_test)
)

print("\nMetrics:")

print(
    f"R2:   {r2:.6f}"
)

print(
    f"MAE:  {mae:.6f}"
)

print(
    f"RMSE: {rmse:.6f}"
)


print("\nError Statistics:")

print(
    f"Mean Error:             {mean_error:.6f}"
)

print(
    f"Median Error:           {median_error:.6f}"
)

print(
    f"Mean Absolute Error:    "
    f"{mean_absolute_error_value:.6f}"
)

print(
    f"Median Absolute Error:  "
    f"{median_absolute_error:.6f}"
)

print(
    f"Maximum Absolute Error: "
    f"{maximum_absolute_error:.6f}"
)


print("\nAbsolute Error Percentiles:")

print(
    f"P50: {percentiles['p50']:.6f}"
)

print(
    f"P75: {percentiles['p75']:.6f}"
)

print(
    f"P90: {percentiles['p90']:.6f}"
)

print(
    f"P95: {percentiles['p95']:.6f}"
)

print(
    f"P99: {percentiles['p99']:.6f}"
)


print("\nPrediction Direction:")

print(
    f"Over Prediction: "
    f"{over_predictions} "
    f"({over_percentage:.2f}%)"
)

print(
    f"Under Prediction: "
    f"{under_predictions} "
    f"({under_percentage:.2f}%)"
)

print(
    f"Exact Prediction: "
    f"{exact_predictions}"
)


# ============================================================
# TOP ERROR PREVIEW
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 HIGHEST ERROR PREDICTIONS")
print("=" * 60)

preview_columns = [
    "actual",
    "predicted",
    "error",
    "absolute_error",
    "error_direction"
]


print(
    top_errors[
        preview_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS FILES CREATED")
print("=" * 60)

print(
    "1.",
    OUTPUT_FILE
)

print(
    "2.",
    SUMMARY_FILE
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETED")
print("=" * 60)

print("Status: PASS")
print("Model:", best_model_name)
print("Target:", target)
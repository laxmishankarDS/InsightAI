# ============================================================
# INSIGHTAI
# CROSS VALIDATION - MODEL STABILITY CHECK
# ============================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    "cross_validation_results.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "cross_validation_report.json"
)


# ============================================================
# SETTINGS
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42

MIN_NUMERIC_CONVERSION_RATIO = 0.95


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("INSIGHTAI - CROSS VALIDATION")
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
    Detect object/category columns that are actually numeric.
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

            conversion_ratio = converted.notna().sum() / non_null

            if conversion_ratio >= MIN_NUMERIC_CONVERSION_RATIO:
                numeric_like.append(col)

    return numeric_like


def convert_numeric_like_columns(df):
    """
    Convert numeric-like columns into numeric dtype.
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


print("\nModel information:")
print("Target:", target)
print("Best Model:", best_model_name)
print("Selected Features:", len(selected_features))
print("Cross Validation Folds:", N_SPLITS)
print("Random State:", RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature dataset...")

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"Feature dataset not found:\n{FEATURE_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

print("Original dataset shape:", df.shape)


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
        f"Target column '{target}' not found in dataset."
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
        "Selected features missing from dataset:\n"
        + str(missing_features)
    )


# ============================================================
# PREPARE X AND Y
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


print("\nEvaluation dataset:")
print("Rows:", len(X))
print("Features:", X.shape[1])


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
# LOAD SAVED MODEL
# ============================================================

print("\nLoading saved best model...")

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Saved model not found:\n{MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Model:", best_model_name)


# ============================================================
# CROSS VALIDATION SETUP
# ============================================================

print("\nPreparing cross validation...")

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# SCORING
# ============================================================

scoring = {
    "r2": "r2",
    "mae": make_scorer(
        mean_absolute_error,
        greater_is_better=False
    ),
    "rmse": make_scorer(
        lambda y_true, y_pred:
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),
        greater_is_better=False
    )
}


# ============================================================
# RUN CROSS VALIDATION
# ============================================================

print("\nRunning 5-fold cross validation...")

results = cross_validate(
    model,
    X,
    y,
    cv=kf,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)


# ============================================================
# EXTRACT RESULTS
# ============================================================

r2_scores = results["test_r2"]

mae_scores = -results["test_mae"]

rmse_scores = -results["test_rmse"]


# ============================================================
# PRINT FOLD RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FOLD RESULTS")
print("=" * 60)

for i in range(N_SPLITS):

    print(f"\nFold {i + 1}")

    print(
        f"R2:   {r2_scores[i]:.6f}"
    )

    print(
        f"MAE:  {mae_scores[i]:.6f}"
    )

    print(
        f"RMSE: {rmse_scores[i]:.6f}"
    )


# ============================================================
# SUMMARY
# ============================================================

mean_r2 = float(np.mean(r2_scores))
std_r2 = float(np.std(r2_scores))

mean_mae = float(np.mean(mae_scores))
std_mae = float(np.std(mae_scores))

mean_rmse = float(np.mean(rmse_scores))
std_rmse = float(np.std(rmse_scores))


# ============================================================
# SUMMARY OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("CROSS VALIDATION SUMMARY")
print("=" * 60)

print(
    f"\nMean R2:   {mean_r2:.6f}"
)

print(
    f"Std R2:    {std_r2:.6f}"
)

print(
    f"\nMean MAE:  {mean_mae:.6f}"
)

print(
    f"Std MAE:   {std_mae:.6f}"
)

print(
    f"\nMean RMSE: {mean_rmse:.6f}"
)

print(
    f"Std RMSE:  {std_rmse:.6f}"
)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

cv_results = pd.DataFrame({
    "fold": range(1, N_SPLITS + 1),
    "r2": r2_scores,
    "mae": mae_scores,
    "rmse": rmse_scores
})


# ============================================================
# ADD SUMMARY ROW
# ============================================================

summary_row = pd.DataFrame({
    "fold": ["mean"],
    "r2": [mean_r2],
    "mae": [mean_mae],
    "rmse": [mean_rmse]
})

cv_results = pd.concat(
    [
        cv_results,
        summary_row
    ],
    ignore_index=True
)


# ============================================================
# SAVE CSV
# ============================================================

cv_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CREATE JSON REPORT
# ============================================================

report = {
    "model": best_model_name,
    "target": target,
    "rows": int(len(X)),
    "features": int(X.shape[1]),
    "folds": N_SPLITS,
    "random_state": RANDOM_STATE,

    "r2": {
        "fold_scores": [
            float(x)
            for x in r2_scores
        ],
        "mean": mean_r2,
        "std": std_r2
    },

    "mae": {
        "fold_scores": [
            float(x)
            for x in mae_scores
        ],
        "mean": mean_mae,
        "std": std_mae
    },

    "rmse": {
        "fold_scores": [
            float(x)
            for x in rmse_scores
        ],
        "mean": mean_rmse,
        "std": std_rmse
    }
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("CROSS VALIDATION FILES CREATED")
print("=" * 60)

print("1.", OUTPUT_FILE)
print("2.", REPORT_FILE)

print("\n" + "=" * 60)
print("CROSS VALIDATION COMPLETED")
print("=" * 60)

print("Status: PASS")
print("Model:", best_model_name)
print("Target:", target)
print("Mean R2:", f"{mean_r2:.6f}")
print("Mean MAE:", f"{mean_mae:.6f}")
print("Mean RMSE:", f"{mean_rmse:.6f}")
# ============================================================
# INSIGHTAI
# FEATURE IMPORTANCE / MODEL EXPLAINABILITY
# ============================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

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
    "feature_importance.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "feature_importance_report.json"
)


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42

N_REPEATS = 5

SCORING = "r2"

MIN_NUMERIC_CONVERSION_RATIO = 0.95


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("INSIGHTAI - FEATURE IMPORTANCE")
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
    Detect object columns containing mostly numeric values.
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

            if (
                conversion_ratio
                >= MIN_NUMERIC_CONVERSION_RATIO
            ):
                numeric_like.append(col)

    return numeric_like


def convert_numeric_like_columns(df):
    """
    Convert numeric-like columns to numeric dtype.
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

print(
    "Target:",
    target
)

print(
    "Best Model:",
    best_model_name
)

print(
    "Selected Features:",
    len(selected_features)
)

print(
    "Permutation Repeats:",
    N_REPEATS
)


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
# RECREATE TEST SET
# ============================================================

print("\nRecreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
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
# LOAD MODEL
# ============================================================

print("\nLoading saved model...")

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Saved model not found:\n{MODEL_FILE}"
    )


model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)

print(
    "Model:",
    best_model_name
)


# ============================================================
# BASELINE PERFORMANCE
# ============================================================

print("\nCalculating baseline performance...")

baseline_predictions = model.predict(
    X_test
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


print(
    f"Baseline R2: {baseline_r2:.6f}"
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance...")

print(
    "This may take some time..."
)


importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring=SCORING,
    n_repeats=N_REPEATS,
    random_state=random_state,
    n_jobs=-1
)


# ============================================================
# CREATE IMPORTANCE DATAFRAME
# ============================================================

importance_df = pd.DataFrame({

    "feature": X_test.columns,

    "importance_mean":
        importance_result.importances_mean,

    "importance_std":
        importance_result.importances_std
})


# ============================================================
# SORT BY IMPORTANCE
# ============================================================

importance_df = (
    importance_df
    .sort_values(
        "importance_mean",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# ADD RANK
# ============================================================

importance_df.insert(
    0,
    "rank",
    range(
        1,
        len(importance_df) + 1
    )
)


# ============================================================
# IMPORTANCE PERCENTAGE
# ============================================================

positive_importance = (
    importance_df[
        "importance_mean"
    ].clip(lower=0)
)

total_positive_importance = (
    positive_importance.sum()
)

if total_positive_importance > 0:

    importance_df[
        "importance_percentage"
    ] = (
        positive_importance
        / total_positive_importance
    ) * 100

else:

    importance_df[
        "importance_percentage"
    ] = 0.0


# ============================================================
# IMPORTANCE INTERPRETATION
# ============================================================

def importance_level(value):

    if value >= 0.10:
        return "high"

    if value >= 0.01:
        return "medium"

    if value > 0:
        return "low"

    return "negative_or_zero"


importance_df["importance_level"] = (
    importance_df[
        "importance_mean"
    ]
    .apply(importance_level)
)


# ============================================================
# SAVE CSV
# ============================================================

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CREATE JSON REPORT
# ============================================================

feature_records = []

for _, row in importance_df.iterrows():

    feature_records.append({

        "rank":
            int(row["rank"]),

        "feature":
            str(row["feature"]),

        "importance_mean":
            float(row["importance_mean"]),

        "importance_std":
            float(row["importance_std"]),

        "importance_percentage":
            float(
                row["importance_percentage"]
            ),

        "importance_level":
            str(row["importance_level"])
    })


# ============================================================
# MOST IMPORTANT FEATURES
# ============================================================

top_features = (
    importance_df
    .head(5)
    ["feature"]
    .tolist()
)


# ============================================================
# NEGATIVE / ZERO IMPORTANCE
# ============================================================

negative_or_zero_features = (
    importance_df[
        importance_df[
            "importance_mean"
        ] <= 0
    ]
    ["feature"]
    .tolist()
)


# ============================================================
# REPORT
# ============================================================

report = {

    "model":
        best_model_name,

    "target":
        target,

    "test_rows":
        int(len(X_test)),

    "selected_features":
        selected_features,

    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "baseline_r2":
        float(baseline_r2),

    "permutation_repeats":
        N_REPEATS,

    "scoring":
        SCORING,

    "top_features":
        top_features,

    "negative_or_zero_features":
        negative_or_zero_features,

    "feature_importance":
        feature_records
}


# ============================================================
# SAVE JSON
# ============================================================

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
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE RESULTS")
print("=" * 60)

print(
    "\nBaseline R2:",
    f"{baseline_r2:.6f}"
)


print(
    "\nFeature ranking:"
)


for _, row in importance_df.iterrows():

    print(
        f"{int(row['rank']):2d}. "
        f"{row['feature']:<25} "
        f"Importance: "
        f"{row['importance_mean']:.6f} "
        f"+/- "
        f"{row['importance_std']:.6f} "
        f"({row['importance_level']})"
    )


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TOP FEATURES")
print("=" * 60)

for i, feature in enumerate(
    top_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# NEGATIVE / ZERO FEATURES
# ============================================================

print("\n" + "=" * 60)
print("NEGATIVE / ZERO IMPORTANCE FEATURES")
print("=" * 60)

if negative_or_zero_features:

    for feature in negative_or_zero_features:

        print(
            "-",
            feature
        )

else:

    print(
        "None"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE FILES CREATED")
print("=" * 60)

print(
    "1.",
    OUTPUT_FILE
)

print(
    "2.",
    REPORT_FILE
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE COMPLETED")
print("=" * 60)

print(
    "Status: PASS"
)

print(
    "Model:",
    best_model_name
)

print(
    "Target:",
    target
)
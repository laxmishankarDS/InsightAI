
# ============================================================
# INSIGHTAI
# SHAP / MODEL EXPLAINABILITY ANALYSIS
#
# Purpose:
# Explain the saved best model and understand:
# 1. Global feature importance
# 2. Feature impact on predictions
# 3. Category-level prediction behaviour
# 4. Individual prediction explanations
#
# Existing model is NOT modified.
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

GLOBAL_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_global_importance.csv"
)

CATEGORY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_category_analysis.csv"
)

LOCAL_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_local_explanations.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_explainability_report.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "profit_margin"

TEST_SIZE = 0.20

RANDOM_STATE = 42

# SHAP calculation can be expensive on 40,000 rows.
# We use a representative sample for explainability.
SHAP_SAMPLE_SIZE = 10000

# Number of individual predictions to explain.
LOCAL_SAMPLE_SIZE = 20


# ============================================================
# HELPER
# ============================================================

def clean_feature_name(name):
    """
    Convert transformed sklearn feature names into
    cleaner readable names.
    """

    name = str(name)

    if "__" in name:
        name = name.split("__", 1)[1]

    return name


def safe_float(value):
    """
    Convert numpy values to normal Python float.
    """

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("INSIGHTAI - SHAP / MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 75)

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # IMPORT SHAP
    # --------------------------------------------------------

    print("\n[1/9] Loading SHAP...")

    try:

        import shap

    except ImportError:

        raise ImportError(
            "\nSHAP is not installed.\n\n"
            "Run this command inside your .venv:\n\n"
            "pip install shap\n"
        )

    print(
        f"SHAP version: {shap.__version__}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n[2/9] Loading dataset...")

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
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n[3/9] Loading saved model and metadata...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(METADATA_PATH):

        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    target = metadata.get(
        "target",
        TARGET_COLUMN
    )

    selected_features = metadata.get(
        "selected_features",
        []
    )

    selected_numeric = metadata.get(
        "selected_numeric",
        []
    )

    selected_categorical = metadata.get(
        "selected_categorical",
        []
    )

    best_model = metadata.get(
        "best_model",
        "Unknown"
    )

    print(
        f"Target: {target}"
    )

    print(
        f"Best model: {best_model}"
    )

    print(
        f"Selected features: {selected_features}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    required_columns = [
        target
    ] + list(
        selected_features
    )

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

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    print("\n[4/9] Preparing exact test split...")

    X = df[
        selected_features
    ].copy()

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    valid_mask = y.notna()

    X = X.loc[
        valid_mask
    ].copy()

    y = y.loc[
        valid_mask
    ].copy()

    # Convert numeric features.
    for column in selected_numeric:

        if column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    # Convert categorical features to string.
    for column in selected_categorical:

        if column in X.columns:

            X[column] = X[column].astype(
                "object"
            )

    indices = np.arange(
        len(X)
    )

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_test = X.iloc[
        test_indices
    ].copy()

    y_test = y.iloc[
        test_indices
    ].copy()

    print(
        f"Test rows: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # VERIFY MODEL
    # --------------------------------------------------------

    print("\n[5/9] Verifying saved model...")

    predictions = model.predict(
        X_test
    )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    baseline_r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"Test R²: {baseline_r2:.6f}"
    )

    # --------------------------------------------------------
    # SAMPLE DATA FOR SHAP
    # --------------------------------------------------------

    print(
        "\n[6/9] Preparing explainability sample..."
    )

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test)
    )

    # Fixed random sample for reproducibility.
    shap_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    shap_sample = shap_sample.reset_index(
        drop=True
    )

    print(
        f"SHAP sample rows: {len(shap_sample):,}"
    )

    # --------------------------------------------------------
    # TRANSFORM FEATURES
    # --------------------------------------------------------

    print(
        "\n[7/9] Transforming model features..."
    )

    if not hasattr(
        model,
        "named_steps"
    ):

        raise ValueError(
            "Saved model is not a sklearn Pipeline. "
            "Pipeline structure is required for this "
            "SHAP analysis."
        )

    if "preprocessor" not in model.named_steps:

        raise ValueError(
            "Model pipeline does not contain "
            "'preprocessor'."
        )

    if "model" not in model.named_steps:

        raise ValueError(
            "Model pipeline does not contain "
            "'model'."
        )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    estimator = model.named_steps[
        "model"
    ]

    transformed_sample = preprocessor.transform(
        shap_sample
    )

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    try:

        transformed_feature_names = (
            preprocessor.get_feature_names_out()
        )

    except Exception:

        transformed_feature_names = [
            f"feature_{i}"
            for i in range(
                transformed_sample.shape[1]
            )
        ]

    transformed_feature_names = [
        clean_feature_name(name)
        for name in transformed_feature_names
    ]

    print(
        f"Transformed features: "
        f"{len(transformed_feature_names)}"
    )

    # --------------------------------------------------------
    # SHAP EXPLAINER
    # --------------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    try:

        explainer = shap.TreeExplainer(
            estimator
        )

        shap_values = explainer.shap_values(
            transformed_sample
        )

    except Exception as error:

        print(
            "\nTreeExplainer failed."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "\nTrying SHAP Explainer..."
        )

        explainer = shap.Explainer(
            estimator,
            transformed_sample
        )

        shap_values = explainer(
            transformed_sample
        ).values

    shap_values = np.asarray(
        shap_values
    )

    # Handle unexpected dimensions.
    if shap_values.ndim == 3:

        shap_values = shap_values[:, :, 0]

    if shap_values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP value shape: "
            + str(shap_values.shape)
        )

    # --------------------------------------------------------
    # GLOBAL IMPORTANCE
    # --------------------------------------------------------

    print(
        "\n[8/9] Creating global feature importance..."
    )

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    mean_shap = np.mean(
        shap_values,
        axis=0
    )

    global_results = []

    for index, feature_name in enumerate(
        transformed_feature_names
    ):

        global_results.append({

            "transformed_feature":
                feature_name,

            "mean_absolute_shap":
                float(
                    mean_abs_shap[index]
                ),

            "mean_shap":
                float(
                    mean_shap[index]
                )
        })

    global_df = pd.DataFrame(
        global_results
    )

    global_df = global_df.sort_values(
        by="mean_absolute_shap",
        ascending=False
    ).reset_index(
        drop=True
    )

    global_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(global_df) + 1
        )
    )

    global_df.to_csv(
        GLOBAL_CSV_PATH,
        index=False
    )

    # --------------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------------

    category_results = []

    if "category" in shap_sample.columns:

        # Identify all transformed columns that belong
        # to category.
        category_feature_indices = [
            index
            for index, name
            in enumerate(
                transformed_feature_names
            )
            if "category" in name.lower()
        ]

        if category_feature_indices:

            category_shap_magnitude = np.sum(
                np.abs(
                    shap_values[
                        :,
                        category_feature_indices
                    ]
                ),
                axis=1
            )

            category_analysis_df = pd.DataFrame({

                "category":
                    shap_sample[
                        "category"
                    ].astype(str).values,

                "category_shap_magnitude":
                    category_shap_magnitude
            })

            grouped = (
                category_analysis_df
                .groupby("category")
                ["category_shap_magnitude"]
                .agg(
                    [
                        "count",
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max"
                    ]
                )
                .reset_index()
            )

            grouped.columns = [
                "category",
                "sample_count",
                "mean_category_shap",
                "median_category_shap",
                "std_category_shap",
                "min_category_shap",
                "max_category_shap"
            ]

            category_results_df = grouped.sort_values(
                by="mean_category_shap",
                ascending=False
            ).reset_index(
                drop=True
            )

            category_results_df.to_csv(
                CATEGORY_CSV_PATH,
                index=False
            )

            category_results = (
                category_results_df
                .to_dict(
                    orient="records"
                )
            )

    # --------------------------------------------------------
    # LOCAL EXPLANATIONS
    # --------------------------------------------------------

    print(
        "\nCreating local prediction explanations..."
    )

    local_size = min(
        LOCAL_SAMPLE_SIZE,
        len(shap_sample)
    )

    local_indices = np.arange(
        local_size
    )

    local_results = []

    for row_index in local_indices:

        row_shap = shap_values[
            row_index
        ]

        top_indices = np.argsort(
            np.abs(row_shap)
        )[::-1][:5]

        row_data = {

            "sample_index":
                int(row_index),

            "actual":
                safe_float(
                    y_test.loc[
                        shap_sample.index[
                            row_index
                        ]
                    ]
                    if shap_sample.index[
                        row_index
                    ] in y_test.index
                    else np.nan
                ),

            "prediction":
                safe_float(
                    model.predict(
                        shap_sample.iloc[
                            [row_index]
                        ]
                    )[0]
                )
        }

        for rank, feature_index in enumerate(
            top_indices,
            start=1
        ):

            row_data[
                f"feature_{rank}"
            ] = transformed_feature_names[
                feature_index
            ]

            row_data[
                f"shap_value_{rank}"
            ] = safe_float(
                row_shap[
                    feature_index
                ]
            )

        local_results.append(
            row_data
        )

    local_df = pd.DataFrame(
        local_results
    )

    local_df.to_csv(
        LOCAL_CSV_PATH,
        index=False
    )

    # --------------------------------------------------------
    # TOP FEATURES
    # --------------------------------------------------------

    top_features = (
        global_df
        .head(15)
        .to_dict(
            orient="records"
        )
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "analysis":
            "shap_explainability",

        "model":
            best_model,

        "target":
            target,

        "test_rows":
            int(len(X_test)),

        "shap_sample_rows":
            int(len(shap_sample)),

        "local_explanation_rows":
            int(local_size),

        "test_r2":
            float(baseline_r2),

        "transformed_feature_count":
            int(
                len(
                    transformed_feature_names
                )
            ),

        "top_features":
            top_features,

        "category_analysis":
            category_results,

        "interpretation_notes": [

            "Mean absolute SHAP measures average magnitude of feature contribution.",

            "Positive mean SHAP indicates an overall tendency to increase predictions, while negative mean SHAP indicates an overall tendency to decrease predictions.",

            "One-hot encoded categorical levels appear as separate transformed features.",

            "Category-level SHAP results should be interpreted together with the category-only baseline and feature-ablation analysis.",

            "SHAP explains the fitted model's behaviour; it does not establish causal relationships."
        ]
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # PRINT GLOBAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TOP SHAP FEATURES")
    print("=" * 75)

    print(
        global_df.head(15).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # CATEGORY RESULTS
    # --------------------------------------------------------

    if category_results:

        print("\n" + "=" * 75)
        print("CATEGORY SHAP ANALYSIS")
        print("=" * 75)

        print(
            pd.DataFrame(
                category_results
            ).to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}"
            )
        )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(
        f"Global importance:\n{GLOBAL_CSV_PATH}"
    )

    print(
        f"Category analysis:\n{CATEGORY_CSV_PATH}"
    )

    print(
        f"Local explanations:\n{LOCAL_CSV_PATH}"
    )

    print(
        f"JSON report:\n{REPORT_PATH}"
    )

    print("\n" + "=" * 75)
    print("SHAP EXPLAINABILITY ANALYSIS: PASS")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


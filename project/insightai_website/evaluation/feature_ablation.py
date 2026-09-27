
# ============================================================
# INSIGHTAI
# FEATURE ABLATION ANALYSIS
#
# Purpose:
# Measure the incremental predictive contribution of features.
#
# No existing model or project file is modified.
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
    "feature_ablation_results.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_ablation_report.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "profit_margin"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

# These are the features selected by the current model-training
# pipeline.

SELECTED_NUMERIC = [
    "unit_price",
    "revenue"
]

SELECTED_CATEGORICAL = [
    "state",
    "region",
    "country",
    "category",
    "sub_category",
    "product_name"
]


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = [

    {
        "name": "Category Only",
        "features": [
            "category"
        ]
    },

    {
        "name": "Category + Unit Price",
        "features": [
            "category",
            "unit_price"
        ]
    },

    {
        "name": "Category + Revenue",
        "features": [
            "category",
            "revenue"
        ]
    },

    {
        "name": "Category + Unit Price + Revenue",
        "features": [
            "category",
            "unit_price",
            "revenue"
        ]
    },

    {
        "name": "Category + Sub Category",
        "features": [
            "category",
            "sub_category"
        ]
    },

    {
        "name": "Category + Product Name",
        "features": [
            "category",
            "product_name"
        ]
    },

    {
        "name": "Category + Geography",
        "features": [
            "category",
            "state",
            "region",
            "country"
        ]
    },

    {
        "name": "All Categorical",
        "features": [
            "state",
            "region",
            "country",
            "category",
            "sub_category",
            "product_name"
        ]
    },

    {
        "name": "All Selected Features",
        "features": [
            "unit_price",
            "revenue",
            "state",
            "region",
            "country",
            "category",
            "sub_category",
            "product_name"
        ]
    }
]


# ============================================================
# HELPER
# ============================================================

def build_pipeline(features):

    numeric_features = [
        feature
        for feature in features
        if feature in SELECTED_NUMERIC
    ]

    categorical_features = [
        feature
        for feature in features
        if feature in SELECTED_CATEGORICAL
    ]

    transformers = []

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    if categorical_features:

        categorical_pipeline = Pipeline(
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
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    return pipeline


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("INSIGHTAI - FEATURE ABLATION ANALYSIS")
    print("=" * 75)

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n[1/7] Loading dataset...")

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
    # VALIDATE FEATURES
    # --------------------------------------------------------

    print("\n[2/7] Validating columns...")

    required_columns = [
        TARGET_COLUMN
    ]

    for experiment in EXPERIMENTS:

        required_columns.extend(
            experiment["features"]
        )

    required_columns = list(
        dict.fromkeys(
            required_columns
        )
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

    print(
        f"Validated columns: {len(required_columns)}"
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    print("\n[3/7] Preparing dataset...")

    analysis_df = df[
        required_columns
    ].copy()

    analysis_df[TARGET_COLUMN] = pd.to_numeric(
        analysis_df[TARGET_COLUMN],
        errors="coerce"
    )

    # Convert numeric features explicitly.
    for column in SELECTED_NUMERIC:

        if column in analysis_df.columns:

            analysis_df[column] = pd.to_numeric(
                analysis_df[column],
                errors="coerce"
            )

    analysis_df = analysis_df.dropna(
        subset=[
            TARGET_COLUMN
        ]
    )

    print(
        f"Usable rows: {len(analysis_df):,}"
    )

    # --------------------------------------------------------
    # EXACT SAME SPLIT
    # --------------------------------------------------------

    print(
        "\n[4/7] Creating exact train/test split..."
    )

    indices = np.arange(
        len(analysis_df)
    )

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_df = analysis_df.iloc[
        train_indices
    ].copy()

    test_df = analysis_df.iloc[
        test_indices
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    print(
        "\n[5/7] Running feature ablation experiments..."
    )

    results = []

    baseline_r2 = None
    baseline_mae = None
    baseline_rmse = None

    for number, experiment in enumerate(
        EXPERIMENTS,
        start=1
    ):

        name = experiment["name"]

        features = experiment["features"]

        print(
            f"\nExperiment {number}/{len(EXPERIMENTS)}"
        )

        print(
            f"Name: {name}"
        )

        print(
            f"Features: {', '.join(features)}"
        )

        X_train = train_df[
            features
        ]

        X_test = test_df[
            features
        ]

        pipeline = build_pipeline(
            features
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
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

        if baseline_r2 is None:

            baseline_r2 = r2
            baseline_mae = mae
            baseline_rmse = rmse

        r2_change = (
            r2 - baseline_r2
        )

        mae_change = (
            baseline_mae - mae
        )

        rmse_change = (
            baseline_rmse - rmse
        )

        results.append({

            "experiment":
                name,

            "feature_count":
                len(features),

            "features":
                ", ".join(features),

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

            "r2_change_vs_category_only":
                float(r2_change),

            "mae_improvement_vs_category_only":
                float(mae_change),

            "rmse_improvement_vs_category_only":
                float(rmse_change)
        })

        print(
            f"R²:   {r2:.6f}"
        )

        print(
            f"MAE:  {mae:.6f}"
        )

        print(
            f"RMSE: {rmse:.6f}"
        )

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    print(
        "\n[6/7] Creating comparison report..."
    )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SORTING
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="r2",
        ascending=False
    ).reset_index(
        drop=True
    )

    results_df.insert(
        0,
        "rank_by_r2",
        np.arange(
            1,
            len(results_df) + 1
        )
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    results_df.to_csv(
        CSV_PATH,
        index=False
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    best_r2_row = results_df.loc[
        results_df["r2"].idxmax()
    ]

    best_mae_row = results_df.loc[
        results_df["mae"].idxmin()
    ]

    best_rmse_row = results_df.loc[
        results_df["rmse"].idxmin()
    ]

    report = {

        "analysis":
            "feature_ablation",

        "target":
            TARGET_COLUMN,

        "dataset_rows":
            int(len(df)),

        "usable_rows":
            int(len(analysis_df)),

        "training_rows":
            int(len(train_df)),

        "test_rows":
            int(len(test_df)),

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "baseline":
            results[0],

        "best_r2_experiment":
            best_r2_row.to_dict(),

        "best_mae_experiment":
            best_mae_row.to_dict(),

        "best_rmse_experiment":
            best_rmse_row.to_dict(),

        "experiments":
            results
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
    # PRINT FINAL TABLE
    # --------------------------------------------------------

    print(
        "\n[7/7] Final results"
    )

    print("\n" + "=" * 75)
    print("FEATURE ABLATION RESULTS")
    print("=" * 75)

    display_columns = [
        "rank_by_r2",
        "experiment",
        "feature_count",
        "r2",
        "mae",
        "rmse",
        "r2_change_vs_category_only",
        "mae_improvement_vs_category_only",
        "rmse_improvement_vs_category_only"
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
    # BEST RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("BEST R² EXPERIMENT")
    print("=" * 75)

    print(
        f"Experiment: "
        f"{best_r2_row['experiment']}"
    )

    print(
        f"R²: "
        f"{best_r2_row['r2']:.6f}"
    )

    print(
        f"MAE: "
        f"{best_r2_row['mae']:.6f}"
    )

    print(
        f"RMSE: "
        f"{best_r2_row['rmse']:.6f}"
    )

    print("\n" + "=" * 75)
    print("BEST MAE EXPERIMENT")
    print("=" * 75)

    print(
        f"Experiment: "
        f"{best_mae_row['experiment']}"
    )

    print(
        f"MAE: "
        f"{best_mae_row['mae']:.6f}"
    )

    print(
        f"R²: "
        f"{best_mae_row['r2']:.6f}"
    )

    print(
        f"RMSE: "
        f"{best_mae_row['rmse']:.6f}"
    )

    print("\n" + "=" * 75)
    print("BEST RMSE EXPERIMENT")
    print("=" * 75)

    print(
        f"Experiment: "
        f"{best_rmse_row['experiment']}"
    )

    print(
        f"RMSE: "
        f"{best_rmse_row['rmse']:.6f}"
    )

    print(
        f"R²: "
        f"{best_rmse_row['r2']:.6f}"
    )

    print(
        f"MAE: "
        f"{best_rmse_row['mae']:.6f}"
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(
        f"CSV:\n{CSV_PATH}"
    )

    print(
        f"JSON:\n{REPORT_PATH}"
    )

    print(
        "\nFEATURE ABLATION ANALYSIS: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


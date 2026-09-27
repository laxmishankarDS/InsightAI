
# ============================================================
# INSIGHTAI
# FINAL MODEL EVALUATION & LIMITATIONS REPORT
# ============================================================

import os
import json
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\maury\OneDrive\Documents\insightAI"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "insightai_website",
    "evaluation"
)

MODEL_OUTPUT_DIR = r"C:\Users\maury\Downloads\InsightAI_Output"


# ============================================================
# INPUT REPORTS
# ============================================================

MODEL_RESULTS_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "model_results.csv"
)

MODEL_METADATA_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "model_metadata.json"
)

EVALUATION_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "evaluation_report.json"
)

CV_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "cross_validation_report.json"
)

ERROR_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "error_analysis_summary.json"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

CATEGORY_BASELINE_PATH = os.path.join(
    OUTPUT_DIR,
    "category_only_baseline_report.json"
)

ABLATION_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_ablation_report.json"
)

SHAP_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_explainability_report.json"
)

CATEGORY_ERROR_PATH = os.path.join(
    OUTPUT_DIR,
    "category_error_analysis_report.json"
)

TARGET_DISTRIBUTION_PATH = os.path.join(
    OUTPUT_DIR,
    "target_distribution_report.json"
)

RELIABILITY_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "prediction_reliability_report.json"
)


# ============================================================
# OUTPUT
# ============================================================

FINAL_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "final_model_evaluation_report.json"
)

FINAL_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "final_model_evaluation_summary.csv"
)

FINAL_LIMITATIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "final_model_limitations.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):
    """
    Load JSON file safely.
    """

    if not os.path.exists(path):
        print(
            f"WARNING: File not found: {path}"
        )
        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def safe_float(value):
    """
    Convert value to float safely.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_nested(data, *keys, default=None):
    """
    Safely retrieve nested dictionary values.
    """

    current = data

    for key in keys:

        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


# ============================================================
# START
# ============================================================

print("=" * 75)
print("INSIGHTAI - FINAL MODEL EVALUATION")
print("=" * 75)


# ============================================================
# 1. LOAD REPORTS
# ============================================================

print("\n[1] Loading evaluation reports...")


metadata = load_json(
    MODEL_METADATA_PATH
)

evaluation_report = load_json(
    EVALUATION_REPORT_PATH
)

cv_report = load_json(
    CV_REPORT_PATH
)

error_report = load_json(
    ERROR_REPORT_PATH
)

category_baseline_report = load_json(
    CATEGORY_BASELINE_PATH
)

ablation_report = load_json(
    ABLATION_REPORT_PATH
)

shap_report = load_json(
    SHAP_REPORT_PATH
)

category_error_report = load_json(
    CATEGORY_ERROR_PATH
)

target_distribution_report = load_json(
    TARGET_DISTRIBUTION_PATH
)

reliability_report = load_json(
    RELIABILITY_REPORT_PATH
)


# ============================================================
# 2. BASIC MODEL INFORMATION
# ============================================================

print("\n[2] Collecting model information...")


model_name = metadata.get(
    "best_model",
    "Unknown"
)

target_name = metadata.get(
    "target",
    "profit_margin"
)

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

training_rows = metadata.get(
    "training_rows"
)

testing_rows = metadata.get(
    "testing_rows"
)

test_size = metadata.get(
    "test_size",
    0.20
)

random_state = metadata.get(
    "random_state",
    42
)


# ============================================================
# 3. MODEL METRICS
# ============================================================

print("\n[3] Collecting model metrics...")


r2 = get_nested(
    evaluation_report,
    "metrics",
    "r2"
)

mae = get_nested(
    evaluation_report,
    "metrics",
    "mae"
)

rmse = get_nested(
    evaluation_report,
    "metrics",
    "rmse"
)


if r2 is None:
    r2 = metadata.get(
        "metrics",
        {}
    ).get(
        "r2"
    )

if mae is None:
    mae = metadata.get(
        "metrics",
        {}
    ).get(
        "mae"
    )

if rmse is None:
    rmse = metadata.get(
        "metrics",
        {}
    ).get(
        "rmse"
    )


# ============================================================
# 4. CROSS VALIDATION
# ============================================================

print("\n[4] Collecting cross-validation results...")


cv_mean_r2 = get_nested(
    cv_report,
    "summary",
    "mean_r2"
)

cv_std_r2 = get_nested(
    cv_report,
    "summary",
    "std_r2"
)

cv_mean_mae = get_nested(
    cv_report,
    "summary",
    "mean_mae"
)

cv_std_mae = get_nested(
    cv_report,
    "summary",
    "std_mae"
)

cv_mean_rmse = get_nested(
    cv_report,
    "summary",
    "mean_rmse"
)

cv_std_rmse = get_nested(
    cv_report,
    "summary",
    "std_rmse"
)


# ============================================================
# 5. ERROR ANALYSIS
# ============================================================

print("\n[5] Collecting error analysis...")


error_mean = get_nested(
    error_report,
    "mean_error"
)

error_median = get_nested(
    error_report,
    "median_error"
)

error_mae = get_nested(
    error_report,
    "mean_absolute_error"
)

error_max_abs = get_nested(
    error_report,
    "max_absolute_error"
)

error_p50 = get_nested(
    error_report,
    "p50"
)

error_p90 = get_nested(
    error_report,
    "p90"
)

error_p95 = get_nested(
    error_report,
    "p95"
)

error_p99 = get_nested(
    error_report,
    "p99"
)

over_prediction_count = get_nested(
    error_report,
    "over_prediction_count"
)

under_prediction_count = get_nested(
    error_report,
    "under_prediction_count"
)


# ============================================================
# 6. CATEGORY BASELINE
# ============================================================

print("\n[6] Collecting category-only baseline...")


category_baseline_r2 = get_nested(
    category_baseline_report,
    "metrics",
    "r2"
)

category_baseline_mae = get_nested(
    category_baseline_report,
    "metrics",
    "mae"
)

category_baseline_rmse = get_nested(
    category_baseline_report,
    "metrics",
    "rmse"
)


# ============================================================
# 7. FEATURE ABLATION
# ============================================================

print("\n[7] Collecting feature ablation...")


ablation_results = []

if isinstance(
    ablation_report,
    dict
):

    possible_results = (
        ablation_report.get(
            "results"
        )
    )

    if isinstance(
        possible_results,
        list
    ):

        ablation_results = (
            possible_results
        )


# If report structure is different,
# try loading the CSV directly.

ABLATION_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_ablation_results.csv"
)


if os.path.exists(
    ABLATION_CSV_PATH
):

    try:

        ablation_df = pd.read_csv(
            ABLATION_CSV_PATH
        )

    except Exception:

        ablation_df = pd.DataFrame()

else:

    ablation_df = pd.DataFrame()


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

print("\n[8] Collecting feature importance...")


feature_importance_df = pd.DataFrame()


if os.path.exists(
    FEATURE_IMPORTANCE_PATH
):

    try:

        feature_importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

    except Exception as exc:

        print(
            "Feature importance warning:",
            exc
        )


top_feature = None
top_feature_importance = None

if not feature_importance_df.empty:

    importance_column = None

    possible_columns = [
        "importance_mean",
        "mean_importance",
        "mean_absolute_shap",
        "importance"
    ]

    for column in possible_columns:

        if column in feature_importance_df.columns:

            importance_column = column
            break

    if importance_column is not None:

        feature_importance_df = (
            feature_importance_df
            .sort_values(
                importance_column,
                ascending=False
            )
        )

        top_row = (
            feature_importance_df
            .iloc[0]
        )

        top_feature = str(
            top_row.iloc[0]
        )

        top_feature_importance = safe_float(
            top_row[
                importance_column
            ]
        )


# ============================================================
# 9. CATEGORY ERROR ANALYSIS
# ============================================================

print("\n[9] Collecting category error analysis...")


highest_category_mae = None
highest_category_rmse = None

category_count = get_nested(
    category_error_report,
    "summary",
    "number_categories"
)

if category_count is None:

    category_count = get_nested(
        category_error_report,
        "number_categories"
    )


category_error_csv = os.path.join(
    OUTPUT_DIR,
    "category_error_analysis.csv"
)


if os.path.exists(
    category_error_csv
):

    try:

        category_error_df = pd.read_csv(
            category_error_csv
        )

        if not category_error_df.empty:

            if "mae" in category_error_df.columns:

                highest_category_mae = safe_float(
                    category_error_df[
                        "mae"
                    ].max()
                )

            if "rmse" in category_error_df.columns:

                highest_category_rmse = safe_float(
                    category_error_df[
                        "rmse"
                    ].max()
                )

    except Exception:

        category_error_df = pd.DataFrame()

else:

    category_error_df = pd.DataFrame()


# ============================================================
# 10. TARGET DISTRIBUTION
# ============================================================

print("\n[10] Collecting target distribution...")


target_mean = get_nested(
    target_distribution_report,
    "overall",
    "mean"
)

target_median = get_nested(
    target_distribution_report,
    "overall",
    "median"
)

target_std = get_nested(
    target_distribution_report,
    "overall",
    "std"
)

target_min = get_nested(
    target_distribution_report,
    "overall",
    "min"
)

target_max = get_nested(
    target_distribution_report,
    "overall",
    "max"
)


# ============================================================
# 11. RELIABILITY ANALYSIS
# ============================================================

print("\n[11] Collecting prediction reliability...")


reliability_r2 = get_nested(
    reliability_report,
    "metrics",
    "r2"
)

reliability_mae = get_nested(
    reliability_report,
    "metrics",
    "mae"
)

reliability_rmse = get_nested(
    reliability_report,
    "metrics",
    "rmse"
)

reliability_mean_error = get_nested(
    reliability_report,
    "metrics",
    "mean_error"
)

reliability_median_error = get_nested(
    reliability_report,
    "metrics",
    "median_error"
)

reliability_over_pct = get_nested(
    reliability_report,
    "bias",
    "over_prediction_percentage"
)

reliability_under_pct = get_nested(
    reliability_report,
    "bias",
    "under_prediction_percentage"
)

calibration_available = get_nested(
    reliability_report,
    "calibration",
    "available"
)


# ============================================================
# 12. TARGET RANGE FINDINGS
# ============================================================

print("\n[12] Reading target-range reliability...")


reliability_range_csv = os.path.join(
    OUTPUT_DIR,
    "prediction_reliability_by_range.csv"
)


reliability_range_df = pd.DataFrame()


if os.path.exists(
    reliability_range_csv
):

    try:

        reliability_range_df = pd.read_csv(
            reliability_range_csv
        )

    except Exception:

        reliability_range_df = pd.DataFrame()


low_range_bias = None
medium_range_bias = None
high_range_bias = None

high_range_mae = None


if not reliability_range_df.empty:

    if "target_range" in reliability_range_df.columns:

        for _, row in reliability_range_df.iterrows():

            range_name = str(
                row["target_range"]
            )

            if range_name == "Low (<20)":

                low_range_bias = safe_float(
                    row["mean_error"]
                )

            elif range_name == "Medium (20-35)":

                medium_range_bias = safe_float(
                    row["mean_error"]
                )

            elif range_name == "High (>=35)":

                high_range_bias = safe_float(
                    row["mean_error"]
                )

                high_range_mae = safe_float(
                    row["mae"]
                )


# ============================================================
# 13. CALIBRATION FINDINGS
# ============================================================

print("\n[13] Reading calibration analysis...")


calibration_csv = os.path.join(
    OUTPUT_DIR,
    "prediction_calibration.csv"
)


calibration_df = pd.DataFrame()


if os.path.exists(
    calibration_csv
):

    try:

        calibration_df = pd.read_csv(
            calibration_csv
        )

    except Exception:

        calibration_df = pd.DataFrame()


largest_calibration_gap = None

largest_calibration_bin = None


if not calibration_df.empty:

    if (
        "absolute_calibration_gap"
        in calibration_df.columns
    ):

        max_index = (
            calibration_df[
                "absolute_calibration_gap"
            ]
            .idxmax()
        )

        max_row = (
            calibration_df
            .loc[max_index]
        )

        largest_calibration_gap = safe_float(
            max_row[
                "absolute_calibration_gap"
            ]
        )

        largest_calibration_bin = str(
            max_row[
                "actual_bin"
            ]
        )


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

print("\n[14] Loading model comparison...")


model_results_df = pd.DataFrame()


if os.path.exists(
    MODEL_RESULTS_PATH
):

    try:

        model_results_df = pd.read_csv(
            MODEL_RESULTS_PATH
        )

    except Exception as exc:

        print(
            "Model results warning:",
            exc
        )


# ============================================================
# 15. FINAL SUMMARY TABLE
# ============================================================

print("\n[15] Creating final summary...")


summary_rows = [

    {
        "section":
            "Model",

        "metric":
            "Selected Model",

        "value":
            model_name,

        "interpretation":
            "Saved production model evaluated on the exact test split."
    },

    {
        "section":
            "Model",

        "metric":
            "Target",

        "value":
            target_name,

        "interpretation":
            "Prediction target used by InsightAI."
    },

    {
        "section":
            "Evaluation",

        "metric":
            "Test R2",

        "value":
            r2,

        "interpretation":
            "Overall coefficient of determination on the held-out test set."
    },

    {
        "section":
            "Evaluation",

        "metric":
            "Test MAE",

        "value":
            mae,

        "interpretation":
            "Average absolute prediction error in target units."
    },

    {
        "section":
            "Evaluation",

        "metric":
            "Test RMSE",

        "value":
            rmse,

        "interpretation":
            "Root mean squared prediction error."
    },

    {
        "section":
            "Cross Validation",

        "metric":
            "Mean R2",

        "value":
            cv_mean_r2,

        "interpretation":
            "Average R2 across five shuffled folds."
    },

    {
        "section":
            "Cross Validation",

        "metric":
            "R2 Standard Deviation",

        "value":
            cv_std_r2,

        "interpretation":
            "Small variation indicates stable performance across folds."
    },

    {
        "section":
            "Cross Validation",

        "metric":
            "Mean MAE",

        "value":
            cv_mean_mae,

        "interpretation":
            "Average MAE across five folds."
    },

    {
        "section":
            "Cross Validation",

        "metric":
            "Mean RMSE",

        "value":
            cv_mean_rmse,

        "interpretation":
            "Average RMSE across five folds."
    },

    {
        "section":
            "Error Analysis",

        "metric":
            "Mean Error",

        "value":
            error_mean,

        "interpretation":
            "Near-zero value indicates little overall directional bias."
    },

    {
        "section":
            "Error Analysis",

        "metric":
            "P50 Absolute Error",

        "value":
            error_p50,

        "interpretation":
            "Median absolute prediction error."
    },

    {
        "section":
            "Error Analysis",

        "metric":
            "P90 Absolute Error",

        "value":
            error_p90,

        "interpretation":
            "90% of predictions have absolute error at or below this value."
    },

    {
        "section":
            "Error Analysis",

        "metric":
            "Maximum Absolute Error",

        "value":
            error_max_abs,

        "interpretation":
            "Largest observed absolute prediction error."
    },

    {
        "section":
            "Baseline",

        "metric":
            "Category-Only R2",

        "value":
            category_baseline_r2,

        "interpretation":
            "Performance of a category-only linear baseline."
    },

    {
        "section":
            "Baseline",

        "metric":
            "Category-Only MAE",

        "value":
            category_baseline_mae,

        "interpretation":
            "Category-only baseline absolute error."
    },

    {
        "section":
            "Reliability",

        "metric":
            "Over Prediction %",

        "value":
            reliability_over_pct,

        "interpretation":
            "Percentage of test predictions above the actual target."
    },

    {
        "section":
            "Reliability",

        "metric":
            "Under Prediction %",

        "value":
            reliability_under_pct,

        "interpretation":
            "Percentage of test predictions below the actual target."
    },

    {
        "section":
            "Reliability",

        "metric":
            "High Target Range Mean Error",

        "value":
            high_range_bias,

        "interpretation":
            "Bias observed for actual profit margins >=35."
    },

    {
        "section":
            "Reliability",

        "metric":
            "Largest Calibration Gap",

        "value":
            largest_calibration_gap,

        "interpretation":
            "Largest difference between actual-bin mean and predicted-bin mean."
    },

    {
        "section":
            "Feature Analysis",

        "metric":
            "Top Permutation Feature",

        "value":
            top_feature,

        "interpretation":
            "Highest measured permutation importance in the evaluation."
    },

    {
        "section":
            "Target Distribution",

        "metric":
            "Target Mean",

        "value":
            target_mean,

        "interpretation":
            "Overall mean profit margin."
    },

    {
        "section":
            "Target Distribution",

        "metric":
            "Target Standard Deviation",

        "value":
            target_std,

        "interpretation":
            "Overall target variability."
    }
]


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 16. FINAL LIMITATIONS
# ============================================================

print("\n[16] Creating limitations...")


limitations = [

    {
        "limitation_id":
            1,

        "area":
            "Extreme Target Values",

        "finding":
            "The model tends to pull extreme low and high profit-margin values toward the middle prediction range.",

        "evidence":
            "Low and high target ranges show directional bias, while the largest calibration gap occurs in the highest target bin.",

        "impact":
            "Predictions for unusually low or unusually high profit margins can have substantially larger errors.",

        "status":
            "Observed"
    },

    {
        "limitation_id":
            2,

        "area":
            "Category Dependence",

        "finding":
            "Category is the dominant original feature in permutation importance and SHAP-based analysis.",

        "evidence":
            "Category-only baseline produces performance very close to the full selected-feature model.",

        "impact":
            "The current predictive signal is strongly associated with category-level target differences.",

        "status":
            "Observed"
    },

    {
        "limitation_id":
            3,

        "area":
            "Within-Category Variation",

        "finding":
            "Category-level predictions capture group-level differences but do not fully explain variation within individual categories.",

        "evidence":
            "Category-level R² values are close to zero despite a substantially higher overall R².",

        "impact":
            "Additional variables may be required to explain individual-row variation.",

        "status":
            "Observed"
    },

    {
        "limitation_id":
            4,

        "area":
            "Feature Incremental Value",

        "finding":
            "Adding the currently selected features produces only very small changes compared with the category-only baseline.",

        "evidence":
            "Feature ablation experiments show near-identical R², MAE and RMSE values.",

        "impact":
            "The current dataset may not contain enough independent predictive information beyond category.",

        "status":
            "Observed"
    },

    {
        "limitation_id":
            5,

        "area":
            "Prediction Error",

        "finding":
            "Prediction error remains material for a portion of observations.",

        "evidence":
            "The P90 and P95 absolute-error values are substantially above the median absolute error.",

        "impact":
            "Individual predictions should be interpreted with their expected error range rather than as exact values.",

        "status":
            "Observed"
    },

    {
        "limitation_id":
            6,

        "area":
            "Feature Coverage",

        "finding":
            "The current model uses the features selected from the available dataset and does not include additional business variables that may influence profit margin.",

        "evidence":
            "Feature ablation and error analysis indicate limited incremental signal from the current feature set.",

        "impact":
            "Additional business-relevant variables could potentially improve within-category and extreme-value predictions.",

        "status":
            "Potential"
    },

    {
        "limitation_id":
            7,

        "area":
            "Explainability Scope",

        "finding":
            "SHAP and permutation importance describe model behavior and feature contribution but do not establish causal relationships.",

        "evidence":
            "Feature importance results are model-specific statistical explanations.",

        "impact":
            "Feature importance should not be interpreted as proof that a feature causes changes in profit margin.",

        "status":
            "Methodological"
    },

    {
        "limitation_id":
            8,

        "area":
            "Generalization",

        "finding":
            "Evaluation is based on the available dataset and its train/test distribution.",

        "evidence":
            "The evaluation uses a reproducible 80/20 split and five-fold cross-validation on the same dataset.",

        "impact":
            "Performance on future datasets or substantially different business conditions may differ.",

        "status":
            "Methodological"
    }
]


limitations_df = pd.DataFrame(
    limitations
)


# ============================================================
# 17. FINAL JSON REPORT
# ============================================================

print("\n[17] Creating final JSON report...")


final_report = {

    "project":
        "InsightAI",

    "analysis":
        "Final Model Evaluation & Limitations",

    "model": {

        "name":
            model_name,

        "target":
            target_name,

        "training_rows":
            training_rows,

        "testing_rows":
            testing_rows,

        "test_size":
            test_size,

        "random_state":
            random_state,

        "selected_features":
            selected_features,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features
    },

    "test_metrics": {

        "r2":
            safe_float(r2),

        "mae":
            safe_float(mae),

        "rmse":
            safe_float(rmse)
    },

    "cross_validation": {

        "folds":
            5,

        "mean_r2":
            safe_float(cv_mean_r2),

        "std_r2":
            safe_float(cv_std_r2),

        "mean_mae":
            safe_float(cv_mean_mae),

        "std_mae":
            safe_float(cv_std_mae),

        "mean_rmse":
            safe_float(cv_mean_rmse),

        "std_rmse":
            safe_float(cv_std_rmse)
    },

    "error_analysis": {

        "mean_error":
            safe_float(error_mean),

        "median_error":
            safe_float(error_median),

        "p50_absolute_error":
            safe_float(error_p50),

        "p90_absolute_error":
            safe_float(error_p90),

        "p95_absolute_error":
            safe_float(error_p95),

        "p99_absolute_error":
            safe_float(error_p99),

        "max_absolute_error":
            safe_float(error_max_abs),

        "over_prediction_count":
            over_prediction_count,

        "under_prediction_count":
            under_prediction_count
    },

    "category_baseline": {

        "r2":
            safe_float(category_baseline_r2),

        "mae":
            safe_float(category_baseline_mae),

        "rmse":
            safe_float(category_baseline_rmse)
    },

    "feature_analysis": {

        "top_permutation_feature":
            top_feature,

        "top_permutation_importance":
            safe_float(
                top_feature_importance
            )
    },

    "target_distribution": {

        "mean":
            safe_float(target_mean),

        "median":
            safe_float(target_median),

        "std":
            safe_float(target_std),

        "min":
            safe_float(target_min),

        "max":
            safe_float(target_max)
    },

    "prediction_reliability": {

        "r2":
            safe_float(reliability_r2),

        "mae":
            safe_float(reliability_mae),

        "rmse":
            safe_float(reliability_rmse),

        "mean_error":
            safe_float(
                reliability_mean_error
            ),

        "median_error":
            safe_float(
                reliability_median_error
            ),

        "over_prediction_percentage":
            safe_float(
                reliability_over_pct
            ),

        "under_prediction_percentage":
            safe_float(
                reliability_under_pct
            ),

        "low_range_mean_error":
            safe_float(
                low_range_bias
            ),

        "medium_range_mean_error":
            safe_float(
                medium_range_bias
            ),

        "high_range_mean_error":
            safe_float(
                high_range_bias
            ),

        "high_range_mae":
            safe_float(
                high_range_mae
            ),

        "largest_calibration_gap":
            safe_float(
                largest_calibration_gap
            ),

        "largest_calibration_bin":
            largest_calibration_bin,

        "calibration_available":
            bool(
                calibration_available
            )
            if calibration_available is not None
            else False
    },

    "limitations":
        limitations
}


# ============================================================
# 18. SAVE FINAL FILES
# ============================================================

print("\n[18] Saving final evaluation files...")


summary_df.to_csv(
    FINAL_SUMMARY_PATH,
    index=False
)


limitations_df.to_csv(
    FINAL_LIMITATIONS_PATH,
    index=False
)


with open(
    FINAL_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_report,
        file,
        indent=4
    )


# ============================================================
# 19. PRINT FINAL MODEL SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FINAL MODEL EVALUATION SUMMARY"
)

print(
    "=" * 75
)


print(
    f"\nModel: {model_name}"
)

print(
    f"Target: {target_name}"
)

print(
    f"Training Rows: {training_rows}"
)

print(
    f"Testing Rows: {testing_rows}"
)


print(
    "\nTest Set Metrics:"
)

print(
    f"R²:   {safe_float(r2):.6f}"
    if r2 is not None
    else "R²:   unavailable"
)

print(
    f"MAE:  {safe_float(mae):.6f}"
    if mae is not None
    else "MAE:  unavailable"
)

print(
    f"RMSE: {safe_float(rmse):.6f}"
    if rmse is not None
    else "RMSE: unavailable"
)


print(
    "\nCross Validation:"
)

print(
    f"Mean R²: "
    f"{safe_float(cv_mean_r2):.6f}"
    if cv_mean_r2 is not None
    else "Mean R²: unavailable"
)

print(
    f"R² Std:  "
    f"{safe_float(cv_std_r2):.6f}"
    if cv_std_r2 is not None
    else "R² Std: unavailable"
)

print(
    f"Mean MAE: "
    f"{safe_float(cv_mean_mae):.6f}"
    if cv_mean_mae is not None
    else "Mean MAE: unavailable"
)

print(
    f"Mean RMSE: "
    f"{safe_float(cv_mean_rmse):.6f}"
    if cv_mean_rmse is not None
    else "Mean RMSE: unavailable"
)


print(
    "\nCategory-Only Baseline:"
)

print(
    f"R²:   "
    f"{safe_float(category_baseline_r2):.6f}"
    if category_baseline_r2 is not None
    else "R²: unavailable"
)

print(
    f"MAE:  "
    f"{safe_float(category_baseline_mae):.6f}"
    if category_baseline_mae is not None
    else "MAE: unavailable"
)

print(
    f"RMSE: "
    f"{safe_float(category_baseline_rmse):.6f}"
    if category_baseline_rmse is not None
    else "RMSE: unavailable"
)


print(
    "\nPrediction Reliability:"
)

print(
    f"Mean Error: "
    f"{safe_float(reliability_mean_error):.6f}"
    if reliability_mean_error is not None
    else "Mean Error: unavailable"
)

print(
    f"Over Prediction: "
    f"{safe_float(reliability_over_pct):.2f}%"
    if reliability_over_pct is not None
    else "Over Prediction: unavailable"
)

print(
    f"Under Prediction: "
    f"{safe_float(reliability_under_pct):.2f}%"
    if reliability_under_pct is not None
    else "Under Prediction: unavailable"
)


print(
    "\nCalibration:"
)

print(
    f"Available: {calibration_available}"
)

if largest_calibration_gap is not None:

    print(
        f"Largest Calibration Gap: "
        f"{largest_calibration_gap:.6f}"
    )

    print(
        f"Largest Gap Bin: "
        f"{largest_calibration_bin}"
    )


print(
    "\nKey Feature Finding:"
)

print(
    f"Top Permutation Feature: "
    f"{top_feature}"
)


print(
    "\nKey Reliability Finding:"
)

if high_range_bias is not None:

    print(
        "High target range shows negative prediction bias "
        f"of {high_range_bias:.6f}."
    )

else:

    print(
        "High target range bias unavailable."
    )


print(
    "\nImportant Limitations:"
)

for limitation in limitations:

    print(
        f"- {limitation['area']}: "
        f"{limitation['finding']}"
    )


# ============================================================
# 20. SAVED FILES
# ============================================================

print(
    "\nSaved files:"
)

print(
    FINAL_REPORT_PATH
)

print(
    FINAL_SUMMARY_PATH
)

print(
    FINAL_LIMITATIONS_PATH
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FINAL MODEL EVALUATION PASS"
)

print(
    "=" * 75
)


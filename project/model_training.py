# ============================================================
# INSIGHTAI - DYNAMIC MODEL TRAINING
# FINAL FORECASTING-READY VERSION
# ============================================================

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsRegressor

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

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

MAX_CATEGORICAL_UNIQUE = 100
REDUNDANT_FEATURE_CORRELATION = 0.9999

MIN_NUMERIC_CONVERSION_RATIO = 0.95
MIN_DATE_CONVERSION_RATIO = 0.70

ENV_TARGET = os.getenv(
    "INSIGHTAI_TARGET",
    ""
).strip().lower()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================
# DATE DETECTION
# ============================================================

def detect_date_columns(df):

    date_columns = []

    date_keywords = [
        "date",
        "datetime",
        "timestamp",
        "time_stamp"
    ]

    engineered_suffixes = (
        "_year",
        "_month",
        "_quarter",
        "_day",
        "_day_of_week",
        "_week",
        "_week_of_year"
    )

    for column in df.columns:

        name = str(column).lower()

        if name.endswith(engineered_suffixes):
            continue

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            date_columns.append(column)
            continue

        if not any(
            keyword in name
            for keyword in date_keywords
        ):
            continue

        converted = pd.to_datetime(
            df[column],
            format="mixed",
            errors="coerce"
        )

        non_empty = df[column].notna().sum()

        if non_empty == 0:
            continue

        ratio = (
            converted.notna().sum()
            / non_empty
        )

        if ratio >= MIN_DATE_CONVERSION_RATIO:

            df[column] = converted

            date_columns.append(column)

    return date_columns


# ============================================================
# NUMERIC-LIKE DETECTION
# ============================================================

def convert_numeric_like_columns(df):

    df = df.copy()

    converted_columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_empty = series.notna().sum()

        if non_empty == 0:
            continue

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

        ratio = (
            converted.notna().sum()
            / non_empty
        )

        if ratio >= MIN_NUMERIC_CONVERSION_RATIO:

            df[column] = converted

            converted_columns.append(column)

    return df, converted_columns


# ============================================================
# ID-LIKE COLUMN DETECTION
# ============================================================

def is_id_like_column(series, column_name):

    name = str(column_name).lower()

    exact_ids = {
        "id",
        "uuid",
        "guid",
        "identifier",
        "order_id",
        "customer_id",
        "transaction_id",
        "invoice_id",
        "record_id",
        "user_id",
        "product_id"
    }

    if name in exact_ids:
        return True

    if name.endswith("_id"):
        return True

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):

        unique_ratio = (
            non_null.nunique()
            / len(non_null)
        )

        if unique_ratio >= 0.98:
            return True

    if pd.api.types.is_numeric_dtype(series):

        unique_count = non_null.nunique()
        total_count = len(non_null)

        if (
            unique_count >= 100
            and unique_count / total_count >= 0.98
        ):

            values = (
                pd.Series(non_null)
                .sort_values()
                .reset_index(drop=True)
            )

            differences = (
                values.diff()
                .dropna()
            )

            if (
                len(differences) > 0
                and (differences == 1).mean() >= 0.95
            ):
                return True

    return False


# ============================================================
# AVAILABLE TARGETS
# ============================================================

def get_available_targets(df):

    targets = []

    for column in df.columns:

        if is_id_like_column(
            df[column],
            column
        ):
            continue

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid_count = numeric.notna().sum()

        unique_count = numeric.nunique(
            dropna=True
        )

        if (
            valid_count > 0
            and unique_count >= 2
        ):
            targets.append(column)

    return targets


# ============================================================
# TARGET DETECTION
# ============================================================

def detect_target(df):

    available_targets = get_available_targets(df)

    if not available_targets:

        raise ValueError(
            "No suitable numeric target found."
        )

    if ENV_TARGET:

        if ENV_TARGET not in df.columns:

            raise ValueError(
                f"\nTarget '{ENV_TARGET}' not found.\n"
                f"Available columns:\n"
                f"{df.columns.tolist()}"
            )

        if ENV_TARGET not in available_targets:

            raise ValueError(
                f"\n'{ENV_TARGET}' is not a suitable "
                "numeric regression target."
            )

        return (
            ENV_TARGET,
            "environment_variable"
        )

    print(
        "\n============================================================"
    )

    print(
        "AVAILABLE NUMERIC TARGETS"
    )

    print(
        "============================================================"
    )

    for index, column in enumerate(
        available_targets,
        start=1
    ):

        print(
            f"{index}. {column}"
        )

    print(
        "\nYou can select a target by:"
    )

    print(
        "1. Number"
    )

    print(
        "2. Column name"
    )

    print(
        "3. Press ENTER for automatic selection"
    )

    choice = input(
        "\nEnter target: "
    ).strip().lower()

    if choice:

        if choice.isdigit():

            index = int(choice) - 1

            if (
                0 <= index
                < len(available_targets)
            ):

                return (
                    available_targets[index],
                    "user_selection"
                )

        if choice in available_targets:

            return (
                choice,
                "user_selection"
            )

        raise ValueError(
            f"Please select one of: "
            f"{available_targets}"
        )

    preferred_names = [
        "target",
        "label",
        "outcome",
        "sales",
        "demand",
        "revenue",
        "profit",
        "amount",
        "price",
        "score",
        "value",
        "quantity"
    ]

    for name in preferred_names:

        if name in available_targets:

            return (
                name,
                "automatic_detection"
            )

    return (
        available_targets[-1],
        "numeric_fallback"
    )


# ============================================================
# TARGET LEAKAGE DETECTION
# ============================================================

def get_target_leakage_columns(
    target,
    columns
):

    target = str(target).lower().strip()

    column_map = {
        str(column).lower(): column
        for column in columns
    }

    leakage_columns = set()

    def add_if_exists(names):

        for name in names:

            if name in column_map:

                leakage_columns.add(
                    column_map[name]
                )

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    if target == "revenue":

        add_if_exists([
            "quantity",
            "qty",
            "units",
            "unit_price",
            "price",
            "price_each",
            "profit",
            "profit_margin",
            "revenue_per_unit"
        ])

    # --------------------------------------------------------
    # PROFIT
    # --------------------------------------------------------

    elif target == "profit":

        add_if_exists([
            "profit_margin"
        ])

    # --------------------------------------------------------
    # PROFIT MARGIN
    # --------------------------------------------------------

    elif target == "profit_margin":

        add_if_exists([
            "profit",
            "revenue"
        ])

    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    elif target in [
        "quantity",
        "qty",
        "units"
    ]:

        add_if_exists([
            "revenue_per_unit"
        ])

    # --------------------------------------------------------
    # UNIT PRICE
    # --------------------------------------------------------

    elif target in [
        "unit_price",
        "price",
        "price_each"
    ]:

        add_if_exists([
            "revenue",
            "revenue_per_unit"
        ])

    # --------------------------------------------------------
    # REVENUE PER UNIT
    # --------------------------------------------------------

    elif target == "revenue_per_unit":

        add_if_exists([
            "revenue",
            "quantity",
            "unit_price"
        ])

    return sorted(
        leakage_columns
    )


# ============================================================
# REDUNDANT NUMERIC FEATURES
# ============================================================

def remove_redundant_numeric_features(
    df,
    features
):

    if len(features) <= 1:

        return (
            features.copy(),
            []
        )

    temp = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    correlation_matrix = temp.corr()

    redundant = set()

    for i in range(len(features)):

        first = features[i]

        if first in redundant:
            continue

        for j in range(
            i + 1,
            len(features)
        ):

            second = features[j]

            if second in redundant:
                continue

            correlation = (
                correlation_matrix.loc[
                    first,
                    second
                ]
            )

            if (
                pd.notna(correlation)
                and abs(correlation)
                >= REDUNDANT_FEATURE_CORRELATION
            ):

                redundant.add(
                    second
                )

    cleaned = [
        column
        for column in features
        if column not in redundant
    ]

    return (
        cleaned,
        sorted(redundant)
    )


# ============================================================
# ONE-HOT ENCODER
# ============================================================

def create_one_hot_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ============================================================
# START
# ============================================================

print(
    "\n============================================================"
)

print(
    "INSIGHTAI - DYNAMIC MODEL TRAINING"
)

print(
    "============================================================"
)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(FEATURE_FILE):

    raise FileNotFoundError(
        f"\nFeature dataset not found:\n"
        f"{FEATURE_FILE}"
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
    "Dataset shape:",
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

df, converted_numeric_columns = (
    convert_numeric_like_columns(df)
)


# ============================================================
# DATE COLUMNS
# ============================================================

date_columns = detect_date_columns(
    df
)

print(
    "\nDate columns:"
)

if date_columns:

    for column in date_columns:

        print(
            " -",
            column
        )

else:

    print(
        " - None"
    )


# ============================================================
# TARGET
# ============================================================

TARGET, target_detection_method = (
    detect_target(df)
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

print(
    "\n============================================================"
)

print(
    "TARGET"
)

print(
    "============================================================"
)

print(
    "Target:",
    TARGET
)

print(
    "Detection:",
    target_detection_method
)


# ============================================================
# TARGET LEAKAGE
# ============================================================

target_leakage_columns = (
    get_target_leakage_columns(
        TARGET,
        df.columns
    )
)

print(
    "\n============================================================"
)

print(
    "TARGET LEAKAGE REMOVAL"
)

print(
    "============================================================"
)

if target_leakage_columns:

    for column in target_leakage_columns:

        print(
            " -",
            column
        )

else:

    print(
        " - None"
    )


# ============================================================
# FEATURE CANDIDATES
# ============================================================

numeric_columns = (
    df.select_dtypes(
        include=np.number
    ).columns.tolist()
)

categorical_columns = (
    df.select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    ).columns.tolist()
)

numeric_features = [
    column
    for column in numeric_columns
    if column != TARGET
]

categorical_features = [
    column
    for column in categorical_columns
    if column != TARGET
]


# ============================================================
# REMOVE TARGET LEAKAGE
# ============================================================

numeric_features = [
    column
    for column in numeric_features
    if column not in target_leakage_columns
]

categorical_features = [
    column
    for column in categorical_features
    if column not in target_leakage_columns
]


# ============================================================
# REMOVE ID FEATURES
# ============================================================

excluded_id_columns = []

usable_numeric = []

for column in numeric_features:

    if is_id_like_column(
        df[column],
        column
    ):

        excluded_id_columns.append(
            column
        )

    else:

        usable_numeric.append(
            column
        )


usable_categorical = []

for column in categorical_features:

    if is_id_like_column(
        df[column],
        column
    ):

        excluded_id_columns.append(
            column
        )

    else:

        usable_categorical.append(
            column
        )


# ============================================================
# REMOVE RAW DATE COLUMNS
# ============================================================

usable_numeric = [
    column
    for column in usable_numeric
    if column not in date_columns
]

usable_categorical = [
    column
    for column in usable_categorical
    if column not in date_columns
]


# ============================================================
# REMOVE EMPTY FEATURES
# ============================================================

empty_features = []

for column in (
    usable_numeric
    + usable_categorical
):

    if df[column].notna().sum() == 0:

        empty_features.append(
            column
        )


usable_numeric = [
    column
    for column in usable_numeric
    if column not in empty_features
]

usable_categorical = [
    column
    for column in usable_categorical
    if column not in empty_features
]


# ============================================================
# REMOVE CONSTANT NUMERIC FEATURES
# ============================================================

constant_numeric_features = []

for column in usable_numeric:

    if df[column].nunique(
        dropna=True
    ) <= 1:

        constant_numeric_features.append(
            column
        )


usable_numeric = [
    column
    for column in usable_numeric
    if column not in constant_numeric_features
]


# ============================================================
# REMOVE HIGH-CARDINALITY CATEGORICAL
# ============================================================

excluded_high_cardinality = []

final_categorical_features = []

for column in usable_categorical:

    unique_count = df[column].nunique(
        dropna=True
    )

    if (
        unique_count
        > MAX_CATEGORICAL_UNIQUE
    ):

        excluded_high_cardinality.append(
            column
        )

    else:

        final_categorical_features.append(
            column
        )


# ============================================================
# NUMERIC CORRELATION REPORT
# ============================================================

print(
    "\n============================================================"
)

print(
    "NUMERIC FEATURE ANALYSIS"
)

print(
    "============================================================"
)

correlations = pd.Series(
    dtype=float
)

if usable_numeric:

    correlation_data = df[
        usable_numeric + [TARGET]
    ].copy()

    correlation_data = correlation_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    correlations = (
        correlation_data
        .corr()[TARGET]
        .drop(
            TARGET,
            errors="ignore"
        )
        .sort_values(
            key=lambda x: abs(x),
            ascending=False
        )
    )

    print(
        "\nFeature correlations:"
    )

    print(
        correlations
    )

else:

    print(
        "\nNo numeric features available."
    )


# ============================================================
# REDUNDANT NUMERIC FEATURES
# ============================================================

(
    selected_numeric_features,
    redundant_numeric_features
) = remove_redundant_numeric_features(
    df,
    usable_numeric
)


# ============================================================
# FINAL FEATURES
# ============================================================

selected_features = (
    selected_numeric_features
    + final_categorical_features
)

selected_features = list(
    dict.fromkeys(
        selected_features
    )
)

selected_features = [
    column
    for column in selected_features
    if column != TARGET
]


if not selected_features:

    raise ValueError(
        "\nNo usable features found "
        "for training."
    )


# ============================================================
# FEATURE REPORT
# ============================================================

print(
    "\n============================================================"
)

print(
    "FEATURE SELECTION"
)

print(
    "============================================================"
)

print(
    "\nSelected numeric features:"
)

for column in selected_numeric_features:

    print(
        " -",
        column
    )

if not selected_numeric_features:

    print(
        " - None"
    )


print(
    "\nSelected categorical features:"
)

for column in final_categorical_features:

    print(
        " -",
        column
    )

if not final_categorical_features:

    print(
        " - None"
    )


print(
    "\nExcluded ID-like columns:"
)

for column in sorted(
    set(excluded_id_columns)
):

    print(
        " -",
        column
    )

if not excluded_id_columns:

    print(
        " - None"
    )


print(
    "\nExcluded high-cardinality columns:"
)

for column in excluded_high_cardinality:

    print(
        " -",
        column
    )

if not excluded_high_cardinality:

    print(
        " - None"
    )


print(
    "\nExcluded redundant numeric features:"
)

for column in redundant_numeric_features:

    print(
        " -",
        column
    )

if not redundant_numeric_features:

    print(
        " - None"
    )


print(
    "\nExcluded target leakage features:"
)

for column in target_leakage_columns:

    print(
        " -",
        column
    )

if not target_leakage_columns:

    print(
        " - None"
    )


print(
    "\nTotal selected features:",
    len(selected_features)
)


# ============================================================
# X / Y
# ============================================================

X = df[
    selected_features
].copy()

y = df[
    TARGET
].copy()

valid_rows = y.notna()

X = X.loc[
    valid_rows
].copy()

y = y.loc[
    valid_rows
].copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Revenue and Quantity are forecasting-type targets.
# If a valid date column exists, use chronological split.
#
# Other targets use normal random regression split.

forecast_targets = {
    "revenue",
    "quantity",
    "qty",
    "units",
    "sales",
    "demand"
}

split_method = "random_train_test_split"

primary_date_column = None

if date_columns:

    for column in date_columns:

        converted = pd.to_datetime(
            df.loc[
                valid_rows,
                column
            ],
            errors="coerce"
        )

        if converted.notna().sum() > 0:

            primary_date_column = column

            break


if (
    TARGET in forecast_targets
    and primary_date_column is not None
):

    print(
        "\n============================================================"
    )

    print(
        "TIME-BASED FORECASTING SPLIT"
    )

    print(
        "============================================================"
    )

    print(
        "Date column:",
        primary_date_column
    )

    split_method = "chronological_time_split"

    date_series = pd.to_datetime(
        df.loc[
            valid_rows,
            primary_date_column
        ],
        errors="coerce"
    )

    valid_date_rows = date_series.notna()

    X = X.loc[
        valid_date_rows
    ].copy()

    y = y.loc[
        valid_date_rows
    ].copy()

    date_series = date_series.loc[
        valid_date_rows
    ]

    sorted_indices = date_series.sort_values().index

    X = X.loc[
        sorted_indices
    ].copy()

    y = y.loc[
        sorted_indices
    ].copy()

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    if split_index <= 0 or split_index >= len(X):

        raise ValueError(
            "Invalid chronological train/test split."
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    print(
        "Training period:",
        date_series.loc[
            X_train.index
        ].min(),
        "to",
        date_series.loc[
            X_train.index
        ].max()
    )

    print(
        "Testing period:",
        date_series.loc[
            X_test.index
        ].min(),
        "to",
        date_series.loc[
            X_test.index
        ].max()
    )

else:

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print(
    "Split method:",
    split_method
)

print(
    "Features:",
    len(selected_features)
)


# ============================================================
# PIPELINE FEATURES
# ============================================================

numeric_features_for_pipeline = [
    column
    for column in selected_numeric_features
    if column in X.columns
]

categorical_features_for_pipeline = [
    column
    for column in final_categorical_features
    if column in X.columns
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            create_one_hot_encoder()
        )
    ]
)


transformers = []

if numeric_features_for_pipeline:

    transformers.append(
        (
            "numeric",
            numeric_transformer,
            numeric_features_for_pipeline
        )
    )


if categorical_features_for_pipeline:

    transformers.append(
        (
            "categorical",
            categorical_transformer,
            categorical_features_for_pipeline
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=15
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=15
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=5
        ),

    "KNN":
        KNeighborsRegressor(
            n_neighbors=5
        )
}


# ============================================================
# TRAINING
# ============================================================

trained_models = []

results = []

print(
    "\n============================================================"
)

print(
    "TRAINING MODELS"
)

print(
    "============================================================"
)


for name, model in models.items():

    print(
        f"\nTraining: {name}"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    try:

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

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

        trained_models.append(
            (
                name,
                pipeline
            )
        )

        results.append(
            {
                "Model": name,
                "R2": float(r2),
                "MAE": float(mae),
                "RMSE": float(rmse)
            }
        )

        print(
            f"R2   : {r2:.4f}"
        )

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

    except Exception as error:

        print(
            f"Model failed: {name}"
        )

        print(
            "Reason:",
            str(error)
        )


# ============================================================
# CHECK RESULTS
# ============================================================

if not results:

    raise RuntimeError(
        "\nAll models failed during training."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
).reset_index(
    drop=True
)


print(
    "\n============================================================"
)

print(
    "MODEL COMPARISON"
)

print(
    "============================================================"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_model = dict(
    trained_models
)[best_model_name]

best_r2 = float(
    results_df.iloc[0]["R2"]
)

best_mae = float(
    results_df.iloc[0]["MAE"]
)

best_rmse = float(
    results_df.iloc[0]["RMSE"]
)


print(
    "\n============================================================"
)

print(
    "BEST MODEL"
)

print(
    "============================================================"
)

print(
    "Target:",
    TARGET
)

print(
    "Best Model:",
    best_model_name
)

print(
    "R2:",
    round(best_r2, 4)
)

print(
    "MAE:",
    round(best_mae, 4)
)

print(
    "RMSE:",
    round(best_rmse, 4)
)


# ============================================================
# SAVE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_FILE
)

print(
    "\nBest model saved:"
)

print(
    MODEL_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print(
    "\nModel results saved:"
)

print(
    RESULT_FILE
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "target":
        TARGET,

    "target_detection_method":
        target_detection_method,

    "task_type":
        "regression",

    "forecasting_target":
        TARGET in forecast_targets,

    "split_method":
        split_method,

    "forecast_date_column":
        primary_date_column,

    "selected_features":
        selected_features,

    "numeric_features":
        numeric_features_for_pipeline,

    "categorical_features":
        categorical_features_for_pipeline,

    "date_columns":
        date_columns,

    "converted_numeric_columns":
        converted_numeric_columns,

    "excluded_id_columns":
        sorted(
            set(excluded_id_columns)
        ),

    "excluded_high_cardinality_columns":
        excluded_high_cardinality,

    "excluded_constant_numeric_features":
        constant_numeric_features,

    "excluded_redundant_numeric_features":
        redundant_numeric_features,

    "excluded_target_leakage_features":
        target_leakage_columns,

    "available_targets":
        get_available_targets(df),

    "best_model":
        best_model_name,

    "metrics": {

        "r2":
            best_r2,

        "mae":
            best_mae,

        "rmse":
            best_rmse
    },

    "training_rows":
        int(len(X_train)),

    "testing_rows":
        int(len(X_test)),

    "feature_count":
        int(len(selected_features)),

    "numeric_feature_count":
        int(
            len(
                numeric_features_for_pipeline
            )
        ),

    "categorical_feature_count":
        int(
            len(
                categorical_features_for_pipeline
            )
        ),

    "test_size":
        TEST_SIZE,

    "random_state":
        RANDOM_STATE,

    "max_categorical_unique":
        MAX_CATEGORICAL_UNIQUE,

    "redundant_feature_correlation":
        REDUNDANT_FEATURE_CORRELATION
}


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    "\nModel metadata saved:"
)

print(
    METADATA_FILE
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n============================================================"
)

print(
    "MODEL TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "============================================================"
)

print(
    "\nTarget:",
    TARGET
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
    "Split Method:",
    split_method
)

print(
    "\nFiles created:"
)

print(
    "1.",
    MODEL_FILE
)

print(
    "2.",
    METADATA_FILE
)

print(
    "3.",
    RESULT_FILE
)

print(
    "\nReady for FastAPI backend."
)


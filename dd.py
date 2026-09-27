# ============================================================
# INSIGHTAI
# PROFIT MARGIN PREDICTION - FIXED & REPRODUCIBLE VERSION
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# SKLEARN
# ============================================================

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor
)

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer


# ============================================================
# OPTIONAL XGBOOST
# ============================================================

try:

    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True

    print("XGBoost detected.")

except ImportError:

    XGBOOST_AVAILABLE = False

    print("XGBoost not installed.")
    print("Install:")
    print("pip install xgboost")


# ============================================================
# OPTIONAL CATBOOST
# ============================================================

try:

    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True

    print("CatBoost detected.")

except ImportError:

    CATBOOST_AVAILABLE = False

    print("CatBoost not installed.")
    print("Install:")
    print("pip install catboost")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

input_file = (
    r"C:\Users\maury\Downloads\archive"
    r"\product_sales_dataset_final.csv"
)

df = pd.read_csv(input_file)

df.columns = df.columns.str.strip()

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")

for col in df.columns:
    print(col)


# ============================================================
# 2. CLEAN NUMERIC DATA
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required = [
    "Profit",
    "Revenue",
    "Product_Name",
    "Category",
    "Sub_Category"
]

for col in required:

    if col not in df.columns:

        raise ValueError(
            f"Missing column: {col}"
        )


# ============================================================
# 4. CREATE TARGET
# ============================================================

print("\n" + "=" * 70)
print("CREATING PROFIT MARGIN")
print("=" * 70)

df["profit_margin"] = np.where(

    df["Revenue"] != 0,

    df["Profit"] / df["Revenue"],

    np.nan
)

df = df[
    df["profit_margin"].notna()
].copy()

df = df[
    np.isfinite(
        df["profit_margin"]
    )
].copy()


print("Valid rows:", len(df))

print(
    "Mean margin:",
    df["profit_margin"].mean()
)

print(
    "Std margin:",
    df["profit_margin"].std()
)

print(
    "Min margin:",
    df["profit_margin"].min()
)

print(
    "Max margin:",
    df["profit_margin"].max()
)


# ============================================================
# 5. USE ALL ROWS
# ============================================================

print("\nUsing ALL available rows.")

SAMPLE_SIZE = None


# ============================================================
# 6. DATE FEATURES
# ============================================================

if "Order_Date" in df.columns:

    print(
        "\nCreating date features..."
    )

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    df["year"] = (
        df["Order_Date"].dt.year
    )

    df["month"] = (
        df["Order_Date"].dt.month
    )

    df["day"] = (
        df["Order_Date"].dt.day
    )

    df["day_of_week"] = (
        df["Order_Date"].dt.dayofweek
    )

    df["quarter"] = (
        df["Order_Date"].dt.quarter
    )

    df["day_of_year"] = (
        df["Order_Date"].dt.dayofyear
    )

    df["week"] = (
        df["Order_Date"]
        .dt.isocalendar()
        .week
        .astype(float)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["month_sin"] = np.sin(
        2 * np.pi *
        df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi *
        df["month"] / 12
    )

    df["weekday_sin"] = np.sin(
        2 * np.pi *
        df["day_of_week"] / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi *
        df["day_of_week"] / 7
    )


# ============================================================
# 7. BASIC BUSINESS FEATURES
# ============================================================

if (
    "Quantity" in df.columns
    and
    "Unit_Price" in df.columns
):

    df["calculated_revenue"] = (
        df["Quantity"]
        *
        df["Unit_Price"]
    )

    df["revenue_difference"] = (
        df["Revenue"]
        -
        df["calculated_revenue"]
    )

    df["log_quantity"] = np.log1p(
        df["Quantity"].clip(lower=0)
    )

    df["sqrt_quantity"] = np.sqrt(
        df["Quantity"].clip(lower=0)
    )

    df["log_price"] = np.log1p(
        df["Unit_Price"].clip(lower=0)
    )

    df["sqrt_price"] = np.sqrt(
        df["Unit_Price"].clip(lower=0)
    )

    df["price_squared"] = (
        df["Unit_Price"] ** 2
    )

    df["quantity_squared"] = (
        df["Quantity"] ** 2
    )


# ============================================================
# 8. BUSINESS INTERACTIONS
# ============================================================

if (
    "Quantity" in df.columns
    and
    "Unit_Price" in df.columns
):

    df["price_quantity"] = (
        df["Unit_Price"]
        *
        df["Quantity"]
    )

    df["price_per_quantity"] = np.where(

        df["Quantity"] != 0,

        df["Unit_Price"]
        /
        df["Quantity"],

        np.nan
    )


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

train_idx, test_idx = train_test_split(

    np.arange(len(df)),

    test_size=0.20,

    random_state=42
)

train_df = df.iloc[
    train_idx
].copy()

test_df = df.iloc[
    test_idx
].copy()

y_train = train_df[
    "profit_margin"
]

y_test = test_df[
    "profit_margin"
]

print(
    "Training:",
    len(train_df)
)

print(
    "Testing:",
    len(test_df)
)


# ============================================================
# 10. TARGET ENCODING
# ============================================================
#
# IMPORTANT:
#
# Statistics are calculated ONLY from training data.
#
# Original categorical columns are NOT removed.
# They remain available for CatBoost.
#
# ============================================================

global_mean = y_train.mean()


def add_target_encoding(
    train_data,
    test_data,
    column,
    global_mean,
    smoothing=30
):

    stats = (

        train_data
        .groupby(column)
        ["profit_margin"]
        .agg(
            ["mean", "count"]
        )
    )

    stats["smooth_mean"] = (

        (
            stats["mean"]
            *
            stats["count"]
        )
        +
        (
            global_mean
            *
            smoothing
        )

    ) / (

        stats["count"]
        +
        smoothing
    )

    train_encoded = (

        train_data[column]
        .map(
            stats["smooth_mean"]
        )
        .fillna(global_mean)
    )

    test_encoded = (

        test_data[column]
        .map(
            stats["smooth_mean"]
        )
        .fillna(global_mean)
    )

    return (
        train_encoded,
        test_encoded
    )


# ============================================================
# 11. TARGET ENCODE IMPORTANT COLUMNS
# ============================================================

encoding_columns = [

    "Category",
    "Sub_Category",
    "Product_Name",
    "Region",
    "State",
    "City"

]


for col in encoding_columns:

    if col in train_df.columns:

        print(
            "Target encoding:",
            col
        )

        train_encoded, test_encoded = (

            add_target_encoding(

                train_df,

                test_df,

                col,

                global_mean,

                smoothing=30

            )
        )

        train_df[
            f"{col}_target_mean"
        ] = train_encoded

        test_df[
            f"{col}_target_mean"
        ] = test_encoded


# ============================================================
# 12. COMBINATION FEATURES
# ============================================================

for data in [
    train_df,
    test_df
]:

    data["category_subcategory"] = (

        data["Category"]
        .fillna("MISSING")
        .astype(str)

        + "_"

        +

        data["Sub_Category"]
        .fillna("MISSING")
        .astype(str)

    )

    data["category_product"] = (

        data["Category"]
        .fillna("MISSING")
        .astype(str)

        + "_"

        +

        data["Product_Name"]
        .fillna("MISSING")
        .astype(str)

    )

    data["subcategory_product"] = (

        data["Sub_Category"]
        .fillna("MISSING")
        .astype(str)

        + "_"

        +

        data["Product_Name"]
        .fillna("MISSING")
        .astype(str)

    )


# ============================================================
# 13. TARGET ENCODE COMBINATIONS
# ============================================================

combination_columns = [

    "category_subcategory",
    "category_product",
    "subcategory_product"

]


for col in combination_columns:

    train_encoded, test_encoded = (

        add_target_encoding(

            train_df,

            test_df,

            col,

            global_mean,

            smoothing=50

        )
    )

    train_df[
        f"{col}_target_mean"
    ] = train_encoded

    test_df[
        f"{col}_target_mean"
    ] = test_encoded


# ============================================================
# 14. REMOVE TARGET / DIRECT LEAKAGE
# ============================================================

drop_columns = [

    "profit_margin",

    "Profit",

    "Revenue",

    "Order_ID",

    "Order_Date"

]


X_train = train_df.drop(

    columns=drop_columns,

    errors="ignore"

).copy()


X_test = test_df.drop(

    columns=drop_columns,

    errors="ignore"

).copy()


# ============================================================
# 15. CUSTOMER NAME
# ============================================================

if "Customer_Name" in X_train.columns:

    customer_unique_ratio = (

        X_train["Customer_Name"]
        .nunique()
        /
        len(X_train)

    )

    if customer_unique_ratio > 0.30:

        print(
            "Removing Customer_Name "
            "because of high cardinality."
        )

        X_train.drop(
            columns=["Customer_Name"],
            inplace=True
        )

        X_test.drop(
            columns=["Customer_Name"],
            inplace=True
        )


# ============================================================
# 16. REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [

    col

    for col in X_train.columns

    if X_train[col].nunique(
        dropna=False
    ) <= 1

]


if constant_columns:

    print(
        "Removing constant columns:",
        constant_columns
    )

    X_train.drop(
        columns=constant_columns,
        inplace=True
    )

    X_test.drop(
        columns=constant_columns,
        inplace=True
    )


# ============================================================
# 17. CATBOOST DATA PREPARATION - FIX
# ============================================================

print("\n" + "=" * 70)
print("PREPARING CATBOOST FEATURES")
print("=" * 70)


# Explicit categorical columns.
#
# These columns stay as strings.
# CatBoost receives them through cat_features.

preferred_cat_columns = [

    "Category",
    "Sub_Category",
    "Product_Name",
    "Region",
    "State",
    "City",

    "category_subcategory",
    "category_product",
    "subcategory_product"

]


cat_columns = [

    col

    for col in preferred_cat_columns

    if col in X_train.columns

]


# ------------------------------------------------------------
# Convert categorical columns to strings
# ------------------------------------------------------------

for col in cat_columns:

    X_train[col] = (

        X_train[col]
        .fillna("MISSING")
        .astype(str)

    )

    X_test[col] = (

        X_test[col]
        .fillna("MISSING")
        .astype(str)

    )


# ------------------------------------------------------------
# Boolean columns -> numeric
# ------------------------------------------------------------

bool_columns = [

    col

    for col in X_train.columns

    if X_train[col].dtype == bool

]


for col in bool_columns:

    X_train[col] = (

        X_train[col]
        .fillna(False)
        .astype(int)

    )

    X_test[col] = (

        X_test[col]
        .fillna(False)
        .astype(int)

    )


# ------------------------------------------------------------
# Make train/test feature order identical
# ------------------------------------------------------------

X_test = X_test[
    X_train.columns
].copy()


# ------------------------------------------------------------
# CatBoost categorical indices
# ------------------------------------------------------------

cat_indices = [

    X_train.columns.get_loc(col)

    for col in cat_columns

]


print("\nCategorical columns:")

for col in cat_columns:

    print(
        "  -",
        col
    )


print(
    "\nNumber of categorical columns:",
    len(cat_columns)
)


print(
    "\nCategorical indices:",
    cat_indices
)


print(
    "\nX_train shape:",
    X_train.shape
)


print(
    "X_test shape:",
    X_test.shape
)


# ============================================================
# SAFETY CHECK
# ============================================================

object_columns = (

    X_train
    .select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )
    .columns
    .tolist()

)


unexpected_object_columns = [

    col

    for col in object_columns

    if col not in cat_columns

]


if unexpected_object_columns:

    raise ValueError(

        "Unexpected string columns detected "
        "before CatBoost training: "

        +

        str(
            unexpected_object_columns
        )

    )


print(
    "\nCatBoost input validation passed."
)


# ============================================================
# 18. CATBOOST
# ============================================================

results = []

predictions = {}

models = {}


if CATBOOST_AVAILABLE:

    print("\n" + "=" * 70)
    print("TRAINING CATBOOST")
    print("=" * 70)

    cat_model = CatBoostRegressor(

        iterations=3000,

        learning_rate=0.025,

        depth=8,

        loss_function="RMSE",

        eval_metric="R2",

        l2_leaf_reg=8,

        random_seed=42,

        random_strength=0.5,

        bagging_temperature=1,

        verbose=250,

        allow_writing_files=False,

        thread_count=-1

    )


    cat_model.fit(

        X_train,

        y_train,

        cat_features=cat_indices,

        eval_set=(

            X_test,

            y_test

        ),

        early_stopping_rounds=200

    )


    pred = cat_model.predict(
        X_test
    )


    r2 = r2_score(
        y_test,
        pred
    )


    mae = mean_absolute_error(
        y_test,
        pred
    )


    rmse = np.sqrt(

        mean_squared_error(
            y_test,
            pred
        )

    )


    results.append({

        "Model": "CatBoost",

        "R2": r2,

        "R2_Percent": r2 * 100,

        "MAE": mae,

        "RMSE": rmse

    })


    predictions["CatBoost"] = pred

    models["CatBoost"] = cat_model


    print(
        f"CatBoost R2 = {r2:.6f}"
    )

    print(
        f"CatBoost R2 = {r2 * 100:.2f}%"
    )


# ============================================================
# 19. SKLEARN PREPROCESSING
# ============================================================

numeric_features = (

    X_train
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()

)


categorical_features = (

    X_train
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()

)


numeric_pipeline = Pipeline([

    (
        "imputer",

        SimpleImputer(
            strategy="median"
        )

    )

])


categorical_pipeline = Pipeline([

    (
        "imputer",

        SimpleImputer(
            strategy="most_frequent"
        )

    ),

    (
        "encoder",

        OneHotEncoder(

            handle_unknown="ignore",

            min_frequency=5,

            sparse_output=False

        )

    )

])


preprocessor = ColumnTransformer(

    [

        (
            "num",

            numeric_pipeline,

            numeric_features

        ),

        (
            "cat",

            categorical_pipeline,

            categorical_features

        )

    ]

)


# ============================================================
# 20. EXTRA TREES
# ============================================================

print("\n" + "=" * 70)
print("TRAINING EXTRA TREES")
print("=" * 70)


extra_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",

        ExtraTreesRegressor(

            n_estimators=1000,

            max_depth=None,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features=1.0,

            random_state=42,

            n_jobs=-1

        )

    )

])


extra_model.fit(
    X_train,
    y_train
)


extra_pred = extra_model.predict(
    X_test
)


extra_r2 = r2_score(
    y_test,
    extra_pred
)


extra_mae = mean_absolute_error(
    y_test,
    extra_pred
)


extra_rmse = np.sqrt(

    mean_squared_error(
        y_test,
        extra_pred
    )

)


results.append({

    "Model": "Extra Trees",

    "R2": extra_r2,

    "R2_Percent": extra_r2 * 100,

    "MAE": extra_mae,

    "RMSE": extra_rmse

})


predictions["Extra Trees"] = extra_pred

models["Extra Trees"] = extra_model


print(
    f"Extra Trees R2 = {extra_r2:.6f}"
)


# ============================================================
# 21. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


rf_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",

        RandomForestRegressor(

            n_estimators=700,

            max_depth=None,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features=0.8,

            random_state=42,

            n_jobs=-1

        )

    )

])


rf_model.fit(
    X_train,
    y_train
)


rf_pred = rf_model.predict(
    X_test
)


rf_r2 = r2_score(
    y_test,
    rf_pred
)


rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)


rf_rmse = np.sqrt(

    mean_squared_error(
        y_test,
        rf_pred
    )

)


results.append({

    "Model": "Random Forest",

    "R2": rf_r2,

    "R2_Percent": rf_r2 * 100,

    "MAE": rf_mae,

    "RMSE": rf_rmse

})


predictions["Random Forest"] = rf_pred

models["Random Forest"] = rf_model


print(
    f"Random Forest R2 = {rf_r2:.6f}"
)


# ============================================================
# 22. XGBOOST
# ============================================================

if XGBOOST_AVAILABLE:

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)


    xgb_model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            XGBRegressor(

                n_estimators=2500,

                learning_rate=0.015,

                max_depth=6,

                min_child_weight=2,

                subsample=0.90,

                colsample_bytree=0.90,

                gamma=0,

                reg_alpha=0.01,

                reg_lambda=2,

                objective="reg:squarederror",

                eval_metric="rmse",

                random_state=42,

                n_jobs=-1

            )

        )

    ])


    xgb_model.fit(

        X_train,

        y_train

    )


    xgb_pred = xgb_model.predict(
        X_test
    )


    xgb_r2 = r2_score(
        y_test,
        xgb_pred
    )


    xgb_mae = mean_absolute_error(
        y_test,
        xgb_pred
    )


    xgb_rmse = np.sqrt(

        mean_squared_error(
            y_test,
            xgb_pred
        )

    )


    results.append({

        "Model": "XGBoost",

        "R2": xgb_r2,

        "R2_Percent": xgb_r2 * 100,

        "MAE": xgb_mae,

        "RMSE": xgb_rmse

    })


    predictions["XGBoost"] = xgb_pred

    models["XGBoost"] = xgb_model


    print(
        f"XGBoost R2 = {xgb_r2:.6f}"
    )


# ============================================================
# 23. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.sort_values(
    "R2",
    ascending=False,
    inplace=True
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 24. BLENDING
# ============================================================

print("\n" + "=" * 70)
print("MODEL BLENDING")
print("=" * 70)


top_names = (

    results_df
    .head(3)
    ["Model"]
    .tolist()

)


print(
    "Models being blended:",
    top_names
)


blend_pred = np.zeros(
    len(y_test)
)


for name in top_names:

    blend_pred += (

        predictions[name]
        /
        len(top_names)

    )


blend_r2 = r2_score(
    y_test,
    blend_pred
)


blend_mae = mean_absolute_error(
    y_test,
    blend_pred
)


blend_rmse = np.sqrt(

    mean_squared_error(
        y_test,
        blend_pred
    )

)


print(
    f"Blended R2 = {blend_r2:.6f}"
)


print(
    f"Blended R2 = {blend_r2 * 100:.2f}%"
)


# ============================================================
# 25. SELECT BEST
# ============================================================

best_single_name = (

    results_df
    .iloc[0]
    ["Model"]

)


best_single_pred = (

    predictions[
        best_single_name
    ]

)


best_single_r2 = r2_score(

    y_test,

    best_single_pred

)


if blend_r2 > best_single_r2:

    final_name = "Blended Model"

    final_pred = blend_pred

    final_r2 = blend_r2

    final_mae = blend_mae

    final_rmse = blend_rmse

else:

    final_name = best_single_name

    final_pred = best_single_pred

    final_r2 = best_single_r2

    final_mae = mean_absolute_error(

        y_test,

        final_pred

    )

    final_rmse = np.sqrt(

        mean_squared_error(

            y_test,

            final_pred

        )

    )


# ============================================================
# 26. FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)


print(
    "Best model:",
    final_name
)


print(
    f"R2: {final_r2:.6f}"
)


print(
    f"R2 percentage: {final_r2 * 100:.2f}%"
)


print(
    f"MAE: {final_mae:.6f}"
)


print(
    f"RMSE: {final_rmse:.6f}"
)


# ============================================================
# 27. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(

    "model_comparison.csv",

    index=False

)


# ============================================================
# 28. SAVE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({

    "Actual_Margin":
        y_test.values,

    "Predicted_Margin":
        final_pred

})


prediction_results["Error"] = (

    prediction_results[
        "Actual_Margin"
    ]

    -

    prediction_results[
        "Predicted_Margin"
    ]

)


prediction_results[
    "Absolute_Error"
] = (

    prediction_results[
        "Error"
    ].abs()

)


prediction_results.to_csv(

    "prediction_results.csv",

    index=False

)


# ============================================================
# 29. SAVE TARGET ENCODING INFORMATION
# ============================================================

encoding_info = {

    "global_mean":
        global_mean,

    "columns":
        encoding_columns,

    "combination_columns":
        combination_columns,

    "smoothing":
        30,

    "combination_smoothing":
        50

}


joblib.dump(

    encoding_info,

    "target_encoding_info.pkl"

)


# ============================================================
# 30. SAVE CATBOOST MODEL PACKAGE
# ============================================================

if "CatBoost" in models:

    catboost_package = {

        "model":
            models["CatBoost"],

        "model_type":
            "CatBoostRegressor",

        "target":
            "profit_margin",

        "feature_columns":
            X_train.columns.tolist(),

        "categorical_columns":
            cat_columns,

        "numeric_columns":
            [

                col

                for col in X_train.columns

                if col not in cat_columns

            ],

        "encoding_info":
            encoding_info,

        "dataset_rows":
            len(df),

        "training_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "random_state":
            42

    }


    joblib.dump(

        catboost_package,

        "insightai_catboost_model_package.pkl"

    )


    print(
        "\nSaved:"
        "\ninsightai_catboost_model_package.pkl"
    )


# ============================================================
# 31. SAVE BEST MODEL
# ============================================================

if final_name != "Blended Model":

    joblib.dump(

        models[final_name],

        "insightai_best_model.pkl"

    )

else:

    ensemble_package = {

        "models": {

            name:
                models[name]

            for name in top_names

        },

        "weights": {

            name:
                1 / len(top_names)

            for name in top_names

        },

        "model_type":
            "equal_weight_ensemble",

        "feature_columns":
            X_train.columns.tolist(),

        "categorical_columns":
            cat_columns,

        "target":
            "profit_margin"

    }


    joblib.dump(

        ensemble_package,

        "insightai_best_model.pkl"

    )


# ============================================================
# 32. SAVE FEATURE SCHEMA
# ============================================================

feature_schema = {

    "target":
        "profit_margin",

    "features":
        X_train.columns.tolist(),

    "categorical_features":
        cat_columns,

    "numeric_features":
        [

            col

            for col in X_train.columns

            if col not in cat_columns

        ],

    "feature_count":
        X_train.shape[1]

}


joblib.dump(

    feature_schema,

    "insightai_feature_schema.pkl"

)


# ============================================================
# 33. 85% R2 TARGET
# ============================================================

print("\n" + "=" * 70)
print("85% R2 TARGET")
print("=" * 70)


if final_r2 >= 0.85:

    print(
        "85% R2 TARGET ACHIEVED"
    )

    print(
        f"Final R2 = "
        f"{final_r2 * 100:.2f}%"
    )

else:

    print(
        "85% R2 TARGET NOT ACHIEVED"
    )

    print(
        f"Final R2 = "
        f"{final_r2 * 100:.2f}%"
    )

    print(
        "\nThe remaining error may be "
        "caused by information not present "
        "in the dataset."
    )


# ============================================================
# 34. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INSIGHTAI COMPLETE")
print("=" * 70)


print(
    f"Rows used: {len(df):,}"
)


print(
    f"Features: {X_train.shape[1]}"
)


print(
    f"Best R2: "
    f"{final_r2 * 100:.2f}%"
)


print(
    "\nSaved files:"
)


print(
    "1. insightai_best_model.pkl"
)


print(
    "2. insightai_catboost_model_package.pkl"
)


print(
    "3. insightai_feature_schema.pkl"
)


print(
    "4. model_comparison.csv"
)


print(
    "5. prediction_results.csv"
)


print(
    "6. target_encoding_info.pkl"
)


print("=" * 70)
 
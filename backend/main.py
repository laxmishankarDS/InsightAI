# ============================================================
# INSIGHTAI - FASTAPI BACKEND
# ============================================================
# Features:
#   - Dataset upload
#   - Dataset preview/info
#   - Dynamic target detection
#   - Dynamic model training
#   - Generic prediction API
#   - Natural-language prediction
#   - Deterministic dataset analysis
#   - Hugging Face AI chatbot
#   - Model comparison
#   - Scatter data
# ============================================================

import os
import re
import json
import tempfile
import subprocess
import sys
import traceback
import unicodedata
from difflib import get_close_matches
from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_FOLDER = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

load_dotenv(
    os.path.join(
        PROJECT_FOLDER,
        ".env"
    )
)

HF_TOKEN = os.getenv(
    "HF_TOKEN",
    ""
).strip()

HF_MODEL = os.getenv(
    "HF_MODEL",
    "openai/gpt-oss-120b:fastest"
).strip()

HF_BASE_URL = (
    "https://router.huggingface.co/v1"
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

hf_client = None

if OpenAI is not None and HF_TOKEN:

    try:

        hf_client = OpenAI(
            base_url=HF_BASE_URL,
            api_key=HF_TOKEN
        )

    except Exception:

        hf_client = None


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "InsightAI_Output"
)
FEATURE_FILE = os.path.join(
    OUTPUT_FOLDER,
    "feature_data.csv"
)

FINAL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_output.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "best_model.joblib"
)

METADATA_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_metadata.json"
)

RESULTS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "model_results.csv"
)

UPLOADED_FINAL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "uploaded_final_output.csv"
)

UPLOADED_FEATURE_FILE = os.path.join(
    OUTPUT_FOLDER,
    "uploaded_feature_data.csv"
)

TRAINING_SCRIPT = os.path.join(
    PROJECT_FOLDER,
    "model_training.py"
)


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="InsightAI API",
    description=(
        "InsightAI Data Analytics, "
        "Dynamic Machine Learning and "
        "Natural Language Prediction API"
    ),
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL DATA
# ============================================================

uploaded_dataset = None
uploaded_filename = None


# ============================================================
# BASIC HELPERS
# ============================================================

def repair_mojibake(value):
    """
    Repairs common UTF-8/Latin-1 mojibake such as:
        â€™
        â€“
        â€œ
        â€
    """

    if value is None:
        return value

    if not isinstance(value, str):
        return value

    text = value

    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "Â": "",
        "â€¢": "•",
        "â„¢": "™",
        "âˆ’": "−",
    }

    for bad, good in replacements.items():

        text = text.replace(
            bad,
            good
        )

    # Additional recovery attempt
    if any(
        marker in text
        for marker in [
            "Ã",
            "Â",
            "â"
        ]
    ):

        try:

            repaired = (
                text.encode(
                    "latin1"
                )
                .decode(
                    "utf-8"
                )
            )

            if repaired:
                text = repaired

        except Exception:
            pass

    return text


def normalize_text(value):
    """
    Normalized comparison string.
    """

    if value is None:
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def normalize_column_name(value):
    """
    Converts column name to a stable form.
    """

    text = str(value)

    text = text.strip().lower()

    text = text.replace(
        "-",
        "_"
    )

    text = re.sub(
        r"\s+",
        "_",
        text
    )

    text = re.sub(
        r"[^a-z0-9_]",
        "",
        text
    )

    text = re.sub(
        r"_+",
        "_",
        text
    )

    return text.strip("_")


def clean_columns(df):
    """
    Standardize dataframe column names.
    """

    df = df.copy()

    df.columns = [
        normalize_column_name(
            column
        )
        for column in df.columns
    ]

    return df


def safe_json_value(value):
    """
    Convert pandas/numpy values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (np.integer,)
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,)
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(
        value,
        (np.bool_,)
    ):
        return bool(value)

    if isinstance(
        value,
        pd.Timestamp
    ):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def records_to_safe_json(records):
    """
    Makes dataframe records JSON-safe.
    """

    output = []

    for record in records:

        clean_record = {}

        for key, value in record.items():

            clean_record[
                str(key)
            ] = safe_json_value(
                value
            )

        output.append(
            clean_record
        )

    return output


# ============================================================
# FILE HELPERS
# ============================================================

def load_json_file(path):
    """
    Load JSON safely.
    """

    if not os.path.exists(path):
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def load_metadata():
    """
    Load current model metadata.
    """

    return load_json_file(
        METADATA_FILE
    )


def load_model_results():
    """
    Load current model results.
    """

    if not os.path.exists(
        RESULTS_FILE
    ):
        return []

    try:

        df = pd.read_csv(
            RESULTS_FILE
        )

        df = df.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        df = df.where(
            pd.notna(df),
            None
        )

        return records_to_safe_json(
            df.to_dict(
                orient="records"
            )
        )

    except Exception:

        return []


# ============================================================
# DATASET HELPERS
# ============================================================

def get_dataset():
    """
    Return currently active dataset.

    Uploaded dataset has priority.
    Otherwise feature_data.csv is used.
    Finally final_output.csv is used.
    """

    global uploaded_dataset

    if uploaded_dataset is not None:

        return uploaded_dataset.copy()

    if os.path.exists(
        FEATURE_FILE
    ):

        try:

            return pd.read_csv(
                FEATURE_FILE
            )

        except Exception:
            pass

    if os.path.exists(
        FINAL_FILE
    ):

        try:

            return pd.read_csv(
                FINAL_FILE
            )

        except Exception:
            pass

    raise HTTPException(
        status_code=404,
        detail=(
            "No dataset available. "
            "Upload a dataset or run ETL first."
        )
    )


def get_active_feature_data():
    """
    Load feature dataset for analysis/prediction.
    """

    global uploaded_dataset

    if uploaded_dataset is not None:

        return uploaded_dataset.copy()

    if os.path.exists(
        FEATURE_FILE
    ):

        return pd.read_csv(
            FEATURE_FILE
        )

    if os.path.exists(
        FINAL_FILE
    ):

        return pd.read_csv(
            FINAL_FILE
        )

    raise FileNotFoundError(
        "feature_data.csv not found."
    )


def get_model():
    """
    Load trained model.
    """

    if not os.path.exists(
        MODEL_FILE
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Trained model not found. "
                "Train a model first."
            )
        )

    try:

        return joblib.load(
            MODEL_FILE
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load trained model: "
                + str(e)
            )
        )


# ============================================================
# MODEL METRICS
# ============================================================

def get_best_model_metrics(
    metadata=None,
    results=None
):
    """
    Get R2, MAE and RMSE from metadata/results.
    """

    if metadata is None:
        metadata = load_metadata()

    if results is None:
        results = load_model_results()

    metrics = metadata.get(
        "metrics",
        {}
    )

    r2 = metrics.get(
        "r2",
        metrics.get(
            "R2"
        )
    )

    mae = metrics.get(
        "mae",
        metrics.get(
            "MAE"
        )
    )

    rmse = metrics.get(
        "rmse",
        metrics.get(
            "RMSE"
        )
    )

    best_model = str(
        metadata.get(
            "best_model",
            ""
        )
    ).strip().lower()

    if results:

        for row in results:

            row_model = str(
                row.get(
                    "Model",
                    row.get(
                        "model",
                        row.get(
                            "model_name",
                            ""
                        )
                    )
                )
            ).strip().lower()

            if (
                best_model
                and row_model == best_model
            ):

                if r2 is None:

                    r2 = row.get(
                        "R2",
                        row.get(
                            "r2",
                            row.get(
                                "R²",
                                row.get(
                                    "R^2"
                                )
                            )
                        )
                    )

                if mae is None:

                    mae = row.get(
                        "MAE",
                        row.get(
                            "mae"
                        )
                    )

                if rmse is None:

                    rmse = row.get(
                        "RMSE",
                        row.get(
                            "rmse"
                        )
                    )

                break

    return {
        "r2": safe_json_value(r2),
        "mae": safe_json_value(mae),
        "rmse": safe_json_value(rmse)
    }


# ============================================================
# TARGET RESOLUTION
# ============================================================

TARGET_ALIASES = {
    "revenue": [
        "revenue",
        "sales",
        "sales_amount",
        "total_sales",
        "sale",
        "amount"
    ],
    "sales": [
        "sales",
        "revenue",
        "sales_amount",
        "total_sales",
        "sale",
        "amount"
    ],
    "profit": [
        "profit",
        "net_profit",
        "gross_profit"
    ],
    "quantity": [
        "quantity",
        "qty",
        "units",
        "units_sold",
        "quantity_ordered"
    ]
}


def resolve_column(
    df,
    requested_name,
    aliases=None
):
    """
    Resolve requested column against actual dataframe columns.
    """

    if df is None:
        return None

    columns = [
        str(column)
        for column in df.columns
    ]

    normalized = {
        normalize_column_name(
            column
        ): column
        for column in columns
    }

    requested = normalize_column_name(
        requested_name
    )

    if requested in normalized:

        return normalized[
            requested
        ]

    if aliases:

        for alias in aliases:

            alias_normalized = (
                normalize_column_name(
                    alias
                )
            )

            if alias_normalized in normalized:

                return normalized[
                    alias_normalized
                ]

    return None


def resolve_target_name(
    df,
    requested_target
):
    """
    Dynamically resolve a target against
    the actual dataset.

    Example:
        revenue -> revenue if present
        revenue -> sales if revenue is absent
    """

    if df is None:
        return None

    requested = normalize_column_name(
        requested_target
    )

    aliases = TARGET_ALIASES.get(
        requested,
        [requested]
    )

    return resolve_column(
        df,
        requested,
        aliases
    )


# ============================================================
# TARGET DETECTION
# ============================================================

def detect_target_from_question(
    question,
    df=None
):
    """
    Detect target requested in a natural-language question.
    """

    if df is None:

        try:
            df = get_active_feature_data()
        except Exception:
            df = None

    question_normalized = normalize_text(
        question
    )

    # First check exact dataset columns
    if df is not None:

        candidates = []

        for column in df.columns:

            normalized_column = (
                normalize_column_name(
                    column
                )
            )

            if normalized_column in [
                "revenue",
                "sales",
                "profit",
                "quantity",
                "qty",
                "units",
                "units_sold"
            ]:

                candidates.append(
                    (
                        normalized_column,
                        column
                    )
                )

        for normalized_column, actual in candidates:

            display_name = str(
                actual
            ).replace(
                "_",
                " "
            )

            if (
                normalized_column
                in question_normalized
            ):

                return actual

            if (
                normalize_text(
                    display_name
                )
                in question_normalized
            ):

                return actual

    # Common target words
    target_patterns = [
        (
            r"\brevenue\b",
            "revenue"
        ),
        (
            r"\bsales\b",
            "sales"
        ),
        (
            r"\bsale\b",
            "sales"
        ),
        (
            r"\bprofit\b",
            "profit"
        ),
        (
            r"\bquantity\b",
            "quantity"
        ),
        (
            r"\bqty\b",
            "quantity"
        ),
        (
            r"\bunits?\b",
            "quantity"
        )
    ]

    for pattern, target in target_patterns:

        if re.search(
            pattern,
            question_normalized
        ):

            resolved = resolve_target_name(
                df,
                target
            )

            if resolved:
                return resolved

            return target

    return None


# ============================================================
# TARGETS ENDPOINT SUPPORT
# ============================================================

EXCLUDED_TARGET_COLUMNS = {
    "order_id",
    "id",
    "row_id",
    "order_date",
    "order_date_year",
    "order_date_month",
    "order_date_quarter",
    "order_date_day_of_week",
    "revenue_per_unit"
}


def get_available_targets(df):
    """
    Find numeric columns suitable as ML targets.
    """

    targets = []

    for column in df.columns:

        normalized = normalize_column_name(
            column
        )

        if normalized in EXCLUDED_TARGET_COLUMNS:
            continue

        series = df[column]

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        valid = int(
            numeric_series.notna().sum()
        )

        unique = int(
            numeric_series.dropna().nunique()
        )

        if (
            valid >= 2
            and unique >= 2
        ):

            targets.append(
                {
                    "name": str(column),
                    "dtype": str(
                        series.dtype
                    ),
                    "valid_values": valid,
                    "unique_values": unique
                }
            )

    return targets


# ============================================================
# DATE HELPERS
# ============================================================

MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12
}


DAY_MAP = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6
}


def extract_year(question):
    """
    Extract a four-digit year.
    """

    matches = re.findall(
        r"\b(19\d{2}|20\d{2}|21\d{2})\b",
        question
    )

    if not matches:
        return None

    return int(
        matches[-1]
    )


def extract_month(question):
    """
    Extract month number.
    """

    text = normalize_text(
        question
    )

    # Month names
    for name, number in MONTH_MAP.items():

        if re.search(
            rf"\b{re.escape(name)}\b",
            text
        ):

            return number

    # Numeric month:
    # month 11
    # month=11
    # month: 11
    match = re.search(
        r"\bmonth\s*(?:is|=|:)?\s*(1[0-2]|[1-9])\b",
        text
    )

    if match:

        return int(
            match.group(1)
        )

    return None


def extract_quarter(question):
    """
    Extract quarter.
    """

    text = normalize_text(
        question
    )

    match = re.search(
        r"\bq([1-4])\b",
        text
    )

    if match:

        return int(
            match.group(1)
        )

    match = re.search(
        r"\bquarter\s*(?:is|=|:)?\s*([1-4])\b",
        text
    )

    if match:

        return int(
            match.group(1)
        )

    return None


def extract_day_of_week(question):
    """
    Extract weekday as:
        Monday = 0
        Sunday = 6
    """

    text = normalize_text(
        question
    )

    for name, number in DAY_MAP.items():

        if re.search(
            rf"\b{re.escape(name)}\b",
            text
        ):

            return number

    return None


# ============================================================
# CATEGORICAL VALUE MATCHING
# ============================================================

def find_dataset_value_in_question(
    question,
    df,
    column
):
    """
    Find an actual value from a dataset column
    inside the user's question.

    This prevents fake values such as:
        Example Product
    from being silently used.
    """

    if df is None:
        return None

    if column not in df.columns:
        return None

    question_normalized = normalize_text(
        question
    )

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # Longest values first.
    # This helps when values contain other values.
    values = sorted(
        values,
        key=lambda x: len(
            normalize_text(x)
        ),
        reverse=True
    )

    for value in values:

        normalized_value = normalize_text(
            value
        )

        if not normalized_value:
            continue

        if len(
            normalized_value
        ) < 2:
            continue

        if normalized_value in question_normalized:

            return value

    return None


# ============================================================
# NUMERIC VALUE EXTRACTION
# ============================================================

def extract_numeric_feature_value(
    question,
    feature
):
    """
    Extract explicit numeric value for a feature.

    Examples:
        quantity 5
        quantity: 5
        quantity = 5
        discount 0.2
    """

    feature_text = (
        normalize_column_name(
            feature
        )
    )

    pattern = (
        rf"\b{re.escape(feature_text)}"
        rf"\s*(?:is|=|:|of)?\s*"
        rf"(-?\d+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        normalize_text(question)
    )

    if not match:
        return None

    try:

        return float(
            match.group(1)
        )

    except Exception:

        return None


# ============================================================
# PREDICTION INTENT
# ============================================================

def is_prediction_question(
    question
):
    """
    Detect prediction intent BEFORE chatbot processing.

    This is the key fix for the previous bug.
    """

    text = normalize_text(
        question
    )

    prediction_words = [
        "predict",
        "prediction",
        "forecast",
        "estimate",
        "estimated",
        "calculate",
        "expected",
        "expect",
        "how much",
        "what will",
        "what is the predicted",
        "give me prediction",
        "give prediction"
    ]

    target_words = [
        "revenue",
        "sales",
        "sale",
        "profit",
        "quantity",
        "qty",
        "units"
    ]

    has_prediction_word = any(
        word in text
        for word in prediction_words
    )

    has_target_word = any(
        re.search(
            rf"\b{re.escape(word)}\b",
            text
        )
        for word in target_words
    )

    # Strong prediction pattern
    if (
        has_prediction_word
        and has_target_word
    ):
        return True

    # If the question begins with prediction language
    if re.search(
        r"\b(predict|forecast|estimate|calculate)\b",
        text
    ):

        return True

    # "How much revenue..."
    if re.search(
        r"\bhow much\b.*\b(revenue|sales|profit|quantity)\b",
        text
    ):

        return True

    # "revenue for California..."
    if re.search(
        r"\b(revenue|sales|profit|quantity)\b.*\bfor\b",
        text
    ):

        return True

    return False


# ============================================================
# EXTRACT PREDICTION INPUTS
# ============================================================

def extract_prediction_inputs(
    question,
    metadata,
    df
):
    """
    Build model input using metadata-defined features.
    """

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

    values = {}

    missing = []

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    year = extract_year(
        question
    )

    month = extract_month(
        question
    )

    quarter = extract_quarter(
        question
    )

    weekday = extract_day_of_week(
        question
    )

    if quarter is None and month is not None:

        quarter = (
            (month - 1) // 3
        ) + 1

    for feature in selected_features:

        normalized_feature = (
            normalize_column_name(
                feature
            )
        )

        # ----------------------------------------------
        # YEAR
        # ----------------------------------------------

        if normalized_feature in [
            "order_date_year",
            "year"
        ]:

            if year is not None:

                values[feature] = float(
                    year
                )

            else:

                missing.append(
                    feature
                )

            continue

        # ----------------------------------------------
        # MONTH
        # ----------------------------------------------

        if normalized_feature in [
            "order_date_month",
            "month"
        ]:

            if month is not None:

                values[feature] = float(
                    month
                )

            else:

                missing.append(
                    feature
                )

            continue

        # ----------------------------------------------
        # QUARTER
        # ----------------------------------------------

        if normalized_feature in [
            "order_date_quarter",
            "quarter",
            "qtr",
            "qtr_id"
        ]:

            if quarter is not None:

                values[feature] = float(
                    quarter
                )

            else:

                missing.append(
                    feature
                )

            continue

        # ----------------------------------------------
        # DAY OF WEEK
        # ----------------------------------------------

        if normalized_feature in [
            "order_date_day_of_week",
            "day_of_week",
            "weekday"
        ]:

            if weekday is not None:

                values[feature] = float(
                    weekday
                )

            else:

                missing.append(
                    feature
                )

            continue

        # ----------------------------------------------
        # NUMERIC FEATURE
        # ----------------------------------------------

        if feature in numeric_features:

            number = (
                extract_numeric_feature_value(
                    question,
                    feature
                )
            )

            if number is not None:

                values[feature] = number

            else:

                missing.append(
                    feature
                )

            continue

        # ----------------------------------------------
        # CATEGORICAL FEATURE
        # ----------------------------------------------

        if feature in categorical_features:

            matched_value = (
                find_dataset_value_in_question(
                    question,
                    df,
                    feature
                )
            )

            if matched_value is not None:

                values[feature] = matched_value

            else:

                missing.append(
                    feature
                )

            continue

        # ----------------------------------------------
        # UNKNOWN FEATURE
        # ----------------------------------------------

        number = (
            extract_numeric_feature_value(
                question,
                feature
            )
        )

        if number is not None:

            values[feature] = number

        else:

            matched_value = (
                find_dataset_value_in_question(
                    question,
                    df,
                    feature
                )
            )

            if matched_value is not None:

                values[feature] = matched_value

            else:

                missing.append(
                    feature
                )

    return values, missing


# ============================================================
# NATURAL LANGUAGE PREDICTION HANDLER
# ============================================================

def handle_natural_language_prediction(
    question
):
    """
    Handle prediction locally.

    IMPORTANT:
    This function NEVER calls Hugging Face.
    """

    try:

        df = get_active_feature_data()

    except Exception as e:

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "error": (
                "Prediction dataset is not available: "
                + str(e)
            ),
            "answer": (
                "Prediction dataset is not available. "
                "Please upload a dataset or prepare "
                "feature_data.csv first."
            )
        }

    metadata = load_metadata()

    if not metadata:

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "error": "Model metadata not found.",
            "answer": (
                "No trained model metadata was found. "
                "Please train a model first."
            )
        }

    if not os.path.exists(
        MODEL_FILE
    ):

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "error": "Model file not found.",
            "answer": (
                "No trained model is available. "
                "Please train a model first."
            )
        }

    # --------------------------------------------------------
    # REQUESTED TARGET
    # --------------------------------------------------------

    requested_target = (
        detect_target_from_question(
            question,
            df
        )
    )

    current_target = metadata.get(
        "target"
    )

    if requested_target is None:

        requested_target = current_target

    resolved_requested_target = (
        resolve_target_name(
            df,
            requested_target
        )
    )

    # --------------------------------------------------------
    # TARGET MISMATCH
    # --------------------------------------------------------

    if (
        current_target
        and requested_target
        and normalize_column_name(
            requested_target
        )
        != normalize_column_name(
            current_target
        )
    ):

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "target": requested_target,
            "current_target": current_target,
            "requires_training": True,
            "answer": (
                f"The currently trained model predicts "
                f"'{current_target}', but your question "
                f"asks for '{requested_target}'. "
                f"Please train the model with "
                f"'{requested_target}' as the target first."
            )
        }

    if (
        current_target
        and resolved_requested_target is None
    ):

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "target": current_target,
            "answer": (
                f"The trained target '{current_target}' "
                f"was not found in the active dataset."
            )
        }

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    selected_features = metadata.get(
        "selected_features",
        []
    )

    if not selected_features:

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "answer": (
                "The trained model has no selected "
                "features in metadata."
            )
        }

    # --------------------------------------------------------
    # EXTRACT INPUTS
    # --------------------------------------------------------

    values, missing = (
        extract_prediction_inputs(
            question,
            metadata,
            df
        )
    )

    if missing:

        readable_missing = [
            str(item)
            for item in missing
        ]

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "target": current_target,
            "model": metadata.get(
                "best_model"
            ),
            "missing_features": readable_missing,
            "required_features": selected_features,
            "answer": (
                "I cannot calculate the prediction yet "
                "because these required inputs are missing: "
                + ", ".join(
                    readable_missing
                )
                + ". Please provide those values."
            )
        }

    # --------------------------------------------------------
    # BUILD MODEL INPUT
    # --------------------------------------------------------

    try:

        input_data = pd.DataFrame(
            [
                [
                    values[feature]
                    for feature in selected_features
                ]
            ],
            columns=selected_features
        )

        # Convert numeric columns
        numeric_features = metadata.get(
            "numeric_features",
            []
        )

        for feature in numeric_features:

            if feature in input_data.columns:

                input_data[
                    feature
                ] = pd.to_numeric(
                    input_data[
                        feature
                    ],
                    errors="coerce"
                )

        # Convert categorical columns
        categorical_features = metadata.get(
            "categorical_features",
            []
        )

        for feature in categorical_features:

            if feature in input_data.columns:

                input_data[
                    feature
                ] = input_data[
                    feature
                ].astype(str)

        # Check numeric conversion
        invalid_numeric = []

        for feature in numeric_features:

            if feature not in input_data.columns:
                continue

            if pd.isna(
                input_data.iloc[
                    0
                ][feature]
            ):

                invalid_numeric.append(
                    feature
                )

        if invalid_numeric:

            return {
                "success": False,
                "provider": "InsightAI Local Model",
                "type": "prediction",
                "target": current_target,
                "answer": (
                    "Invalid numeric values were supplied "
                    "for: "
                    + ", ".join(
                        invalid_numeric
                    )
                )
            }

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = joblib.load(
            MODEL_FILE
        )

        # ----------------------------------------------------
        # REAL MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )

        value = prediction[0]

        value = safe_json_value(
            value
        )

        return {
            "success": True,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "target": current_target,
            "model": metadata.get(
                "best_model"
            ),
            "prediction": value,
            "input": {
                feature: safe_json_value(
                    values.get(feature)
                )
                for feature in selected_features
            },
            "model_features": selected_features,
            "answer": (
                f"Predicted {current_target}: "
                f"{value:.2f}"
                if isinstance(
                    value,
                    (int, float)
                )
                else (
                    f"Predicted {current_target}: "
                    f"{value}"
                )
            )
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "success": False,
            "provider": "InsightAI Local Model",
            "type": "prediction",
            "target": current_target,
            "error": str(e),
            "answer": (
                "The local model could not calculate "
                "the prediction. Error: "
                + str(e)
            )
        }


# ============================================================
# DETERMINISTIC ANALYSIS
# ============================================================

def find_numeric_column(
    df,
    names
):
    """
    Dynamically find a numeric column using aliases.
    """

    if df is None:
        return None

    for name in names:

        column = resolve_column(
            df,
            name,
            [name]
        )

        if column is None:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if numeric.notna().sum() > 0:

            return column

    return None


def build_dataset_context():
    """
    Build a compact, factual dataset/model context for AI chat.
    All values are calculated from the active dataset and current
    model metadata. No dataset statistics are invented.
    """

    try:
        df = get_active_feature_data()
    except Exception:
        df = None

    metadata = load_metadata()
    results = load_model_results()
    metrics = get_best_model_metrics(
        metadata,
        results
    )

    if df is None:
        return {
            "dataset_available": False,
            "model": metadata.get("best_model", "unknown"),
            "target": metadata.get("target", "unknown")
        }

    column_names = [str(column) for column in df.columns]
    numeric_columns = [
        str(column)
        for column in df.select_dtypes(include=np.number).columns
    ]
    categorical_columns = [
        str(column)
        for column in df.select_dtypes(exclude=np.number).columns
    ]

    context = {
        "dataset_available": True,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": column_names,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target": metadata.get("target", "unknown"),
        "task_type": metadata.get("task_type", "regression"),
        "best_model": metadata.get("best_model", "unknown"),
        "selected_features": metadata.get("selected_features", []),
        "numeric_features": metadata.get("numeric_features", []),
        "categorical_features": metadata.get("categorical_features", []),
        "metrics": metrics
    }

    # Date range when a date-like column exists.
    date_column = resolve_column(
        df,
        "order_date",
        ["order_date", "orderdate", "date"]
    )

    if date_column:
        parsed_dates = pd.to_datetime(
            df[date_column],
            errors="coerce"
        ).dropna()

        if not parsed_dates.empty:
            context["date_column"] = str(date_column)
            context["date_start"] = parsed_dates.min().date().isoformat()
            context["date_end"] = parsed_dates.max().date().isoformat()

    # Useful numeric summaries.
    for key, aliases in {
        "revenue": TARGET_ALIASES["revenue"],
        "profit": TARGET_ALIASES["profit"],
        "quantity": TARGET_ALIASES["quantity"]
    }.items():
        column = resolve_column(df, key, aliases)
        if column:
            numeric = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()
            if not numeric.empty:
                context[f"{key}_column"] = str(column)
                context[f"{key}_total"] = safe_json_value(numeric.sum())
                context[f"{key}_average"] = safe_json_value(numeric.mean())

    return context


def _analysis_response(answer, **extra):
    """Create a consistent local InsightAI analysis response."""
    result = {
        "success": True,
        "provider": "InsightAI Analysis",
        "type": "analysis",
        "answer": repair_mojibake(answer)
    }
    result.update(extra)
    return result



def _question_column_candidates(df, question):
    """
    Dynamically match words in a question to real dataset columns.
    No fixed dataset schema is required.
    """
    text = normalize_text(question)
    columns = [str(c) for c in df.columns]
    matches = []

    # Exact normalized column/name phrase matches first.
    for col in columns:
        ncol = normalize_column_name(col)
        display = normalize_text(str(col).replace("_", " "))
        if ncol and re.search(rf"\b{re.escape(ncol)}\b", text):
            matches.append((col, 100.0))
        elif display and display in text:
            matches.append((col, 95.0))

    # Common semantic aliases.
    aliases = {
        "revenue": ["revenue", "sales", "sale", "selling", "turnover", "amount"],
        "profit": ["profit", "earnings", "net profit", "gross profit"],
        "quantity": ["quantity", "qty", "units", "units sold", "volume"],
        "product": ["product", "products", "item", "items", "product name", "product names", "item name", "item names"],
        "category": ["category", "type", "segment"],
        "subcategory": ["sub category", "subcategory", "sub-category"],
        "state": ["state", "province"],
        "region": ["region", "area", "zone"],
        "city": ["city", "town"],
        "customer": ["customer", "customers", "client", "clients", "buyer", "buyers"],
        "date": ["date", "day", "order date", "transaction date"],
        "month": ["month"],
        "year": ["year"],
    }
    for col in columns:
        ncol = normalize_column_name(col)
        for key, words in aliases.items():
            if ncol == key or any(
                normalize_column_name(w) == ncol for w in words
            ):
                if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
                    matches.append((col, 90.0))

    # Fuzzy match only against meaningful multi-character tokens.
    tokens = [t for t in re.findall(r"[a-z0-9_]+", text) if len(t) >= 4]
    normalized_columns = {normalize_column_name(c): c for c in columns}
    for token in tokens:
        close = get_close_matches(token, list(normalized_columns.keys()), n=1, cutoff=0.88)
        if close:
            matches.append((normalized_columns[close[0]], 70.0))

    # Preserve order by score and uniqueness.
    seen = set()
    result = []
    for col, score in sorted(matches, key=lambda x: -x[1]):
        if col not in seen:
            result.append((col, score))
            seen.add(col)
    return result


def _dynamic_metric_column(df, question):
    """
    Find the most likely numeric metric mentioned in the question.
    """
    candidates = _question_column_candidates(df, question)
    numeric = []
    for col, score in candidates:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            numeric.append((col, score))
    if numeric:
        return numeric[0][0]

    # If no explicit metric is mentioned, use common business measures.
    for alias in ["revenue", "sales", "profit", "quantity", "amount"]:
        col = resolve_column(df, alias, TARGET_ALIASES.get(alias, [alias]))
        if col is not None and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            return col
    return None


def _dynamic_group_column(df, question, metric=None):
    """
    Find a categorical/date dimension for 'by category', 'per state',
    'for each region', etc.
    """
    text = normalize_text(question)
    candidates = _question_column_candidates(df, question)

    # Explicit "by/per/for each <dimension>" gets priority.
    for col, _ in candidates:
        n = normalize_column_name(col)
        pretty = normalize_text(str(col).replace("_", " "))
        patterns = [
            rf"\bby\s+{re.escape(n)}\b",
            rf"\bper\s+{re.escape(n)}\b",
            rf"\bfor each\s+{re.escape(n)}\b",
            rf"\bacross\s+{re.escape(n)}\b",
            rf"\bfrom each\s+{re.escape(n)}\b",
        ]
        if any(re.search(p, text) for p in patterns) or pretty in text:
            if col != metric:
                return col

    # If the question says "which category/state/product..." choose that dimension.
    for col, _ in candidates:
        if col == metric:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col

    return None


def _dynamic_value_filter(df, question):
    """
    Find one or more real categorical values mentioned in the question.
    Returns a boolean mask when a strong value match exists.
    """
    text = normalize_text(question)
    best = None

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        values = (
            df[col].dropna().astype(str).str.strip().unique().tolist()
        )
        values = sorted(
            values,
            key=lambda v: len(normalize_text(v)),
            reverse=True
        )

        for value in values:
            nv = normalize_text(value)
            if len(nv) < 2:
                continue
            if nv in text:
                # Avoid matching very generic values such as "south" only
                # when the column itself is not relevant.
                score = len(nv)
                if best is None or score > best[0]:
                    best = (score, col, value)

    if best is None:
        return None

    _, col, value = best
    return df[col].astype(str).str.strip().map(normalize_text) == normalize_text(value)


def _format_dynamic_number(value):
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def dynamic_dataset_analysis(question):
    """
    General-purpose, schema-agnostic dataframe analysis.

    It intentionally runs after the stable explicit handlers in
    deterministic_analysis(), so existing answers and prediction behavior
    remain unchanged.
    """
    try:
        df = get_active_feature_data()
    except Exception:
        return None

    if df is None or df.empty:
        return None

    text = normalize_text(question)
    metric = _dynamic_metric_column(df, question)
    group = _dynamic_group_column(df, question, metric)

    # ---- top/bottom/highest/lowest by a dimension -----------------
    wants_top = bool(re.search(r"\b(top|highest|most|largest|maximum|max)\b", text))
    wants_bottom = bool(re.search(r"\b(bottom|lowest|least|smallest|minimum|min)\b", text))
    ranking_words = wants_top or wants_bottom

    if ranking_words and metric and group and not pd.api.types.is_numeric_dtype(df[group]):
        numeric = pd.to_numeric(df[metric], errors="coerce")
        temp = pd.DataFrame({
            "_group": df[group].astype(str),
            "_value": numeric
        }).dropna()
        if not temp.empty:
            totals = temp.groupby("_group")["_value"].sum().sort_values(
                ascending=wants_bottom
            )
            n_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", text)
            n = int(n_match.group(1)) if n_match else 1
            selected = totals.head(max(1, min(n, 20)))
            label = "lowest" if wants_bottom else "highest"
            lines = [
                f"{idx}: {_format_dynamic_number(val)}"
                for idx, val in selected.items()
            ]
            answer = (
                f"Based on the current dataset, the {label} "
                f"{group.replace('_', ' ')} values by total "
                f"{metric.replace('_', ' ')} are: "
                + "; ".join(lines)
                + "."
            )
            return _analysis_response(
                answer,
                operation="rank",
                metric=metric,
                group_by=group,
                rows_analyzed=int(len(temp)),
                results=[
                    {"group": str(idx), "value": safe_json_value(val)}
                    for idx, val in selected.items()
                ]
            )

    # ---- grouped aggregate: "revenue by category", "profit per state" ---
    aggregate_words = (
        "by " in text or "per " in text or "for each " in text
        or "across " in text or "breakdown" in text
        or "break down" in text
    )
    if metric and group and aggregate_words and not pd.api.types.is_numeric_dtype(df[group]):
        numeric = pd.to_numeric(df[metric], errors="coerce")
        temp = pd.DataFrame({
            "_group": df[group].astype(str),
            "_value": numeric
        }).dropna()
        if not temp.empty:
            # Mean when question explicitly asks average/mean; otherwise sum.
            if "average" in text or "mean" in text:
                grouped = temp.groupby("_group")["_value"].mean().sort_values(ascending=False)
                agg_name = "average"
            else:
                grouped = temp.groupby("_group")["_value"].sum().sort_values(ascending=False)
                agg_name = "total"
            # Avoid returning hundreds of categories.
            limited = grouped.head(20)
            lines = [
                f"{idx}: {_format_dynamic_number(val)}"
                for idx, val in limited.items()
            ]
            return _analysis_response(
                f"{agg_name.title()} {metric.replace('_', ' ')} by "
                f"{group.replace('_', ' ')}: " + "; ".join(lines) + ".",
                operation="groupby",
                metric=metric,
                group_by=group,
                aggregation=agg_name,
                results=[
                    {"group": str(idx), "value": safe_json_value(val)}
                    for idx, val in limited.items()
                ]
            )

    # ---- generic metric operations --------------------------------
    if metric:
        numeric = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not numeric.empty:
            filtered = numeric
            mask = _dynamic_value_filter(df, question)
            if mask is not None and len(mask) == len(df):
                filtered = pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()

            if len(filtered) > 0:
                if re.search(r"\b(average|mean|avg)\b", text):
                    value = filtered.mean()
                    return _analysis_response(
                        f"The average {metric.replace('_', ' ')} is {_format_dynamic_number(value)}.",
                        operation="average", metric=metric,
                        value=safe_json_value(value), rows_analyzed=int(len(filtered))
                    )
                if re.search(r"\b(total|sum)\b", text):
                    value = filtered.sum()
                    return _analysis_response(
                        f"The total {metric.replace('_', ' ')} is {_format_dynamic_number(value)}.",
                        operation="sum", metric=metric,
                        value=safe_json_value(value), rows_analyzed=int(len(filtered))
                    )
                if re.search(r"\b(maximum|maximum value|max|highest)\b", text):
                    value = filtered.max()
                    return _analysis_response(
                        f"The maximum {metric.replace('_', ' ')} is {_format_dynamic_number(value)}.",
                        operation="max", metric=metric,
                        value=safe_json_value(value)
                    )
                if re.search(r"\b(minimum|minimum value|min|lowest)\b", text):
                    value = filtered.min()
                    return _analysis_response(
                        f"The minimum {metric.replace('_', ' ')} is {_format_dynamic_number(value)}.",
                        operation="min", metric=metric,
                        value=safe_json_value(value)
                    )

    # ---- row count / unique count for a mentioned dimension --------
    if re.search(r"\b(how many|number of|count)\b", text):
        candidates = _question_column_candidates(df, question)
        for col, _ in candidates:
            if col in df.columns:
                unique = int(df[col].nunique(dropna=True))
                return _analysis_response(
                    f"There are {unique:,} unique values in {col.replace('_', ' ')}.",
                    operation="nunique", column=col, value=unique
                )

    # ---- correlation between two numeric columns -------------------
    if "correlation" in text or "correlate" in text or "relationship between" in text:
        numeric_cols = list(df.select_dtypes(include=np.number).columns)
        mentioned = [c for c, _ in _question_column_candidates(df, question) if c in numeric_cols]
        if len(mentioned) >= 2:
            a, b = mentioned[:2]
            corr = pd.to_numeric(df[a], errors="coerce").corr(
                pd.to_numeric(df[b], errors="coerce")
            )
            if pd.notna(corr):
                return _analysis_response(
                    f"The correlation between {a.replace('_', ' ')} and "
                    f"{b.replace('_', ' ')} is {float(corr):.4f}.",
                    operation="correlation", columns=[a, b],
                    value=safe_json_value(corr)
                )

    return None


def deterministic_analysis(question):
    """
    Handle factual dataset/model questions locally before HF.
    This keeps answers grounded in the actual active dataset.
    """

    try:
        df = get_active_feature_data()
    except Exception:
        return None

    text = normalize_text(question)
    rows = int(df.shape[0])
    columns = int(df.shape[1])
    metadata = load_metadata()
    results = load_model_results()
    metrics = get_best_model_metrics(metadata, results)

    # --------------------------------------------------------
    # EXPLAIN / SUMMARY
    # --------------------------------------------------------
    if (
        "explain my dataset" in text
        or "explain the dataset" in text
        or "describe my dataset" in text
        or "describe the dataset" in text
        or "tell me about my dataset" in text
        or "tell me about the dataset" in text
        or "dataset overview" in text
        or "dataset summary" in text
    ):
        column_names = [str(column) for column in df.columns]
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        target = metadata.get("target", "not trained")
        best_model = metadata.get("best_model", "not available")
        r2 = metrics.get("r2")

        metric_text = (
            f" R² is {float(r2):.4f}."
            if isinstance(r2, (int, float))
            else ""
        )

        answer = (
            f"Your active dataset contains {rows:,} rows and "
            f"{columns:,} columns. Key fields include "
            f"{', '.join(column_names[:14])}"
            + (" and additional engineered fields." if len(column_names) > 14 else ".")
            + f" Missing values: {missing:,}. Duplicate rows: {duplicates:,}. "
            f"The current prediction target is '{target}' and the current best model is "
            f"'{best_model}'.{metric_text}"
        )

        return _analysis_response(
            answer,
            rows=rows,
            column_count=columns,
            columns=column_names,
            missing_values=missing,
            duplicate_rows=duplicates,
            target=target,
            best_model=best_model,
            metrics=metrics
        )

    # --------------------------------------------------------
    # DATASET SIZE
    # --------------------------------------------------------
    if (
        "how many rows" in text
        or "number of rows" in text
        or "total rows" in text
        or "how many records" in text
        or "number of records" in text
        or "how large is the dataset" in text
    ):
        return _analysis_response(
            f"The dataset contains {rows:,} rows.",
            rows=rows
        )

    if (
        "how many columns" in text
        or "number of columns" in text
        or "total columns" in text
    ):
        return _analysis_response(
            f"The dataset contains {columns:,} columns.",
            column_count=columns
        )

    if (
        "dataset size" in text
        or "shape of dataset" in text
        or "dataset shape" in text
    ):
        return _analysis_response(
            f"The dataset has {rows:,} rows and {columns:,} columns.",
            rows=rows,
            column_count=columns
        )

    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------
    if (
        "what columns" in text
        or "list columns" in text
        or "show columns" in text
        or "column names" in text
        or "columns are in" in text
    ):
        column_names = [str(column) for column in df.columns]
        return _analysis_response(
            "Dataset columns: " + ", ".join(column_names),
            columns=column_names
        )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------
    if (
        "missing values" in text
        or "missing data" in text
        or "null values" in text
        or "nulls" in text
    ):
        total_missing = int(df.isna().sum().sum())
        return _analysis_response(
            f"The dataset contains {total_missing:,} missing values.",
            missing_values=total_missing
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------
    if "duplicate" in text or "duplicates" in text:
        duplicate_rows = int(df.duplicated().sum())
        return _analysis_response(
            f"The dataset contains {duplicate_rows:,} duplicate rows.",
            duplicate_rows=duplicate_rows
        )

    # --------------------------------------------------------
    # CURRENT MODEL / TARGET / METRICS
    # --------------------------------------------------------
    if (
        "current model" in text
        or "best model" in text
        or "which model" in text
        or "model are you using" in text
    ):
        best_model = metadata.get("best_model", "not available")
        target = metadata.get("target", "not available")
        return _analysis_response(
            f"The current best model is '{best_model}' and it predicts '{target}'.",
            best_model=best_model,
            target=target,
            metrics=metrics
        )

    if (
        "prediction target" in text
        or "current target" in text
        or "what is the target" in text
        or "what target" in text
    ):
        target = metadata.get("target", "not available")
        return _analysis_response(
            f"The current prediction target is '{target}'.",
            target=target
        )

    if (
        "r2" in text
        or "r²" in text
        or "r squared" in text
        or "model score" in text
        or "model performance" in text
        or "model metrics" in text
    ):
        r2 = metrics.get("r2")
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        parts = []
        if isinstance(r2, (int, float)):
            parts.append(f"R²: {float(r2):.4f}")
        if isinstance(mae, (int, float)):
            parts.append(f"MAE: {float(mae):.4f}")
        if isinstance(rmse, (int, float)):
            parts.append(f"RMSE: {float(rmse):.4f}")
        if not parts:
            return _analysis_response("Model metrics are not available yet.")
        return _analysis_response(
            "Current model metrics — " + " | ".join(parts) + ".",
            target=metadata.get("target"),
            best_model=metadata.get("best_model"),
            metrics=metrics
        )

    # --------------------------------------------------------
    # AVERAGES / TOTALS
    # --------------------------------------------------------
    numeric_requests = [
        ("revenue", ["revenue", "sales", "sales_amount", "total_sales"]),
        ("profit", ["profit", "net_profit", "gross_profit"]),
        ("quantity", ["quantity", "qty", "units", "units_sold"])
    ]

    for label, aliases in numeric_requests:
        if (
            f"average {label}" in text
            or f"mean {label}" in text
            or f"total {label}" in text
            or f"sum {label}" in text
        ):
            column = find_numeric_column(df, aliases)
            if column:
                numeric = pd.to_numeric(df[column], errors="coerce").dropna()
                if not numeric.empty:
                    if "average" in text or "mean" in text:
                        value = numeric.mean()
                        answer = f"The average {column} is {value:,.2f}."
                    else:
                        value = numeric.sum()
                        answer = f"The total {column} is {value:,.2f}."
                    return _analysis_response(
                        answer,
                        column=column,
                        value=safe_json_value(value)
                    )

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------
    if (
        "business insights" in text
        or "business insight" in text
        or "key insights" in text
        or "give me insights" in text
        or "important insights" in text
        or "analyze my business" in text
        or "business analysis" in text
    ):
        insights = []
        revenue_column = find_numeric_column(
            df, ["revenue", "sales", "sales_amount", "total_sales"]
        )
        profit_column = find_numeric_column(
            df, ["profit", "net_profit", "gross_profit"]
        )
        quantity_column = find_numeric_column(
            df, ["quantity", "qty", "units", "units_sold"]
        )

        if revenue_column:
            revenue = pd.to_numeric(df[revenue_column], errors="coerce").dropna()
            if not revenue.empty:
                insights.append(
                    f"Total {revenue_column}: {revenue.sum():,.2f}; average per row: {revenue.mean():,.2f}."
                )

                for group_name in ["category", "region", "state"]:
                    group_column = resolve_column(df, group_name, [group_name])
                    if group_column:
                        grouped = pd.DataFrame({
                            "group": df[group_column].astype(str),
                            "value": revenue.reindex(df.index)
                        }).dropna()
                        totals = grouped.groupby("group")["value"].sum().sort_values(ascending=False)
                        if not totals.empty:
                            top_name = str(totals.index[0])
                            top_value = float(totals.iloc[0])
                            insights.append(
                                f"Highest total {revenue_column} by {group_column}: {top_name} ({top_value:,.2f})."
                            )
                            break

        if profit_column:
            profit = pd.to_numeric(df[profit_column], errors="coerce").dropna()
            if not profit.empty:
                insights.append(
                    f"Total {profit_column}: {profit.sum():,.2f}; average per row: {profit.mean():,.2f}."
                )

        if quantity_column:
            quantity = pd.to_numeric(df[quantity_column], errors="coerce").dropna()
            if not quantity.empty:
                insights.append(
                    f"Average {quantity_column}: {quantity.mean():,.2f}."
                )

        insights.append(
            f"Data quality: {int(df.isna().sum().sum()):,} missing values and {int(df.duplicated().sum()):,} duplicate rows."
        )

        if metadata.get("target"):
            insights.append(
                f"Current ML target: {metadata.get('target')}; best model: {metadata.get('best_model', 'not available')}."
            )

        return _analysis_response(
            "Here are the main data-driven insights: " + " ".join(insights),
            insights=insights,
            target=metadata.get("target"),
            best_model=metadata.get("best_model"),
            metrics=metrics
        )

    return None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": (
            "InsightAI FastAPI Backend "
            "is running"
        ),
        "version": "2.0.0",
        "backend": "FastAPI"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "model_available": os.path.exists(
            MODEL_FILE
        ),
        "metadata_available": os.path.exists(
            METADATA_FILE
        ),
        "results_available": os.path.exists(
            RESULTS_FILE
        )
    }


# ============================================================
# UPLOAD
# ============================================================

@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):

    global uploaded_dataset
    global uploaded_filename

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    original_filename = file.filename

    filename = (
        original_filename
        .lower()
        .strip()
    )

    allowed_extensions = (
        ".csv",
        ".xlsx",
        ".xls",
        ".json"
    )

    if not filename.endswith(
        allowed_extensions
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Use CSV, XLSX, XLS or JSON."
            )
        )

    try:

        contents = await file.read()

        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        if filename.endswith(
            ".csv"
        ):

            df = pd.read_csv(
                BytesIO(contents)
            )

        # ----------------------------------------------------
        # READ EXCEL
        # ----------------------------------------------------

        elif filename.endswith(
            (
                ".xlsx",
                ".xls"
            )
        ):

            df = pd.read_excel(
                BytesIO(contents)
            )

        # ----------------------------------------------------
        # READ JSON
        # ----------------------------------------------------

        else:

            try:

                df = pd.read_json(
                    BytesIO(contents)
                )

            except Exception:

                data = json.loads(
                    contents.decode(
                        "utf-8"
                    )
                )

                if isinstance(
                    data,
                    list
                ):

                    df = pd.DataFrame(
                        data
                    )

                else:

                    df = pd.json_normalize(
                        data
                    )

        if df.empty:

            raise HTTPException(
                status_code=400,
                detail="Uploaded dataset is empty."
            )

        # ----------------------------------------------------
        # CLEAN COLUMN NAMES
        # ----------------------------------------------------

        df = clean_columns(
            df
        )

        # ----------------------------------------------------
        # BASIC CLEANING
        # ----------------------------------------------------

        df = df.dropna(
            how="all"
        )

        df = df.drop_duplicates()

        # ----------------------------------------------------
        # DATE FEATURES
        # ----------------------------------------------------

        date_candidates = [
            "order_date",
            "orderdate",
            "date",
            "order_date_time"
        ]

        date_column = None

        for candidate in date_candidates:

            if candidate in df.columns:

                date_column = candidate
                break

        if date_column:

            parsed_date = pd.to_datetime(
                df[date_column],
                errors="coerce"
            )

            df[
                "order_date_year"
            ] = parsed_date.dt.year

            df[
                "order_date_month"
            ] = parsed_date.dt.month

            df[
                "order_date_quarter"
            ] = parsed_date.dt.quarter

            df[
                "order_date_day_of_week"
            ] = parsed_date.dt.dayofweek

        # ----------------------------------------------------
        # NUMERIC CLEANING
        # ----------------------------------------------------

        numeric_candidates = [
            "quantity",
            "qty",
            "units",
            "units_sold",
            "quantity_ordered",
            "unit_price",
            "price_each",
            "revenue",
            "sales",
            "sales_amount",
            "total_sales",
            "profit",
            "net_profit",
            "gross_profit",
            "discount",
            "cost"
        ]

        converted_numeric_columns = []

        for column in numeric_candidates:

            if column not in df.columns:
                continue

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if converted.notna().sum() > 0:

                df[
                    column
                ] = converted

                converted_numeric_columns.append(
                    column
                )

        # ----------------------------------------------------
        # PROFIT MARGIN
        # ----------------------------------------------------

        revenue_column = resolve_column(
            df,
            "revenue",
            TARGET_ALIASES[
                "revenue"
            ]
        )

        profit_column = resolve_column(
            df,
            "profit",
            TARGET_ALIASES[
                "profit"
            ]
        )

        if (
            revenue_column
            and profit_column
        ):

            revenue_numeric = pd.to_numeric(
                df[revenue_column],
                errors="coerce"
            )

            profit_numeric = pd.to_numeric(
                df[profit_column],
                errors="coerce"
            )

            valid_revenue = (
                revenue_numeric
                .replace(
                    0,
                    np.nan
                )
            )

            df[
                "profit_margin"
            ] = (
                profit_numeric
                / valid_revenue
                * 100
            )

        # ----------------------------------------------------
        # REVENUE PER UNIT
        # ----------------------------------------------------

        quantity_column = resolve_column(
            df,
            "quantity",
            TARGET_ALIASES[
                "quantity"
            ]
        )

        if (
            revenue_column
            and quantity_column
        ):

            revenue_numeric = pd.to_numeric(
                df[revenue_column],
                errors="coerce"
            )

            quantity_numeric = pd.to_numeric(
                df[quantity_column],
                errors="coerce"
            )

            valid_quantity = (
                quantity_numeric
                .replace(
                    0,
                    np.nan
                )
            )

            df[
                "revenue_per_unit"
            ] = (
                revenue_numeric
                / valid_quantity
            )

        # ----------------------------------------------------
        # STORE ACTIVE DATASET
        # ----------------------------------------------------

        uploaded_dataset = df.copy()

        uploaded_filename = (
            original_filename
        )

        # ----------------------------------------------------
        # SAVE UPLOADED DATASET
        # ----------------------------------------------------

        df.to_csv(
            UPLOADED_FINAL_FILE,
            index=False
        )

        df.to_csv(
            UPLOADED_FEATURE_FILE,
            index=False
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        missing_values = int(
            df.isna().sum().sum()
        )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        numeric_columns = [
            str(column)
            for column in df.select_dtypes(
                include=np.number
            ).columns
        ]

        return {
            "success": True,
            "status": "success",
            "filename": original_filename,
            "rows": int(
                df.shape[0]
            ),
            "columns": int(
                df.shape[1]
            ),
            "column_names": [
                str(column)
                for column in df.columns
            ],
            "numeric_columns": numeric_columns,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "converted_numeric_columns": (
                converted_numeric_columns
            ),
            "saved_files": [
                UPLOADED_FINAL_FILE,
                UPLOADED_FEATURE_FILE
            ],
            "message": (
                "Dataset uploaded and processed successfully."
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process uploaded file: "
                + str(e)
            )
        )


# ============================================================
# DATASET PREVIEW
# ============================================================

@app.get("/dataset/preview")
def dataset_preview(
    rows: int = 10
):

    df = get_dataset()

    rows = max(
        1,
        min(
            int(rows),
            100
        )
    )

    preview = (
        df.head(rows)
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .where(
            lambda x: pd.notna(x),
            ""
        )
    )

    return {
        "success": True,
        "status": "success",
        "rows": len(preview),
        "columns": [
            str(column)
            for column in df.columns
        ],
        "data": preview.to_dict(
            orient="records"
        )
    }


# ============================================================
# DATASET INFO
# ============================================================

@app.get("/dataset/info")
def dataset_info():

    df = get_dataset()

    numeric_columns = [
        str(column)
        for column in df.select_dtypes(
            include=np.number
        ).columns
    ]

    categorical_columns = [
        str(column)
        for column in df.select_dtypes(
            exclude=np.number
        ).columns
    ]

    missing_by_column = {}

    for column in df.columns:

        count = int(
            df[column].isna().sum()
        )

        if count > 0:

            missing_by_column[
                str(column)
            ] = count

    duplicate_rows = int(
        df.duplicated().sum()
    )

    total_cells = (
        df.shape[0]
        * df.shape[1]
    )

    missing_cells = int(
        df.isna().sum().sum()
    )

    if total_cells > 0:

        completeness = (
            (
                total_cells
                - missing_cells
            )
            / total_cells
            * 100
        )

    else:

        completeness = 0.0

    return {
        "success": True,
        "status": "success",
        "rows": int(
            df.shape[0]
        ),
        "columns": int(
            df.shape[1]
        ),
        "column_count": int(
            df.shape[1]
        ),
        "column_names": [
            str(column)
            for column in df.columns
        ],
        "numeric_columns": numeric_columns,
        "numeric_features": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": missing_cells,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "duplicates": duplicate_rows,
        "completeness": round(
            completeness,
            2
        ),
        "preview": records_to_safe_json(
            df.head(10).to_dict(
                orient="records"
            )
        )
    }


# ============================================================
# TARGETS
# ============================================================

@app.get("/targets")
def targets():

    df = get_dataset()

    available_targets = (
        get_available_targets(
            df
        )
    )

    return {
        "success": True,
        "targets": available_targets,
        "count": len(
            available_targets
        )
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/model/status")
def model_status():

    metadata = load_metadata()

    results = load_model_results()

    metrics = get_best_model_metrics(
        metadata,
        results
    )

    model_exists = os.path.exists(
        MODEL_FILE
    )

    metadata_exists = os.path.exists(
        METADATA_FILE
    )

    results_exists = os.path.exists(
        RESULTS_FILE
    )

    return {
        "success": True,
        "model_exists": model_exists,
        "metadata_exists": metadata_exists,
        "results_exists": results_exists,
        "model_path": MODEL_FILE,
        "metadata_path": METADATA_FILE,
        "results_path": RESULTS_FILE,
        "target": metadata.get(
            "target"
        ),
        "task_type": metadata.get(
            "task_type",
            "regression"
        ),
        "best_model": metadata.get(
            "best_model"
        ),
        "selected_features": metadata.get(
            "selected_features",
            []
        ),
        "numeric_features": metadata.get(
            "numeric_features",
            []
        ),
        "categorical_features": metadata.get(
            "categorical_features",
            []
        ),
        "metrics": metrics,
        "results": results,
        "model_results": results
    }


# ============================================================
# PREDICTION INFO
# ============================================================

@app.get("/predict/info")
def prediction_info():

    metadata = load_metadata()

    if not metadata:

        return {
            "success": False,
            "message": (
                "No trained model metadata found."
            )
        }

    return {
        "success": True,
        "target": metadata.get(
            "target"
        ),
        "task_type": metadata.get(
            "task_type",
            "regression"
        ),
        "best_model": metadata.get(
            "best_model"
        ),
        "selected_features": metadata.get(
            "selected_features",
            []
        ),
        "numeric_features": metadata.get(
            "numeric_features",
            []
        ),
        "categorical_features": metadata.get(
            "categorical_features",
            []
        )
    }


# ============================================================
# TRAIN MODEL
# ============================================================

@app.post("/train")
def train_model(
    target: str
):

    global uploaded_dataset

    target = str(
        target
    ).strip()

    if not target:

        raise HTTPException(
            status_code=400,
            detail="Target is required."
        )

    # --------------------------------------------------------
    # ACTIVE DATASET
    # --------------------------------------------------------

    try:

        if uploaded_dataset is not None:

            df = uploaded_dataset.copy()

            # Preserve current project training workflow.
            df.to_csv(
                FEATURE_FILE,
                index=False
            )

        else:

            if os.path.exists(
                FEATURE_FILE
            ):

                df = pd.read_csv(
                    FEATURE_FILE
                )

            elif os.path.exists(
                FINAL_FILE
            ):

                df = pd.read_csv(
                    FINAL_FILE
                )

            else:

                raise HTTPException(
                    status_code=404,
                    detail=(
                        "No feature dataset available."
                    )
                )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load training dataset: "
                + str(e)
            )
        )

    df = clean_columns(
        df
    )

    # --------------------------------------------------------
    # RESOLVE TARGET
    # --------------------------------------------------------

    resolved_target = resolve_target_name(
        df,
        target
    )

    if resolved_target is None:

        available_numeric = []

        for column in df.columns:

            numeric = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if (
                numeric.notna().sum() >= 2
                and numeric.dropna().nunique() >= 2
            ):

                available_numeric.append(
                    str(column)
                )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Target '{target}' was not "
                    "found in the active dataset."
                ),
                "requested_target": target,
                "available_numeric_targets": (
                    available_numeric
                )
            }
        )

    # --------------------------------------------------------
    # TARGET VALIDATION
    # --------------------------------------------------------

    target_series = pd.to_numeric(
        df[resolved_target],
        errors="coerce"
    )

    valid_values = target_series.notna()

    if valid_values.sum() < 2:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Target '{resolved_target}' "
                "does not contain enough numeric values."
            )
        )

    if (
        target_series[
            valid_values
        ].nunique()
        < 2
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Target '{resolved_target}' "
                "must contain at least two "
                "unique values."
            )
        )

    # --------------------------------------------------------
    # ENVIRONMENT FOR TRAINING SCRIPT
    # --------------------------------------------------------

    env = os.environ.copy()

    env[
        "INSIGHTAI_TARGET"
    ] = str(
        resolved_target
    )

    env[
        "INSIGHTAI_FEATURE_FILE"
    ] = FEATURE_FILE

    # --------------------------------------------------------
    # CHECK TRAINING SCRIPT
    # --------------------------------------------------------

    if not os.path.exists(
        TRAINING_SCRIPT
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "model_training.py was not found at: "
                + TRAINING_SCRIPT
            )
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    try:

        process = subprocess.run(
            [
                sys.executable,
                TRAINING_SCRIPT
            ],
            cwd=PROJECT_FOLDER,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800
        )

    except subprocess.TimeoutExpired:

        raise HTTPException(
            status_code=504,
            detail=(
                "Model training exceeded the "
                "30 minute timeout."
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to start model training: "
                + str(e)
            )
        )

    # --------------------------------------------------------
    # TRAINING FAILURE
    # --------------------------------------------------------

    if process.returncode != 0:

        training_output = (
            process.stdout
            + "\n"
            + process.stderr
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Model training failed.",
                "target": resolved_target,
                "training_output": (
                    training_output[-12000:]
                )
            }
        )

    # --------------------------------------------------------
    # VERIFY OUTPUT
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_FILE
    ):

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Training completed but "
                    "best_model.joblib was not created."
                ),
                "training_output": (
                    process.stdout[-12000:]
                )
            }
        )

    # --------------------------------------------------------
    # READ NEW METADATA
    # --------------------------------------------------------

    metadata = load_metadata()

    results = load_model_results()

    metrics = get_best_model_metrics(
        metadata,
        results
    )

    return {
        "success": True,
        "status": "success",
        "message": (
            "Model training completed successfully."
        ),
        "target": metadata.get(
            "target",
            resolved_target
        ),
        "task_type": metadata.get(
            "task_type",
            "regression"
        ),
        "best_model": metadata.get(
            "best_model"
        ),
        "selected_features": metadata.get(
            "selected_features",
            []
        ),
        "numeric_features": metadata.get(
            "numeric_features",
            []
        ),
        "categorical_features": metadata.get(
            "categorical_features",
            []
        ),
        "metrics": metrics,
        "results": results,
        "training_output": (
            process.stdout[-12000:]
        )
    }


# ============================================================
# GENERIC PREDICT API
# ============================================================

@app.post("/predict")
def predict(
    request: dict[str, Any]
):

    metadata = load_metadata()

    if not metadata:

        raise HTTPException(
            status_code=404,
            detail=(
                "Model metadata not found. "
                "Train the model first."
            )
        )

    if not os.path.exists(
        MODEL_FILE
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Trained model not found. "
                "Train the model first."
            )
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

    target = metadata.get(
        "target",
        "prediction"
    )

    if not selected_features:

        raise HTTPException(
            status_code=500,
            detail=(
                "Selected model features "
                "were not found in metadata."
            )
        )

    if not isinstance(
        request,
        dict
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction request must be a JSON object."
            )
        )

    # --------------------------------------------------------
    # MISSING FEATURES
    # --------------------------------------------------------

    missing = [
        feature
        for feature in selected_features
        if feature not in request
    ]

    if missing:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Required prediction features "
                    "are missing."
                ),
                "missing_features": missing,
                "required_features": selected_features
            }
        )

    try:

        input_values = {}

        for feature in selected_features:

            input_values[
                feature
            ] = request[
                feature
            ]

        input_data = pd.DataFrame(
            [
                [
                    input_values[
                        feature
                    ]
                    for feature in selected_features
                ]
            ],
            columns=selected_features
        )

        # Numeric
        for feature in numeric_features:

            if feature in input_data.columns:

                input_data[
                    feature
                ] = pd.to_numeric(
                    input_data[
                        feature
                    ],
                    errors="coerce"
                )

        # Categorical
        for feature in categorical_features:

            if feature in input_data.columns:

                input_data[
                    feature
                ] = input_data[
                    feature
                ].astype(str)

        # Validate numeric values
        for feature in numeric_features:

            if feature not in input_data.columns:
                continue

            value = input_data.iloc[
                0
            ][feature]

            if pd.isna(value):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid numeric value "
                        f"for feature '{feature}'."
                    )
                )

        model = joblib.load(
            MODEL_FILE
        )

        prediction = model.predict(
            input_data
        )

        value = safe_json_value(
            prediction[0]
        )

        return {
            "success": True,
            "status": "success",
            "target": target,
            "model": metadata.get(
                "best_model"
            ),
            "prediction": value,
            "input": {
                feature: safe_json_value(
                    input_values[
                        feature
                    ]
                )
                for feature in selected_features
            },
            "model_features": selected_features
        }

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                + str(e)
            )
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.get("/model/comparison")
def model_comparison():

    if not os.path.exists(
        RESULTS_FILE
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "model_results.csv not found."
            )
        )

    results = load_model_results()

    metadata = load_metadata()

    metrics = get_best_model_metrics(
        metadata,
        results
    )

    return {
        "success": True,
        "status": "success",
        "target": metadata.get(
            "target"
        ),
        "best_model": metadata.get(
            "best_model"
        ),
        "metrics": metrics,
        "rows": results,
        "results": results,
        "model_results": results
    }


# ============================================================
# SCATTER DATA
# ============================================================

@app.get("/dataset/scatter")
def dataset_scatter():

    df = get_dataset()

    # Try revenue/sales
    x_column = find_numeric_column(
        df,
        [
            "revenue",
            "sales",
            "sales_amount",
            "total_sales"
        ]
    )

    # Try profit margin first
    y_column = find_numeric_column(
        df,
        [
            "profit_margin",
            "margin",
            "profit_percentage"
        ]
    )

    # If profit margin unavailable, use profit
    if y_column is None:

        y_column = find_numeric_column(
            df,
            [
                "profit",
                "net_profit",
                "gross_profit"
            ]
        )

    if (
        x_column is None
        or y_column is None
    ):

        return {
            "success": True,
            "status": "empty",
            "x_column": x_column,
            "y_column": y_column,
            "data": []
        }

    working = df[
        [
            x_column,
            y_column
        ]
    ].copy()

    working[
        x_column
    ] = pd.to_numeric(
        working[
            x_column
        ],
        errors="coerce"
    )

    working[
        y_column
    ] = pd.to_numeric(
        working[
            y_column
        ],
        errors="coerce"
    )

    working = working.dropna()

    # Limit chart payload
    if len(working) > 3000:

        working = working.sample(
            n=3000,
            random_state=42
        )

    data = []

    for _, row in working.iterrows():

        data.append(
            {
                "x": safe_json_value(
                    row[
                        x_column
                    ]
                ),
                "y": safe_json_value(
                    row[
                        y_column
                    ]
                )
            }
        )

    return {
        "success": True,
        "status": "success",
        "x_column": x_column,
        "y_column": y_column,
        "count": len(data),
        "data": data
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api")
def api_info():

    return {
        "success": True,
        "name": "InsightAI",
        "backend": "FastAPI",
        "version": "2.0.0",
        "services": [
            "/",
            "/health",
            "/upload",
            "/dataset/preview",
            "/dataset/info",
            "/dataset/scatter",
            "/targets",
            "/model/status",
            "/predict/info",
            "/train",
            "/predict",
            "/model/comparison",
            "/chat"
        ]
    }


# ============================================================
# CHAT REQUEST
# ============================================================


# ============================================================
# FUTURE MONTH FORECASTING
# ============================================================

def is_future_forecast_question(question):
    """Detect future month/year forecasting requests."""
    text = normalize_text(question)
    future_words = re.search(
        r"\b(next|future|upcoming|coming|forecast|predict)\b",
        text
    )
    month_words = re.search(
        r"\b(month|months|monthly)\b",
        text
    )
    relative_words = re.search(
        r"\b(one|1|two|2|three|3|four|4|five|5|six|6|one|2|3|6)\s+months?\b",
        text
    )
    return bool(future_words and (month_words or relative_words))


def extract_forecast_horizon(question):
    """Return number of future months requested; default is one."""
    text = normalize_text(question)

    word_to_number = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }

    match = re.search(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b", text)
    if match:
        raw = match.group(1)
        return word_to_number.get(raw, int(raw) if raw.isdigit() else 1)

    if re.search(r"\bnext\s+month\b|\bone\s+month\b", text):
        return 1

    return 1


def _monthly_forecast_values(df, target_column, horizon):
    """Forecast future monthly totals using trend + annual seasonality."""
    work = df[[target_column]].copy()

    date_column = resolve_column(
        df,
        "order_date",
        ["order_date", "orderdate", "date", "transaction_date"]
    )

    if date_column is None:
        return None, "A date column is required for future month forecasting."

    work["_date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )
    work["_value"] = pd.to_numeric(
        work[target_column],
        errors="coerce"
    )
    work = work.dropna(subset=["_date", "_value"])

    if work.empty:
        return None, f"No valid date and '{target_column}' values are available."

    monthly = (
        work.set_index("_date")["_value"]
        .resample("MS")
        .sum()
        .dropna()
    )

    if len(monthly) < 3:
        return None, "At least 3 historical months are required for a future forecast."

    # Use a compact trend + annual seasonal model.
    y = monthly.to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)
    period = 12.0
    X = np.column_stack([
        np.ones(len(y)),
        t,
        np.sin(2 * np.pi * t / period),
        np.cos(2 * np.pi * t / period),
    ])

    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        coef = np.array([float(y.mean()), 0.0, 0.0, 0.0])

    last_month = monthly.index.max()
    future_dates = pd.date_range(
        last_month + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    future_t = np.arange(len(y), len(y) + horizon, dtype=float)
    XF = np.column_stack([
        np.ones(horizon),
        future_t,
        np.sin(2 * np.pi * future_t / period),
        np.cos(2 * np.pi * future_t / period),
    ])
    predictions = XF @ coef

    # Prevent impossible negative business totals.
    predictions = np.maximum(predictions, 0.0)

    return {
        "historical_months": int(len(monthly)),
        "history_start": monthly.index.min().strftime("%Y-%m"),
        "history_end": monthly.index.max().strftime("%Y-%m"),
        "future_dates": future_dates,
        "predictions": predictions,
        "historical_monthly": monthly,
    }, None


def handle_future_forecast(question):
    """Handle natural-language future monthly revenue/profit/quantity forecasts."""
    try:
        df = get_active_feature_data()
    except Exception as e:
        return {
            "success": False,
            "provider": "InsightAI Forecast Engine",
            "type": "future_forecast",
            "error": str(e),
            "answer": "Forecast dataset is not available. Please upload or prepare the dataset first."
        }

    target = detect_target_from_question(question, df)
    if target is None:
        metadata = load_metadata()
        target = metadata.get("target")

    # If the user only says “next two months” and no active model metadata
    # is available, use the common business target present in the dataset.
    if target is None:
        for fallback_target in ["revenue", "sales", "profit", "quantity"]:
            fallback_column = resolve_target_name(df, fallback_target)
            if fallback_column is not None:
                target = fallback_column
                break

    resolved_target = resolve_target_name(df, target) if target else None
    if resolved_target is None:
        return {
            "success": False,
            "provider": "InsightAI Forecast Engine",
            "type": "future_forecast",
            "answer": "I could not identify which numeric column you want to forecast. Please mention revenue, profit, quantity, or another numeric dataset column."
        }

    if not pd.api.types.is_numeric_dtype(df[resolved_target]):
        numeric = pd.to_numeric(df[resolved_target], errors="coerce")
        if numeric.notna().sum() == 0:
            return {
                "success": False,
                "provider": "InsightAI Forecast Engine",
                "type": "future_forecast",
                "answer": f"The column '{resolved_target}' does not contain usable numeric values for forecasting."
            }

    horizon = max(1, min(extract_forecast_horizon(question), 12))
    result, error = _monthly_forecast_values(df, resolved_target, horizon)
    if result is None:
        return {
            "success": False,
            "provider": "InsightAI Forecast Engine",
            "type": "future_forecast",
            "target": resolved_target,
            "answer": error
        }

    rows = []
    for date_value, prediction in zip(result["future_dates"], result["predictions"]):
        rows.append({
            "month": date_value.strftime("%B %Y"),
            "forecast": round(float(prediction), 2)
        })

    total = float(np.sum(result["predictions"]))
    target_label = str(resolved_target).replace("_", " ")
    model = load_metadata().get("best_model", "not used")

    lines = [
        f"{row['month']}: {_format_dynamic_number(row['forecast'])}"
        for row in rows
    ]

    answer = (
        f"Future {target_label} forecast for the next {horizon} "
        f"month{'s' if horizon != 1 else ''}: "
        + "; ".join(lines)
        + f". Total forecast: {_format_dynamic_number(total)}. "
        f"This forecast uses the dataset's historical monthly totals "
        f"({result['history_start']} to {result['history_end']}) with "
        f"trend and annual seasonality; it is separate from the current "
        f"row-level ML model ({model})."
    )

    return {
        "success": True,
        "provider": "InsightAI Forecast Engine",
        "type": "future_forecast",
        "target": resolved_target,
        "horizon_months": horizon,
        "history_start": result["history_start"],
        "history_end": result["history_end"],
        "forecast": rows,
        "total_forecast": round(total, 2),
        "answer": answer
    }

def extract_question_from_payload(
    payload
):
    """
    Supports:
        {"question": "..."}
        {"message": "..."}
        {"prompt": "..."}
    """

    if not isinstance(
        payload,
        dict
    ):

        return ""

    for key in [
        "question",
        "message",
        "prompt",
        "query"
    ]:

        value = payload.get(
            key
        )

        if value is not None:

            return repair_mojibake(
                str(value)
            ).strip()

    return ""


# ============================================================
# HUGGING FACE CHAT
# ============================================================

def ask_huggingface(
    question
):
    """
    Call Hugging Face only for normal
    conversational/analysis questions.

    Prediction questions never reach here.
    """

    if hf_client is None:

        return {
            "success": False,
            "provider": "Hugging Face",
            "error": (
                "Hugging Face client is not configured."
            ),
            "answer": (
                "AI chat is not configured. "
                "The local InsightAI prediction "
                "and analysis services are still available."
            )
        }

    metadata = load_metadata()

    target = metadata.get(
        "target",
        "unknown"
    )

    best_model = metadata.get(
        "best_model",
        "unknown"
    )

    try:
        dataset_context = build_dataset_context()
    except Exception:
        dataset_context = {
            "dataset_available": False
        }

    context = (
        "You are InsightAI, an AI data analytics assistant. "
        "Answer using the factual dataset and model context below. "
        "Do not invent statistics, columns, dates, metrics, or predictions. "
        "If a requested fact is not present in the context, say that it is not available. "
        "Explain technical results in simple language when useful. "
        "For exact predictions, the local InsightAI prediction service handles them.\n\n"
        "CURRENT INSIGHTAI CONTEXT:\n"
        + json.dumps(
            dataset_context,
            ensure_ascii=False,
            default=str
        )
    )

    try:

        response = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": context
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        answer = ""

        if (
            response
            and response.choices
        ):

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        answer = repair_mojibake(
            answer
        )

        return {
            "success": True,
            "provider": "Hugging Face",
            "model": HF_MODEL,
            "type": "chat",
            "target": target,
            "best_model": best_model,
            "analysis_used": False,
            "answer": answer
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "success": False,
            "provider": "Hugging Face",
            "model": HF_MODEL,
            "error": str(e),
            "answer": (
                "Hugging Face AI service could not "
                "process the request. "
                "The local InsightAI services are still available."
            )
        }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    payload: dict[str, Any]
):

    question = (
        extract_question_from_payload(
            payload
        )
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide a question."
            )
        )

    # ========================================================
    # CRITICAL ORDER
    # ========================================================
    # Prediction MUST be checked first.
    #
    # Previous bug:
    #     question -> HF
    #
    # Correct:
    #     question
    #        ↓
    #     prediction intent?
    #        ↓ YES
    #     local model
    #
    # ========================================================

    if is_future_forecast_question(
        question
    ):

        result = handle_future_forecast(
            question
        )

        if "answer" in result:
            result["answer"] = repair_mojibake(
                result["answer"]
            )

        return result

    if is_prediction_question(
        question
    ):

        result = (
            handle_natural_language_prediction(
                question
            )
        )

        # Repair every visible answer
        if "answer" in result:

            result[
                "answer"
            ] = repair_mojibake(
                result[
                    "answer"
                ]
            )

        return result

    # --------------------------------------------------------
    # Deterministic analysis
    # --------------------------------------------------------

    analysis_result = (
        deterministic_analysis(
            question
        )
    )

    if analysis_result is not None:

        if "answer" in analysis_result:

            analysis_result[
                "answer"
            ] = repair_mojibake(
                analysis_result[
                    "answer"
                ]
            )

        return analysis_result

    # --------------------------------------------------------
    # Dynamic dataset analysis
    # --------------------------------------------------------

    dynamic_result = dynamic_dataset_analysis(
        question
    )

    if dynamic_result is not None:

        if "answer" in dynamic_result:

            dynamic_result[
                "answer"
            ] = repair_mojibake(
                dynamic_result[
                    "answer"
                ]
            )

        return dynamic_result

    # --------------------------------------------------------
    # Normal AI chat
    # --------------------------------------------------------

    return ask_huggingface(
        question
    )


# ============================================================
# STARTUP INFORMATION
# ============================================================

@app.on_event(
    "startup"
)
def startup_event():

    print()
    print("=" * 70)
    print("INSIGHTAI FASTAPI BACKEND")
    print("=" * 70)

    print(
        "Project folder:",
        PROJECT_FOLDER
    )

    print(
        "Output folder:",
        OUTPUT_FOLDER
    )

    print(
        "Model available:",
        os.path.exists(
            MODEL_FILE
        )
    )

    print(
        "Metadata available:",
        os.path.exists(
            METADATA_FILE
        )
    )

    metadata = load_metadata()

    if metadata:

        print(
            "Current target:",
            metadata.get(
                "target"
            )
        )

        print(
            "Best model:",
            metadata.get(
                "best_model"
            )
        )

        print(
            "Selected features:",
            len(
                metadata.get(
                    "selected_features",
                    []
                )
            )
        )

    print(
        "HF configured:",
        bool(
            hf_client
        )
    )

    print("=" * 70)
    print()


# ============================================================
# END
# ============================================================

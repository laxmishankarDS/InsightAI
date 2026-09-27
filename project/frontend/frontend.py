import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightAI - Profit Margin Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 0;
}

.dashboard-subtitle {
    font-size: 17px;
    color: #6b7280;
    margin-top: 5px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
}

.metric-title {
    color: #6b7280;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #111827;
    font-size: 30px;
    font-weight: 800;
    margin-top: 5px;
}

.metric-description {
    color: #9ca3af;
    font-size: 12px;
    margin-top: 3px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #111827;
    margin-top: 20px;
    margin-bottom: 10px;
}

.success-box {
    background: #ecfdf5;
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 15px;
    color: #065f46;
}

.warning-box {
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 15px;
    color: #92400e;
}

.info-box {
    background: #eff6ff;
    border: 1px solid #3b82f6;
    border-radius: 12px;
    padding: 15px;
    color: #1e40af;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_FILE = "insightai_best_model.pkl"
MODEL_RESULTS = "model_comparison.csv"
PREDICTIONS_FILE = "prediction_results.csv"
ENGINEERED_FILE = "insightai_engineered_dataset.csv"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(path):

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_resource
def load_model(path):

    if not os.path.exists(path):
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def metric_card(title, value, description=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_number(value):

    if value is None:
        return "N/A"

    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"

    return f"{value:,}"


# ============================================================
# LOAD DATA
# ============================================================

model_results = load_csv(MODEL_RESULTS)
prediction_results = load_csv(PREDICTIONS_FILE)
engineered_data = load_csv(ENGINEERED_FILE)
feature_importance = load_csv(FEATURE_IMPORTANCE_FILE)
best_model = load_model(MODEL_FILE)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 InsightAI")

    st.markdown(
        "### Profit Margin Intelligence"
    )

    st.markdown("---")

    st.markdown("### Dashboard")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🤖 Model Performance",
            "🎯 Predictions",
            "🔥 Feature Importance",
            "📊 Data Analysis",
            "📋 Data Explorer"
        ]
    )

    st.markdown("---")

    st.markdown("### Files")

    files = [
        MODEL_FILE,
        MODEL_RESULTS,
        PREDICTIONS_FILE,
        ENGINEERED_FILE,
        FEATURE_IMPORTANCE_FILE
    ]

    for file in files:

        if os.path.exists(file):
            st.success(f"✓ {file}")
        else:
            st.warning(f"✗ {file}")

    st.markdown("---")

    st.caption(
        "InsightAI Profit Margin Prediction"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">InsightAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'AI-powered Profit Margin Prediction & Business Intelligence Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK FILES
# ============================================================

if model_results is None:

    st.error(
        "model_comparison.csv was not found."
    )

    st.info(
        "Run your training script first so that the "
        "generated CSV files are available."
    )

    st.stop()


# ============================================================
# CALCULATE MAIN METRICS
# ============================================================

model_results = model_results.copy()

model_results["R2"] = pd.to_numeric(
    model_results["R2"],
    errors="coerce"
)

model_results["R2_Percent"] = pd.to_numeric(
    model_results["R2_Percent"],
    errors="coerce"
)

model_results["MAE"] = pd.to_numeric(
    model_results["MAE"],
    errors="coerce"
)

model_results["RMSE"] = pd.to_numeric(
    model_results["RMSE"],
    errors="coerce"
)

model_results = model_results.dropna(
    subset=["R2"]
)

best_row = model_results.iloc[
    model_results["R2"].idxmax()
]

best_model_name = best_row["Model"]
best_r2 = best_row["R2"]
best_mae = best_row["MAE"]
best_rmse = best_row["RMSE"]


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        metric_card(
            "Best R²",
            f"{best_r2:.2%}",
            "Prediction accuracy"
        )

    with col2:

        metric_card(
            "MAE",
            f"{best_mae:.5f}",
            "Mean absolute error"
        )

    with col3:

        metric_card(
            "RMSE",
            f"{best_rmse:.5f}",
            "Root mean squared error"
        )

    with col4:

        metric_card(
            "Models",
            len(model_results),
            "Models evaluated"
        )

    with col5:

        if engineered_data is not None:

            rows = len(engineered_data)

        else:

            rows = "N/A"

        metric_card(
            "Dataset Rows",
            format_number(rows),
            "Rows used/generated"
        )

    st.markdown("")


    # --------------------------------------------------------
    # TARGET CHECK
    # --------------------------------------------------------

    if best_r2 >= 0.85:

        st.markdown(
            f"""
            <div class="success-box">
                <strong>✓ 85% R² TARGET ACHIEVED</strong><br>
                Current R²: <strong>{best_r2:.2%}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="warning-box">
                <strong>85% R² TARGET NOT ACHIEVED</strong><br>
                Current R²: <strong>{best_r2:.2%}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Best Performing Model</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "Model",
            best_model_name
        )

        st.metric(
            "R² Score",
            f"{best_r2:.4f}"
        )

    with col2:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=best_r2 * 100,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 32
                    }
                },
                title={
                    "text": "R² Performance"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": "#2563eb"
                    },
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "#fee2e2"
                        },
                        {
                            "range": [50, 70],
                            "color": "#fef3c7"
                        },
                        {
                            "range": [70, 85],
                            "color": "#dbeafe"
                        },
                        {
                            "range": [85, 100],
                            "color": "#dcfce7"
                        }
                    ],
                    "threshold": {
                        "line": {
                            "color": "#16a34a",
                            "width": 4
                        },
                        "thickness": 0.75,
                        "value": 85
                    }
                }
            )
        )

        gauge.update_layout(
            height=280,
            margin=dict(
                l=30,
                r=30,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            gauge,
            width="stretch"
        )


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    chart = px.bar(
        model_results.sort_values(
            "R2",
            ascending=True
        ),
        x="R2",
        y="Model",
        orientation="h",
        text="R2_Percent",
        color="R2",
        color_continuous_scale="Blues"
    )

    chart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    chart.update_layout(
        xaxis_title="R² Score",
        yaxis_title="",
        height=450
    )

    st.plotly_chart(
        chart,
        width="stretch"
    )


    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Metrics</div>',
        unsafe_allow_html=True
    )

    display_df = model_results.copy()

    display_df["R2"] = display_df["R2"].map(
        lambda x: f"{x:.4f}"
    )

    display_df["R2_Percent"] = display_df[
        "R2_Percent"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    display_df["MAE"] = display_df[
        "MAE"
    ].map(
        lambda x: f"{x:.6f}"
    )

    display_df["RMSE"] = display_df[
        "RMSE"
    ].map(
        lambda x: f"{x:.6f}"
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance Analysis</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            model_results.sort_values(
                "R2",
                ascending=False
            ),
            x="Model",
            y="R2",
            color="R2",
            color_continuous_scale="Viridis",
            text="R2_Percent"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            title="R² Score by Model",
            yaxis_title="R²",
            xaxis_title="",
            yaxis=dict(
                range=[
                    max(
                        0,
                        model_results["R2"].min() - 0.05
                    ),
                    1
                ]
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    with col2:

        fig = px.bar(
            model_results.sort_values(
                "MAE"
            ),
            x="Model",
            y="MAE",
            color="MAE",
            color_continuous_scale="RdYlGn_r"
        )

        fig.update_layout(
            title="Mean Absolute Error",
            yaxis_title="MAE",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    fig = px.bar(
        model_results.sort_values(
            "RMSE"
        ),
        x="Model",
        y="RMSE",
        color="RMSE",
        color_continuous_scale="Oranges"
    )

    fig.update_layout(
        title="RMSE by Model",
        yaxis_title="RMSE",
        xaxis_title=""
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


    # --------------------------------------------------------
    # METRIC RADAR
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Metric Profile</div>',
        unsafe_allow_html=True
    )

    radar_df = model_results.copy()

    max_mae = radar_df["MAE"].max()
    max_rmse = radar_df["RMSE"].max()

    radar_df["MAE_Score"] = (
        1 - radar_df["MAE"] / max_mae
    )

    radar_df["RMSE_Score"] = (
        1 - radar_df["RMSE"] / max_rmse
    )

    radar_df["R2_Score"] = radar_df["R2"].clip(
        0,
        1
    )

    selected_models = st.multiselect(
        "Select models",
        radar_df["Model"].tolist(),
        default=radar_df["Model"].tolist()
    )

    radar_fig = go.Figure()

    for model in selected_models:

        row = radar_df[
            radar_df["Model"] == model
        ].iloc[0]

        radar_fig.add_trace(
            go.Scatterpolar(
                r=[
                    row["R2_Score"],
                    row["MAE_Score"],
                    row["RMSE_Score"]
                ],
                theta=[
                    "R²",
                    "MAE Performance",
                    "RMSE Performance"
                ],
                fill="toself",
                name=model
            )
        )

    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        height=500
    )

    st.plotly_chart(
        radar_fig,
        width="stretch"
    )


# ============================================================
# PREDICTIONS
# ============================================================

elif page == "🎯 Predictions":

    st.markdown(
        '<div class="section-title">Actual vs Predicted Profit Margin</div>',
        unsafe_allow_html=True
    )

    if prediction_results is None:

        st.error(
            "prediction_results.csv not found."
        )

    else:

        pred = prediction_results.copy()

        actual = pd.to_numeric(
            pred["Actual_Margin"],
            errors="coerce"
        )

        predicted = pd.to_numeric(
            pred["Predicted_Margin"],
            errors="coerce"
        )

        error = actual - predicted

        # ----------------------------------------------------
        # Prediction KPIs
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Actual Mean",
                f"{actual.mean():.4f}"
            )

        with col2:

            st.metric(
                "Predicted Mean",
                f"{predicted.mean():.4f}"
            )

        with col3:

            st.metric(
                "Mean Error",
                f"{error.mean():.4f}"
            )

        with col4:

            st.metric(
                "Max Absolute Error",
                f"{error.abs().max():.4f}"
            )


        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        scatter = go.Figure()

        scatter.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode="markers",
                marker=dict(
                    size=6,
                    color=np.abs(error),
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(
                        title="Absolute Error"
                    )
                ),
                name="Predictions"
            )
        )

        min_val = min(
            actual.min(),
            predicted.min()
        )

        max_val = max(
            actual.max(),
            predicted.max()
        )

        scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(
                    color="red",
                    dash="dash",
                    width=2
                ),
                name="Perfect Prediction"
            )
        )

        scatter.update_layout(
            title="Actual vs Predicted Profit Margin",
            xaxis_title="Actual Profit Margin",
            yaxis_title="Predicted Profit Margin",
            height=600
        )

        st.plotly_chart(
            scatter,
            width="stretch"
        )


        # ----------------------------------------------------
        # ERROR DISTRIBUTION
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                x=error,
                nbins=50,
                marginal="box",
                color_discrete_sequence=[
                    "#2563eb"
                ]
            )

            fig.update_layout(
                title="Prediction Error Distribution",
                xaxis_title="Actual - Predicted",
                yaxis_title="Frequency"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            fig = px.scatter(
                x=predicted,
                y=error,
                color=np.abs(error),
                color_continuous_scale="Turbo"
            )

            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="red"
            )

            fig.update_layout(
                title="Prediction Error vs Predicted Margin",
                xaxis_title="Predicted Margin",
                yaxis_title="Error"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


        # ----------------------------------------------------
        # PREDICTION LINE
        # ----------------------------------------------------

        sample_size = min(
            len(pred),
            500
        )

        sample = pred.head(
            sample_size
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=sample["Actual_Margin"],
                mode="lines",
                name="Actual",
                line=dict(
                    color="#2563eb"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                y=sample["Predicted_Margin"],
                mode="lines",
                name="Predicted",
                line=dict(
                    color="#f97316"
                )
            )
        )

        fig.update_layout(
            title=f"Actual vs Predicted Sequence ({sample_size} rows)",
            xaxis_title="Observation",
            yaxis_title="Profit Margin",
            height=450
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.download_button(
            "⬇ Download Prediction Results",
            data=pred.to_csv(index=False),
            file_name="prediction_results.csv",
            mime="text/csv"
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔥 Feature Importance":

    st.markdown(
        '<div class="section-title">Feature Importance</div>',
        unsafe_allow_html=True
    )

    if feature_importance is None:

        st.warning(
            "feature_importance.csv was not generated."
        )

        st.info(
            "Feature importance is currently exported "
            "only when the selected model is CatBoost."
        )

    else:

        fi = feature_importance.copy()

        fi["Importance"] = pd.to_numeric(
            fi["Importance"],
            errors="coerce"
        )

        fi = fi.dropna(
            subset=["Importance"]
        )

        fi = fi.sort_values(
            "Importance",
            ascending=False
        )

        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=min(
                50,
                len(fi)
            ),
            value=min(
                20,
                len(fi)
            )
        )

        top_features = fi.head(
            top_n
        ).sort_values(
            "Importance"
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            title=f"Top {top_n} Most Important Features",
            height=max(
                450,
                top_n * 25
            ),
            yaxis_title="",
            xaxis_title="Importance"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


        # ----------------------------------------------------
        # PIE / TREEMAP
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Feature Contribution</div>',
            unsafe_allow_html=True
        )

        treemap_data = fi.head(
            min(30, len(fi))
        )

        fig = px.treemap(
            treemap_data,
            path=["Feature"],
            values="Importance",
            color="Importance",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


        st.dataframe(
            fi,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "⬇ Download Feature Importance",
            data=fi.to_csv(index=False),
            file_name="feature_importance.csv",
            mime="text/csv"
        )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.markdown(
        '<div class="section-title">Profit Margin Data Analysis</div>',
        unsafe_allow_html=True
    )

    if engineered_data is None:

        st.error(
            "insightai_engineered_dataset.csv not found."
        )

    else:

        data = engineered_data.copy()

        # ----------------------------------------------------
        # TARGET DISTRIBUTION
        # ----------------------------------------------------

        if "profit_margin" in data.columns:

            margin = pd.to_numeric(
                data["profit_margin"],
                errors="coerce"
            ).dropna()

            st.markdown(
                "### Profit Margin Distribution"
            )

            fig = px.histogram(
                x=margin,
                nbins=60,
                marginal="box",
                color_discrete_sequence=[
                    "#2563eb"
                ]
            )

            fig.update_layout(
                xaxis_title="Profit Margin",
                yaxis_title="Number of Records",
                height=500
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


            # ------------------------------------------------
            # BOX PLOT
            # ------------------------------------------------

            fig = px.box(
                x=margin,
                points="outliers",
                color_discrete_sequence=[
                    "#7c3aed"
                ]
            )

            fig.update_layout(
                title="Profit Margin Box Plot",
                xaxis_title="Profit Margin"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


            # ------------------------------------------------
            # TARGET STATISTICS
            # ------------------------------------------------

            stats = pd.DataFrame({
                "Statistic": [
                    "Count",
                    "Mean",
                    "Median",
                    "Std",
                    "Minimum",
                    "1% Quantile",
                    "25% Quantile",
                    "75% Quantile",
                    "99% Quantile",
                    "Maximum"
                ],
                "Value": [
                    margin.count(),
                    margin.mean(),
                    margin.median(),
                    margin.std(),
                    margin.min(),
                    margin.quantile(0.01),
                    margin.quantile(0.25),
                    margin.quantile(0.75),
                    margin.quantile(0.99),
                    margin.max()
                ]
            })

            st.markdown(
                "### Target Statistics"
            )

            st.dataframe(
                stats,
                width="stretch",
                hide_index=True
            )


        # ----------------------------------------------------
        # NUMERIC CORRELATIONS
        # ----------------------------------------------------

        numeric = data.select_dtypes(
            include=np.number
        )

        if (
            "profit_margin" in numeric.columns
            and len(numeric.columns) > 1
        ):

            correlations = (
                numeric
                .corr()["profit_margin"]
                .drop("profit_margin")
                .sort_values()
            )

            st.markdown(
                "### Feature Correlation with Profit Margin"
            )

            corr_fig = px.bar(
                x=correlations.values,
                y=correlations.index,
                orientation="h",
                color=correlations.values,
                color_continuous_scale="RdBu"
            )

            corr_fig.update_layout(
                xaxis_title="Correlation",
                yaxis_title="Feature",
                height=max(
                    500,
                    len(correlations) * 18
                )
            )

            st.plotly_chart(
                corr_fig,
                width="stretch"
            )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "📋 Data Explorer":

    st.markdown(
        '<div class="section-title">Feature-Engineered Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    if engineered_data is None:

        st.error(
            "Engineered dataset not found."
        )

    else:

        data = engineered_data.copy()

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        st.write(
            f"Dataset shape: "
            f"**{data.shape[0]:,} rows × "
            f"{data.shape[1]:,} columns**"
        )

        selected_columns = st.multiselect(
            "Select columns",
            data.columns.tolist(),
            default=data.columns.tolist()[
                :min(10, len(data.columns))
            ]
        )

        if selected_columns:

            display_data = data[
                selected_columns
            ]

        else:

            display_data = data


        # ----------------------------------------------------
        # ROW LIMIT
        # ----------------------------------------------------

        row_limit = st.slider(
            "Rows to display",
            10,
            min(
                5000,
                len(display_data)
            ),
            min(
                100,
                len(display_data)
            )
        )

        st.dataframe(
            display_data.head(row_limit),
            width="stretch",
            height=600
        )


        # ----------------------------------------------------
        # DATA TYPES
        # ----------------------------------------------------

        st.markdown(
            "### Column Information"
        )

        info_df = pd.DataFrame({
            "Column": data.columns,
            "Data Type": [
                str(dtype)
                for dtype in data.dtypes
            ],
            "Missing Values": [
                data[col].isna().sum()
                for col in data.columns
            ],
            "Unique Values": [
                data[col].nunique()
                for col in data.columns
            ]
        })

        st.dataframe(
            info_df,
            width="stretch",
            hide_index=True
        )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.download_button(
            "⬇ Download Engineered Dataset",
            data=data.to_csv(index=False),
            file_name="insightai_engineered_dataset.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#9ca3af;">
        <b>InsightAI</b> • Profit Margin Prediction Dashboard
        <br>
        Machine Learning • Business Intelligence • Predictive Analytics
    </div>
    """,
    unsafe_allow_html=True
)
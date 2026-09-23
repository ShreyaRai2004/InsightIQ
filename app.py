import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data
from src.validation import validate
from src.features import add_features
from src.model import train, load_saved
from src.analysis import anomalies
from src.explain import explain


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="InsightIQ — Predictive Business Analytics Platform",
    page_icon="IQ",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px 34px;
        border-radius: 22px;
        background: linear-gradient(135deg, #222b75, #111a3d);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 52px;
        margin: 0;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.8;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "result" not in st.session_state:
    st.session_state.result = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## InsightIQ")
    st.caption("Predictive Business Analytics Platform")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"]
    )

    if st.button(
        "Load sample dataset",
        use_container_width=True
    ):
        try:
            st.session_state.df = load_data(
                "data/sample/InsightIQ_Business_Data.xlsx"
            )
            st.session_state.result = None
            st.rerun()

        except Exception as e:
            st.error(f"Unable to load sample dataset: {e}")

    st.divider()

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Performance",
            "Prediction",
            "Explainability",
            "Anomalies",
            "Forecasting",
            "Risk Analysis",
            "What-if Simulation"
        ]
    )


# ---------------------------------------------------------
# HANDLE UPLOADED FILE
# ---------------------------------------------------------

if uploaded_file is not None:

    try:
        st.session_state.df = load_data(uploaded_file)
        st.session_state.result = None

    except Exception as e:
        st.error(f"Unable to load the uploaded file: {e}")


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

df = st.session_state.df

st.markdown(
    """
    <div class="hero">
        <h1>InsightIQ</h1>
        <p>Predictive Business Analytics Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# NO DATA LOADED
# ---------------------------------------------------------

if df is None:

    st.info(
        "Load the included sample dataset or upload your own "
        "business CSV/Excel file to begin."
    )

    st.stop()


# ---------------------------------------------------------
# DATA VALIDATION
# ---------------------------------------------------------

issues = validate(df)

if issues:

    for issue in issues:
        st.error(issue)

    st.stop()


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

df = add_features(df)


# ---------------------------------------------------------
# MAIN KPI SECTION
# ---------------------------------------------------------

total_revenue = pd.to_numeric(
    df["Revenue"],
    errors="coerce"
).sum()

total_cost = pd.to_numeric(
    df["Cost"],
    errors="coerce"
).sum()

total_profit = pd.to_numeric(
    df["Profit"],
    errors="coerce"
).sum()


if total_revenue > 0:
    overall_margin = (
        total_profit / total_revenue
    ) * 100
else:
    overall_margin = 0.0


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Revenue",
    f"₹{total_revenue:,.0f}"
)

c2.metric(
    "Cost",
    f"₹{total_cost:,.0f}"
)

c3.metric(
    "Profit",
    f"₹{total_profit:,.0f}"
)

c4.metric(
    "Margin",
    f"{overall_margin:.1f}%"
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.subheader("Business Overview")

    if "Date" in df.columns:

        trend = (
            df.groupby("Date", as_index=False)["Profit"]
            .sum()
        )

        st.plotly_chart(
            px.line(
                trend,
                x="Date",
                y="Profit",
                title="Profit Trend"
            ),
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    category_profit = (
        df.groupby("Category", as_index=False)["Profit"]
        .sum()
    )

    col1.plotly_chart(
        px.bar(
            category_profit,
            x="Category",
            y="Profit",
            title="Profit by Category"
        ),
        use_container_width=True
    )

    region_profit = (
        df.groupby("Region", as_index=False)["Profit"]
        .sum()
    )

    col2.plotly_chart(
        px.bar(
            region_profit,
            x="Region",
            y="Profit",
            title="Profit by Region"
        ),
        use_container_width=True
    )


# =========================================================
# PERFORMANCE
# =========================================================

elif page == "Performance":

    st.subheader("Performance Analysis")

    if "Date" in df.columns:

        performance_df = df.copy()

        performance_df["Period"] = (
            performance_df["Date"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly = (
            performance_df
            .groupby("Period", as_index=False)[
                ["Revenue", "Cost", "Profit"]
            ]
            .sum()
        )

        st.plotly_chart(
            px.line(
                monthly,
                x="Period",
                y=["Revenue", "Cost", "Profit"],
                title="Monthly Business Performance"
            ),
            use_container_width=True
        )

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.subheader("Predictive Modeling")

    st.caption(
        "The prediction model uses operational and historical input "
        "fields available before profit is known. Revenue, Cost, "
        "Operating Cost, and Profit-derived information are excluded "
        "from prediction inputs to reduce target leakage."
    )

    if st.button(
        "Train / Refresh Models",
        type="primary"
    ):

        with st.spinner("Training models..."):
            st.session_state.result = train(df)

    result = (
        st.session_state.result
        or load_saved()
    )

    if result:

        st.success(
            f"Best model: {result['best_model_name']}"
        )

        st.dataframe(
            result["comparison"].round(4),
            use_container_width=True
        )

        metrics = result["metrics"][
            result["best_model_name"]
        ]

        x, y, z = st.columns(3)

        x.metric(
            "Test MAE",
            f"₹{metrics['Test MAE']:,.2f}"
        )

        y.metric(
            "Test RMSE",
            f"₹{metrics['Test RMSE']:,.2f}"
        )

        z.metric(
            "Test R²",
            f"{metrics['Test R2']:.4f}"
        )

        if metrics["Overfit Warning"]:

            st.warning(
                "Train/test performance gap suggests possible "
                "overfitting."
            )

    else:

        st.info(
            "Train the models to see actual evaluation results."
        )


# =========================================================
# EXPLAINABILITY
# =========================================================

elif page == "Explainability":

    st.subheader("Explainability")

    result = (
        st.session_state.result
        or load_saved()
    )

    if result:

        try:

            shap_result = explain(
                df,
                result
            )

            st.dataframe(
                shap_result,
                use_container_width=True
            )

            chart_data = (
                shap_result
                .sort_values("Mean |SHAP|")
            )

            st.plotly_chart(
                px.bar(
                    chart_data,
                    x="Mean |SHAP|",
                    y="Feature",
                    orientation="h",
                    title="Global SHAP Feature Importance"
                ),
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"SHAP explanation failed: {e}"
            )

    else:

        st.info(
            "Train the model first."
        )


# =========================================================
# ANOMALY DETECTION
# =========================================================

elif page == "Anomalies":

    st.subheader("Anomaly Detection")

    try:

        anomaly_data = anomalies(df)

        anomaly_count = int(
            (anomaly_data["Anomaly"] == -1).sum()
        )

        total_rows = len(anomaly_data)

        anomaly_percentage = (
            anomaly_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Anomalies Detected",
            anomaly_count
        )

        col2.metric(
            "Anomaly Rate",
            f"{anomaly_percentage:.1f}%"
        )

        chart_data = anomaly_data.copy()

        chart_data["Status"] = (
            chart_data["Anomaly"]
            .map({
                1: "Normal",
                -1: "Anomaly"
            })
        )

        st.plotly_chart(
            px.scatter(
                chart_data,
                x="Revenue",
                y="Profit",
                color="Status",
                title="Revenue vs Profit — Anomaly Detection"
            ),
            use_container_width=True
        )

        st.subheader("Detected Anomalies")

        st.dataframe(
            anomaly_data[
                anomaly_data["Anomaly"] == -1
            ],
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Anomaly detection failed: {e}"
        )


# =========================================================
# FORECASTING
# =========================================================

elif page == "Forecasting":

    st.subheader("Profit Forecasting")

    try:

        from statsmodels.tsa.holtwinters import (
            ExponentialSmoothing
        )

        if "Date" not in df.columns:
            raise ValueError(
                "Date column is required."
            )

        time_series = (
            df.set_index("Date")["Profit"]
            .resample("MS")
            .sum()
        )

        if len(time_series) < 12:
            raise ValueError(
                "At least 12 monthly observations are required."
            )

        model = ExponentialSmoothing(
            time_series,
            trend="add",
            damped_trend=True,
            initialization_method="estimated"
        )

        fitted_model = model.fit()

        forecast = fitted_model.forecast(6)

        historical = pd.DataFrame(
            {
                "Period": time_series.index,
                "Profit": time_series.values,
                "Type": "Historical"
            }
        )

        future = pd.DataFrame(
            {
                "Period": forecast.index,
                "Profit": forecast.values,
                "Type": "Forecast"
            }
        )

        combined = pd.concat(
            [historical, future],
            ignore_index=True
        )

        fig = px.line(
            combined,
            x="Period",
            y="Profit",
            color="Type",
            title="Monthly Profit Forecast"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            future.drop(columns=["Type"]),
            use_container_width=True
        )

    except Exception as e:

        st.warning(str(e))


# =========================================================
# RISK ANALYSIS
# =========================================================

elif page == "Risk Analysis":

    st.subheader("Risk Analysis")

    # -----------------------------------------------------
    # PROFIT MARGIN
    # -----------------------------------------------------

    revenue = pd.to_numeric(
        df["Revenue"],
        errors="coerce"
    )

    profit = pd.to_numeric(
        df["Profit"],
        errors="coerce"
    )

    total_revenue = revenue.sum()
    total_profit = profit.sum()

    if total_revenue > 0:

        margin = (
            total_profit /
            total_revenue
        ) * 100

    else:

        margin = 0.0


    # -----------------------------------------------------
    # DISCOUNT RATE
    # -----------------------------------------------------

    if (
        "Discount" in df.columns
        and "Revenue" in df.columns
    ):

        discount = pd.to_numeric(
            df["Discount"],
            errors="coerce"
        )

        valid = (
            revenue > 0
        ) & discount.notna()

        if valid.any():

            discount_rates = (
                discount[valid] /
                revenue[valid]
            ) * 100

            avg_discount_rate = (
                discount_rates.mean()
            )

        else:

            avg_discount_rate = 0.0

    else:

        avg_discount_rate = 0.0


    # -----------------------------------------------------
    # DISPLAY RISK METRICS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(
        "Profit Margin",
        f"{margin:.1f}%"
    )

    col2.metric(
        "Average Discount Rate",
        f"{avg_discount_rate:.1f}%"
    )


    # -----------------------------------------------------
    # RULE-BASED RISK ASSESSMENT
    # -----------------------------------------------------

    if margin < 5:

        st.error(
            "High margin pressure based on the configured "
            "risk threshold."
        )

    elif margin < 12:

        st.warning(
            "Moderate margin pressure based on the configured "
            "risk threshold."
        )

    else:

        st.success(
            "Profit margin is above the configured risk threshold."
        )


# =========================================================
# WHAT-IF SIMULATION
# =========================================================

elif page == "What-if Simulation":

    st.subheader("What-if Simulation")

    result = (
        st.session_state.result
        or load_saved()
    )

    if not result:

        st.info(
            "Train the model first."
        )

    else:

        base = (
            df
            .drop(
                columns=[
                    "Profit",
                    "Gross_Margin"
                ],
                errors="ignore"
            )
            .iloc[[0]]
            .copy()
        )


        # -------------------------------------------------
        # QUANTITY
        # -------------------------------------------------

        if "Quantity" in base.columns:

            default_quantity = int(
                pd.to_numeric(
                    base["Quantity"].iloc[0],
                    errors="coerce"
                )
            )

            quantity = st.slider(
                "Quantity",
                1,
                50,
                default_quantity
            )

        else:

            quantity = None


        # -------------------------------------------------
        # DISCOUNT
        # -------------------------------------------------

        if "Discount" in base.columns:

            default_discount = int(
                pd.to_numeric(
                    base["Discount"].iloc[0],
                    errors="coerce"
                )
            )

            discount = st.slider(
                "Discount",
                0,
                30,
                min(default_discount, 30)
            )

        else:

            discount = None


        # -------------------------------------------------
        # SCENARIO DATA
        # -------------------------------------------------

        scenario = base.copy()

        if quantity is not None:
            scenario["Quantity"] = quantity

        if discount is not None:
            scenario["Discount"] = discount


        # -------------------------------------------------
        # PREDICTION FUNCTION
        # -------------------------------------------------

        def predict_profit(input_data):

            data = input_data.copy()

            if "Date" in data.columns:

                date_values = pd.to_datetime(
                    data["Date"],
                    errors="coerce"
                )

                data["Year"] = date_values.dt.year
                data["Month"] = date_values.dt.month
                data["Quarter"] = date_values.dt.quarter
                data["DayOfWeek"] = date_values.dt.dayofweek

                data = data.drop(
                    columns=["Date"]
                )

            return float(
                result["model"].predict(data)[0]
            )


        # -------------------------------------------------
        # BASELINE VS SCENARIO
        # -------------------------------------------------

        baseline_prediction = predict_profit(
            base
        )

        scenario_prediction = predict_profit(
            scenario
        )

        change = (
            scenario_prediction -
            baseline_prediction
        )


        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Baseline",
            f"₹{baseline_prediction:,.2f}"
        )

        col2.metric(
            "Scenario",
            f"₹{scenario_prediction:,.2f}"
        )

        col3.metric(
            "Change",
            f"₹{change:,.2f}"
        )
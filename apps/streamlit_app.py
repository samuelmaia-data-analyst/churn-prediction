"""
Churn Prediction Dashboard - Streamlit
Author: Samuel de Andrade Maia
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from apps.dashboard_runtime import (
    DashboardRuntime,
    SidebarState,
    build_filtered_views,
    build_prediction_payload,
    load_data,
    load_predictor,
    summarise_metrics,
)
from src.modeling.predictor import ChurnPredictor
from src.runtime.config import PipelineConfig

RUNTIME_CONFIG = PipelineConfig.from_runtime(run_id="streamlit")
DASHBOARD_RUNTIME = DashboardRuntime.from_config(RUNTIME_CONFIG)
COLOR_BG_START = "#f7f9fc"
COLOR_BG_END = "#eef3f9"
COLOR_PRIMARY = "#164e63"
COLOR_SECONDARY = "#0f766e"
COLOR_ALERT = "#dc2626"
COLOR_TEXT = "#0b1f33"
COLOR_MUTED = "#5b6b80"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
        :root {{
            --bg-start: {COLOR_BG_START};
            --bg-end: {COLOR_BG_END};
            --primary: {COLOR_PRIMARY};
            --secondary: {COLOR_SECONDARY};
            --alert: {COLOR_ALERT};
            --text: {COLOR_TEXT};
            --muted: {COLOR_MUTED};
        }}
        .stApp {{
            background:
                radial-gradient(circle at 8% 6%, rgba(22, 78, 99, 0.10), transparent 36%),
                radial-gradient(circle at 85% 0%, rgba(15, 118, 110, 0.12), transparent 38%),
                linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
            color: var(--text);
        }}
        html, body, [class*="css"] {{
            font-family: "Space Grotesk", sans-serif;
        }}
        h1, h2, h3,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2 {{
            color: var(--text);
            letter-spacing: -0.02em;
        }}
        [data-testid="stHeaderActionElements"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(
                200deg,
                rgba(22, 78, 99, 0.96) 0%,
                rgba(11, 31, 51, 0.97) 100%
            );
        }}
        [data-testid="stSidebar"] * {{
            color: #f8fafc;
        }}
        [data-testid="stSidebar"] a {{
            color: #99f6e4 !important;
        }}
        .hero {{
            background: linear-gradient(125deg, rgba(22, 78, 99, 0.96), rgba(15, 118, 110, 0.92));
            border-radius: 20px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 14px 42px rgba(11, 31, 51, 0.25);
            margin-bottom: 0.9rem;
        }}
        .hero-title {{
            color: #f8fafc;
            margin: 0;
            font-size: clamp(1.45rem, 4.5vw, 2.3rem);
            font-weight: 700;
            line-height: 1.15;
        }}
        .hero-subtitle {{
            color: rgba(248, 250, 252, 0.92);
            margin-top: 0.45rem;
            font-size: 0.98rem;
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0.2rem 0 0.65rem 0;
        }}
        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(22, 78, 99, 0.18);
            border-radius: 14px;
            padding: 0.55rem 0.75rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--muted);
            font-weight: 600;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--primary);
            font-size: 1.55rem;
            font-weight: 700;
        }}
        .stButton > button {{
            border-radius: 10px;
            border: 1px solid rgba(22, 78, 99, 0.22);
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #fff;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(22, 78, 99, 0.25);
        }}
        .stButton > button:hover {{
            border-color: rgba(22, 78, 99, 0.35);
            background: linear-gradient(135deg, #0f4254, #0a5a53);
            color: #fff;
        }}
        .risk-box {{
            border-radius: 12px;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(22, 78, 99, 0.20);
            background: rgba(255, 255, 255, 0.72);
            color: var(--text);
            font-size: 0.96rem;
        }}
        .status-box {{
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.65rem;
            background: rgba(255,255,255,0.06);
        }}
        code {{
            font-family: "JetBrains Mono", monospace !important;
        }}
        @media (max-width: 820px) {{
            .hero {{
                border-radius: 14px;
                padding: 1rem 1rem;
            }}
            [data-testid="stMetric"] {{
                padding: 0.55rem 0.65rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">Churn Prediction Control Room</h1>
            <p class="hero-subtitle">
                Operational dashboard for churn monitoring, risk exploration,
                and customer-level inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(runtime: DashboardRuntime) -> SidebarState:
    df: pd.DataFrame | None = None
    predictor: ChurnPredictor | None = None
    model_loaded = False

    with st.sidebar:
        st.markdown("## Runtime Status")

        if runtime.data_path.exists():
            df = load_data(runtime.data_path)
            st.success(f"Dataset ready: {len(df):,} rows")
        else:
            st.error("Dataset not found")

        if runtime.bundle_path.exists():
            predictor = load_predictor(runtime.bundle_path)
            model_loaded = predictor.is_ready
            if model_loaded:
                st.success(f"Inference bundle ready: {runtime.bundle_path.name}")
            else:
                st.error("Bundle loaded but incompatible with the current predictor contract.")
        else:
            st.warning(
                "Inference bundle not found. Run the training pipeline to enable predictions."
            )

        st.markdown(
            f"""
            <div class="status-box">
                <strong>Environment</strong><br>{RUNTIME_CONFIG.environment}<br><br>
                <strong>Run ID</strong><br><code>{RUNTIME_CONFIG.run_id}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            ### Maintainer
            **Samuel de Andrade Maia**

            [GitHub](https://github.com/samuelmaia-analytics)

            [LinkedIn](https://linkedin.com/in/samuelmaia-analytics)
            """
        )

    return SidebarState(dataframe=df, predictor=predictor, model_loaded=model_loaded)


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    st.markdown("---")
    st.markdown('<div class="section-title">Exploration Filters</div>', unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        st.caption("Controls the left-side chart")
        contract_options = ["Todos"]
        if "Contract" in df.columns:
            contract_options += sorted(df["Contract"].dropna().unique().tolist())
        selected_contract = st.selectbox("Contract", contract_options)

    with filter_col2:
        st.caption("Controls the right-side chart")
        internet_options = ["Todos"]
        if "InternetService" in df.columns:
            internet_options += sorted(df["InternetService"].dropna().unique().tolist())
        selected_internet = st.selectbox("Internet service", internet_options)

    left_chart_df, right_chart_df, preview_df = build_filtered_views(
        df=df,
        selected_contract=selected_contract,
        selected_internet=selected_internet,
    )
    return left_chart_df, right_chart_df, preview_df, selected_contract, selected_internet


def render_metrics(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Portfolio Snapshot</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    metrics = summarise_metrics(df)

    with col1:
        st.metric("Total Customers", f"{metrics['total_customers']:,}")

    with col2:
        st.metric("Churn Rate", f"{metrics['churn_rate']:.1f}%")

    with col3:
        st.metric("Average Monthly Charge", f"${metrics['avg_monthly']:.2f}")

    with col4:
        st.metric("Average Tenure", f"{metrics['avg_tenure']:.1f} months")

    st.caption(
        "This overview is computed from the raw telco customer dataset. "
        "Detailed prioritization and value-at-risk views are available "
        "in the dedicated Streamlit pages."
    )


def render_charts(
    left_chart_df: pd.DataFrame,
    right_chart_df: pd.DataFrame,
    selected_contract: str,
    selected_internet: str,
) -> None:
    st.markdown('<div class="section-title">Risk Exploration</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Churn Distribution by Contract: {selected_contract}")
        if "Churn" in left_chart_df.columns:
            churn_counts = (
                left_chart_df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
            )
            fig1 = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Cancellation split across selected contract cohort",
                color=churn_counts.index,
                color_discrete_map={"Yes": COLOR_ALERT, "No": COLOR_PRIMARY},
                hole=0.48,
            )
            fig1.update_layout(
                margin=dict(l=10, r=10, t=48, b=10),
                legend_title_text="Churn",
                title_font=dict(size=18),
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Column 'Churn' not available for this chart.")

    with col2:
        st.subheader(f"Contract Mix by Internet Service: {selected_internet}")
        if {"Contract", "Churn"}.issubset(right_chart_df.columns):
            contract_order = ["Month-to-month", "One year", "Two year"]
            contract_churn_rate = (
                right_chart_df.assign(churn_yes=right_chart_df["Churn"].eq("Yes").astype(float))
                .groupby("Contract", as_index=True)["churn_yes"]
                .mean()
                .mul(100)
                .reindex(contract_order, fill_value=0.0)
            )
            fig2 = px.bar(
                x=contract_churn_rate.index,
                y=contract_churn_rate.values,
                title="Churn rate by contract type",
                labels={"x": "Contract type", "y": "Rate (%)"},
                color_discrete_sequence=[COLOR_SECONDARY],
            )
            fig2.update_traces(marker_line_color="#083344", marker_line_width=1.0)
            fig2.update_layout(margin=dict(l=10, r=10, t=48, b=10), title_font=dict(size=18))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Required columns for contract view are unavailable.")


def render_data_preview(filtered_df: pd.DataFrame) -> None:
    with st.expander("Preview source records", expanded=False):
        columns_to_show = [
            "customerID",
            "gender",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "InternetService",
            "Churn",
        ]
        valid_cols = [col for col in columns_to_show if col in filtered_df.columns]
        st.dataframe(filtered_df[valid_cols].head(20), use_container_width=True)


def render_prediction(predictor: ChurnPredictor) -> None:
    st.markdown("---")
    st.markdown(
        '<div class="section-title">Customer-Level Prediction</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Enter customer profile", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col3:
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.5)
            total = st.number_input("Total charges ($)", 0.0, 10000.0, 786.0)
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            phone_service = st.selectbox("Phone service", ["Yes", "No"])

        if st.button("Score customer"):
            try:
                payload = build_prediction_payload(
                    gender=gender,
                    senior=senior,
                    partner=partner,
                    dependents=dependents,
                    tenure=tenure,
                    phone_service=phone_service,
                    internet_service=internet_service,
                    contract=contract,
                    paperless=paperless,
                    payment=payment,
                    monthly=monthly,
                    total=total,
                )
                result = predictor.predict_from_dict(payload)
                proba = result.probability

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=proba * 100,
                            title={"text": "Churn probability"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": COLOR_PRIMARY},
                                "steps": [
                                    {"range": [0, 45], "color": "rgba(13, 148, 136, 0.35)"},
                                    {"range": [45, 70], "color": "rgba(245, 158, 11, 0.28)"},
                                    {"range": [70, 100], "color": "rgba(220, 38, 38, 0.30)"},
                                ],
                            },
                        )
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=56, b=0), height=280)
                    st.plotly_chart(fig, use_container_width=True)

                with result_col2:
                    if result.risk_level == "Alto":
                        st.error(f"High churn risk detected ({proba:.1%})")
                    else:
                        st.success(f"Lower churn risk detected ({proba:.1%})")
                    st.markdown(
                        """
                        <div class="risk-box">
                            Use this score to trigger outreach, pricing review,
                            or contract intervention. The intended operating model
                            is prioritization by risk and value, not raw score
                            inspection alone.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #355070; padding: 0.35rem 0 0.8rem 0;'>
            <p>Developed by <strong>Samuel de Andrade Maia</strong></p>
            <p>2026 - Churn Prediction Data Product</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Churn Prediction Control Room",
        page_icon="CR",
        layout="wide",
    )

    pio.templates.default = "plotly_white"
    inject_styles()
    render_header()
    sidebar_state = render_sidebar(DASHBOARD_RUNTIME)
    df = sidebar_state.dataframe
    predictor = sidebar_state.predictor
    model_loaded = sidebar_state.model_loaded

    if df is None:
        st.error(f"Dataset not found at: {DASHBOARD_RUNTIME.data_path}")
        st.stop()

    render_metrics(df)

    left_chart_df, right_chart_df, preview_df, selected_contract, selected_internet = apply_filters(
        df
    )
    if left_chart_df.empty and right_chart_df.empty:
        st.warning("No records matched the selected filters.")
        st.stop()

    render_charts(left_chart_df, right_chart_df, selected_contract, selected_internet)
    render_data_preview(preview_df)

    if model_loaded and predictor is not None:
        render_prediction(predictor)

    render_footer()


if __name__ == "__main__":
    main()

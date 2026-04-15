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
    build_dashboard_status,
    build_filtered_views,
    build_prediction_payload,
    format_risk_level,
    load_best_available_dataframe,
    load_dashboard_assets,
    load_predictor,
    summarise_metrics,
)
from apps.dashboard_ui import (
    COLOR_ALERT,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    configure_dashboard_page,
    inject_global_styles,
    render_footer,
    render_page_hero,
    render_sidebar_summary,
    render_status_banner,
    section_container,
)
from src.modeling.predictor import ChurnPredictor
from src.runtime.config import PipelineConfig

RUNTIME_CONFIG = PipelineConfig.from_runtime(run_id="streamlit")
DASHBOARD_RUNTIME = DashboardRuntime.from_config(RUNTIME_CONFIG)


def render_operational_shell() -> None:
    assets = load_dashboard_assets()
    status = build_dashboard_status(assets)
    render_page_hero(
        eyebrow="Control Room",
        title="Churn Prediction Control Room",
        subtitle=(
            "A production-minded workspace for portfolio monitoring, cohort exploration, "
            "and customer-level scoring built from canonical churn artifacts."
        ),
        meta="Portfolio dashboard | SaaS-style analytical workspace",
    )

    if status["fallback"]:
        title = "Fallback artifacts currently in use"
        body = (
            "The dashboard is rendering synthetic or degraded outputs because the canonical "
            "pipeline artifacts are not fully available."
        )
    elif status["ready"]:
        title = "Canonical dashboard artifacts are ready"
        body = "The dashboard is currently backed by the canonical reporting and model outputs."
    else:
        title = "Artifacts are partially available"
        body = "Some dashboard views may remain limited until the canonical pipeline completes."

    render_status_banner(
        title=title,
        body=body,
        status_items=[
            ("Environment", str(status["environment"])),
            ("Schema", str(status["schema_version"])),
            ("Run ID", str(status["run_id"])),
            ("Generated", str(status["generated_at_utc"])),
            ("EDA", "Ready" if status["eda_ready"] else "Missing"),
            ("Governance", "Ready" if status["governance_ready"] else "Missing"),
        ],
    )


def render_sidebar(runtime: DashboardRuntime) -> SidebarState:
    assets = load_dashboard_assets()
    dataframe: pd.DataFrame | None = None
    predictor: ChurnPredictor | None = None
    model_loaded = False

    with st.sidebar:
        st.markdown("## Workspace Status")
        st.caption("Runtime health, dataset readiness, and inference bundle availability.")

        dataframe, data_source = load_best_available_dataframe(runtime, assets)
        if dataframe is not None:
            st.success(f"Dataset ready with {len(dataframe):,} rows")
            if data_source.startswith("raw:"):
                st.caption(f"Source: {runtime.data_path.name}")
            elif data_source.startswith("silver:"):
                st.warning("Using silver layer because raw input is unavailable.")
            else:
                st.warning("Using prioritization artifact as fallback data source.")
        else:
            st.error("No data source available (raw, silver, or prioritization).")

        if runtime.bundle_path.exists():
            predictor = load_predictor(runtime.bundle_path)
            model_loaded = predictor.is_ready
            if model_loaded:
                st.success(f"Inference bundle ready: {runtime.bundle_path.name}")
            else:
                st.error(
                    "Inference bundle exists but does not match the current predictor contract."
                )
        else:
            st.warning("Inference bundle not found. Run the training pipeline to enable scoring.")

        if assets.eda_ready:
            st.success("EDA artifacts ready")
        else:
            st.warning("EDA artifacts missing (run pipeline to generate eda_profile and report).")

        if assets.governance_ready:
            st.success("Governance artifacts ready")
        else:
            st.warning("Governance artifacts missing (run pipeline to generate public view).")

        render_sidebar_summary(RUNTIME_CONFIG)

        with st.expander("Navigation Notes", expanded=False):
            st.markdown(
                """
                - `Control Room`: high-level monitoring and customer scoring
                - `Executive Overview`: board-level summary and artifact export
                - `Risk and Growth`: cohort mix and opportunity view
                - `Prioritization`: operational action list
                - `Simulation`: campaign impact scenario planning
                """
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

    return SidebarState(dataframe=dataframe, predictor=predictor, model_loaded=model_loaded)


def render_portfolio_metrics(dataframe: pd.DataFrame) -> None:
    metrics = summarise_metrics(dataframe)
    columns = st.columns(4)
    columns[0].metric("Total Customers", f"{metrics['total_customers']:,}")
    columns[1].metric("Churn Rate", f"{metrics['churn_rate']:.1f}%")
    columns[2].metric("Average Monthly Charge", f"${metrics['avg_monthly']:.2f}")
    columns[3].metric("Average Tenure", f"{metrics['avg_tenure']:.1f} months")


def render_filter_bar(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        contract_options = ["All"]
        if "Contract" in dataframe.columns:
            contract_options += sorted(dataframe["Contract"].dropna().unique().tolist())
        selected_contract = st.selectbox(
            "Contract cohort",
            contract_options,
            help="Filters the churn distribution chart and the preview table.",
        )

    with filter_col2:
        internet_options = ["All"]
        if "InternetService" in dataframe.columns:
            internet_options += sorted(dataframe["InternetService"].dropna().unique().tolist())
        selected_internet = st.selectbox(
            "Internet service segment",
            internet_options,
            help="Filters the contract mix chart and the preview table.",
        )

    return (
        *build_filtered_views(dataframe, selected_contract, selected_internet),
        selected_contract,
        selected_internet,
    )


def render_overview_tab(dataframe: pd.DataFrame) -> None:
    with section_container(
        "Portfolio Snapshot",
        (
            "Use this summary as the first stop for portfolio health before moving "
            "to prioritization and simulation."
        ),
    ):
        render_portfolio_metrics(dataframe)
        st.caption(
            "These KPIs are derived from the raw telco dataset and provide a compact "
            "view of customer base risk exposure."
        )

    with section_container(
        "Execution Notes",
        (
            "The dashboard is designed around an operating model where the score "
            "supports action, not isolated model inspection."
        ),
    ):
        left_col, right_col = st.columns((1.25, 1))
        with left_col:
            st.markdown(
                """
                **What this workspace is optimized for**

                - identifying exposure in the current customer portfolio
                - exploring risk distribution by contract and service mix
                - preparing action lists for retention teams
                - testing simple intervention scenarios before campaign rollout
                """
            )
        with right_col:
            st.markdown(
                """
                **Suggested reading flow**

                1. review portfolio KPIs here
                2. inspect artifact-backed executive outputs
                3. move to prioritization for action lists
                4. use simulation to size campaign impact
                """
            )


def render_exploration_tab(
    dataframe: pd.DataFrame,
    left_chart_df: pd.DataFrame,
    right_chart_df: pd.DataFrame,
    preview_df: pd.DataFrame,
    selected_contract: str,
    selected_internet: str,
) -> None:
    with section_container(
        "Cohort Filters",
        (
            "Filter the portfolio by contract and internet service to compare churn "
            "exposure across commercial segments."
        ),
    ):
        st.caption(
            "The controls above update the cohort-level charts and the source-record "
            "preview in the same view."
        )

    with section_container(
        "Risk Exploration",
        "Compare churn mix and contract-level churn rates without leaving the dashboard.",
    ):
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader(f"Churn Distribution | {selected_contract}")
            if "Churn" in left_chart_df.columns:
                churn_counts = (
                    left_chart_df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
                )
                figure = px.pie(
                    values=churn_counts.values,
                    names=churn_counts.index,
                    title="Churn split within the selected contract cohort",
                    color=churn_counts.index,
                    color_discrete_map={"Yes": COLOR_ALERT, "No": COLOR_PRIMARY},
                    hole=0.52,
                )
                figure.update_layout(
                    margin=dict(l=10, r=10, t=48, b=10),
                    legend_title_text="Churn",
                    title_font=dict(size=18),
                )
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.info("Column 'Churn' is not available for this chart.")

        with chart_col2:
            st.subheader(f"Contract Mix | {selected_internet}")
            if {"Contract", "Churn"}.issubset(right_chart_df.columns):
                contract_order = ["Month-to-month", "One year", "Two year"]
                contract_churn_rate = (
                    right_chart_df.assign(churn_yes=right_chart_df["Churn"].eq("Yes").astype(float))
                    .groupby("Contract", as_index=True)["churn_yes"]
                    .mean()
                    .mul(100)
                    .reindex(contract_order, fill_value=0.0)
                )
                figure = px.bar(
                    x=contract_churn_rate.index,
                    y=contract_churn_rate.values,
                    title="Churn rate by contract type",
                    labels={"x": "Contract type", "y": "Rate (%)"},
                    color_discrete_sequence=[COLOR_SECONDARY],
                )
                figure.update_traces(marker_line_color="#083344", marker_line_width=1.0)
                figure.update_layout(margin=dict(l=10, r=10, t=48, b=10), title_font=dict(size=18))
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.info("The required columns for the contract view are unavailable.")

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
        valid_columns = [column for column in columns_to_show if column in preview_df.columns]
        st.dataframe(preview_df[valid_columns].head(20), use_container_width=True, hide_index=True)


def render_scoring_tab(predictor: ChurnPredictor | None, model_loaded: bool) -> None:
    if not model_loaded or predictor is None:
        st.info(
            "Scoring is unavailable until the inference bundle is generated by the "
            "training pipeline."
        )
        return

    with section_container(
        "Customer-Level Scoring",
        (
            "Capture a representative profile and inspect the resulting churn risk "
            "before pushing the customer into downstream action."
        ),
    ):
        with st.form("customer_scoring_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])

            with col2:
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
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
                monthly = st.number_input(
                    "Monthly charges ($)", min_value=0.0, max_value=200.0, value=65.5
                )
                total = st.number_input(
                    "Total charges ($)", min_value=0.0, max_value=10000.0, value=786.0
                )
                internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                phone_service = st.selectbox("Phone service", ["Yes", "No"])

            submitted = st.form_submit_button("Score Customer", use_container_width=True)

        if not submitted:
            return

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
            probability = result.probability
            risk_level = format_risk_level(result.risk_level)

            result_col1, result_col2 = st.columns((1.05, 0.95))
            with result_col1:
                figure = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
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
                figure.update_layout(margin=dict(l=10, r=10, t=56, b=0), height=280)
                st.plotly_chart(figure, use_container_width=True)

            with result_col2:
                st.metric("Risk Level", risk_level)
                st.metric("Probability", f"{probability:.1%}")
                if risk_level == "High":
                    st.error("Escalate this customer into the highest-priority retention queue.")
                elif risk_level == "Medium":
                    st.warning("This customer warrants guided outreach and commercial review.")
                else:
                    st.success("This customer can remain outside the immediate intervention queue.")
                st.caption(
                    "Use the probability together with value and contract context. "
                    "The expected operating mode is prioritization by risk and "
                    "business value."
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main() -> None:
    configure_dashboard_page(page_title="Churn Prediction Control Room", page_icon="CR")
    pio.templates.default = "plotly_white"
    inject_global_styles()
    render_operational_shell()

    sidebar_state = render_sidebar(DASHBOARD_RUNTIME)
    dataframe = sidebar_state.dataframe

    if dataframe is None:
        st.error(
            "Dashboard data unavailable. Expected at least one source: "
            f"raw={DASHBOARD_RUNTIME.data_path} or silver={DASHBOARD_RUNTIME.silver_path}."
        )
        st.stop()

    with section_container(
        "Exploration Filters",
        (
            "Adjust the active cohort before reviewing the portfolio overview, "
            "cohort charts, and source records."
        ),
    ):
        left_chart_df, right_chart_df, preview_df, selected_contract, selected_internet = (
            render_filter_bar(dataframe)
        )

    overview_tab, exploration_tab, scoring_tab = st.tabs(
        ["Overview", "Explore Cohorts", "Score Customer"]
    )

    with overview_tab:
        render_overview_tab(dataframe)

    with exploration_tab:
        if left_chart_df.empty and right_chart_df.empty:
            st.warning("No records matched the selected filters.")
        else:
            render_exploration_tab(
                dataframe,
                left_chart_df,
                right_chart_df,
                preview_df,
                selected_contract,
                selected_internet,
            )

    with scoring_tab:
        render_scoring_tab(sidebar_state.predictor, sidebar_state.model_loaded)

    render_footer()


if __name__ == "__main__":
    main()

from __future__ import annotations

import plotly.express as px
import streamlit as st

from apps.dashboard_runtime import (
    build_dashboard_status,
    build_portfolio_summary,
    build_risk_distribution,
    load_dashboard_assets,
)
from apps.dashboard_ui import (
    COLOR_ALERT,
    COLOR_SECONDARY,
    configure_dashboard_page,
    inject_global_styles,
    render_footer,
    render_page_hero,
    render_status_banner,
    section_container,
)

configure_dashboard_page(page_title="Risk and Growth", page_icon="RG")
inject_global_styles()

assets = load_dashboard_assets()
status = build_dashboard_status(assets)
dataframe = assets.prioritization

render_page_hero(
    eyebrow="Risk and Growth",
    title="Portfolio Exposure and Growth Lens",
    subtitle=(
        "Inspect customer risk mix, contract concentration, and next-purchase opportunity in one "
        "artifact-backed workspace."
    ),
    meta="Portfolio analytics | Growth lens | Contract mix",
)

if dataframe.empty:
    st.warning("Run the pipeline to generate customer_prioritization.csv.")
    st.stop()

render_status_banner(
    title="Prioritization artifact status",
    body=(
        "This page reads from the same prioritization artifact used by the downstream action list."
        if not status["fallback"]
        else "Fallback prioritization artifacts are currently being used."
    ),
    status_items=[
        ("Environment", str(status["environment"])),
        ("Schema", str(status["schema_version"])),
        ("Run ID", str(status["run_id"])),
    ],
)

summary = build_portfolio_summary(dataframe)
with section_container(
    "Portfolio KPIs",
    (
        "Use this summary to quantify the size of the immediate churn problem "
        "before moving into campaign design."
    ),
):
    summary_columns = st.columns(4)
    summary_columns[0].metric("High-Risk Customers", f"{summary['high_risk_customers']:,}")
    summary_columns[1].metric("Revenue at Risk", f"${summary['high_risk_revenue']:,.2f}")
    summary_columns[2].metric("Month-to-Month Share", f"{summary['month_to_month_share']:.1f}%")
    summary_columns[3].metric("Avg. Next Purchase", f"${summary['avg_next_purchase']:,.2f}")

distribution_tab, opportunity_tab = st.tabs(["Risk Distribution", "Growth Opportunity"])

with distribution_tab:
    with section_container(
        "Customer Distribution by Risk Segment",
        "This chart shows the portfolio mix across high, medium, and low churn segments.",
    ):
        risk_distribution = build_risk_distribution(dataframe)
        figure = px.bar(
            risk_distribution,
            x="risk_segment",
            y="customers",
            title="Customer distribution by risk segment",
            labels={"risk_segment": "Risk segment", "customers": "Customers"},
            color="risk_segment",
            color_discrete_map={"high": COLOR_ALERT, "medium": "#f59e0b", "low": "#0f766e"},
        )
        figure.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(figure, use_container_width=True)

with opportunity_tab:
    with section_container(
        "Risk vs. Next Purchase Prediction",
        (
            "Plot churn probability against predicted next purchase value to find "
            "accounts that justify intervention budget."
        ),
    ):
        figure = px.scatter(
            dataframe,
            x="churn_probability",
            y="next_purchase_prediction",
            color="Contract",
            title="Risk versus next purchase prediction",
            labels={
                "churn_probability": "Churn probability",
                "next_purchase_prediction": "Next purchase prediction",
            },
            color_discrete_sequence=[COLOR_SECONDARY, "#38bdf8", "#f59e0b"],
        )
        figure.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(figure, use_container_width=True)

render_footer()

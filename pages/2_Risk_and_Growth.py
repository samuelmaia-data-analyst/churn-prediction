from __future__ import annotations

import plotly.express as px
import streamlit as st

from apps.dashboard_runtime import (
    build_portfolio_summary,
    build_risk_distribution,
    load_dashboard_assets,
)

st.set_page_config(page_title="Risk and Growth", page_icon="RG", layout="wide")
st.title("Risk and Growth")
st.caption("Portfolio-level risk distribution and forward revenue opportunity view.")

assets = load_dashboard_assets()
df = assets.prioritization
if df.empty:
    st.warning("Run the pipeline to generate customer_prioritization.csv.")
    st.stop()

summary = build_portfolio_summary(df)
sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
sum_col1.metric("High Risk Customers", f"{summary['high_risk_customers']:,}")
sum_col2.metric("Revenue at Risk", f"${summary['high_risk_revenue']:,.2f}")
sum_col3.metric("Month-to-Month Share", f"{summary['month_to_month_share']:.1f}%")
sum_col4.metric("Avg. Next Purchase", f"${summary['avg_next_purchase']:,.2f}")

risk_distribution = build_risk_distribution(df)
col1, col2 = st.columns(2)
with col1:
    fig_risk = px.bar(
        risk_distribution,
        x="risk_segment",
        y="customers",
        title="Customer Distribution by Risk Segment",
        labels={"risk_segment": "Risk segment", "customers": "Customers"},
        color="risk_segment",
        color_discrete_map={"high": "#dc2626", "medium": "#f59e0b", "low": "#0f766e"},
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    fig_growth = px.scatter(
        df,
        x="churn_probability",
        y="next_purchase_prediction",
        color="Contract",
        title="Risk vs. Next Purchase Prediction",
        labels={
            "churn_probability": "Churn probability",
            "next_purchase_prediction": "Next purchase prediction",
        },
    )
    st.plotly_chart(fig_growth, use_container_width=True)

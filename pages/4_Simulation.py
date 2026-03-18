from __future__ import annotations

import streamlit as st

from apps.dashboard_runtime import load_dashboard_assets, simulate_retention_impact

st.set_page_config(page_title="Simulation", page_icon="SM", layout="wide")
st.title("Simulation")
st.caption(
    "Scenario view for retention effectiveness and revenue " "recovered from high-risk cohorts."
)

assets = load_dashboard_assets()
df = assets.prioritization
if df.empty:
    st.warning("Run the pipeline to generate customer_prioritization.csv.")
    st.stop()

retention_effectiveness = st.slider(
    "Retention campaign effectiveness (%)",
    min_value=0,
    max_value=100,
    value=25,
    help="Estimated percentage of revenue at risk recovered by the campaign.",
)

simulation = simulate_retention_impact(df, retention_effectiveness=retention_effectiveness)

col1, col2, col3 = st.columns(3)
col1.metric("Baseline Revenue at Risk", f"${simulation['baseline_revenue_risk']:,.2f}")
col2.metric("Recovered Revenue", f"${simulation['recovered_revenue']:,.2f}")
col3.metric("Remaining Revenue at Risk", f"${simulation['remaining_revenue_risk']:,.2f}")

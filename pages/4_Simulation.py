from __future__ import annotations

import streamlit as st

from src.dashboard_data import load_prioritization

st.set_page_config(page_title="Simulation", page_icon="SM", layout="wide")
st.title("Simulation")
st.caption("Simulacao executiva de impacto de retencao")

df = load_prioritization()
if df.empty:
    st.warning("Rode o pipeline para gerar customer_prioritization.csv.")
    st.stop()

baseline_revenue_risk = df.loc[df["churn_probability"] >= 0.7, "MonthlyCharges"].sum()

retention_effectiveness = st.slider(
    "Efetividade da campanha de retencao (%)",
    min_value=0,
    max_value=100,
    value=25,
    help="Percentual estimado de recuperacao da receita em risco.",
)

recovered = baseline_revenue_risk * (retention_effectiveness / 100)
remaining = baseline_revenue_risk - recovered

col1, col2, col3 = st.columns(3)
col1.metric("Baseline Revenue at Risk", f"${baseline_revenue_risk:,.2f}")
col2.metric("Recovered Revenue (simulated)", f"${recovered:,.2f}")
col3.metric("Remaining Revenue at Risk", f"${remaining:,.2f}")

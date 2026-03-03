from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard_data import load_prioritization

st.set_page_config(page_title="Risk and Growth", page_icon="RG", layout="wide")
st.title("Risk and Growth")
st.caption("Distribuicao de risco e potencial de receita futura")

df = load_prioritization()
if df.empty:
    st.warning("Rode o pipeline para gerar customer_prioritization.csv.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    fig_risk = px.histogram(
        df,
        x="churn_probability",
        nbins=20,
        title="Distribution of Churn Probability",
        labels={"churn_probability": "Churn Probability"},
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    fig_growth = px.scatter(
        df,
        x="churn_probability",
        y="next_purchase_prediction",
        color="Contract",
        title="Risk vs Next Purchase Prediction",
        labels={
            "churn_probability": "Churn Probability",
            "next_purchase_prediction": "Next Purchase Prediction",
        },
    )
    st.plotly_chart(fig_growth, use_container_width=True)

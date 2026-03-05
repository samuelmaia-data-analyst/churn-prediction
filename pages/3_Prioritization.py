from __future__ import annotations

import streamlit as st

from src.dashboard_data import PRIORITIZATION_PATH, load_prioritization

st.set_page_config(page_title="Prioritization", page_icon="PR", layout="wide")
st.title("Prioritization")
st.caption("Clientes priorizados por risco de churn e recomendação de ação")

df = load_prioritization()
if df.empty:
    st.warning("Rode o pipeline para gerar customer_prioritization.csv.")
    st.stop()

top_n = st.slider(
    "Quantidade de clientes priorizados",
    min_value=10,
    max_value=min(len(df), 500),
    value=min(50, len(df)),
    help="Seleciona quantos clientes de maior risco exibir.",
)

prioritized = df.head(top_n)
st.dataframe(prioritized, use_container_width=True)

with open(PRIORITIZATION_PATH, "rb") as fp:
    st.download_button(
        "Download customer_prioritization.csv",
        data=fp,
        file_name="customer_prioritization.csv",
        mime="text/csv",
    )

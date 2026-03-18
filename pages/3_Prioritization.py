from __future__ import annotations

import streamlit as st

from apps.dashboard_runtime import load_dashboard_assets

st.set_page_config(page_title="Prioritization", page_icon="PR", layout="wide")
st.title("Prioritization")
st.caption("Operational list of customers ranked by churn risk and recommended action.")

assets = load_dashboard_assets()
df = assets.prioritization
if df.empty:
    st.warning("Run the pipeline to generate customer_prioritization.csv.")
    st.stop()

top_n = st.slider(
    "Number of prioritized customers",
    min_value=10,
    max_value=min(len(df), 500),
    value=min(50, len(df)),
    help="Select how many high-risk customers to display.",
)

prioritized = df.head(top_n)
st.dataframe(prioritized, use_container_width=True)

with open(assets.prioritization_path, "rb") as fp:
    st.download_button(
        "Download customer_prioritization.csv",
        data=fp,
        file_name="customer_prioritization.csv",
        mime="text/csv",
    )

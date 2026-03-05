from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.dashboard_data import KPI_PATH, REPORT_PATH, load_executive_report, load_kpis

st.set_page_config(page_title="Executive Overview", page_icon="📈", layout="wide")
st.title("Executive Overview")
st.caption("Painel C-level de KPIs e desempenho do pipeline")

kpis_df = load_kpis()
report = load_executive_report()

if kpis_df.empty or not report:
    st.warning("Rode o pipeline para gerar dados em data/gold e reports.")
    st.stop()

kpis = report.get("kpis", {})
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{int(kpis.get('total_customers', 0)):,}")
col2.metric("Churn Rate", f"{kpis.get('churn_rate', 0.0):.2%}")
col3.metric("High Risk Customers", f"{int(kpis.get('high_risk_customers', 0)):,}")
col4.metric("Revenue at Risk", f"${kpis.get('revenue_at_risk', 0.0):,.2f}")

st.subheader("Model Metrics")
st.json(report.get("model_metrics", {}))
metrics = report.get("model_metrics", {})

baseline = metrics.get("baseline_model", {})
st.markdown("### Baseline Model")
st.markdown(f"- Logistic Regression")
st.markdown(f"- ROC-AUC: `{float(baseline.get('roc_auc', 0.0)):.2f}`")

st.markdown("### Model Comparison")
comparison_df = pd.DataFrame(metrics.get("model_comparison", []))
if not comparison_df.empty and {"model", "roc_auc"}.issubset(comparison_df.columns):
    st.dataframe(
        comparison_df.rename(columns={"model": "Model", "roc_auc": "ROC-AUC"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Model comparison unavailable.")

st.markdown("### Top Drivers of Churn")
for driver in metrics.get("top_drivers_of_churn", []):
    st.markdown(f"- {driver}")

st.markdown("### Key Insights")
for insight in metrics.get("key_insights", []):
    st.markdown(f"- {insight}")

st.markdown("### Pipeline Visual")
st.code(metrics.get("pipeline_visual", "Raw -> Bronze -> Silver -> Gold"))

with open(REPORT_PATH, "rb") as fp:
    st.download_button(
        "Download executive_report.json",
        data=fp,
        file_name="executive_report.json",
        mime="application/json",
    )

with open(KPI_PATH, "rb") as fp:
    st.download_button(
        "Download kpi_summary.csv",
        data=fp,
        file_name="kpi_summary.csv",
        mime="text/csv",
    )

st.code(json.dumps(report.get("top_10_priorities", []), ensure_ascii=False, indent=2))

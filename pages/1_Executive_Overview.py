from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard_data import KPI_PATH, REPORT_PATH, load_executive_report, load_kpis

st.set_page_config(page_title="Executive Overview", page_icon=":bar_chart:", layout="wide")
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

metrics = report.get("model_metrics", {})

st.subheader("Model Summary")
baseline = metrics.get("baseline_model", {})
st.markdown("### Baseline Model")
st.markdown("- Logistic Regression")
st.markdown(f"- ROC-AUC: `{float(baseline.get('roc_auc', 0.0)):.3f}`")

st.markdown("### Model Comparison")
comparison_df = pd.DataFrame(metrics.get("model_comparison", []))
if not comparison_df.empty and {"model", "roc_auc"}.issubset(comparison_df.columns):
    comparison_df = comparison_df.rename(columns={"model": "Model", "roc_auc": "ROC-AUC"})
    comparison_df["ROC-AUC"] = comparison_df["ROC-AUC"].astype(float).round(3)
    st.dataframe(
        comparison_df.sort_values("ROC-AUC", ascending=False),
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
st.markdown("""
```mermaid
flowchart LR
    A[Raw] --> B[Bronze]
    B --> C[Silver]
    C --> D[Gold]
```
""")

st.markdown("### Top 10 Prioridades")
top10_df = pd.DataFrame(report.get("top_10_priorities", []))
if not top10_df.empty:
    if "churn_probability" in top10_df.columns:
        top10_df["churn_probability"] = (top10_df["churn_probability"] * 100).round(2)
    columns = [
        "customerID",
        "churn_probability",
        "next_purchase_prediction",
        "MonthlyCharges",
        "Contract",
    ]
    existing_columns = [c for c in columns if c in top10_df.columns]
    st.dataframe(top10_df[existing_columns], use_container_width=True, hide_index=True)
else:
    st.info("Top priorities unavailable.")

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

from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_runtime import load_dashboard_assets

st.set_page_config(page_title="Executive Overview", page_icon="EO", layout="wide")
st.title("Executive Overview")
st.caption(
    "Board-level KPI view with operating context, model summary, " "and downloadable artifacts."
)

assets = load_dashboard_assets()

if not assets.is_ready:
    st.warning("Run the pipeline to generate gold outputs and reporting artifacts for this view.")
    st.stop()

kpis = assets.report.get("kpis", {})
metadata = assets.report_metadata
metrics = assets.report.get("model_metrics", {})

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Environment", str(metadata.get("environment", "unknown")))
status_col2.metric("Schema Version", str(metadata.get("schema_version", "unknown")))
status_col3.metric("Run ID", str(metadata.get("run_id", "unknown")))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{int(kpis.get('total_customers', 0)):,}")
col2.metric("Churn Rate", f"{kpis.get('churn_rate', 0.0):.2%}")
col3.metric("High Risk Customers", f"{int(kpis.get('high_risk_customers', 0)):,}")
col4.metric("Revenue at Risk", f"${kpis.get('revenue_at_risk', 0.0):,.2f}")

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
    st.info("Model comparison metadata is unavailable.")

st.markdown("### Top Drivers of Churn")
for driver in metrics.get("top_drivers_of_churn", []):
    st.markdown(f"- {driver}")

st.markdown("### Key Insights")
for insight in metrics.get("key_insights", []):
    st.markdown(f"- {insight}")

st.markdown("### Executive Operating Model")
st.caption("Strategy, prioritization, execution, and value realization in one operating view.")
st.markdown(
    """
```mermaid
flowchart LR
    A[Strategic Targets\nRevenue Retention Margin]
    --> B[Decision Intelligence\nChurn Risk & Value at Risk]
    B --> C{Capital Allocation Gate\nApprove Hold Reject}
    C --> D[Commercial Execution\nSales CS Marketing Programs]
    D --> E[Value Realization\nRetention Revenue Margin]
    E --> F[Executive Cockpit\nKPI ROI SLA]
    F --> G[Weekly Operating Review]
    G -. Rebalance budget and capacity .-> C
    G -. Model and policy feedback .-> B
```
"""
)

gov_col1, gov_col2, gov_col3 = st.columns(3)
gov_col1.metric("Decision Cadence", "Weekly")
gov_col2.metric("Allocation Gate", "Approve / Hold / Reject")
gov_col3.metric("Primary Owners", "CCO / CFO / RevOps")

st.markdown("### Top 10 Priority Accounts")
top10_df = pd.DataFrame(assets.report.get("top_10_priorities", []))
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
    st.info("Priority list is unavailable.")

with open(assets.report_path, "rb") as fp:
    st.download_button(
        "Download executive_report.json",
        data=fp,
        file_name="executive_report.json",
        mime="application/json",
    )

with open(assets.kpi_path, "rb") as fp:
    st.download_button(
        "Download kpi_summary.csv",
        data=fp,
        file_name="kpi_summary.csv",
        mime="text/csv",
    )

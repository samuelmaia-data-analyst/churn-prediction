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

st.markdown("### Executive Operating Model")
st.caption("Board-grade view: strategy, allocation, execution, and value realization")
st.markdown("""
```mermaid
flowchart LR
    A[Strategic Targets\nRevenue Retention Margin] --> B[Decision Intelligence\nChurn Risk Value at Risk]
    B --> C{Capital Allocation Gate\nApprove Hold Reject}
    C --> D[Commercial Execution\nSales CS Marketing Programs]
    D --> E[Value Realization\nRetention Revenue Margin]
    E --> F[Executive Cockpit\nKPI ROI SLA]
    F --> G[Weekly Operating Review]
    G -. Rebalance budget and capacity .-> C
    G -. Model and policy feedback .-> B
```
""")

gov_col1, gov_col2, gov_col3 = st.columns(3)
gov_col1.metric("Decision Cadence", "Weekly")
gov_col2.metric("Allocation Gate", "Approve / Hold / Reject")
gov_col3.metric("Primary Owners", "CCO / CFO / RevOps")

with st.expander("Operator View (Owners + SLAs + Controls)"):
    st.caption("Execution detail for operating teams")
    st.markdown("""
```mermaid
flowchart LR
    subgraph O1[Data and ML Factory]
        A1[Ingestion and Standardization\nOwner Data Engineering\nSLA Daily 07:00 UTC] --> A2[Quality and Contract Gates\nOwner Data Governance\nSLA <2 percent failed checks]
        A2 --> A3[Scoring and Value at Risk\nOwner Data Science\nSLA AUC >= 0.82]
    end

    subgraph O2[Portfolio Governance]
        A3 --> B1[Prioritization Engine\nOwner RevOps\nSLA Top 10 published by 09:00]
        B1 --> B2{Investment Committee Gate\nOwner CCO CFO}
    end

    subgraph O3[Field Execution]
        B2 --> C1[Playbook Activation\nOwner Sales and CS\nSLA First touch <24h]
        C1 --> C2[Customer Outcomes\nSave Upsell Cross-sell]
        C2 --> C3[Outcome Ledger\nOwner Finance\nSLA Weekly close]
    end

    C3 --> D1[Executive Cockpit\nOwner Strategy Office]
    D1 --> D2[Weekly Operating Review]
    D2 -. Drift incidents and policy breaches .-> D3[Model Risk Monitoring\nOwner MRM]
    D3 -. Controlled retrain and recalibration .-> A3
    D2 -. Reallocate budget and capacity .-> B2
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

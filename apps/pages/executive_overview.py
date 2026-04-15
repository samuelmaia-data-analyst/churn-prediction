from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_runtime import build_dashboard_status, load_dashboard_assets
from apps.dashboard_ui import (
    configure_dashboard_page,
    inject_global_styles,
    render_download_actions,
    render_footer,
    render_page_hero,
    render_status_banner,
    section_container,
)

configure_dashboard_page(page_title="Executive Overview", page_icon="EO")
inject_global_styles()

assets = load_dashboard_assets()
status = build_dashboard_status(assets)

render_page_hero(
    eyebrow="Executive Overview",
    title="Executive Churn Briefing",
    subtitle=(
        "Board-level KPI view, model context, and top-priority accounts backed by the canonical "
        "reporting layer."
    ),
    meta="Leadership view | KPI summary | Export-ready artifacts",
)

if not assets.is_ready:
    st.warning("Run the pipeline to generate gold outputs and reporting artifacts for this view.")
    st.stop()

if status["fallback"]:
    banner_title = "Fallback dashboard artifacts detected"
    banner_body = (
        "Executive outputs are available, but they were generated through the fallback path."
    )
else:
    banner_title = "Canonical executive artifacts available"
    banner_body = "This page is backed by the canonical reporting and gold-layer outputs."

render_status_banner(
    title=banner_title,
    body=banner_body,
    status_items=[
        ("Environment", str(status["environment"])),
        ("Schema", str(status["schema_version"])),
        ("Run ID", str(status["run_id"])),
    ],
)

kpis = assets.report.get("kpis", {})
metrics = assets.report.get("model_metrics", {})

with section_container(
    "Executive KPI Snapshot",
    "A compact commercial view of portfolio risk and revenue exposure for operating review.",
):
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Customers", f"{int(kpis.get('total_customers', 0)):,}")
    metric_columns[1].metric("Churn Rate", f"{kpis.get('churn_rate', 0.0):.2%}")
    metric_columns[2].metric("High-Risk Customers", f"{int(kpis.get('high_risk_customers', 0)):,}")
    metric_columns[3].metric("Revenue at Risk", f"${kpis.get('revenue_at_risk', 0.0):,.2f}")

summary_tab, priorities_tab, exports_tab = st.tabs(
    ["Model Summary", "Priority Accounts", "Downloads"]
)

with summary_tab:
    with section_container(
        "Model Summary",
        (
            "Show enough model context for leadership alignment without turning "
            "this page into a data science notebook."
        ),
    ):
        baseline = metrics.get("baseline_model", {})
        overview_col1, overview_col2 = st.columns((0.95, 1.05))

        with overview_col1:
            st.markdown("**Baseline Model**")
            st.markdown("- Logistic Regression")
            st.markdown(f"- ROC-AUC: `{float(baseline.get('roc_auc', 0.0)):.3f}`")
            st.markdown("**Top Drivers of Churn**")
            for driver in metrics.get("top_drivers_of_churn", []):
                st.markdown(f"- {driver}")

        with overview_col2:
            st.markdown("**Key Insights**")
            for insight in metrics.get("key_insights", []):
                st.markdown(f"- {insight}")
            st.markdown("**Executive Operating Model**")
            st.markdown(
                """
                1. detect exposure through churn score and value-at-risk
                2. allocate commercial capacity to the highest-priority cohort
                3. execute retention and pricing interventions
                4. review recovered revenue and rebalance policy weekly
                """
            )

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

with priorities_tab:
    with section_container(
        "Top 10 Priority Accounts",
        "These accounts represent the highest-value candidates for immediate retention action.",
    ):
        top10_df = pd.DataFrame(assets.report.get("top_10_priorities", []))
        if not top10_df.empty:
            if "churn_probability" in top10_df.columns:
                top10_df["churn_probability"] = top10_df["churn_probability"].map(
                    lambda value: f"{value:.2%}"
                )
            existing_columns = [
                column
                for column in [
                    "customerID",
                    "churn_probability",
                    "next_purchase_prediction",
                    "MonthlyCharges",
                    "Contract",
                ]
                if column in top10_df.columns
            ]
            st.dataframe(top10_df[existing_columns], use_container_width=True, hide_index=True)
        else:
            st.info("Priority list is unavailable.")

with exports_tab:
    with section_container(
        "Export Actions",
        (
            "Use these downloads to move the executive summary and KPI snapshot "
            "into leadership workflows."
        ),
    ):
        render_download_actions(
            [
                (
                    "Download executive_report.json",
                    assets.report_path,
                    "executive_report.json",
                    "application/json",
                ),
                ("Download kpi_summary.csv", assets.kpi_path, "kpi_summary.csv", "text/csv"),
            ]
        )

render_footer()

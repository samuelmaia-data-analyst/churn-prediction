from __future__ import annotations

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

configure_dashboard_page(page_title="Prioritization", page_icon="PR")
inject_global_styles()

assets = load_dashboard_assets()
status = build_dashboard_status(assets)
dataframe = assets.prioritization

render_page_hero(
    eyebrow="Prioritization",
    title="Operational Retention Queue",
    subtitle=(
        "A ranked customer list for actioning churn risk through retention "
        "outreach, pricing review, and contract intervention."
    ),
    meta="Action list | Ranked customers | Download-ready",
)

if dataframe.empty:
    st.warning("Run the pipeline to generate customer_prioritization.csv.")
    st.stop()

render_status_banner(
    title="Prioritization artifact status",
    body=(
        "This queue is backed by the canonical prioritization artifact."
        if not status["fallback"]
        else "Fallback dashboard artifacts are currently populating this queue."
    ),
    status_items=[
        ("Environment", str(status["environment"])),
        ("Schema", str(status["schema_version"])),
        ("Run ID", str(status["run_id"])),
    ],
)

with section_container(
    "Queue Controls",
    (
        "Keep the list short enough to be actioned by the commercial team in "
        "the current operating cycle."
    ),
):
    top_n = st.slider(
        "Number of prioritized customers",
        min_value=10,
        max_value=min(len(dataframe), 500),
        value=min(50, len(dataframe)),
        help="Select how many customers to keep in the visible retention queue.",
    )

with section_container(
    "Priority Queue",
    (
        "Each row should be interpretable by an operator without needing to "
        "inspect raw model features."
    ),
):
    prioritized = dataframe.head(top_n)
    columns = [
        "customerID",
        "churn_probability",
        "risk_segment",
        "value_segment",
        "action_recommendation",
        "MonthlyCharges",
        "next_purchase_prediction",
    ]
    available_columns = [column for column in columns if column in prioritized.columns]
    display_df = prioritized[available_columns].copy()
    if "churn_probability" in display_df.columns:
        display_df["churn_probability"] = display_df["churn_probability"].map(
            lambda value: f"{value:.2%}"
        )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with section_container(
    "Export Actions",
    "Use the CSV export to move the ranked queue into campaign tooling or analyst workflows.",
):
    render_download_actions(
        [
            (
                "Download customer_prioritization.csv",
                assets.prioritization_path,
                "customer_prioritization.csv",
                "text/csv",
            )
        ]
    )

render_footer()

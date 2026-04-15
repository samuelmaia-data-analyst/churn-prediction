from __future__ import annotations

import streamlit as st

from apps.dashboard_runtime import (
    build_dashboard_status,
    load_dashboard_assets,
    simulate_retention_impact,
)
from apps.dashboard_ui import (
    configure_dashboard_page,
    inject_global_styles,
    render_footer,
    render_page_hero,
    render_status_banner,
    section_container,
)

configure_dashboard_page(page_title="Simulation", page_icon="SM")
inject_global_styles()

assets = load_dashboard_assets()
status = build_dashboard_status(assets)
dataframe = assets.prioritization

render_page_hero(
    eyebrow="Simulation",
    title="Retention Impact Simulator",
    subtitle=(
        "Estimate how much revenue at risk could be recovered under different intervention "
        "effectiveness assumptions."
    ),
    meta="Scenario planning | Revenue impact | Campaign assumptions",
)

if dataframe.empty:
    st.warning("Run the pipeline to generate customer_prioritization.csv.")
    st.stop()

render_status_banner(
    title="Simulation artifact status",
    body=(
        "The simulation is using the canonical prioritization artifact."
        if not status["fallback"]
        else "Fallback prioritization artifacts are currently being used for simulation."
    ),
    status_items=[
        ("Environment", str(status["environment"])),
        ("Schema", str(status["schema_version"])),
        ("Run ID", str(status["run_id"])),
    ],
)

with section_container(
    "Scenario Controls",
    (
        "Use this view to test whether campaign effectiveness assumptions justify "
        "the operational cost of intervention."
    ),
):
    retention_effectiveness = st.slider(
        "Retention campaign effectiveness (%)",
        min_value=0,
        max_value=100,
        value=25,
        help="Estimated percentage of revenue at risk recovered by the campaign.",
    )

simulation = simulate_retention_impact(dataframe, retention_effectiveness=retention_effectiveness)

with section_container(
    "Scenario Outcome",
    (
        "This output converts the selected effectiveness assumption into a "
        "simplified commercial result."
    ),
):
    result_columns = st.columns(3)
    result_columns[0].metric(
        "Baseline Revenue at Risk",
        f"${simulation['baseline_revenue_risk']:,.2f}",
    )
    result_columns[1].metric("Recovered Revenue", f"${simulation['recovered_revenue']:,.2f}")
    result_columns[2].metric(
        "Remaining Revenue at Risk",
        f"${simulation['remaining_revenue_risk']:,.2f}",
    )
    st.caption(
        "This is a directional scenario model. It is useful for comparing "
        "intervention assumptions, not for replacing a full campaign ROI model."
    )

render_footer()

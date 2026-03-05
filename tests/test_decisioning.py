from __future__ import annotations

import pandas as pd

from src.decisioning import (
    build_action_playbook,
    customer_value_segment,
    threshold_for_value_segment,
)


def test_threshold_strategy_by_value_segment() -> None:
    assert customer_value_segment(95.0) == "high_ltv"
    assert customer_value_segment(50.0) == "low_ltv"
    assert threshold_for_value_segment("high_ltv") == 0.65
    assert threshold_for_value_segment("low_ltv") == 0.80


def test_action_playbook_has_actionable_contract() -> None:
    recommendations = pd.DataFrame(
        [
            {"value_segment": "high_ltv", "risk_segment": "high"},
            {"value_segment": "high_ltv", "risk_segment": "high"},
            {"value_segment": "low_ltv", "risk_segment": "high"},
            {"value_segment": "low_ltv", "risk_segment": "medium"},
        ]
    )
    playbook = build_action_playbook(recommendations)

    assert {"Segment", "Risk", "Action", "Expected ROI"}.issubset(playbook.columns)
    assert (
        playbook.loc[(playbook["Segment"] == "High LTV") & (playbook["Risk"] == "High"), "Action"]
        .iloc[0]
        == "Call retention"
    )
    assert (
        playbook.loc[
            (playbook["Segment"] == "High LTV") & (playbook["Risk"] == "High"), "Expected ROI"
        ].iloc[0]
        == "+$200/customer"
    )

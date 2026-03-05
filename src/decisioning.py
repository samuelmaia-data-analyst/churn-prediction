from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DecisionPolicy:
    name: str
    fp_cost: float
    fn_cost: float
    description: str


POLICIES: dict[str, DecisionPolicy] = {
    "campanha_cara": DecisionPolicy(
        name="campanha_cara",
        fp_cost=12.0,
        fn_cost=3.0,
        description="Contato caro; prioriza precisao e reduz falso positivo.",
    ),
    "balanceada": DecisionPolicy(
        name="balanceada",
        fp_cost=5.0,
        fn_cost=5.0,
        description="Trade-off simetrico entre falso positivo e falso negativo.",
    ),
    "campanha_barata": DecisionPolicy(
        name="campanha_barata",
        fp_cost=2.0,
        fn_cost=8.0,
        description="Contato barato; prioriza recall e reduz falso negativo.",
    ),
}

THRESHOLD_BY_VALUE_SEGMENT: dict[str, float] = {
    "high_ltv": 0.65,
    "low_ltv": 0.80,
}


def customer_value_segment(next_purchase: float) -> str:
    return "high_ltv" if next_purchase >= 80.0 else "low_ltv"


def threshold_for_value_segment(value_segment: str) -> float:
    return THRESHOLD_BY_VALUE_SEGMENT.get(value_segment, THRESHOLD_BY_VALUE_SEGMENT["low_ltv"])


def get_policy(policy_name: str = "balanceada") -> DecisionPolicy:
    return POLICIES.get(policy_name, POLICIES["balanceada"])


def decision_threshold(policy: DecisionPolicy) -> float:
    # Trigger retention when expected FN loss exceeds expected FP spend.
    return policy.fp_cost / (policy.fp_cost + policy.fn_cost)


def risk_segment(probability: float, threshold: float) -> str:
    medium_threshold = max(0.35, threshold - 0.20)
    if probability >= threshold:
        return "high"
    if probability >= medium_threshold:
        return "medium"
    return "low"


def action_for_segment(segment: str, value_segment: str) -> str:
    if segment == "high" and value_segment == "high_ltv":
        return "Call retention"
    if segment == "high":
        return "Retention offer by email"
    if segment == "medium" and value_segment == "high_ltv":
        return "Proactive loyalty outreach"
    if segment == "medium":
        return "Automated nurture journey"
    return "Monitor and upsell trigger"


def build_action_playbook(recommendations: pd.DataFrame) -> pd.DataFrame:
    assumptions: dict[tuple[str, str], dict[str, float | str]] = {
        ("high_ltv", "high"): {"action": "Call retention", "expected_roi_usd": 200.0},
        ("low_ltv", "high"): {"action": "Retention offer by email", "expected_roi_usd": 90.0},
        ("high_ltv", "medium"): {"action": "Proactive loyalty outreach", "expected_roi_usd": 80.0},
        ("low_ltv", "medium"): {"action": "Automated nurture journey", "expected_roi_usd": 35.0},
        ("high_ltv", "low"): {"action": "Monitor and upsell trigger", "expected_roi_usd": 20.0},
        ("low_ltv", "low"): {"action": "Monitor and upsell trigger", "expected_roi_usd": 8.0},
    }
    playbook = (
        recommendations[["value_segment", "risk_segment"]]
        .value_counts()
        .reset_index(name="Customers")
        .rename(columns={"value_segment": "Segment", "risk_segment": "Risk"})
    )
    playbook["Action"] = playbook.apply(
        lambda row: str(assumptions[(row["Segment"], row["Risk"])]["action"]),
        axis=1,
    )
    playbook["Expected ROI"] = playbook.apply(
        lambda row: (
            f"+${assumptions[(row['Segment'], row['Risk'])]['expected_roi_usd']:.0f}/customer"
        ),
        axis=1,
    )
    playbook["expected_roi_usd_per_customer"] = playbook.apply(
        lambda row: float(assumptions[(row["Segment"], row["Risk"])]["expected_roi_usd"]),
        axis=1,
    )
    playbook["total_expected_roi_usd"] = (
        playbook["expected_roi_usd_per_customer"] * playbook["Customers"]
    )

    playbook["Segment"] = playbook["Segment"].map(
        {"high_ltv": "High LTV", "low_ltv": "Low LTV"}
    )
    playbook["Risk"] = playbook["Risk"].str.capitalize()
    playbook["risk_order"] = playbook["Risk"].map({"High": 0, "Medium": 1, "Low": 2}).fillna(99)
    playbook = playbook.sort_values(["risk_order", "Segment"]).reset_index(drop=True)
    playbook = playbook.drop(columns=["risk_order"])

    return playbook[
        ["Segment", "Risk", "Action", "Expected ROI", "Customers", "total_expected_roi_usd"]
    ]

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


def action_for_segment(segment: str, next_purchase: float) -> str:
    if segment == "high" and next_purchase >= 80:
        return "Contato imediato + oferta premium de retencao"
    if segment == "high":
        return "Contato imediato + desconto de retencao"
    if segment == "medium":
        return "Campanha de engajamento proativa"
    return "Monitoramento e nutricao de relacionamento"


def build_action_playbook(
    recommendations: pd.DataFrame,
    policy: DecisionPolicy,
) -> pd.DataFrame:
    assumptions = {
        "Contato imediato + oferta premium de retencao": {"cost": 120.0, "lift": 0.35},
        "Contato imediato + desconto de retencao": {"cost": 70.0, "lift": 0.25},
        "Campanha de engajamento proativa": {"cost": 18.0, "lift": 0.12},
        "Monitoramento e nutricao de relacionamento": {"cost": 4.0, "lift": 0.04},
    }

    top_10 = recommendations.head(10).copy().reset_index(drop=True)
    top_10["rank"] = top_10.index + 1
    top_10["unit_cost_usd"] = top_10["action_recommendation"].map(
        lambda action: assumptions[action]["cost"]
    )
    top_10["expected_impact_usd"] = top_10.apply(
        lambda row: row["churn_probability"]
        * float(row["MonthlyCharges"])
        * 12.0
        * assumptions[row["action_recommendation"]]["lift"],
        axis=1,
    )
    top_10["expected_roi"] = (top_10["expected_impact_usd"] - top_10["unit_cost_usd"]) / top_10[
        "unit_cost_usd"
    ]
    top_10["decision_policy"] = policy.name

    return top_10[
        [
            "rank",
            "customerID",
            "action_recommendation",
            "unit_cost_usd",
            "expected_impact_usd",
            "expected_roi",
            "decision_policy",
        ]
    ]

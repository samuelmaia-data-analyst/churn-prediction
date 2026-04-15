from __future__ import annotations

import pandas as pd
import pytest

from src.contracts.output_contracts import validate_kpi_contract, validate_prioritization_contract


def test_validate_prioritization_contract_rejects_out_of_range_probability() -> None:
    frame = pd.DataFrame(
        [
            {
                "customerID": "C-001",
                "churn_probability": 1.2,
                "next_purchase_prediction": 100.0,
                "MonthlyCharges": 80.0,
                "Contract": "Month-to-month",
                "value_segment": "high",
                "decision_threshold": 0.65,
                "risk_segment": "high",
                "action_recommendation": "call",
                "decision_policy": "balanceada",
                "base_decision_threshold": 0.65,
            }
        ]
    )

    with pytest.raises(ValueError, match="churn_probability"):
        validate_prioritization_contract(frame)


def test_validate_kpi_contract_rejects_more_than_one_row() -> None:
    frame = pd.DataFrame(
        [
            {
                "total_customers": 100,
                "churn_rate": 0.2,
                "high_risk_customers": 10,
                "revenue_at_risk": 1200.0,
                "avg_next_purchase_prediction": 200.0,
            },
            {
                "total_customers": 100,
                "churn_rate": 0.2,
                "high_risk_customers": 10,
                "revenue_at_risk": 1200.0,
                "avg_next_purchase_prediction": 200.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="1 linha"):
        validate_kpi_contract(frame)

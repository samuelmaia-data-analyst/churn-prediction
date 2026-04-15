from __future__ import annotations

import pandas as pd

from src.pipelines.governance import build_public_prioritization_view


def test_build_public_prioritization_view_removes_identifier_in_strict_mode() -> None:
    frame = pd.DataFrame(
        [
            {
                "customerID": "C-001",
                "churn_probability": 0.8,
                "risk_segment": "high",
            }
        ]
    )

    strict_public = build_public_prioritization_view(frame, salt="abc", strict_mode=True)
    standard_public = build_public_prioritization_view(frame, salt="abc", strict_mode=False)

    assert "customer_token" in strict_public.columns
    assert "customerID" not in strict_public.columns
    assert "customerID" in standard_public.columns

from __future__ import annotations

import pandas as pd

from apps.dashboard_runtime import build_filtered_views, build_prediction_payload, summarise_metrics


def test_build_prediction_payload_returns_canonical_inference_contract() -> None:
    payload = build_prediction_payload(
        gender="Male",
        senior=0,
        partner="Yes",
        dependents="No",
        tenure=12,
        phone_service="Yes",
        internet_service="Fiber optic",
        contract="Month-to-month",
        paperless="Yes",
        payment="Electronic check",
        monthly=65.5,
        total=786.0,
    )

    assert payload["Contract"] == "Month-to-month"
    assert payload["InternetService"] == "Fiber optic"
    assert payload["MonthlyCharges"] == 65.5
    assert payload["TotalCharges"] == 786.0
    assert payload["MultipleLines"] == "No"


def test_summarise_metrics_returns_expected_dashboard_kpis() -> None:
    df = pd.DataFrame(
        {
            "Churn": ["Yes", "No", "Yes"],
            "MonthlyCharges": [10.0, 20.0, 30.0],
            "tenure": [1, 2, 3],
        }
    )

    metrics = summarise_metrics(df)

    assert metrics["total_customers"] == 3
    assert metrics["churn_rate"] > 0
    assert metrics["avg_monthly"] == 20.0
    assert metrics["avg_tenure"] == 2.0


def test_build_filtered_views_splits_chart_and_preview_contexts() -> None:
    df = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year"],
            "InternetService": ["DSL", "Fiber optic"],
        }
    )

    left_df, right_df, preview_df = build_filtered_views(
        df=df,
        selected_contract="Month-to-month",
        selected_internet="Fiber optic",
    )

    assert len(left_df) == 1
    assert len(right_df) == 1
    assert preview_df.empty

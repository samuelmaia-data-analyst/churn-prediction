from __future__ import annotations

from pathlib import Path

import pandas as pd

from apps.dashboard_runtime import (
    DashboardAssets,
    DashboardRuntime,
    build_dashboard_status,
    build_filtered_views,
    build_portfolio_summary,
    build_prediction_payload,
    build_risk_distribution,
    format_risk_level,
    load_best_available_dataframe,
    normalize_filter_value,
    simulate_retention_impact,
    summarise_metrics,
)


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


def test_build_filtered_views_accepts_all_aliases() -> None:
    df = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year"],
            "InternetService": ["DSL", "Fiber optic"],
        }
    )

    left_df, right_df, preview_df = build_filtered_views(
        df=df,
        selected_contract="All",
        selected_internet="Todos",
    )

    assert len(left_df) == 2
    assert len(right_df) == 2
    assert len(preview_df) == 2


def test_build_dashboard_status_exposes_fallback_metadata() -> None:
    assets = DashboardAssets(
        report={
            "metadata": {
                "run_id": "dashboard-fallback",
                "generated_at_utc": "2026-03-18T12:00:00Z",
                "environment": "dev",
                "schema_version": "v1",
            }
        },
        kpis=pd.DataFrame({"metric": [1]}),
        prioritization=pd.DataFrame({"customerID": ["0001"]}),
        report_path=Path("unused"),
        kpi_path=Path("unused"),
        prioritization_path=Path("unused"),
    )

    status = build_dashboard_status(assets)

    assert status["ready"] is True
    assert status["fallback"] is True
    assert status["environment"] == "dev"
    assert status["schema_version"] == "v1"
    assert status["run_id"] == "dashboard-fallback"


def test_risk_level_and_filter_helpers_normalize_runtime_values() -> None:
    assert normalize_filter_value("Todos") == "All"
    assert normalize_filter_value("All") == "All"
    assert format_risk_level("Alto") == "High"
    assert format_risk_level("medium") == "Medium"


def test_build_portfolio_summary_aggregates_high_risk_and_revenue() -> None:
    df = pd.DataFrame(
        {
            "risk_segment": ["high", "medium", "high"],
            "MonthlyCharges": [100.0, 20.0, 50.0],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "next_purchase_prediction": [120.0, 80.0, 90.0],
        }
    )

    summary = build_portfolio_summary(df)

    assert summary["high_risk_customers"] == 2
    assert summary["high_risk_revenue"] == 150.0
    assert summary["month_to_month_share"] > 0


def test_build_risk_distribution_preserves_expected_order() -> None:
    df = pd.DataFrame({"risk_segment": ["low", "high", "medium", "high"]})

    distribution = build_risk_distribution(df)

    assert distribution["risk_segment"].tolist() == ["high", "medium", "low"]
    assert distribution["customers"].tolist() == [2, 1, 1]


def test_simulate_retention_impact_returns_recovered_and_remaining_values() -> None:
    df = pd.DataFrame(
        {
            "churn_probability": [0.9, 0.8, 0.5],
            "MonthlyCharges": [100.0, 50.0, 30.0],
        }
    )

    simulation = simulate_retention_impact(df, retention_effectiveness=20)

    assert simulation["baseline_revenue_risk"] == 150.0
    assert simulation["recovered_revenue"] == 30.0
    assert simulation["remaining_revenue_risk"] == 120.0


def test_load_best_available_dataframe_falls_back_to_prioritization_when_files_missing() -> None:
    assets = DashboardAssets(
        report={"metadata": {"run_id": "dashboard-fallback"}},
        kpis=pd.DataFrame({"metric": [1]}),
        prioritization=pd.DataFrame({"customerID": ["C-001"], "MonthlyCharges": [50.0]}),
        report_path=Path("unused"),
        kpi_path=Path("unused"),
        prioritization_path=Path("unused"),
    )
    runtime = DashboardRuntime(
        data_path=Path("missing_raw.csv"),
        silver_path=Path("missing_silver.csv"),
        bundle_path=Path("unused"),
    )

    dataframe, source = load_best_available_dataframe(runtime, assets)

    assert dataframe is not None
    assert len(dataframe) == 1
    assert source == "prioritization:fallback"

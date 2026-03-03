from __future__ import annotations

import importlib

import pandas as pd

import src.dashboard_data as dashboard_data


def test_ensure_dashboard_outputs_with_synthetic_fallback(tmp_path, monkeypatch) -> None:
    module = importlib.reload(dashboard_data)

    monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "reports" / "executive_report.json")
    monkeypatch.setattr(
        module, "PRIORITIZATION_PATH", tmp_path / "data" / "gold" / "customer_prioritization.csv"
    )
    monkeypatch.setattr(module, "KPI_PATH", tmp_path / "data" / "gold" / "kpi_summary.csv")
    monkeypatch.setattr(module, "RAW_PATH", tmp_path / "data" / "raw" / "missing.csv")

    module.ensure_dashboard_outputs()

    assert module.REPORT_PATH.exists()
    assert module.PRIORITIZATION_PATH.exists()
    assert module.KPI_PATH.exists()

    prioritization = pd.read_csv(module.PRIORITIZATION_PATH)
    assert len(prioritization) > 0
    assert {"churn_probability", "next_purchase_prediction", "action_recommendation"}.issubset(
        prioritization.columns
    )

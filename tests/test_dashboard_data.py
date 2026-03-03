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


def test_ensure_dashboard_outputs_prefers_pipeline_when_raw_exists(tmp_path, monkeypatch) -> None:
    module = importlib.reload(dashboard_data)

    monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "reports" / "executive_report.json")
    monkeypatch.setattr(
        module, "PRIORITIZATION_PATH", tmp_path / "data" / "gold" / "customer_prioritization.csv"
    )
    monkeypatch.setattr(module, "KPI_PATH", tmp_path / "data" / "gold" / "kpi_summary.csv")
    monkeypatch.setattr(module, "RAW_PATH", tmp_path / "data" / "raw" / "dataset.csv")

    module.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    module.RAW_PATH.write_text("dummy", encoding="utf-8")

    called = {"pipeline": 0, "fallback": 0}

    def fake_pipeline() -> bool:
        called["pipeline"] += 1
        module.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        module.PRIORITIZATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        module.KPI_PATH.parent.mkdir(parents=True, exist_ok=True)
        module.REPORT_PATH.write_text("{}", encoding="utf-8")
        pd.DataFrame([{"x": 1}]).to_csv(module.PRIORITIZATION_PATH, index=False)
        pd.DataFrame([{"x": 1}]).to_csv(module.KPI_PATH, index=False)
        return True

    def fake_fallback(_df) -> None:
        called["fallback"] += 1

    monkeypatch.setattr(module, "_generate_outputs_from_pipeline", fake_pipeline)
    monkeypatch.setattr(module, "_generate_outputs_from_raw_or_synthetic", fake_fallback)

    module.ensure_dashboard_outputs()

    assert called["pipeline"] == 1
    assert called["fallback"] == 0

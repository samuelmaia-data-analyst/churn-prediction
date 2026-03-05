from __future__ import annotations

import pandas as pd

from src.config import PipelineConfig
from src.monitoring import run_drift_monitoring


def test_drift_monitoring_creates_baseline_and_alert_file(tmp_path) -> None:
    config = PipelineConfig(data_dir=tmp_path / "data", seed=42, log_level="INFO")
    baseline = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5],
            "MonthlyCharges": [20, 30, 40, 50, 60],
            "TotalCharges": [20, 60, 120, 200, 300],
            "churn_probability": [0.1, 0.2, 0.3, 0.2, 0.1],
        }
    )

    result = run_drift_monitoring(config, baseline)

    assert result["status"] == "cold_start"
    assert config.drift_reference_path.exists()
    assert config.drift_alert_path.exists()


def test_drift_monitoring_alerts_on_large_shift(tmp_path) -> None:
    config = PipelineConfig(data_dir=tmp_path / "data", seed=42, log_level="INFO")
    baseline = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5] * 20,
            "MonthlyCharges": [20, 30, 40, 50, 60] * 20,
            "TotalCharges": [20, 60, 120, 200, 300] * 20,
            "churn_probability": [0.1, 0.2, 0.3, 0.2, 0.1] * 20,
        }
    )
    current = pd.DataFrame(
        {
            "tenure": [24, 36, 48, 60, 72] * 20,
            "MonthlyCharges": [110, 115, 120, 118, 119] * 20,
            "TotalCharges": [500, 1000, 1500, 2000, 2500] * 20,
            "churn_probability": [0.8, 0.85, 0.9, 0.88, 0.87] * 20,
        }
    )

    run_drift_monitoring(config, baseline)
    result = run_drift_monitoring(config, current)

    assert result["status"] == "alert"
    assert result["alert"] is True
    assert any(feature["drift"] for feature in result["features"])

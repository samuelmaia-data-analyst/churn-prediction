from __future__ import annotations

import pandas as pd

from monitoring.drift_detection import run_detection


def test_run_detection_returns_alert_on_shift() -> None:
    baseline = pd.DataFrame({"x": [1, 2, 3, 4, 5] * 20})
    current = pd.DataFrame({"x": [50, 55, 60, 65, 70] * 20})

    result = run_detection(baseline, current)

    assert result["status"] == "alert"
    assert result["alert"] is True
    assert len(result["features"]) == 1

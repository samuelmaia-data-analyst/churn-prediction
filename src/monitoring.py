from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import PipelineConfig

DEFAULT_DRIFT_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "churn_probability",
]


def _to_numeric_series(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return numeric.to_numpy(dtype=float)


def population_stability_index(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    baseline_values = _to_numeric_series(baseline)
    current_values = _to_numeric_series(current)
    if len(baseline_values) == 0 or len(current_values) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(baseline_values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    baseline_hist, _ = np.histogram(baseline_values, bins=edges)
    current_hist, _ = np.histogram(current_values, bins=edges)
    baseline_pct = np.clip(baseline_hist / max(baseline_hist.sum(), 1), 1e-6, None)
    current_pct = np.clip(current_hist / max(current_hist.sum(), 1), 1e-6, None)
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def ks_statistic(baseline: pd.Series, current: pd.Series) -> float:
    baseline_values = np.sort(_to_numeric_series(baseline))
    current_values = np.sort(_to_numeric_series(current))
    if len(baseline_values) == 0 or len(current_values) == 0:
        return 0.0

    values = np.sort(np.concatenate([baseline_values, current_values]))
    baseline_cdf = np.searchsorted(baseline_values, values, side="right") / len(baseline_values)
    current_cdf = np.searchsorted(current_values, values, side="right") / len(current_values)
    return float(np.max(np.abs(baseline_cdf - current_cdf)))


def run_drift_monitoring(config: PipelineConfig, current_df: pd.DataFrame) -> dict[str, object]:
    config.monitoring_dir.mkdir(parents=True, exist_ok=True)

    available_features = [f for f in DEFAULT_DRIFT_FEATURES if f in current_df.columns]
    current_snapshot = current_df[available_features].copy()
    status = "ok"
    alert = False
    per_feature: list[dict[str, object]] = []

    if not config.drift_reference_path.exists():
        current_snapshot.to_csv(config.drift_reference_path, index=False)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "cold_start",
            "alert": False,
            "reason": "Baseline de drift criada na primeira execucao.",
            "features": [],
        }
        with open(config.drift_alert_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        return payload

    baseline = pd.read_csv(config.drift_reference_path)
    for feature in available_features:
        if feature not in baseline.columns:
            continue
        psi = population_stability_index(baseline[feature], current_snapshot[feature])
        ks = ks_statistic(baseline[feature], current_snapshot[feature])
        drift_flag = psi >= 0.20 or ks >= 0.15
        alert = alert or drift_flag
        per_feature.append(
            {
                "feature": feature,
                "psi": round(float(psi), 4),
                "ks": round(float(ks), 4),
                "drift": drift_flag,
            }
        )

    if alert:
        status = "alert"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "alert": alert,
        "thresholds": {"psi_alert": 0.20, "ks_alert": 0.15},
        "features": per_feature,
    }
    with open(config.drift_alert_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload

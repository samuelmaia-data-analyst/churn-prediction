from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.monitoring import ks_statistic, population_stability_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect model drift with PSI and KS.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline CSV path.")
    parser.add_argument("--current", type=Path, required=True, help="Current CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/drift_alert.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def run_detection(baseline: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    common = sorted(set(baseline.columns).intersection(set(current.columns)))
    results: list[dict[str, object]] = []
    alert = False

    for feature in common:
        psi = population_stability_index(baseline[feature], current[feature])
        ks = ks_statistic(baseline[feature], current[feature])
        drift = psi >= 0.20 or ks >= 0.15
        alert = alert or drift
        results.append(
            {
                "feature": feature,
                "psi": round(float(psi), 4),
                "ks": round(float(ks), 4),
                "drift": drift,
            }
        )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "alert" if alert else "ok",
        "alert": alert,
        "thresholds": {"psi_alert": 0.20, "ks_alert": 0.15},
        "features": results,
    }


def main() -> None:
    args = parse_args()
    baseline_df = pd.read_csv(args.baseline)
    current_df = pd.read_csv(args.current)
    payload = run_detection(baseline_df, current_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"Drift status: {payload['status']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

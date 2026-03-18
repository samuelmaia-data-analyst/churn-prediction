from __future__ import annotations

import json
from pathlib import Path

from src.cli.pipeline import run_pipeline
from src.config import PipelineConfig


def test_run_pipeline_persists_execution_metadata_and_data_quality(tmp_path: Path) -> None:
    project_data_dir = tmp_path / "data"
    raw_dir = project_data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw_dir.joinpath(source_dataset.name).write_bytes(source_dataset.read_bytes())

    summary = run_pipeline(
        data_dir=str(project_data_dir),
        log_level="INFO",
        mlflow_tracking_uri="disabled",
        environment="test",
    )
    config = PipelineConfig.from_runtime(
        data_dir=project_data_dir,
        log_level="INFO",
        mlflow_tracking_uri="disabled",
        environment="test",
        run_id=str(summary["run_id"]),
    )

    execution_payload = json.loads(config.execution_metadata_path.read_text(encoding="utf-8"))
    quality_payload = json.loads(config.data_quality_report_path.read_text(encoding="utf-8"))

    assert execution_payload["run_id"] == summary["run_id"]
    assert execution_payload["environment"] == "test"
    assert execution_payload["drift_status"] in {"ok", "cold_start", "alert"}
    assert execution_payload["metrics"]["churn_roc_auc"] >= 0.0
    assert quality_payload["rows"] > 0
    assert quality_payload["invalid_churn_labels"] == 0

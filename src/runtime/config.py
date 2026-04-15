from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


def _load_dotenv(dotenv_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not dotenv_path.exists():
        return values

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    seed: int
    log_level: str
    mlflow_tracking_uri: str = "file:./mlruns"
    decision_policy: str = "balanceada"
    environment: str = "dev"
    run_id: str = "local"
    artifacts_dir: Path | None = None
    model_registry_dir: Path | None = None
    raw_filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    test_size: float = 0.2

    @classmethod
    def from_runtime(
        cls,
        data_dir: Path | None = None,
        seed: int = 42,
        log_level: str = "INFO",
        mlflow_tracking_uri: str | None = None,
        decision_policy: str | None = None,
        environment: str | None = None,
        run_id: str | None = None,
        dotenv_path: Path = Path(".env"),
    ) -> "PipelineConfig":
        dotenv_values = _load_dotenv(dotenv_path)
        default_data_dir = data_dir or Path("data")

        def _setting(name: str, default: str) -> str:
            return os.getenv(name, dotenv_values.get(name, default))

        return cls(
            data_dir=Path(_setting("CHURN_DATA_DIR", str(data_dir or "data"))),
            seed=int(_setting("CHURN_SEED", str(seed))),
            log_level=_setting("CHURN_LOG_LEVEL", log_level).upper(),
            mlflow_tracking_uri=_setting(
                "CHURN_MLFLOW_TRACKING_URI", mlflow_tracking_uri or "file:./mlruns"
            ),
            decision_policy=_setting("CHURN_DECISION_POLICY", decision_policy or "balanceada"),
            environment=_setting("CHURN_ENV", environment or "dev"),
            run_id=_setting("CHURN_RUN_ID", run_id or str(uuid4())),
            artifacts_dir=Path(
                _setting("CHURN_ARTIFACTS_DIR", str(default_data_dir.parent / "artifacts"))
            ),
            model_registry_dir=Path(
                _setting("CHURN_MODEL_REGISTRY_DIR", str(default_data_dir.parent / "models"))
            ),
        )

    @property
    def resolved_artifacts_dir(self) -> Path:
        return self.artifacts_dir or (self.data_dir.parent / "artifacts")

    @property
    def resolved_model_registry_dir(self) -> Path:
        return self.model_registry_dir or (self.data_dir.parent / "models")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def reports_dir(self) -> Path:
        return self.resolved_artifacts_dir / "reports"

    @property
    def models_dir(self) -> Path:
        return self.resolved_artifacts_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.resolved_artifacts_dir / "logs"

    @property
    def monitoring_dir(self) -> Path:
        return self.resolved_artifacts_dir / "monitoring"

    @property
    def metadata_dir(self) -> Path:
        return self.resolved_artifacts_dir / "metadata"

    @property
    def raw_input_path(self) -> Path:
        return self.raw_dir / self.raw_filename

    @property
    def bronze_output_path(self) -> Path:
        return self.bronze_dir / "customer_churn_bronze.csv"

    @property
    def silver_output_path(self) -> Path:
        return self.silver_dir / "customer_churn_silver.csv"

    @property
    def executive_report_path(self) -> Path:
        return self.reports_dir / "executive_report.json"

    @property
    def data_quality_report_path(self) -> Path:
        return self.reports_dir / "data_quality_report.json"

    @property
    def model_card_path(self) -> Path:
        return self.reports_dir / "model_card.md"

    @property
    def executive_brief_path(self) -> Path:
        return self.reports_dir / "executive_brief.md"

    @property
    def action_playbook_path(self) -> Path:
        return self.reports_dir / "action_playbook.md"

    @property
    def gold_manifest_path(self) -> Path:
        return self.gold_dir / "_manifest.json"

    @property
    def churn_model_path(self) -> Path:
        return self.models_dir / "enterprise_churn_model.joblib"

    @property
    def next_purchase_model_path(self) -> Path:
        return self.models_dir / "enterprise_next_purchase_model.joblib"

    @property
    def enterprise_bundle_path(self) -> Path:
        return self.models_dir / "enterprise_churn_bundle.joblib"

    @property
    def versioned_model_path(self) -> Path:
        return self.resolved_model_registry_dir / "model_v1.pkl"

    @property
    def model_metadata_path(self) -> Path:
        return self.resolved_model_registry_dir / "model_metadata.json"

    @property
    def model_registry_manifest_path(self) -> Path:
        return self.resolved_model_registry_dir / "registry_manifest.json"

    @property
    def execution_metadata_path(self) -> Path:
        return self.metadata_dir / f"pipeline_run_{self.run_id}.json"

    @property
    def latest_execution_metadata_path(self) -> Path:
        return self.metadata_dir / "latest_run.json"

    @property
    def lineage_manifest_path(self) -> Path:
        return self.metadata_dir / f"lineage_run_{self.run_id}.json"

    @property
    def latest_lineage_manifest_path(self) -> Path:
        return self.metadata_dir / "latest_lineage.json"

    @property
    def drift_reference_path(self) -> Path:
        return self.monitoring_dir / "drift_reference.csv"

    @property
    def drift_alert_path(self) -> Path:
        return self.monitoring_dir / "drift_alert.json"

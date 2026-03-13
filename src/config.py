from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    seed: int
    log_level: str
    mlflow_tracking_uri: str = "file:./mlruns"
    decision_policy: str = "balanceada"

    raw_filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    test_size: float = 0.2

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
        return Path("artifacts/reports")

    @property
    def models_dir(self) -> Path:
        return Path("artifacts/models")

    @property
    def model_registry_dir(self) -> Path:
        return Path("models")

    @property
    def logs_dir(self) -> Path:
        return Path("artifacts/logs")

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
    def model_card_path(self) -> Path:
        return self.reports_dir / "model_card.md"

    @property
    def executive_brief_path(self) -> Path:
        return self.reports_dir / "executive_brief.md"

    @property
    def action_playbook_path(self) -> Path:
        return self.reports_dir / "action_playbook.md"

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
        return self.model_registry_dir / "model_v1.pkl"

    @property
    def model_metadata_path(self) -> Path:
        return self.model_registry_dir / "model_metadata.json"

    @property
    def monitoring_dir(self) -> Path:
        return self.data_dir.parent / "reports"

    @property
    def drift_reference_path(self) -> Path:
        return self.monitoring_dir / "drift_reference.csv"

    @property
    def drift_alert_path(self) -> Path:
        return self.monitoring_dir / "drift_alert.json"

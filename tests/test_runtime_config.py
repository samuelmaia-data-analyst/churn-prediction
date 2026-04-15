from __future__ import annotations

from pathlib import Path

from src.runtime.config import PipelineConfig


def test_pipeline_config_reads_environment_from_dotenv(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "CHURN_DATA_DIR=custom-data",
                "CHURN_ARTIFACTS_DIR=custom-artifacts",
                "CHURN_MODEL_REGISTRY_DIR=custom-models",
                "CHURN_ENV=prod",
                "CHURN_DECISION_POLICY=campanha_cara",
                "CHURN_MLFLOW_TRACKING_URI=disabled",
                "CHURN_LGPD_MODE=strict",
                "CHURN_GOV_RETENTION_DAYS=180",
                "CHURN_LGPD_SALT=top-secret",
            ]
        ),
        encoding="utf-8",
    )

    config = PipelineConfig.from_runtime(dotenv_path=dotenv_path, run_id="test-run")

    assert config.data_dir == Path("custom-data")
    assert config.artifacts_dir == Path("custom-artifacts")
    assert config.model_registry_dir == Path("custom-models")
    assert config.environment == "prod"
    assert config.decision_policy == "campanha_cara"
    assert config.mlflow_tracking_uri == "disabled"
    assert config.lgpd_mode == "strict"
    assert config.governance_retention_days == 180
    assert config.lgpd_salt == "top-secret"
    assert config.execution_metadata_path == Path(
        "custom-artifacts/metadata/pipeline_run_test-run.json"
    )
    assert config.lineage_manifest_path == Path(
        "custom-artifacts/metadata/lineage_run_test-run.json"
    )
    assert config.governance_manifest_path == Path(
        "custom-artifacts/metadata/governance_run_test-run.json"
    )

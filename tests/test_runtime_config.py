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
    assert config.execution_metadata_path == Path(
        "custom-artifacts/metadata/pipeline_run_test-run.json"
    )
    assert config.lineage_manifest_path == Path(
        "custom-artifacts/metadata/lineage_run_test-run.json"
    )

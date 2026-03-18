from __future__ import annotations

from src.config import PipelineConfig
from src.ingestion import build_bronze_layer
from src.modeling.pipeline import train_models_and_score
from src.modeling.predictor import ChurnPredictor
from src.transformation import build_silver_layer
from tests.test_data import build_raw_df


def test_training_outputs_business_metrics_and_persisted_artifacts(tmp_path) -> None:
    config = PipelineConfig(data_dir=tmp_path / "data", seed=42, log_level="INFO")
    silver = build_silver_layer(build_bronze_layer(build_raw_df(120)))

    outputs = train_models_and_score(config, silver)

    assert 0.0 <= outputs.metrics["churn_precision"] <= 1.0
    assert 0.0 <= outputs.metrics["churn_recall"] <= 1.0
    assert 0.0 <= outputs.metrics["churn_f1"] <= 1.0
    assert 0.0 <= outputs.metrics["churn_roc_auc"] <= 1.0
    assert "confusion_matrix" in outputs.metrics
    assert "risk_profiles" in outputs.metrics
    assert config.enterprise_bundle_path.exists()
    assert config.model_metadata_path.exists()
    assert config.model_registry_manifest_path.exists()


def test_predictor_uses_bundle_generated_by_training_pipeline(tmp_path) -> None:
    config = PipelineConfig(data_dir=tmp_path / "data", seed=42, log_level="INFO")
    silver = build_silver_layer(build_bronze_layer(build_raw_df(120)))
    train_models_and_score(config, silver)

    predictor = ChurnPredictor(bundle_path=config.enterprise_bundle_path)
    payload = build_raw_df(5).drop(columns=["Churn", "customerID"]).iloc[0].to_dict()
    result = predictor.predict_from_dict(payload)

    assert result.churn in {"Sim", "Não"}
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_level in {"Baixo", "Médio", "Alto"}

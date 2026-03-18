from __future__ import annotations

from src.ml import train_models_and_score
from src.pipelines.ingestion import build_bronze_layer
from src.pipelines.reporting import build_business_outputs
from src.pipelines.transformation import build_silver_layer
from src.runtime.config import PipelineConfig
from tests.test_enterprise_pipeline import build_dataset


def test_executive_report_contract_is_stable(tmp_path) -> None:
    config = PipelineConfig(data_dir=tmp_path / "data", seed=42, log_level="INFO")
    raw = build_dataset(120)
    bronze = build_bronze_layer(raw)
    silver = build_silver_layer(bronze)
    model_outputs = train_models_and_score(config, silver)
    report_outputs = build_business_outputs(config, model_outputs.scored_df, model_outputs.metrics)

    payload = report_outputs.executive_report.to_dict()

    assert set(payload.keys()) == {"metadata", "kpis", "model_metrics", "top_10_priorities"}
    assert set(payload["metadata"].keys()) == {
        "schema_version",
        "generated_at_utc",
        "run_id",
        "environment",
    }
    assert set(payload["kpis"].keys()) == {
        "total_customers",
        "churn_rate",
        "high_risk_customers",
        "revenue_at_risk",
        "avg_next_purchase_prediction",
    }
    assert len(payload["top_10_priorities"]) <= 10

from __future__ import annotations

from tests.test_enterprise_pipeline import build_dataset

from src.config import PipelineConfig
from src.ingestion import build_bronze_layer
from src.ml import train_models_and_score
from src.reporting import build_business_outputs
from src.transformation import build_silver_layer


def test_executive_report_contract_is_stable(tmp_path) -> None:
    config = PipelineConfig(data_dir=tmp_path / "data", seed=42, log_level="INFO")
    raw = build_dataset(120)
    bronze = build_bronze_layer(raw)
    silver = build_silver_layer(bronze)
    model_outputs = train_models_and_score(config, silver)
    report_outputs = build_business_outputs(model_outputs.scored_df, model_outputs.metrics)

    payload = report_outputs.executive_report.to_dict()

    assert set(payload.keys()) == {"kpis", "model_metrics", "top_10_priorities"}
    assert set(payload["kpis"].keys()) == {
        "total_customers",
        "churn_rate",
        "high_risk_customers",
        "revenue_at_risk",
        "avg_next_purchase_prediction",
    }
    assert len(payload["top_10_priorities"]) <= 10

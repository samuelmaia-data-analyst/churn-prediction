from __future__ import annotations

from src.ml import train_models_and_score
from src.pipelines.eda import build_eda_profile
from src.pipelines.governance import build_governance_manifest, build_public_prioritization_view
from src.pipelines.ingestion import build_bronze_layer
from src.pipelines.reporting import build_business_outputs
from src.pipelines.transformation import build_silver_layer
from src.runtime.config import PipelineConfig
from tests.test_enterprise_pipeline import build_dataset


def test_eda_profile_and_governance_manifest_have_expected_keys(tmp_path) -> None:
    config = PipelineConfig(
        data_dir=tmp_path / "data",
        seed=42,
        log_level="INFO",
        lgpd_mode="strict",
    )
    silver = build_silver_layer(build_bronze_layer(build_dataset(120)))
    profile = build_eda_profile(silver)
    model_outputs = train_models_and_score(config, silver)
    report_outputs = build_business_outputs(config, model_outputs.scored_df, model_outputs.metrics)

    governance = build_governance_manifest(config, silver, report_outputs.recommendations)
    public = build_public_prioritization_view(
        report_outputs.recommendations, salt="salt", strict_mode=True
    )

    assert {"rows", "columns", "numeric_summary", "top_categories"}.issubset(profile)
    assert governance["framework"] == "LGPD"
    assert governance["controls"]["strict_mode_identifier_removed"] is True
    assert "customerID" not in public.columns

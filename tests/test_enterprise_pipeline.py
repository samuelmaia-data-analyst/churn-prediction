from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.ingestion import build_bronze_layer
from src.ml import train_models_and_score
from src.reporting import build_business_outputs, persist_business_outputs
from src.transformation import build_silver_layer
from src.warehouse import build_star_schema


def build_dataset(rows: int = 120) -> pd.DataFrame:
    records = []
    for i in range(rows):
        records.append(
            {
                "customerID": f"CUST-{i:04d}",
                "gender": "Male" if i % 2 == 0 else "Female",
                "SeniorCitizen": i % 2,
                "Partner": "Yes" if i % 3 else "No",
                "Dependents": "No" if i % 4 else "Yes",
                "tenure": (i % 60) + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if i % 2 == 0 else "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes" if i % 2 == 0 else "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if i % 2 == 0 else "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check" if i % 3 == 0 else "Mailed check",
                "MonthlyCharges": 35 + i * 0.3,
                "TotalCharges": " " if i == 0 else str((35 + i * 0.3) * ((i % 60) + 1)),
                "Churn": "Yes" if i % 4 == 0 else "No",
            }
        )
    return pd.DataFrame(records)


def test_layer_contracts_and_star_schema() -> None:
    raw = build_dataset(60)
    bronze = build_bronze_layer(raw)
    silver = build_silver_layer(bronze)
    schema = build_star_schema(silver)

    assert {"ingested_at", "source_system"}.issubset(bronze.columns)
    assert silver["TotalCharges"].dtype.kind in {"f", "i"}
    assert silver["Churn"].isin([0, 1]).all()
    assert schema.fact_customer_churn["customer_sk"].notna().all()
    assert schema.fact_customer_churn["contract_sk"].notna().all()
    assert schema.fact_customer_churn["service_sk"].notna().all()


def test_ml_outputs_and_executive_report_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = PipelineConfig(data_dir=data_dir, seed=42, log_level="INFO")
    dataset = build_dataset(120)

    bronze = build_bronze_layer(dataset)
    silver = build_silver_layer(bronze)
    model_outputs = train_models_and_score(config, silver)
    report_outputs = build_business_outputs(model_outputs.scored_df, model_outputs.metrics)
    persist_business_outputs(config, report_outputs)

    assert {"churn_probability", "next_purchase_prediction"}.issubset(
        model_outputs.scored_df.columns
    )
    assert 0.0 <= model_outputs.metrics["churn_f1"] <= 1.0
    assert 0.0 <= model_outputs.metrics["churn_roc_auc"] <= 1.0
    assert model_outputs.metrics["next_purchase_mae"] >= 0.0
    assert "baseline_model" in model_outputs.metrics
    assert "model_comparison" in model_outputs.metrics
    assert "top_drivers_of_churn" in model_outputs.metrics
    assert "key_insights" in model_outputs.metrics
    assert model_outputs.metrics["pipeline_visual"] == "Raw -> Bronze -> Silver -> Gold"
    assert len(model_outputs.metrics["model_comparison"]) == 3

    assert config.executive_report_path.exists()
    with open(config.executive_report_path, "r", encoding="utf-8") as fp:
        report = json.load(fp)
    assert "kpis" in report
    assert "model_metrics" in report
    assert "top_10_priorities" in report
    assert (config.gold_dir / "customer_prioritization.csv").exists()
    assert (config.gold_dir / "kpi_summary.csv").exists()

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd

try:
    from prefect import flow, task
except ImportError:  # pragma: no cover

    def task(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def flow(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


from src.config import PipelineConfig
from src.ingestion import build_bronze_layer, load_raw_dataset, persist_bronze
from src.logging_utils import configure_logging
from src.ml import ModelOutputs, train_models_and_score
from src.monitoring import run_drift_monitoring
from src.reporting import ReportOutputs, build_business_outputs, persist_business_outputs
from src.transformation import build_silver_layer, persist_silver
from src.warehouse import StarSchema, build_star_schema, persist_star_schema

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline enterprise em camadas (raw -> bronze -> silver -> gold)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Diretório base de dados"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de log",
    )
    return parser.parse_args()


@task(retries=2, retry_delay_seconds=3)
def bronze_task(config: PipelineConfig) -> pd.DataFrame:
    raw_df = load_raw_dataset(config)
    bronze_df = build_bronze_layer(raw_df)
    persist_bronze(config, bronze_df)
    return bronze_df


@task(retries=2, retry_delay_seconds=3)
def silver_task(config: PipelineConfig, bronze_df: pd.DataFrame) -> pd.DataFrame:
    silver_df = build_silver_layer(bronze_df)
    persist_silver(config, silver_df)
    return silver_df


@task(retries=1, retry_delay_seconds=2)
def warehouse_task(config: PipelineConfig, silver_df: pd.DataFrame) -> StarSchema:
    schema = build_star_schema(silver_df)
    persist_star_schema(config, schema)
    return schema


@task(retries=1, retry_delay_seconds=2)
def ml_task(config: PipelineConfig, silver_df: pd.DataFrame) -> ModelOutputs:
    return train_models_and_score(config, silver_df)


@task(retries=1, retry_delay_seconds=2)
def reporting_task(config: PipelineConfig, model_outputs: ModelOutputs) -> ReportOutputs:
    outputs = build_business_outputs(model_outputs.scored_df, model_outputs.metrics)
    persist_business_outputs(config, outputs)
    return outputs


@task(retries=1, retry_delay_seconds=2)
def monitoring_task(config: PipelineConfig, model_outputs: ModelOutputs) -> dict[str, object]:
    return run_drift_monitoring(config, model_outputs.scored_df)


@flow(name="enterprise-churn-pipeline")
def run_pipeline(
    seed: int = 42,
    data_dir: str = "data",
    log_level: str = "INFO",
    mlflow_tracking_uri: str = "file:./mlruns",
) -> None:
    run_id = str(uuid4())
    config = PipelineConfig(
        data_dir=Path(data_dir),
        seed=seed,
        log_level=log_level,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    configure_logging(level=config.log_level, log_dir=config.logs_dir, run_id=run_id)

    started_at = time.perf_counter()
    logger.info(
        "pipeline_start run_id=%s seed=%s data_dir=%s",
        run_id,
        config.seed,
        config.data_dir,
    )

    bronze_df = bronze_task(config)
    silver_df = silver_task(config, bronze_df)
    warehouse_task(config, silver_df)
    model_outputs = ml_task(config, silver_df)
    reporting_task(config, model_outputs)
    drift_result = monitoring_task(config, model_outputs)

    elapsed_seconds = time.perf_counter() - started_at
    logger.info(
        (
            "pipeline_done run_id=%s duration_seconds=%.2f churn_f1=%.4f "
            "churn_auc=%.4f drift_status=%s"
        ),
        run_id,
        elapsed_seconds,
        model_outputs.metrics["churn_f1"],
        model_outputs.metrics["churn_roc_auc"],
        drift_result.get("status", "unknown"),
    )


def main() -> None:
    args = parse_args()
    run_pipeline(
        seed=args.seed,
        data_dir=str(args.data_dir),
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

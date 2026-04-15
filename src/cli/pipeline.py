from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import pandas as pd

from src.ml import ModelOutputs, train_models_and_score
from src.pipelines.decisioning import POLICIES
from src.pipelines.ingestion import build_bronze_layer, load_raw_dataset, persist_bronze
from src.pipelines.monitoring import run_drift_monitoring
from src.pipelines.reporting import ReportOutputs, build_business_outputs, persist_business_outputs
from src.pipelines.transformation import build_silver_layer, persist_silver
from src.pipelines.warehouse import StarSchema, build_star_schema, persist_star_schema
from src.runtime.config import PipelineConfig
from src.runtime.logging import configure_logging
from src.utils.io import file_sha256, write_json_atomic

logger = logging.getLogger(__name__)
TaskResult = TypeVar("TaskResult")
NON_RETRYABLE_EXCEPTIONS = (FileNotFoundError, ValueError)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline enterprise em camadas (raw -> bronze -> silver -> gold)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Diretorio base de dados"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de log",
    )
    parser.add_argument(
        "--decision-policy",
        type=str,
        default="balanceada",
        choices=sorted(POLICIES.keys()),
        help="Politica de custo usada para definir o threshold global do classificador.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        help="Ambiente de execucao usado para particionar configuracoes e observabilidade.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="file:./mlruns",
        help="Tracking URI do MLflow. Use 'disabled' para desabilitar logging de experimento.",
    )
    return parser.parse_args()


def execute_with_retry(
    task_name: str,
    fn: Callable[..., TaskResult],
    *args: object,
    retries: int,
    retry_delay_seconds: int,
    retry_backoff_multiplier: float = 1.5,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    **kwargs: object,
) -> TaskResult:
    retryable_exceptions = retryable_exceptions or (Exception,)
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            logger.info("task_start name=%s attempt=%s", task_name, attempt)
            result = fn(*args, **kwargs)
            logger.info("task_done name=%s attempt=%s", task_name, attempt)
            return result
        except Exception as exc:
            retryable = isinstance(exc, retryable_exceptions) and not isinstance(
                exc, NON_RETRYABLE_EXCEPTIONS
            )
            logger.exception(
                "task_failed name=%s attempt=%s retryable=%s",
                task_name,
                attempt,
                retryable,
            )
            if not retryable or attempt >= attempts:
                raise
            wait_seconds = retry_delay_seconds * (retry_backoff_multiplier ** (attempt - 1))
            logger.warning(
                "task_retry_wait name=%s attempt=%s wait_seconds=%.2f",
                task_name,
                attempt,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Unexpected retry exhaustion for task {task_name}")


def bronze_task(config: PipelineConfig) -> pd.DataFrame:
    raw_df = load_raw_dataset(config)
    bronze_df = build_bronze_layer(raw_df)
    persist_bronze(config, bronze_df)
    return bronze_df


def silver_task(config: PipelineConfig, bronze_df: pd.DataFrame) -> pd.DataFrame:
    silver_df = build_silver_layer(bronze_df)
    persist_silver(config, silver_df)
    return silver_df


def warehouse_task(config: PipelineConfig, silver_df: pd.DataFrame) -> StarSchema:
    schema = build_star_schema(silver_df)
    persist_star_schema(config, schema)
    return schema


def ml_task(config: PipelineConfig, silver_df: pd.DataFrame) -> ModelOutputs:
    return train_models_and_score(config, silver_df)


def reporting_task(config: PipelineConfig, model_outputs: ModelOutputs) -> ReportOutputs:
    outputs = build_business_outputs(config, model_outputs.scored_df, model_outputs.metrics)
    persist_business_outputs(config, outputs)
    return outputs


def monitoring_task(config: PipelineConfig, model_outputs: ModelOutputs) -> dict[str, object]:
    return run_drift_monitoring(config, model_outputs.scored_df)


def _summarize_stage_result(result: object) -> dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        return {"rows": int(result.shape[0]), "columns": int(result.shape[1])}
    if isinstance(result, StarSchema):
        return {
            "dim_customer_rows": int(len(result.dim_customer)),
            "dim_contract_rows": int(len(result.dim_contract)),
            "dim_service_rows": int(len(result.dim_service)),
            "fact_rows": int(len(result.fact_customer_churn)),
        }
    if isinstance(result, ModelOutputs):
        metrics = result.metrics
        return {
            "scored_rows": int(len(result.scored_df)),
            "churn_f1": float(metrics.get("churn_f1", 0.0)),
            "churn_roc_auc": float(metrics.get("churn_roc_auc", 0.0)),
        }
    if isinstance(result, ReportOutputs):
        return {
            "recommendation_rows": int(len(result.recommendations)),
            "kpi_rows": int(len(result.kpi_summary)),
            "playbook_rows": int(len(result.action_playbook)),
        }
    if isinstance(result, dict):
        summary: dict[str, Any] = {}
        for key in ("status", "alert", "reason"):
            if key in result:
                summary[key] = result[key]
        if "features" in result and isinstance(result["features"], list):
            summary["features_count"] = len(result["features"])
        return summary
    return {"result_type": type(result).__name__}


def _execute_stage(
    stage_name: str,
    fn: Callable[..., TaskResult],
    *args: object,
    retries: int,
    retry_delay_seconds: int,
    **kwargs: object,
) -> tuple[TaskResult, dict[str, Any]]:
    started_at = time.perf_counter()
    result = execute_with_retry(
        stage_name,
        fn,
        *args,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        **kwargs,
    )
    duration_seconds = round(time.perf_counter() - started_at, 3)
    metadata = {"duration_seconds": duration_seconds}
    metadata.update(_summarize_stage_result(result))
    return result, metadata


def run_pipeline(
    seed: int = 42,
    data_dir: str = "data",
    log_level: str = "INFO",
    mlflow_tracking_uri: str = "file:./mlruns",
    decision_policy: str = "balanceada",
    environment: str = "dev",
) -> dict[str, object]:
    run_id = str(uuid4())
    config = PipelineConfig.from_runtime(
        data_dir=Path(data_dir),
        seed=seed,
        log_level=log_level,
        mlflow_tracking_uri=mlflow_tracking_uri,
        decision_policy=decision_policy,
        environment=environment,
        run_id=run_id,
    )
    configure_logging(
        level=config.log_level,
        log_dir=config.logs_dir,
        run_id=config.run_id,
        environment=config.environment,
    )

    started_at = time.perf_counter()
    logger.info(
        "pipeline_start run_id=%s seed=%s data_dir=%s environment=%s policy=%s",
        config.run_id,
        config.seed,
        config.data_dir,
        config.environment,
        config.decision_policy,
    )
    stage_metadata: dict[str, dict[str, Any]] = {}

    try:
        bronze_df, stage_metadata["bronze"] = _execute_stage(
            "bronze_task",
            bronze_task,
            config,
            retries=2,
            retry_delay_seconds=3,
        )
        silver_df, stage_metadata["silver"] = _execute_stage(
            "silver_task",
            silver_task,
            config,
            bronze_df,
            retries=2,
            retry_delay_seconds=3,
        )
        _, stage_metadata["warehouse"] = _execute_stage(
            "warehouse_task",
            warehouse_task,
            config,
            silver_df,
            retries=1,
            retry_delay_seconds=2,
        )
        model_outputs, stage_metadata["modeling"] = _execute_stage(
            "ml_task",
            ml_task,
            config,
            silver_df,
            retries=1,
            retry_delay_seconds=2,
        )
        _, stage_metadata["reporting"] = _execute_stage(
            "reporting_task",
            reporting_task,
            config,
            model_outputs,
            retries=1,
            retry_delay_seconds=2,
        )
        drift_result, stage_metadata["monitoring"] = _execute_stage(
            "monitoring_task",
            monitoring_task,
            config,
            model_outputs,
            retries=1,
            retry_delay_seconds=2,
        )
    except Exception:
        logger.exception("pipeline_failed run_id=%s", config.run_id)
        raise

    elapsed_seconds = time.perf_counter() - started_at
    raw_digest = file_sha256(config.raw_input_path) if config.raw_input_path.exists() else "missing"
    execution_summary = {
        "run_id": config.run_id,
        "environment": config.environment,
        "data_dir": str(config.data_dir),
        "artifacts_dir": str(config.resolved_artifacts_dir),
        "decision_policy": config.decision_policy,
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "duration_seconds": round(elapsed_seconds, 2),
        "metrics": {
            "churn_f1": model_outputs.metrics["churn_f1"],
            "churn_roc_auc": model_outputs.metrics["churn_roc_auc"],
        },
        "drift_status": drift_result.get("status", "unknown"),
        "input": {
            "raw_path": str(config.raw_input_path),
            "raw_sha256": raw_digest,
        },
        "stages": stage_metadata,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    write_json_atomic(config.execution_metadata_path, execution_summary)
    write_json_atomic(config.latest_execution_metadata_path, execution_summary)
    logger.info(
        (
            "pipeline_done run_id=%s duration_seconds=%.2f churn_f1=%.4f "
            "churn_auc=%.4f drift_status=%s"
        ),
        config.run_id,
        elapsed_seconds,
        model_outputs.metrics["churn_f1"],
        model_outputs.metrics["churn_roc_auc"],
        drift_result.get("status", "unknown"),
    )
    return execution_summary


def main() -> None:
    args = parse_args()
    run_pipeline(
        seed=args.seed,
        data_dir=str(args.data_dir),
        log_level=args.log_level,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        decision_policy=args.decision_policy,
        environment=args.environment,
    )


if __name__ == "__main__":
    main()

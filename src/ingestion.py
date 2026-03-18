from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from src.config import PipelineConfig
from src.utils.io import write_csv_atomic
from src.validation import persist_validation_report, validate_raw_dataframe

logger = logging.getLogger(__name__)


def load_raw_dataset(config: PipelineConfig) -> pd.DataFrame:
    if not config.raw_input_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {config.raw_input_path}. "
            "Baixe do Kaggle e coloque em data/raw."
        )
    df = pd.read_csv(config.raw_input_path)
    report = validate_raw_dataframe(df)
    persist_validation_report(report, config.data_quality_report_path)
    logger.info("raw_loaded rows=%s cols=%s", df.shape[0], df.shape[1])
    logger.info("raw_validated %s", report.to_dict())
    return df


def build_bronze_layer(raw_df: pd.DataFrame) -> pd.DataFrame:
    bronze = raw_df.copy()
    bronze["ingested_at"] = datetime.now(UTC).isoformat()
    bronze["source_system"] = "kaggle_telco_churn"
    return bronze


def persist_bronze(config: PipelineConfig, bronze_df: pd.DataFrame) -> None:
    write_csv_atomic(config.bronze_output_path, bronze_df)
    logger.info("bronze_saved path=%s", config.bronze_output_path)

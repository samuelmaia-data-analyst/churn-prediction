"""Compatibility layer for processed dataset export using the canonical pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import PipelineConfig
from src.ingestion import build_bronze_layer, load_raw_dataset
from src.transformation import build_silver_layer

logger = logging.getLogger(__name__)


def main() -> None:
    config = PipelineConfig(data_dir=Path("data"), seed=42, log_level="INFO")
    raw_df = load_raw_dataset(config)
    bronze_df = build_bronze_layer(raw_df)
    silver_df = build_silver_layer(bronze_df)

    output_path = config.data_dir / "processed" / "telco_churn_processed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silver_df.to_csv(output_path, index=False)
    logger.info("processed_dataset_saved path=%s rows=%s", output_path, len(silver_df))


if __name__ == "__main__":
    main()

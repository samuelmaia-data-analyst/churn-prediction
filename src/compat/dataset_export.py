from __future__ import annotations

import logging
from pathlib import Path

from src.cli.save_processed_data import export_processed_dataset
from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def export_processed_dataset_legacy(
    data_dir: Path = Path("data"),
    output_path: Path = Path("data/processed/telco_churn_processed.csv"),
) -> Path:
    """Compatibility wrapper for the old processed-dataset export entrypoint."""
    config = PipelineConfig.from_runtime(data_dir=data_dir, run_id="legacy-process-data")
    exported_path = export_processed_dataset(config=config, output_path=output_path, sample_size=0)
    logger.info("legacy_processed_dataset_saved path=%s", exported_path)
    return exported_path

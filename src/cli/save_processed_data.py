from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipelines.ingestion import build_bronze_layer, load_raw_dataset
from src.pipelines.transformation import build_silver_layer
from src.runtime.config import PipelineConfig
from src.utils.io import write_csv_atomic

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta um dataset processado usando o fluxo canonico raw -> bronze -> silver."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/telco_churn_processed.csv"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Numero maximo de linhas para exportacao. Use 0 para exportar tudo.",
    )
    return parser.parse_args()


def export_processed_dataset(
    config: PipelineConfig,
    output_path: Path,
    sample_size: int = 2000,
) -> Path:
    raw_df = load_raw_dataset(config)
    bronze_df = build_bronze_layer(raw_df)
    silver_df = build_silver_layer(bronze_df)

    if sample_size > 0:
        silver_df = silver_df.sample(n=min(sample_size, len(silver_df)), random_state=config.seed)

    write_csv_atomic(output_path, silver_df)
    logger.info("processed_dataset_saved path=%s rows=%s", output_path, len(silver_df))
    return output_path


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_runtime(data_dir=args.data_dir, run_id="process-data")
    export_processed_dataset(
        config=config,
        output_path=args.output_path,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()

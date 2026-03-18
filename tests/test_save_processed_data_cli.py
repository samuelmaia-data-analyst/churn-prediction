from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cli.save_processed_data import export_processed_dataset
from src.runtime.config import PipelineConfig


def test_export_processed_dataset_uses_canonical_silver_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw_dir.joinpath(source_dataset.name).write_bytes(source_dataset.read_bytes())

    config = PipelineConfig.from_runtime(data_dir=data_dir, run_id="processed-export")
    output_path = tmp_path / "exports" / "processed.csv"

    exported_path = export_processed_dataset(config=config, output_path=output_path, sample_size=50)
    exported_df = pd.read_csv(exported_path)

    assert exported_path.exists()
    assert len(exported_df) == 50
    assert exported_df["Churn"].isin([0, 1]).all()
    assert exported_df["TotalCharges"].dtype.kind in {"f", "i"}

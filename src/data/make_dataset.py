from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def load_data(self) -> pd.DataFrame:
        dataset_path = Path(self.config["data"]["raw_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

        df = pd.read_csv(dataset_path)
        logger.info("Dados carregados: %s linhas, %s colunas", df.shape[0], df.shape[1])
        return df

    def save_processed(self, df: pd.DataFrame) -> Path:
        output_path = Path(self.config["data"]["processed_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Dados salvos em: %s", output_path)
        return output_path

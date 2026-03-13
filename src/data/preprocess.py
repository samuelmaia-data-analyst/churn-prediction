from __future__ import annotations

import logging

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.ingestion import build_bronze_layer
from src.transformation import build_silver_layer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Compatibility layer over the canonical bronze/silver pipeline."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        bronze = build_bronze_layer(df)
        silver = build_silver_layer(bronze)
        if "customerID" in silver.columns:
            silver = silver.drop(columns=["customerID"])
        return silver

    def split_data(self, df: pd.DataFrame):
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y,
        )

        logger.info("Treino: %s | Teste: %s", X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test

from __future__ import annotations

import logging

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()

        if "customerID" in cleaned.columns:
            cleaned = cleaned.drop(columns=["customerID"])

        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        median_total_charges = cleaned["TotalCharges"].median()
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(median_total_charges)

        cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})
        if cleaned["Churn"].isna().any():
            raise ValueError("Coluna Churn contem valores inesperados. Esperado: Yes/No")

        return cleaned

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

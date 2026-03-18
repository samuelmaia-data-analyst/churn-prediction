from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer

from src.feature_engineering import engineer_features
from src.modeling.churn import build_preprocessor


class FeatureEngineer:
    """Legacy facade kept for backward compatibility with the canonical feature pipeline."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names: list[str] | None = None

    def create_preprocessor(self) -> ColumnTransformer:
        self.preprocessor = build_preprocessor()
        return self.preprocessor

    def fit_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.preprocessor is None:
            self.create_preprocessor()

        train_featured = engineer_features(X_train)
        test_featured = engineer_features(X_test)
        X_train_proc = self.preprocessor.fit_transform(train_featured)
        X_test_proc = self.preprocessor.transform(test_featured)
        self.feature_names = list(self.preprocessor.get_feature_names_out())

        X_train_proc_df = pd.DataFrame(
            X_train_proc, columns=self.feature_names, index=X_train.index
        )
        X_test_proc_df = pd.DataFrame(X_test_proc, columns=self.feature_names, index=X_test.index)
        return X_train_proc_df, X_test_proc_df

    def save_preprocessor(self, path: str = "models/preprocessor.joblib") -> None:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessador nao treinado. Rode fit_transform antes de salvar.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path)

    def load_preprocessor(self, path: str = "models/preprocessor.joblib") -> None:
        self.preprocessor = joblib.load(path)
        self.feature_names = list(self.preprocessor.get_feature_names_out())

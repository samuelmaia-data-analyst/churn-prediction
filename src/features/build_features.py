from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names: list[str] | None = None

    def create_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["features"]["numerical_features"]),
                ("cat", categorical_transformer, self.config["features"]["categorical_features"]),
            ]
        )
        return self.preprocessor

    def fit_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.preprocessor is None:
            self.create_preprocessor()

        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)
        self._generate_feature_names()

        assert self.feature_names is not None

        X_train_proc_df = pd.DataFrame(
            X_train_proc, columns=self.feature_names, index=X_train.index
        )
        X_test_proc_df = pd.DataFrame(X_test_proc, columns=self.feature_names, index=X_test.index)

        return X_train_proc_df, X_test_proc_df

    def _generate_feature_names(self) -> None:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessador nao foi inicializado")

        numeric_features = list(self.config["features"]["numerical_features"])
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        categorical_features = self.config["features"]["categorical_features"]
        encoded = list(cat_encoder.get_feature_names_out(categorical_features))

        self.feature_names = numeric_features + encoded

    def save_preprocessor(self, path: str = "models/preprocessor.joblib") -> None:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessador nao treinado. Rode fit_transform antes de salvar.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path)

    def load_preprocessor(self, path: str = "models/preprocessor.joblib") -> None:
        self.preprocessor = joblib.load(path)

from __future__ import annotations

from pathlib import Path

import joblib
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class ModelTrainer:
    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.models: dict = {}
        self.results: dict = {}
        self.best_model: str | None = None

    def get_models(self) -> dict:
        seed = self.config["data"]["random_state"]
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=seed
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=seed
            ),
        }

    def evaluate(self, model, X_test, y_test) -> dict[str, float]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

    def train_all(self, X_train, y_train, X_test, y_test) -> dict:
        for name, model in self.get_models().items():
            model.fit(X_train, y_train)
            self.models[name] = model
            self.results[name] = self.evaluate(model, X_test, y_test)

        self.best_model = max(
            self.results,
            key=lambda name: (self.results[name]["f1"], self.results[name]["roc_auc"]),
        )
        return self.results

    def save_model(self, model_name: str | None = None) -> Path:
        target_name = model_name or self.best_model
        if not target_name:
            raise RuntimeError("Nenhum modelo treinado para salvar")
        if target_name not in self.models:
            raise ValueError(f"Modelo não encontrado: {target_name}")

        path = Path(self.config["models"]["save_path"]) / f"{target_name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[target_name], path)
        return path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import yaml
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.models = {}
        self.results = {}
        self.best_model = None

    def get_models(self):
        """Retorna dicionário com modelos"""
        return {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=self.config['data']['random_state']
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config['data']['random_state']
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config['data']['random_state']
            )
        }

    def evaluate(self, model, X_test, y_test):
        """Avalia o modelo"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }

    def train_all(self, X_train, y_train, X_test, y_test):
        """Treina todos os modelos"""

        for name, model in self.get_models().items():
            logger.info(f"Treinando {name}...")

            # Treinar
            model.fit(X_train, y_train)
            self.models[name] = model

            # Avaliar
            metrics = self.evaluate(model, X_test, y_test)
            self.results[name] = metrics

            logger.info(f"{name} - F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")

        # Melhor modelo (baseado em F1)
        self.best_model = max(self.results, key=lambda x: self.results[x]['f1'])
        logger.info(f"Melhor modelo: {self.best_model}")

        return self.results

    def save_model(self, model_name=None):
        """Salva o modelo"""
        if model_name is None:
            model_name = self.best_model

        path = Path(self.config['models']['save_path']) / f"{model_name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], path)
        logger.info(f"Modelo salvo em: {path}")
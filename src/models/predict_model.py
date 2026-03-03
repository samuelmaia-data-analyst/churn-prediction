from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/LogisticRegression.joblib")
DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@dataclass(frozen=True)
class PredictionResult:
    churn: str
    probability: float
    risk_level: str


class ChurnPredictor:
    """Facade para inferencia de churn com artefatos persistidos."""

    def __init__(
        self,
        model: Any | None = None,
        preprocessor: Any | None = None,
        model_path: Path = DEFAULT_MODEL_PATH,
        preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def load_artifacts(self) -> None:
        if not self.model_path.exists() or not self.preprocessor_path.exists():
            missing = []
            if not self.model_path.exists():
                missing.append(str(self.model_path))
            if not self.preprocessor_path.exists():
                missing.append(str(self.preprocessor_path))
            raise FileNotFoundError(f"Artefatos nao encontrados: {', '.join(missing)}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability > 0.6:
            return "Alto"
        if probability > 0.3:
            return "Medio"
        return "Baixo"

    def predict_from_dict(self, customer_data: Mapping[str, Any]) -> PredictionResult:
        if not self.is_ready:
            self.load_artifacts()

        customer_df = pd.DataFrame([dict(customer_data)])
        transformed = self.preprocessor.transform(customer_df)

        prediction = int(self.model.predict(transformed)[0])
        probability = float(self.model.predict_proba(transformed)[0][1])

        return PredictionResult(
            churn="Sim" if prediction == 1 else "Nao",
            probability=round(probability, 4),
            risk_level=self._risk_level(probability),
        )

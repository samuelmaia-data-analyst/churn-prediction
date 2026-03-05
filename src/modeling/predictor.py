from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("artifacts/models/LogisticRegression.joblib")
DEFAULT_PREPROCESSOR_PATH = Path("artifacts/models/preprocessor.joblib")
LEGACY_MODEL_PATH = Path("models/LogisticRegression.joblib")
LEGACY_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@dataclass(frozen=True)
class PredictionResult:
    churn: str
    probability: float
    risk_level: str


class ChurnPredictor:
    """Facade para inferência de churn com artefatos persistidos."""

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

    @staticmethod
    def _resolve_artifact(primary: Path, legacy: Path) -> Path:
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy
        return primary

    def load_artifacts(self) -> None:
        model_path = self._resolve_artifact(self.model_path, LEGACY_MODEL_PATH)
        preprocessor_path = self._resolve_artifact(self.preprocessor_path, LEGACY_PREPROCESSOR_PATH)

        if not model_path.exists() or not preprocessor_path.exists():
            missing = []
            if not model_path.exists():
                missing.append(str(model_path))
            if not preprocessor_path.exists():
                missing.append(str(preprocessor_path))
            raise FileNotFoundError(f"Artefatos não encontrados: {', '.join(missing)}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability > 0.6:
            return "Alto"
        if probability > 0.3:
            return "Médio"
        return "Baixo"

    def predict_from_dict(self, customer_data: Mapping[str, Any]) -> PredictionResult:
        if not self.is_ready:
            self.load_artifacts()

        customer_df = pd.DataFrame([dict(customer_data)])
        transformed = self.preprocessor.transform(customer_df)

        prediction = int(self.model.predict(transformed)[0])
        probability = float(self.model.predict_proba(transformed)[0][1])

        return PredictionResult(
            churn="Sim" if prediction == 1 else "Não",
            probability=round(probability, 4),
            risk_level=self._risk_level(probability),
        )

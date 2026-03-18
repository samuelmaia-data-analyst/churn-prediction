from __future__ import annotations

from src.models import ChurnPredictor, ModelTrainer, PredictionResult, tune_random_forest


def test_legacy_model_wrappers_resolve_to_compat_exports() -> None:
    assert ChurnPredictor.__name__ == "ChurnPredictor"
    assert ModelTrainer.__name__ == "ModelTrainer"
    assert PredictionResult.__name__ == "PredictionResult"
    assert callable(tune_random_forest)

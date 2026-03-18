"""Compatibility wrapper package. Prefer importing from `src.compat` or `src.modeling`."""

from src.compat.modeling import ChurnPredictor, ModelTrainer, PredictionResult, tune_random_forest

__all__ = ["ChurnPredictor", "PredictionResult", "ModelTrainer", "tune_random_forest"]

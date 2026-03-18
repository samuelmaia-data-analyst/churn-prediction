"""Compatibility exports for legacy modeling imports."""

from src.modeling.predictor import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    LEGACY_MODEL_PATH,
    LEGACY_PREPROCESSOR_PATH,
    ChurnPredictor,
    PredictionResult,
)
from src.modeling.trainer import ModelTrainer
from src.modeling.tuner import tune_random_forest

__all__ = [
    "ChurnPredictor",
    "PredictionResult",
    "ModelTrainer",
    "tune_random_forest",
    "DEFAULT_BUNDLE_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PREPROCESSOR_PATH",
    "LEGACY_MODEL_PATH",
    "LEGACY_PREPROCESSOR_PATH",
]

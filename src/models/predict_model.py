"""Compatibility layer: use src.modeling.predictor moving forward."""

from src.modeling.predictor import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    LEGACY_MODEL_PATH,
    LEGACY_PREPROCESSOR_PATH,
    ChurnPredictor,
    PredictionResult,
)

__all__ = [
    "ChurnPredictor",
    "PredictionResult",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PREPROCESSOR_PATH",
    "LEGACY_MODEL_PATH",
    "LEGACY_PREPROCESSOR_PATH",
]

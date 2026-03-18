"""Compatibility wrapper. Prefer importing predictor classes from `src.compat` or `src.modeling`."""

from src.compat.modeling import (
    DEFAULT_BUNDLE_PATH,
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
    "DEFAULT_BUNDLE_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PREPROCESSOR_PATH",
    "LEGACY_MODEL_PATH",
    "LEGACY_PREPROCESSOR_PATH",
]

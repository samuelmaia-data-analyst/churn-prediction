"""Compatibility layer for legacy imports kept only for backward compatibility."""

from src.compat.dataset_export import export_processed_dataset_legacy
from src.compat.features import FeatureEngineer
from src.compat.modeling import ChurnPredictor, ModelTrainer, PredictionResult, tune_random_forest
from src.compat.preprocessing import DataPreprocessor

__all__ = [
    "ChurnPredictor",
    "DataPreprocessor",
    "FeatureEngineer",
    "ModelTrainer",
    "PredictionResult",
    "export_processed_dataset_legacy",
    "tune_random_forest",
]

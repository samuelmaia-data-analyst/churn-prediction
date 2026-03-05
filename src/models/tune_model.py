"""Compatibility layer: use src.modeling.tuner moving forward."""

from src.modeling.tuner import tune_random_forest

__all__ = ["tune_random_forest"]

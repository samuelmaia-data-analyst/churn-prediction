"""Compatibility wrapper. Prefer importing tuners from `src.compat` or `src.modeling`."""

from src.compat.modeling import tune_random_forest

__all__ = ["tune_random_forest"]

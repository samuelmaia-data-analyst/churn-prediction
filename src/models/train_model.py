"""Compatibility wrapper. Prefer importing trainer classes from `src.compat` or `src.modeling`."""

from src.compat.modeling import ModelTrainer

__all__ = ["ModelTrainer"]

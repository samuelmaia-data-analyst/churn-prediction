from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict


class BaselineModel(TypedDict):
    name: str
    roc_auc: float


class ModelComparisonRow(TypedDict):
    model: str
    roc_auc: float


class FeatureImportanceRow(TypedDict):
    feature: str
    importance: float


@dataclass(frozen=True)
class ExecutiveMetrics:
    churn_f1: float
    churn_roc_auc: float
    next_purchase_mae: float
    baseline_model: BaselineModel
    model_comparison: list[ModelComparisonRow]
    selected_churn_model: str
    feature_importance: list[FeatureImportanceRow]
    top_drivers_of_churn: list[str]
    key_insights: list[str]
    pipeline_visual: str
    model_comparison_note: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.model_comparison_note is None:
            payload.pop("model_comparison_note", None)
        return payload

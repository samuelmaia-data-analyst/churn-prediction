from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict


class KPIValues(TypedDict):
    total_customers: int
    churn_rate: float
    high_risk_customers: int
    revenue_at_risk: float
    avg_next_purchase_prediction: float


class ReportMetadata(TypedDict):
    schema_version: str
    generated_at_utc: str
    run_id: str
    environment: str


@dataclass(frozen=True)
class ExecutiveReport:
    metadata: ReportMetadata
    kpis: KPIValues
    model_metrics: dict[str, object]
    top_10_priorities: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

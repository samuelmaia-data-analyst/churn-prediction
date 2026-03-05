from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.config import PipelineConfig
from src.contracts import ExecutiveReport


@dataclass(frozen=True)
class ReportOutputs:
    executive_report: ExecutiveReport
    recommendations: pd.DataFrame
    kpi_summary: pd.DataFrame


def _recommend_action(probability: float, next_purchase: float) -> str:
    if probability >= 0.7 and next_purchase >= 80:
        return "Contato imediato + oferta premium de retenção"
    if probability >= 0.7:
        return "Contato imediato + desconto de retenção"
    if probability >= 0.45:
        return "Campanha de engajamento proativa"
    return "Monitoramento e nutrição de relacionamento"


def build_business_outputs(scored_df: pd.DataFrame, metrics: Mapping[str, object]) -> ReportOutputs:
    recommendations = scored_df[
        [
            "customerID",
            "churn_probability",
            "next_purchase_prediction",
            "MonthlyCharges",
            "Contract",
        ]
    ].copy()
    recommendations["action_recommendation"] = recommendations.apply(
        lambda row: _recommend_action(row["churn_probability"], row["next_purchase_prediction"]),
        axis=1,
    )
    recommendations = recommendations.sort_values("churn_probability", ascending=False).reset_index(
        drop=True
    )

    high_risk = recommendations[recommendations["churn_probability"] >= 0.7]
    kpis = {
        "total_customers": int(len(scored_df)),
        "churn_rate": float(scored_df["Churn"].mean()),
        "high_risk_customers": int(len(high_risk)),
        "revenue_at_risk": float(high_risk["MonthlyCharges"].sum()),
        "avg_next_purchase_prediction": float(scored_df["next_purchase_prediction"].mean()),
    }

    kpi_summary = pd.DataFrame([kpis])
    executive_report = ExecutiveReport(
        kpis=kpis,
        model_metrics=dict(metrics),
        top_10_priorities=recommendations.head(10).to_dict(orient="records"),
    )

    return ReportOutputs(
        executive_report=executive_report,
        recommendations=recommendations,
        kpi_summary=kpi_summary,
    )


def _render_model_card(executive_report: ExecutiveReport) -> str:
    report = executive_report.to_dict()
    kpis = report.get("kpis", {})
    model_metrics = report.get("model_metrics", {})
    baseline = model_metrics.get("baseline_model", {})
    comparison = model_metrics.get("model_comparison", [])
    top_drivers = model_metrics.get("top_drivers_of_churn", [])
    insights = model_metrics.get("key_insights", [])
    pipeline_visual = model_metrics.get("pipeline_visual", "Raw -> Bronze -> Silver -> Gold")

    comparison_rows = "\n".join(
        f"| {row.get('model', '-') } | {float(row.get('roc_auc', 0.0)):.3f} |" for row in comparison
    )
    drivers_rows = "\n".join(f"- {driver}" for driver in top_drivers)
    insights_rows = "\n".join(f"- {insight}" for insight in insights)

    return f"""# Model Card - Churn Prediction

## Baseline Model
- Logistic Regression
- ROC-AUC: {float(baseline.get("roc_auc", 0.0)):.3f}

## Model Comparison
| Model | ROC-AUC |
|---|---:|
{comparison_rows}

## Top Drivers of Churn
{drivers_rows}

## Key Insights
{insights_rows}

## KPI Snapshot
- Total Customers: {int(kpis.get("total_customers", 0))}
- Churn Rate: {float(kpis.get("churn_rate", 0.0)):.2%}
- High Risk Customers: {int(kpis.get("high_risk_customers", 0))}
- Revenue at Risk: ${float(kpis.get("revenue_at_risk", 0.0)):,.2f}

## Pipeline Visual
```mermaid
flowchart LR
    A[Raw] --> B[Bronze]
    B --> C[Silver]
    C --> D[Gold]
```

Referencia textual: `{pipeline_visual}`
"""


def _render_executive_brief(executive_report: ExecutiveReport, recommendations: pd.DataFrame) -> str:
    report = executive_report.to_dict()
    kpis = report.get("kpis", {})
    model_metrics = report.get("model_metrics", {})

    high_risk = recommendations[recommendations["churn_probability"] >= 0.7]
    medium_risk = recommendations[
        (recommendations["churn_probability"] >= 0.45)
        & (recommendations["churn_probability"] < 0.7)
    ]
    low_risk = recommendations[recommendations["churn_probability"] < 0.45]

    month_to_month = recommendations[recommendations["Contract"].eq("Month-to-month")]
    month_to_month_risk = (
        float(month_to_month["churn_probability"].mean()) if not month_to_month.empty else 0.0
    )

    top_drivers = model_metrics.get("top_drivers_of_churn", [])
    key_insights = model_metrics.get("key_insights", [])

    return f"""# Executive Brief - Churn Strategy

## Executive Summary
- Customers analyzed: {int(kpis.get("total_customers", 0))}
- Churn rate: {float(kpis.get("churn_rate", 0.0)):.2%}
- High-risk customers: {int(kpis.get("high_risk_customers", 0))}
- Revenue at risk: ${float(kpis.get("revenue_at_risk", 0.0)):,.2f}

## Risk Segmentation Plan
| Segment | Criteria | Customers | Recommended Action |
|---|---|---:|---|
| High | churn_probability >= 0.70 | {len(high_risk)} | Immediate retention contact + premium offer |
| Medium | 0.45 <= churn_probability < 0.70 | {len(medium_risk)} | Proactive engagement campaign |
| Low | churn_probability < 0.45 | {len(low_risk)} | Relationship nurture and monitoring |

## Contract Insight
- Month-to-month average churn probability: {month_to_month_risk:.2%}
- Strategic interpretation: month-to-month contracts should be prioritized in retention waves.

## Top Drivers of Churn
{chr(10).join(f"- {driver}" for driver in top_drivers)}

## Key Insights
{chr(10).join(f"- {insight}" for insight in key_insights)}

## Pipeline
```mermaid
flowchart LR
    A[Raw] --> B[Bronze]
    B --> C[Silver]
    C --> D[Gold]
```
"""


def persist_business_outputs(config: PipelineConfig, outputs: ReportOutputs) -> None:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.gold_dir.mkdir(parents=True, exist_ok=True)

    with open(config.executive_report_path, "w", encoding="utf-8") as fp:
        json.dump(outputs.executive_report.to_dict(), fp, ensure_ascii=False, indent=2)
    with open(config.model_card_path, "w", encoding="utf-8") as fp:
        fp.write(_render_model_card(outputs.executive_report))
    with open(config.executive_brief_path, "w", encoding="utf-8") as fp:
        fp.write(_render_executive_brief(outputs.executive_report, outputs.recommendations))

    outputs.kpi_summary.to_csv(config.gold_dir / "kpi_summary.csv", index=False)
    outputs.recommendations.to_csv(config.gold_dir / "customer_prioritization.csv", index=False)

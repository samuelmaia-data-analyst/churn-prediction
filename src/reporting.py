from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.config import PipelineConfig
from src.contracts import ExecutiveReport
from src.decisioning import (
    action_for_segment,
    build_action_playbook,
    customer_value_segment,
    decision_threshold,
    get_policy,
    risk_segment,
    threshold_for_value_segment,
)


@dataclass(frozen=True)
class ReportOutputs:
    executive_report: ExecutiveReport
    recommendations: pd.DataFrame
    kpi_summary: pd.DataFrame
    action_playbook: pd.DataFrame


def build_business_outputs(scored_df: pd.DataFrame, metrics: Mapping[str, object]) -> ReportOutputs:
    policy = get_policy("balanceada")
    base_threshold = decision_threshold(policy)

    recommendations = scored_df[
        [
            "customerID",
            "churn_probability",
            "next_purchase_prediction",
            "MonthlyCharges",
            "Contract",
        ]
    ].copy()

    recommendations["value_segment"] = recommendations["next_purchase_prediction"].apply(
        lambda p: customer_value_segment(float(p))
    )
    recommendations["decision_threshold"] = recommendations["value_segment"].apply(
        threshold_for_value_segment
    )
    recommendations["risk_segment"] = recommendations.apply(
        lambda row: risk_segment(float(row["churn_probability"]), float(row["decision_threshold"])),
        axis=1,
    )
    recommendations["action_recommendation"] = recommendations.apply(
        lambda row: action_for_segment(row["risk_segment"], row["value_segment"]),
        axis=1,
    )
    recommendations["decision_policy"] = policy.name
    recommendations["base_decision_threshold"] = base_threshold
    recommendations = recommendations.sort_values("churn_probability", ascending=False).reset_index(
        drop=True
    )

    high_risk = recommendations[recommendations["risk_segment"] == "high"]
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
    action_playbook = build_action_playbook(recommendations)

    return ReportOutputs(
        executive_report=executive_report,
        recommendations=recommendations,
        kpi_summary=kpi_summary,
        action_playbook=action_playbook,
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


def _render_executive_brief(
    executive_report: ExecutiveReport, recommendations: pd.DataFrame
) -> str:
    report = executive_report.to_dict()
    kpis = report.get("kpis", {})
    model_metrics = report.get("model_metrics", {})

    high_risk = recommendations[recommendations["risk_segment"] == "high"]
    medium_risk = recommendations[recommendations["risk_segment"] == "medium"]
    low_risk = recommendations[recommendations["risk_segment"] == "low"]

    month_to_month = recommendations[recommendations["Contract"].eq("Month-to-month")]
    month_to_month_risk = (
        float(month_to_month["churn_probability"].mean()) if not month_to_month.empty else 0.0
    )

    top_drivers = model_metrics.get("top_drivers_of_churn", [])
    key_insights = model_metrics.get("key_insights", [])
    segmentation_rows = "\n".join(
        [
            (
                f"| High | Threshold by LTV (0.65 or 0.80) | {len(high_risk)} | "
                "Call retention / retention offer by email |"
            ),
            (
                f"| Medium | Threshold minus 0.20 band | {len(medium_risk)} | "
                "Proactive loyalty outreach / automated nurture journey |"
            ),
            (
                f"| Low | Below medium-risk band | {len(low_risk)} | "
                "Monitor and upsell trigger |"
            ),
        ]
    )

    return f"""# Executive Brief - Churn Strategy

## Executive Summary
- Customers analyzed: {int(kpis.get("total_customers", 0))}
- Churn rate: {float(kpis.get("churn_rate", 0.0)):.2%}
- High-risk customers: {int(kpis.get("high_risk_customers", 0))}
- Revenue at risk: ${float(kpis.get("revenue_at_risk", 0.0)):,.2f}

## Risk Segmentation Plan
| Segment | Criteria | Customers | Recommended Action |
|---|---|---:|---|
{segmentation_rows}

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


def _render_action_playbook(playbook: pd.DataFrame) -> str:
    header = (
        "| Segment | Risk | Action | Expected ROI | Customers | Total Expected ROI (USD) |\n"
        "|---|---|---|---|---:|---:|"
    )
    rows = []
    for _, row in playbook.iterrows():
        template = (
            "| {segment} | {risk} | {action} | {expected_roi} | {customers} | {total_roi:.2f} |"
        )
        rows.append(
            template.format(
                segment=row["Segment"],
                risk=row["Risk"],
                action=row["Action"],
                expected_roi=row["Expected ROI"],
                customers=int(row["Customers"]),
                total_roi=float(row["total_expected_roi_usd"]),
            )
        )
    return "# Action Playbook\n\n" + header + "\n" + "\n".join(rows) + "\n"


def persist_business_outputs(config: PipelineConfig, outputs: ReportOutputs) -> None:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.gold_dir.mkdir(parents=True, exist_ok=True)

    with open(config.executive_report_path, "w", encoding="utf-8") as fp:
        json.dump(outputs.executive_report.to_dict(), fp, ensure_ascii=False, indent=2)
    with open(config.model_card_path, "w", encoding="utf-8") as fp:
        fp.write(_render_model_card(outputs.executive_report))
    with open(config.executive_brief_path, "w", encoding="utf-8") as fp:
        fp.write(_render_executive_brief(outputs.executive_report, outputs.recommendations))
    with open(config.action_playbook_path, "w", encoding="utf-8") as fp:
        fp.write(_render_action_playbook(outputs.action_playbook))

    outputs.kpi_summary.to_csv(config.gold_dir / "kpi_summary.csv", index=False)
    outputs.recommendations.to_csv(config.gold_dir / "customer_prioritization.csv", index=False)
    outputs.action_playbook.to_csv(config.gold_dir / "action_playbook.csv", index=False)

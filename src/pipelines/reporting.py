from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.contracts import (
    ArtifactEntry,
    ArtifactManifest,
    ExecutiveReport,
    validate_action_playbook_contract,
    validate_executive_report_contract,
    validate_kpi_contract,
    validate_prioritization_contract,
)
from src.pipelines.decisioning import (
    action_for_segment,
    build_action_playbook,
    customer_value_segment,
    decision_threshold,
    get_policy,
    risk_segment,
    threshold_for_value_segment,
)
from src.runtime.config import PipelineConfig
from src.utils.io import write_csv_atomic, write_json_atomic, write_text_atomic


@dataclass(frozen=True)
class ReportOutputs:
    executive_report: ExecutiveReport
    recommendations: pd.DataFrame
    kpi_summary: pd.DataFrame
    action_playbook: pd.DataFrame


def build_business_outputs(
    config: PipelineConfig,
    scored_df: pd.DataFrame,
    metrics: Mapping[str, object],
) -> ReportOutputs:
    policy_name = str(metrics.get("decision_policy", {}).get("name", "balanceada"))
    policy = get_policy(policy_name)
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
        metadata={
            "schema_version": "1.1.0",
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "run_id": config.run_id,
            "environment": config.environment,
        },
        kpis=kpis,
        model_metrics=dict(metrics),
        top_10_priorities=recommendations.head(10).to_dict(orient="records"),
    )
    action_playbook = build_action_playbook(recommendations)

    validate_executive_report_contract(executive_report)
    validate_prioritization_contract(recommendations)
    validate_kpi_contract(kpi_summary)
    validate_action_playbook_contract(action_playbook)

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
    confusion = model_metrics.get("confusion_matrix", {})
    threshold = float(model_metrics.get("decision_threshold", 0.5))
    decision_policy = model_metrics.get("decision_policy", {})
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

## Classification Metrics
- Precision: {float(model_metrics.get("churn_precision", 0.0)):.3f}
- Recall: {float(model_metrics.get("churn_recall", 0.0)):.3f}
- F1-score: {float(model_metrics.get("churn_f1", 0.0)):.3f}
- ROC-AUC: {float(model_metrics.get("churn_roc_auc", 0.0)):.3f}
- Decision threshold: {threshold:.2f}
- Decision policy: {decision_policy.get("name", "-")}

## Model Comparison
| Model | ROC-AUC |
|---|---:|
{comparison_rows}

## Top Drivers of Churn
{drivers_rows}

## Key Insights
{insights_rows}

## Confusion Matrix
- True Negatives: {int(confusion.get("tn", 0))}
- False Positives: {int(confusion.get("fp", 0))}
- False Negatives: {int(confusion.get("fn", 0))}
- True Positives: {int(confusion.get("tp", 0))}

## Business Decision Policy
- False positive cost: {float(decision_policy.get("fp_cost", 0.0)):.2f}
- False negative cost: {float(decision_policy.get("fn_cost", 0.0)):.2f}
- Policy rationale: {decision_policy.get("description", "-")}

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

## Limitations and Governance
- Dataset de referencia e sintetico para caso de negocio, nao prova causalidade.
- Performance pode degradar com drift de mix de clientes, canais ou precificacao.
- Threshold deve ser revisado conforme custo operacional, budget e capacidade comercial.
- Recomendado monitorar drift, recalibrar threshold e auditar impacto financeiro por campanha.
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
    risk_profiles = model_metrics.get("risk_profiles", [])
    cost_interpretation = model_metrics.get("cost_interpretation", {})
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
            (f"| Low | Below medium-risk band | {len(low_risk)} | " "Monitor and upsell trigger |"),
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

## Risk Profiles with Highest Average Churn
{chr(10).join(_format_risk_profile(row) for row in risk_profiles)}

## Key Insights
{chr(10).join(f"- {insight}" for insight in key_insights)}

## Error Cost Interpretation
- False positive: {cost_interpretation.get("false_positive", "-")}
- False negative: {cost_interpretation.get("false_negative", "-")}
- Trade-off: {cost_interpretation.get("business_tradeoff", "-")}

## Pipeline
```mermaid
flowchart LR
    A[Raw] --> B[Bronze]
    B --> C[Silver]
    C --> D[Gold]
```
"""


def _format_risk_profile(row: Mapping[str, object]) -> str:
    return (
        f"- {row['contract']} + {row['internet_service']}: "
        f"avg churn prob {float(row['avg_churn_probability']):.2%}, "
        f"avg monthly charges ${float(row['avg_monthly_charges']):.2f}"
    )


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
    write_json_atomic(config.executive_report_path, outputs.executive_report.to_dict())
    write_text_atomic(config.model_card_path, _render_model_card(outputs.executive_report))
    write_text_atomic(
        config.executive_brief_path,
        _render_executive_brief(outputs.executive_report, outputs.recommendations),
    )
    write_text_atomic(config.action_playbook_path, _render_action_playbook(outputs.action_playbook))

    write_csv_atomic(config.gold_dir / "kpi_summary.csv", outputs.kpi_summary)
    write_csv_atomic(config.gold_dir / "customer_prioritization.csv", outputs.recommendations)
    write_csv_atomic(config.gold_dir / "action_playbook.csv", outputs.action_playbook)
    gold_manifest = ArtifactManifest(
        schema_version="1.0.0",
        artifact_type="gold_layer",
        generated_at_utc=pd.Timestamp.utcnow().isoformat(),
        run_id=config.run_id,
        environment=config.environment,
        entries=[
            ArtifactEntry(
                name="kpi_summary",
                path=str((config.gold_dir / "kpi_summary.csv").as_posix()),
                format="csv",
                rows=int(len(outputs.kpi_summary)),
            ),
            ArtifactEntry(
                name="customer_prioritization",
                path=str((config.gold_dir / "customer_prioritization.csv").as_posix()),
                format="csv",
                rows=int(len(outputs.recommendations)),
            ),
            ArtifactEntry(
                name="action_playbook",
                path=str((config.gold_dir / "action_playbook.csv").as_posix()),
                format="csv",
                rows=int(len(outputs.action_playbook)),
            ),
        ],
    )
    write_json_atomic(config.gold_manifest_path, gold_manifest.to_dict())

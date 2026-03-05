from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.config import PipelineConfig


@dataclass(frozen=True)
class ReportOutputs:
    executive_report: dict
    recommendations: pd.DataFrame
    kpi_summary: pd.DataFrame


def _recommend_action(probability: float, next_purchase: float) -> str:
    if probability >= 0.7 and next_purchase >= 80:
        return "Contato imediato + oferta premium de retencao"
    if probability >= 0.7:
        return "Contato imediato + desconto de retencao"
    if probability >= 0.45:
        return "Campanha de engajamento proativa"
    return "Monitoramento e nutricao de relacionamento"


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
    executive_report = {
        "kpis": kpis,
        "model_metrics": dict(metrics),
        "top_10_priorities": recommendations.head(10).to_dict(orient="records"),
    }

    return ReportOutputs(
        executive_report=executive_report,
        recommendations=recommendations,
        kpi_summary=kpi_summary,
    )


def persist_business_outputs(config: PipelineConfig, outputs: ReportOutputs) -> None:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.gold_dir.mkdir(parents=True, exist_ok=True)

    with open(config.executive_report_path, "w", encoding="utf-8") as fp:
        json.dump(outputs.executive_report, fp, ensure_ascii=False, indent=2)

    outputs.kpi_summary.to_csv(config.gold_dir / "kpi_summary.csv", index=False)
    outputs.recommendations.to_csv(config.gold_dir / "customer_prioritization.csv", index=False)

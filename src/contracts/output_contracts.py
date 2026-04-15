from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.contracts.report_schema import ExecutiveReport

REQUIRED_PRIORITIZATION_COLUMNS = {
    "customerID",
    "churn_probability",
    "next_purchase_prediction",
    "MonthlyCharges",
    "Contract",
    "value_segment",
    "decision_threshold",
    "risk_segment",
    "action_recommendation",
    "decision_policy",
    "base_decision_threshold",
}

REQUIRED_KPI_COLUMNS = {
    "total_customers",
    "churn_rate",
    "high_risk_customers",
    "revenue_at_risk",
    "avg_next_purchase_prediction",
}

REQUIRED_ACTION_PLAYBOOK_COLUMNS = {
    "Segment",
    "Risk",
    "Action",
    "Expected ROI",
    "Customers",
    "total_expected_roi_usd",
}


def _ensure_columns(df: pd.DataFrame, required: set[str], artifact_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{artifact_name} sem colunas obrigatorias: {', '.join(missing)}")


def _ensure_non_empty(df: pd.DataFrame, artifact_name: str) -> None:
    if df.empty:
        raise ValueError(f"{artifact_name} vazio.")


def _ensure_probability_bounds(series: pd.Series, column_name: str) -> None:
    if ((series < 0.0) | (series > 1.0)).any():
        raise ValueError(f"{column_name} fora do intervalo [0, 1].")


def _ensure_allowed_values(series: pd.Series, allowed: Iterable[str], column_name: str) -> None:
    invalid = sorted(set(series.astype(str)) - set(allowed))
    if invalid:
        raise ValueError(f"{column_name} com valores invalidos: {', '.join(invalid)}")


def validate_executive_report_contract(report: ExecutiveReport) -> None:
    payload = report.to_dict()
    expected_root = {"metadata", "kpis", "model_metrics", "top_10_priorities"}
    if set(payload.keys()) != expected_root:
        raise ValueError("executive_report com estrutura raiz invalida.")

    metadata = payload["metadata"]
    expected_metadata = {"schema_version", "generated_at_utc", "run_id", "environment"}
    if set(metadata.keys()) != expected_metadata:
        raise ValueError("executive_report.metadata invalido.")

    kpis = payload["kpis"]
    if set(kpis.keys()) != REQUIRED_KPI_COLUMNS:
        raise ValueError("executive_report.kpis invalido.")


def validate_prioritization_contract(df: pd.DataFrame) -> None:
    _ensure_non_empty(df, "customer_prioritization")
    _ensure_columns(df, REQUIRED_PRIORITIZATION_COLUMNS, "customer_prioritization")
    _ensure_probability_bounds(df["churn_probability"], "churn_probability")
    _ensure_allowed_values(df["risk_segment"], {"high", "medium", "low"}, "risk_segment")
    if df["customerID"].astype(str).str.strip().eq("").any():
        raise ValueError("customer_prioritization com customerID vazio.")


def validate_kpi_contract(df: pd.DataFrame) -> None:
    _ensure_non_empty(df, "kpi_summary")
    _ensure_columns(df, REQUIRED_KPI_COLUMNS, "kpi_summary")
    if len(df) != 1:
        raise ValueError("kpi_summary deve ter exatamente 1 linha.")
    _ensure_probability_bounds(df["churn_rate"], "churn_rate")


def validate_action_playbook_contract(df: pd.DataFrame) -> None:
    _ensure_non_empty(df, "action_playbook")
    _ensure_columns(df, REQUIRED_ACTION_PLAYBOOK_COLUMNS, "action_playbook")
    if (df["Customers"] < 0).any():
        raise ValueError("action_playbook com Customers negativo.")

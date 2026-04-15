from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.modeling.predictor import ChurnPredictor
from src.pipelines.dashboard_data import (
    KPI_PATH,
    PRIORITIZATION_PATH,
    REPORT_PATH,
    load_executive_report,
    load_kpis,
    load_prioritization,
)
from src.runtime.config import PipelineConfig

DEFAULT_CONFIG = PipelineConfig.from_runtime(mlflow_tracking_uri="disabled", run_id="dashboard")
EDA_PROFILE_PATH = DEFAULT_CONFIG.eda_profile_path
EDA_REPORT_PATH = DEFAULT_CONFIG.eda_report_path
GOVERNANCE_PATH = DEFAULT_CONFIG.latest_governance_manifest_path
PUBLIC_PRIORITIZATION_PATH = DEFAULT_CONFIG.gold_dir / "customer_prioritization_public.csv"


@dataclass(frozen=True)
class DashboardRuntime:
    data_path: Path
    silver_path: Path
    bundle_path: Path

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "DashboardRuntime":
        return cls(
            data_path=config.raw_input_path,
            silver_path=config.silver_output_path,
            bundle_path=config.enterprise_bundle_path,
        )


@dataclass(frozen=True)
class SidebarState:
    dataframe: pd.DataFrame | None
    predictor: ChurnPredictor | None
    model_loaded: bool


@dataclass(frozen=True)
class DashboardAssets:
    report: dict[str, Any]
    kpis: pd.DataFrame
    prioritization: pd.DataFrame
    report_path: Path
    kpi_path: Path
    prioritization_path: Path
    eda_profile_path: Path = EDA_PROFILE_PATH
    eda_report_path: Path = EDA_REPORT_PATH
    governance_path: Path = GOVERNANCE_PATH
    public_prioritization_path: Path = PUBLIC_PRIORITIZATION_PATH

    @property
    def is_ready(self) -> bool:
        return bool(self.report) and not self.kpis.empty and not self.prioritization.empty

    @property
    def report_metadata(self) -> dict[str, Any]:
        return self.report.get("metadata", {}) if self.report else {}

    @property
    def is_fallback(self) -> bool:
        run_id = str(self.report_metadata.get("run_id", ""))
        return run_id == "dashboard-fallback"

    @property
    def generated_at_utc(self) -> str:
        return str(self.report_metadata.get("generated_at_utc", "unknown"))

    @property
    def environment(self) -> str:
        return str(self.report_metadata.get("environment", "unknown"))

    @property
    def schema_version(self) -> str:
        return str(self.report_metadata.get("schema_version", "unknown"))

    @property
    def eda_ready(self) -> bool:
        return self.eda_profile_path.exists() and self.eda_report_path.exists()

    @property
    def governance_ready(self) -> bool:
        return self.governance_path.exists() and self.public_prioritization_path.exists()


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df_loaded = pd.read_csv(path)
    if "TotalCharges" in df_loaded.columns:
        df_loaded["TotalCharges"] = pd.to_numeric(df_loaded["TotalCharges"], errors="coerce")
        df_loaded["TotalCharges"] = df_loaded["TotalCharges"].fillna(
            df_loaded["TotalCharges"].median()
        )
    return df_loaded


@st.cache_resource(show_spinner=False)
def load_predictor(bundle_path: Path) -> ChurnPredictor:
    predictor = ChurnPredictor(bundle_path=bundle_path)
    predictor.load_artifacts()
    return predictor


@st.cache_data(show_spinner=False)
def load_dashboard_assets() -> DashboardAssets:
    return DashboardAssets(
        report=load_executive_report(),
        kpis=load_kpis(),
        prioritization=load_prioritization(),
        report_path=REPORT_PATH,
        kpi_path=KPI_PATH,
        prioritization_path=PRIORITIZATION_PATH,
    )


def load_best_available_dataframe(
    runtime: DashboardRuntime, assets: DashboardAssets
) -> tuple[pd.DataFrame | None, str]:
    if runtime.data_path.exists():
        return load_data(runtime.data_path), f"raw:{runtime.data_path.name}"

    if runtime.silver_path.exists():
        return load_data(runtime.silver_path), f"silver:{runtime.silver_path.name}"

    if not assets.prioritization.empty:
        return assets.prioritization.copy(), "prioritization:fallback"

    return None, "missing"


def build_dashboard_status(assets: DashboardAssets) -> dict[str, str | bool]:
    return {
        "ready": assets.is_ready,
        "fallback": assets.is_fallback,
        "eda_ready": assets.eda_ready,
        "governance_ready": assets.governance_ready,
        "environment": assets.environment,
        "schema_version": assets.schema_version,
        "generated_at_utc": assets.generated_at_utc,
        "run_id": str(assets.report_metadata.get("run_id", "unknown")),
    }


def normalize_filter_value(selected_value: str) -> str:
    return "All" if selected_value in {"All", "Todos"} else selected_value


def format_risk_level(risk_level: str) -> str:
    risk_map = {
        "alto": "High",
        "high": "High",
        "medio": "Medium",
        "médio": "Medium",
        "medium": "Medium",
        "baixo": "Low",
        "low": "Low",
    }
    return risk_map.get(risk_level.strip().lower(), risk_level)


def build_prediction_payload(
    *,
    gender: str,
    senior: int,
    partner: str,
    dependents: str,
    tenure: int,
    phone_service: str,
    internet_service: str,
    contract: str,
    paperless: str,
    payment: str,
    monthly: float,
    total: float,
) -> dict[str, object]:
    return {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def summarise_metrics(df: pd.DataFrame) -> dict[str, Any]:
    churn_rate = 0.0
    if "Churn" in df.columns and len(df) > 0:
        churn_rate = (df["Churn"].value_counts().get("Yes", 0) / len(df)) * 100
    return {
        "total_customers": len(df),
        "churn_rate": churn_rate,
        "avg_monthly": df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else 0.0,
        "avg_tenure": df["tenure"].mean() if "tenure" in df.columns else 0.0,
    }


def build_filtered_views(
    df: pd.DataFrame,
    selected_contract: str,
    selected_internet: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_contract = normalize_filter_value(selected_contract)
    normalized_internet = normalize_filter_value(selected_internet)

    left_chart_df = df.copy()
    if normalized_contract != "All" and "Contract" in left_chart_df.columns:
        left_chart_df = left_chart_df[left_chart_df["Contract"] == normalized_contract]

    right_chart_df = df.copy()
    if normalized_internet != "All" and "InternetService" in right_chart_df.columns:
        right_chart_df = right_chart_df[right_chart_df["InternetService"] == normalized_internet]

    preview_df = df.copy()
    if normalized_contract != "All" and "Contract" in preview_df.columns:
        preview_df = preview_df[preview_df["Contract"] == normalized_contract]
    if normalized_internet != "All" and "InternetService" in preview_df.columns:
        preview_df = preview_df[preview_df["InternetService"] == normalized_internet]

    return left_chart_df, right_chart_df, preview_df


def build_portfolio_summary(prioritization: pd.DataFrame) -> dict[str, Any]:
    if prioritization.empty:
        return {
            "high_risk_customers": 0,
            "high_risk_revenue": 0.0,
            "month_to_month_share": 0.0,
            "avg_next_purchase": 0.0,
        }

    high_risk = prioritization[prioritization["risk_segment"].eq("high")]
    month_to_month_share = float(prioritization["Contract"].eq("Month-to-month").mean()) * 100
    return {
        "high_risk_customers": int(len(high_risk)),
        "high_risk_revenue": float(high_risk["MonthlyCharges"].sum()),
        "month_to_month_share": month_to_month_share,
        "avg_next_purchase": float(prioritization["next_purchase_prediction"].mean()),
    }


def build_risk_distribution(prioritization: pd.DataFrame) -> pd.DataFrame:
    if prioritization.empty or "risk_segment" not in prioritization.columns:
        return pd.DataFrame(columns=["risk_segment", "customers"])

    risk_order = ["high", "medium", "low"]
    distribution = (
        prioritization["risk_segment"]
        .value_counts()
        .rename_axis("risk_segment")
        .reset_index(name="customers")
    )
    distribution["risk_segment"] = pd.Categorical(
        distribution["risk_segment"], categories=risk_order, ordered=True
    )
    return distribution.sort_values("risk_segment").reset_index(drop=True)


def simulate_retention_impact(
    prioritization: pd.DataFrame,
    retention_effectiveness: int,
    risk_threshold: float = 0.7,
) -> dict[str, float]:
    if prioritization.empty:
        return {
            "baseline_revenue_risk": 0.0,
            "recovered_revenue": 0.0,
            "remaining_revenue_risk": 0.0,
        }

    baseline_revenue_risk = float(
        prioritization.loc[
            prioritization["churn_probability"] >= risk_threshold,
            "MonthlyCharges",
        ].sum()
    )
    recovered = baseline_revenue_risk * (retention_effectiveness / 100)
    remaining = baseline_revenue_risk - recovered
    return {
        "baseline_revenue_risk": baseline_revenue_risk,
        "recovered_revenue": recovered,
        "remaining_revenue_risk": remaining,
    }

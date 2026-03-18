from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.modeling.predictor import ChurnPredictor
from src.runtime.config import PipelineConfig


@dataclass(frozen=True)
class DashboardRuntime:
    data_path: Path
    bundle_path: Path

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "DashboardRuntime":
        return cls(data_path=config.raw_input_path, bundle_path=config.enterprise_bundle_path)


@dataclass(frozen=True)
class SidebarState:
    dataframe: pd.DataFrame | None
    predictor: ChurnPredictor | None
    model_loaded: bool


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
    left_chart_df = df.copy()
    if selected_contract != "Todos" and "Contract" in left_chart_df.columns:
        left_chart_df = left_chart_df[left_chart_df["Contract"] == selected_contract]

    right_chart_df = df.copy()
    if selected_internet != "Todos" and "InternetService" in right_chart_df.columns:
        right_chart_df = right_chart_df[right_chart_df["InternetService"] == selected_internet]

    preview_df = df.copy()
    if selected_contract != "Todos" and "Contract" in preview_df.columns:
        preview_df = preview_df[preview_df["Contract"] == selected_contract]
    if selected_internet != "Todos" and "InternetService" in preview_df.columns:
        preview_df = preview_df[preview_df["InternetService"] == selected_internet]

    return left_chart_df, right_chart_df, preview_df

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.runtime.config import PipelineConfig
from src.utils.io import write_json_atomic

IDENTIFIER_COLUMNS = {"customerID"}
SENSITIVE_COLUMNS = {"SeniorCitizen"}


def pseudonymize_value(value: object, *, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}::{value}".encode("utf-8")).hexdigest()
    return digest[:16]


def build_public_prioritization_view(
    recommendations: pd.DataFrame, *, salt: str, strict_mode: bool
) -> pd.DataFrame:
    public_df = recommendations.copy()
    public_df["customer_token"] = public_df["customerID"].map(
        lambda value: pseudonymize_value(value, salt=salt)
    )
    if strict_mode:
        public_df = public_df.drop(columns=["customerID"])
    return public_df


def build_governance_manifest(
    config: PipelineConfig,
    silver_df: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> dict[str, Any]:
    available_columns = set(silver_df.columns)
    identifiers = sorted(IDENTIFIER_COLUMNS & available_columns)
    sensitive = sorted(SENSITIVE_COLUMNS & available_columns)
    strict_mode = config.lgpd_mode == "strict"

    manifest = {
        "schema_version": "1.0.0",
        "framework": "LGPD",
        "run_id": config.run_id,
        "environment": config.environment,
        "lawful_basis": "legitimate_interest",
        "retention_days": config.governance_retention_days,
        "privacy_mode": config.lgpd_mode,
        "dataset_classification": {
            "identifier_columns": identifiers,
            "sensitive_columns": sensitive,
            "behavioral_columns": sorted(
                column
                for column in [
                    "Contract",
                    "InternetService",
                    "MonthlyCharges",
                    "TotalCharges",
                    "Churn",
                ]
                if column in available_columns
            ),
        },
        "controls": {
            "data_minimization": True,
            "pseudonymization_applied": True,
            "strict_mode_identifier_removed": strict_mode,
        },
        "artifacts": {
            "governance_manifest": str(config.governance_manifest_path),
            "public_prioritization": str(config.gold_dir / "customer_prioritization_public.csv"),
            "restricted_prioritization": str(config.gold_dir / "customer_prioritization.csv"),
        },
        "counts": {
            "silver_rows": int(len(silver_df)),
            "prioritization_rows": int(len(recommendations)),
        },
    }
    return manifest


def persist_governance_manifest(config: PipelineConfig, manifest: dict[str, Any]) -> None:
    write_json_atomic(config.governance_manifest_path, manifest)
    write_json_atomic(config.latest_governance_manifest_path, manifest)

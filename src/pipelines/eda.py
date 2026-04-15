from __future__ import annotations

from typing import Any

import pandas as pd

from src.runtime.config import PipelineConfig
from src.utils.io import write_json_atomic, write_text_atomic


def build_eda_profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    numeric_summary: dict[str, dict[str, float]] = {}
    if numeric_columns:
        described = df[numeric_columns].describe().transpose()
        for column, row in described.iterrows():
            numeric_summary[column] = {
                "mean": float(row.get("mean", 0.0)),
                "std": float(row.get("std", 0.0)),
                "min": float(row.get("min", 0.0)),
                "max": float(row.get("max", 0.0)),
            }

    top_categories: dict[str, list[dict[str, object]]] = {}
    for column in categorical_columns:
        counts = df[column].astype(str).value_counts(dropna=False).head(5)
        top_categories[column] = [
            {"value": value, "count": int(count)} for value, count in counts.items()
        ]

    missingness = {
        column: round(float(df[column].isna().mean()), 4)
        for column in df.columns
        if df[column].isna().any()
    }

    churn_by_contract: list[dict[str, object]] = []
    if {"Contract", "Churn"}.issubset(df.columns):
        grouped = (
            df.groupby("Contract", dropna=False)["Churn"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        churn_by_contract = [
            {"contract": str(row["Contract"]), "churn_rate": float(row["Churn"])}
            for _, row in grouped.iterrows()
        ]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missingness_rate": missingness,
        "numeric_summary": numeric_summary,
        "top_categories": top_categories,
        "churn_by_contract": churn_by_contract,
    }


def render_eda_report(profile: dict[str, Any]) -> str:
    top_missing = sorted(
        profile["missingness_rate"].items(), key=lambda item: item[1], reverse=True
    )[:5]
    top_missing_rows = (
        "\n".join(f"- {name}: {rate:.2%}" for name, rate in top_missing)
        if top_missing
        else "- none"
    )
    churn_rows = (
        "\n".join(
            f"- {item['contract']}: {item['churn_rate']:.2%}"
            for item in profile["churn_by_contract"]
        )
        if profile["churn_by_contract"]
        else "- unavailable"
    )
    return f"""# EDA Report

## Dataset Shape
- Rows: {profile['rows']}
- Columns: {profile['columns']}
- Numeric columns: {len(profile['numeric_columns'])}
- Categorical columns: {len(profile['categorical_columns'])}

## Missingness (Top 5)
{top_missing_rows}

## Churn Rate by Contract
{churn_rows}
"""


def persist_eda_outputs(config: PipelineConfig, profile: dict[str, Any]) -> None:
    write_json_atomic(config.eda_profile_path, profile)
    write_text_atomic(config.eda_report_path, render_eda_report(profile))

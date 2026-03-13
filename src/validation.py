from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.transformation import REQUIRED_COLUMNS


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    columns: int
    duplicate_customer_ids: int
    missing_total_charges: int
    invalid_churn_labels: int
    churn_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def validate_raw_dataframe(df: pd.DataFrame) -> ValidationReport:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Schema invalido: colunas obrigatorias ausentes: {missing}")

    duplicate_customer_ids = int(df["customerID"].duplicated().sum())
    if duplicate_customer_ids > 0:
        raise ValueError(f"Schema invalido: {duplicate_customer_ids} customerIDs duplicados.")

    invalid_churn_labels = int((~df["Churn"].isin(["Yes", "No"])).sum())
    if invalid_churn_labels > 0:
        raise ValueError(
            f"Schema invalido: {invalid_churn_labels} registros com Churn fora de Yes/No."
        )

    missing_total_charges = int(pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum())
    churn_rate = float(df["Churn"].eq("Yes").mean())

    return ValidationReport(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        duplicate_customer_ids=duplicate_customer_ids,
        missing_total_charges=missing_total_charges,
        invalid_churn_labels=invalid_churn_labels,
        churn_rate=churn_rate,
    )


def validate_training_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset de treino vazio apos a preparacao.")
    if df["Churn"].nunique() < 2:
        raise ValueError("Target invalido: e necessario haver churners e non-churners.")
    if (df["tenure"] < 0).any():
        raise ValueError("Coluna tenure contem valores negativos.")
    if (df["MonthlyCharges"] < 0).any() or (df["TotalCharges"] < 0).any():
        raise ValueError("Colunas de cobranca contem valores negativos.")

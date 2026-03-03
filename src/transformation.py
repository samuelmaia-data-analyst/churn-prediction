from __future__ import annotations

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover

    class _Check:
        @staticmethod
        def isin(_values):
            return None

        @staticmethod
        def greater_than_or_equal_to(_value):
            return None

    class _Schema:
        def validate(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    class _PanderaFallback:
        Check = _Check

        @staticmethod
        def Column(*_args, **_kwargs):
            return None

        @staticmethod
        def DataFrameSchema(*_args, **_kwargs):
            return _Schema()

    pa = _PanderaFallback()

from src.config import PipelineConfig

REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

BRONZE_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, nullable=False),
        "SeniorCitizen": pa.Column(int, nullable=False),
        "Partner": pa.Column(str, nullable=False),
        "Dependents": pa.Column(str, nullable=False),
        "tenure": pa.Column(int, nullable=False),
        "PhoneService": pa.Column(str, nullable=False),
        "MultipleLines": pa.Column(str, nullable=False),
        "InternetService": pa.Column(str, nullable=False),
        "OnlineSecurity": pa.Column(str, nullable=False),
        "OnlineBackup": pa.Column(str, nullable=False),
        "DeviceProtection": pa.Column(str, nullable=False),
        "TechSupport": pa.Column(str, nullable=False),
        "StreamingTV": pa.Column(str, nullable=False),
        "StreamingMovies": pa.Column(str, nullable=False),
        "Contract": pa.Column(str, nullable=False),
        "PaperlessBilling": pa.Column(str, nullable=False),
        "PaymentMethod": pa.Column(str, nullable=False),
        "MonthlyCharges": pa.Column(float, nullable=False),
        "TotalCharges": pa.Column(object, nullable=True),
        "Churn": pa.Column(str, checks=pa.Check.isin(["Yes", "No"]), nullable=False),
        "ingested_at": pa.Column(str, nullable=False),
        "source_system": pa.Column(str, nullable=False),
    },
    strict=False,
)

SILVER_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, nullable=False),
        "SeniorCitizen": pa.Column(int, nullable=False),
        "Partner": pa.Column(str, nullable=False),
        "Dependents": pa.Column(str, nullable=False),
        "tenure": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0), nullable=False),
        "PhoneService": pa.Column(str, nullable=False),
        "MultipleLines": pa.Column(str, nullable=False),
        "InternetService": pa.Column(str, nullable=False),
        "OnlineSecurity": pa.Column(str, nullable=False),
        "OnlineBackup": pa.Column(str, nullable=False),
        "DeviceProtection": pa.Column(str, nullable=False),
        "TechSupport": pa.Column(str, nullable=False),
        "StreamingTV": pa.Column(str, nullable=False),
        "StreamingMovies": pa.Column(str, nullable=False),
        "Contract": pa.Column(str, nullable=False),
        "PaperlessBilling": pa.Column(str, nullable=False),
        "PaymentMethod": pa.Column(str, nullable=False),
        "MonthlyCharges": pa.Column(
            float, checks=pa.Check.greater_than_or_equal_to(0), nullable=False
        ),
        "TotalCharges": pa.Column(
            float, checks=pa.Check.greater_than_or_equal_to(0), nullable=False
        ),
        "Churn": pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
        "ingested_at": pa.Column(str, nullable=False),
        "source_system": pa.Column(str, nullable=False),
    },
    strict=False,
)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Colunas obrigatorias ausentes: {missing_str}")


def build_silver_layer(bronze_df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(bronze_df)
    BRONZE_SCHEMA.validate(bronze_df)
    silver = bronze_df.copy()

    silver["TotalCharges"] = pd.to_numeric(silver["TotalCharges"], errors="coerce")
    silver["TotalCharges"] = silver["TotalCharges"].fillna(silver["TotalCharges"].median())
    silver["Churn"] = silver["Churn"].map({"Yes": 1, "No": 0})

    if silver["Churn"].isna().any():
        raise ValueError("Coluna Churn contem valores invalidos. Esperado: Yes/No.")

    SILVER_SCHEMA.validate(silver)
    return silver


def persist_silver(config: PipelineConfig, silver_df: pd.DataFrame) -> None:
    config.silver_dir.mkdir(parents=True, exist_ok=True)
    silver_df.to_csv(config.silver_output_path, index=False)

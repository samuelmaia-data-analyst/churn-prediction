from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import PipelineConfig


@dataclass(frozen=True)
class StarSchema:
    dim_customer: pd.DataFrame
    dim_contract: pd.DataFrame
    dim_service: pd.DataFrame
    fact_customer_churn: pd.DataFrame


def build_star_schema(silver_df: pd.DataFrame) -> StarSchema:
    dim_customer = (
        silver_df[
            [
                "customerID",
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_customer["customer_sk"] = dim_customer.index + 1

    dim_contract = (
        silver_df[["Contract", "PaperlessBilling", "PaymentMethod"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_contract["contract_sk"] = dim_contract.index + 1

    dim_service = (
        silver_df[
            [
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_service["service_sk"] = dim_service.index + 1

    fact = silver_df[
        [
            "customerID",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "MonthlyCharges",
            "TotalCharges",
            "tenure",
            "Churn",
        ]
    ].copy()

    fact = fact.merge(dim_customer[["customerID", "customer_sk"]], on="customerID", how="left")
    fact = fact.merge(
        dim_contract[["Contract", "PaperlessBilling", "PaymentMethod", "contract_sk"]],
        on=["Contract", "PaperlessBilling", "PaymentMethod"],
        how="left",
    )
    fact = fact.merge(
        dim_service[
            [
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "service_sk",
            ]
        ],
        on=[
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ],
        how="left",
    )

    fact_customer_churn = fact[
        [
            "customer_sk",
            "contract_sk",
            "service_sk",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]
    ].rename(columns={"Churn": "churn_label"})

    return StarSchema(
        dim_customer=dim_customer,
        dim_contract=dim_contract,
        dim_service=dim_service,
        fact_customer_churn=fact_customer_churn,
    )


def persist_star_schema(config: PipelineConfig, schema: StarSchema) -> None:
    config.gold_dir.mkdir(parents=True, exist_ok=True)
    schema.dim_customer.to_csv(config.gold_dir / "dim_customer.csv", index=False)
    schema.dim_contract.to_csv(config.gold_dir / "dim_contract.csv", index=False)
    schema.dim_service.to_csv(config.gold_dir / "dim_service.csv", index=False)
    schema.fact_customer_churn.to_csv(config.gold_dir / "fact_customer_churn.csv", index=False)

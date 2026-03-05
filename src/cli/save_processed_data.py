from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_PATH = Path("data/processed/telco_churn_processed.csv")


def main() -> None:
    print("Starting data processing...")

    if not RAW_PATH.parent.exists():
        raise FileNotFoundError(
            "Missing 'data/raw' directory. Expected CSV at "
            "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing input dataset: {RAW_PATH}")

    print(f"Loading raw data from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    print(f"After cleaning: {df.shape[0]} rows")

    columns_to_keep = [
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
    ]
    df = df[columns_to_keep]

    sample_df = df.sample(n=min(2000, len(df)), random_state=42)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved processed sample to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

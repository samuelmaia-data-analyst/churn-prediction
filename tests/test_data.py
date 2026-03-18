from __future__ import annotations

import pandas as pd
import pytest

from src.pipelines.feature_engineering import engineer_features
from src.pipelines.ingestion import build_bronze_layer
from src.pipelines.transformation import build_silver_layer
from src.pipelines.validation import validate_raw_dataframe, validate_training_dataframe


def build_raw_df(rows: int = 40) -> pd.DataFrame:
    records = []
    for i in range(rows):
        records.append(
            {
                "customerID": f"CUST-{i:03d}",
                "gender": "Male" if i % 2 == 0 else "Female",
                "SeniorCitizen": i % 2,
                "Partner": "Yes" if i % 3 else "No",
                "Dependents": "No",
                "tenure": 5 + i,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if i % 2 == 0 else "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if i % 2 == 0 else "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 40.0 + i,
                "TotalCharges": " " if i == 0 else str((40.0 + i) * (5 + i)),
                "Churn": "Yes" if i % 2 == 0 else "No",
            }
        )
    return pd.DataFrame(records)


def test_validation_and_silver_layer_keep_training_dataset_reliable() -> None:
    raw = build_raw_df()

    report = validate_raw_dataframe(raw)
    bronze = build_bronze_layer(raw)
    silver = build_silver_layer(bronze)
    validate_training_dataframe(silver)

    assert report.rows == len(raw)
    assert report.invalid_churn_labels == 0
    assert "customerID" in silver.columns
    assert silver["TotalCharges"].dtype.kind in {"f", "i"}
    assert silver["Churn"].isin([0, 1]).all()


def test_validate_raw_dataframe_rejects_invalid_target_label() -> None:
    raw = build_raw_df()
    raw.loc[0, "Churn"] = "Maybe"

    with pytest.raises(ValueError, match="Churn"):
        validate_raw_dataframe(raw)


def test_engineer_features_adds_business_features() -> None:
    silver = build_silver_layer(build_bronze_layer(build_raw_df()))
    featured = engineer_features(silver)

    assert {"charges_per_tenure", "service_count", "is_month_to_month", "tenure_band"}.issubset(
        featured.columns
    )
    assert featured["service_count"].ge(0).all()

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.compat.dataset_export import export_processed_dataset_legacy
from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer


def test_legacy_preprocessor_wrapper_uses_compat_layer() -> None:
    preprocessor = DataPreprocessor()
    raw_df = pd.DataFrame(
        [
            {
                "customerID": "CUST-001",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0,
                "TotalCharges": "500.0",
                "Churn": "No",
            }
        ]
    )

    cleaned = preprocessor.clean_data(raw_df)

    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].isin([0, 1]).all()


def test_legacy_feature_engineer_wrapper_uses_compat_layer() -> None:
    feature_engineer = FeatureEngineer()
    X_train = pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
            }
        ]
    )
    X_test = X_train.copy()

    train_transformed, test_transformed = feature_engineer.fit_transform(X_train, X_test)

    assert not train_transformed.empty
    assert train_transformed.shape[1] == test_transformed.shape[1]


def test_legacy_dataset_export_uses_canonical_exporter(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw_dir.joinpath(source_dataset.name).write_bytes(source_dataset.read_bytes())

    output_path = tmp_path / "processed" / "legacy.csv"
    exported = export_processed_dataset_legacy(data_dir=data_dir, output_path=output_path)

    assert exported.exists()

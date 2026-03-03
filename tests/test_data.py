from __future__ import annotations

import pandas as pd

from src.data.preprocess import DataPreprocessor


def build_raw_df() -> pd.DataFrame:
    rows = []
    for i in range(40):
        rows.append(
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
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 40.0 + i,
                "TotalCharges": " " if i == 0 else str((40.0 + i) * (5 + i)),
                "Churn": "Yes" if i % 2 == 0 else "No",
            }
        )
    return pd.DataFrame(rows)


def test_clean_data_maps_target_and_totalcharges() -> None:
    preprocessor = DataPreprocessor()
    raw = build_raw_df()

    clean = preprocessor.clean_data(raw)

    assert "customerID" not in clean.columns
    assert clean["TotalCharges"].dtype.kind in {"f", "i"}
    assert clean["Churn"].isin([0, 1]).all()


def test_split_data_returns_train_and_test() -> None:
    preprocessor = DataPreprocessor()
    clean = preprocessor.clean_data(build_raw_df())

    X_train, X_test, y_train, y_test = preprocessor.split_data(clean)

    assert len(X_train) > len(X_test)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

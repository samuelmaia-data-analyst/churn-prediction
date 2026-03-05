from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.predict_model import ChurnPredictor
from src.models.train_model import ModelTrainer
from tests.test_data import build_raw_df

FEATURES = [
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
]


def build_trained_objects():
    raw = build_raw_df()
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
    X = raw[FEATURES]
    y = raw["Churn"].map({"Yes": 1, "No": 0})

    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [f for f in FEATURES if f not in numeric]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    transformed = preprocessor.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(transformed, y)

    return model, preprocessor


def test_model_trainer_selects_best_model() -> None:
    trainer = ModelTrainer()
    data_preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    cleaned = data_preprocessor.clean_data(build_raw_df())
    X_train, X_test, y_train, y_test = data_preprocessor.split_data(cleaned)
    X_train_proc, X_test_proc = feature_engineer.fit_transform(X_train, X_test)

    results = trainer.train_all(X_train_proc, y_train, X_test_proc, y_test)

    assert trainer.best_model in results
    assert "f1" in results[trainer.best_model]


def test_churn_predictor_returns_business_payload() -> None:
    model, preprocessor = build_trained_objects()
    predictor = ChurnPredictor(model=model, preprocessor=preprocessor)

    payload = build_raw_df().iloc[1][FEATURES].to_dict()
    payload["TotalCharges"] = float(payload["TotalCharges"])
    result = predictor.predict_from_dict(payload)

    assert result.churn in {"Sim", "Não"}
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_level in {"Baixo", "Médio", "Alto"}

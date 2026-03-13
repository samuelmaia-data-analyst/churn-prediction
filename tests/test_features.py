from __future__ import annotations

from src.feature_engineering import MODEL_FEATURES, engineer_features
from src.modeling.churn import build_churn_models
from tests.test_data import build_raw_df
from src.ingestion import build_bronze_layer
from src.transformation import build_silver_layer


def test_model_pipeline_accepts_raw_business_features_and_engineers_the_rest() -> None:
    silver = build_silver_layer(build_bronze_layer(build_raw_df(80)))
    X = silver.drop(columns=["Churn"])
    y = silver["Churn"]
    models, _ = build_churn_models(seed=42)

    model = models["Logistic"]
    model.fit(X, y)
    probabilities = model.predict_proba(X.head(5))[:, 1]

    assert len(probabilities) == 5
    assert all(0.0 <= score <= 1.0 for score in probabilities)


def test_engineered_feature_space_contains_expected_model_columns() -> None:
    silver = build_silver_layer(build_bronze_layer(build_raw_df(20)))
    featured = engineer_features(silver)

    assert set(MODEL_FEATURES).issubset(featured.columns)

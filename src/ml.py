from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PipelineConfig

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # pragma: no cover
    mlflow = None


@dataclass(frozen=True)
class ModelOutputs:
    scored_df: pd.DataFrame
    metrics: dict[str, float]


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

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [feature for feature in FEATURES if feature not in NUMERIC_FEATURES]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _add_next_purchase_target(df: pd.DataFrame) -> pd.Series:
    contract_factor = np.where(df["Contract"].eq("Month-to-month"), 1.04, 1.015)
    service_factor = np.where(df["InternetService"].eq("Fiber optic"), 1.025, 1.0)
    loyalty_factor = np.where(df["tenure"] >= 24, 1.01, 1.0)
    return df["MonthlyCharges"] * contract_factor * service_factor * loyalty_factor


def train_models_and_score(config: PipelineConfig, silver_df: pd.DataFrame) -> ModelOutputs:
    X = silver_df[FEATURES].copy()
    y_churn = silver_df["Churn"].astype(int)
    y_next_purchase = _add_next_purchase_target(silver_df)

    preprocessor = _build_preprocessor()
    churn_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=250, random_state=config.seed)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_churn,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y_churn,
    )
    churn_model.fit(X_train, y_train)
    churn_pred = churn_model.predict(X_test)
    churn_prob = churn_model.predict_proba(X_test)[:, 1]

    next_purchase_model = Pipeline(
        steps=[
            ("prep", _build_preprocessor()),
            ("reg", RandomForestRegressor(n_estimators=200, random_state=config.seed)),
        ]
    )
    X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(
        X, y_next_purchase, test_size=config.test_size, random_state=config.seed
    )
    next_purchase_model.fit(X_np_train, y_np_train)
    np_pred_test = next_purchase_model.predict(X_np_test)

    scored = silver_df.copy()
    scored["churn_probability"] = churn_model.predict_proba(X)[:, 1]
    scored["next_purchase_prediction"] = next_purchase_model.predict(X)

    metrics = {
        "churn_f1": float(f1_score(y_test, churn_pred)),
        "churn_roc_auc": float(roc_auc_score(y_test, churn_prob)),
        "next_purchase_mae": float(mean_absolute_error(y_np_test, np_pred_test)),
    }

    config.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(churn_model, config.churn_model_path)
    joblib.dump(next_purchase_model, config.next_purchase_model_path)

    if mlflow is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        with mlflow.start_run(run_name="churn-enterprise-pipeline"):
            mlflow.log_param("seed", config.seed)
            mlflow.log_param("test_size", config.test_size)
            mlflow.log_metric("churn_f1", metrics["churn_f1"])
            mlflow.log_metric("churn_roc_auc", metrics["churn_roc_auc"])
            mlflow.log_metric("next_purchase_mae", metrics["next_purchase_mae"])
            mlflow.sklearn.log_model(churn_model, artifact_path="churn_model")
            mlflow.sklearn.log_model(next_purchase_model, artifact_path="next_purchase_model")

    return ModelOutputs(scored_df=scored, metrics=metrics)

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import PipelineConfig
from src.contracts import ExecutiveMetrics
from src.modeling.churn import (
    BUSINESS_FEATURE_NAMES,
    FEATURES,
    MODEL_DISPLAY_NAMES,
    PIPELINE_VISUAL,
    XGBClassifier,
    add_next_purchase_target,
    build_churn_models,
    build_preprocessor,
    month_to_month_risk_ratio,
    top_feature_drivers,
)

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # pragma: no cover
    mlflow = None


@dataclass(frozen=True)
class ModelOutputs:
    scored_df: pd.DataFrame
    metrics: dict[str, object]


def _build_metrics_payload(
    y_test: pd.Series,
    churn_pred: np.ndarray,
    churn_prob: np.ndarray,
    y_np_test: pd.Series,
    np_pred_test: np.ndarray,
    comparison_rows: list[dict[str, float | str]],
    top_drivers: list[tuple[str, float]],
    selected_model_name: str,
    risk_ratio: float,
) -> dict[str, object]:
    sorted_comparison = sorted(comparison_rows, key=lambda row: float(row["roc_auc"]), reverse=True)
    baseline_auc = float(
        next(row["roc_auc"] for row in sorted_comparison if row["model"] == MODEL_DISPLAY_NAMES["Logistic"])
    )
    metrics = ExecutiveMetrics(
        churn_f1=float(f1_score(y_test, churn_pred)),
        churn_roc_auc=float(roc_auc_score(y_test, churn_prob)),
        next_purchase_mae=float(mean_absolute_error(y_np_test, np_pred_test)),
        baseline_model={"name": MODEL_DISPLAY_NAMES["Logistic"], "roc_auc": baseline_auc},
        model_comparison=[
            {"model": str(row["model"]), "roc_auc": float(row["roc_auc"])}
            for row in sorted_comparison
        ],
        selected_churn_model=selected_model_name,
        feature_importance=[
            {"feature": feature, "importance": float(importance)} for feature, importance in top_drivers
        ],
        top_drivers_of_churn=[BUSINESS_FEATURE_NAMES.get(feature, feature) for feature, _ in top_drivers],
        key_insights=[f"Customers with month-to-month contracts show {risk_ratio:.1f}x higher churn risk."],
        pipeline_visual=PIPELINE_VISUAL,
        model_comparison_note=(
            "XGBoost unavailable; using GradientBoosting fallback." if XGBClassifier is None else None
        ),
    )
    return metrics.to_dict()


def train_models_and_score(config: PipelineConfig, silver_df: pd.DataFrame) -> ModelOutputs:
    X = silver_df[FEATURES].copy()
    y_churn = silver_df["Churn"].astype(int)
    y_next_purchase = add_next_purchase_target(silver_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_churn,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y_churn,
    )

    churn_models, model_aliases = build_churn_models(config.seed)
    comparison_rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_key, churn_model in churn_models.items():
        churn_model.fit(X_train, y_train)
        fitted_models[model_key] = churn_model
        churn_prob = churn_model.predict_proba(X_test)[:, 1]
        churn_auc = float(roc_auc_score(y_test, churn_prob))
        comparison_rows.append(
            {"model_key": model_key, "model": model_aliases[model_key], "roc_auc": churn_auc}
        )

    best_key = max(comparison_rows, key=lambda row: row["roc_auc"])["model_key"]
    champion_model = fitted_models[best_key]
    churn_pred = champion_model.predict(X_test)
    churn_prob = champion_model.predict_proba(X_test)[:, 1]

    next_purchase_model = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("reg", RandomForestRegressor(n_estimators=200, random_state=config.seed)),
        ]
    )
    X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(
        X, y_next_purchase, test_size=config.test_size, random_state=config.seed
    )
    next_purchase_model.fit(X_np_train, y_np_train)
    np_pred_test = next_purchase_model.predict(X_np_test)

    scored = silver_df.copy()
    scored["churn_probability"] = champion_model.predict_proba(X)[:, 1]
    scored["next_purchase_prediction"] = next_purchase_model.predict(X)

    top_drivers = top_feature_drivers(champion_model, top_n=3)
    risk_ratio = month_to_month_risk_ratio(silver_df)

    metrics = _build_metrics_payload(
        y_test=y_test,
        churn_pred=churn_pred,
        churn_prob=churn_prob,
        y_np_test=y_np_test,
        np_pred_test=np_pred_test,
        comparison_rows=comparison_rows,
        top_drivers=top_drivers,
        selected_model_name=model_aliases[best_key],
        risk_ratio=risk_ratio,
    )

    config.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, config.churn_model_path)
    joblib.dump(next_purchase_model, config.next_purchase_model_path)

    if mlflow is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        with mlflow.start_run(run_name="churn-enterprise-pipeline"):
            mlflow.log_param("seed", config.seed)
            mlflow.log_param("test_size", config.test_size)
            mlflow.log_metric("churn_f1", metrics["churn_f1"])
            mlflow.log_metric("churn_roc_auc", metrics["churn_roc_auc"])
            mlflow.log_metric("next_purchase_mae", metrics["next_purchase_mae"])
            mlflow.sklearn.log_model(champion_model, artifact_path="churn_model")
            mlflow.sklearn.log_model(next_purchase_model, artifact_path="next_purchase_model")

    return ModelOutputs(scored_df=scored, metrics=metrics)

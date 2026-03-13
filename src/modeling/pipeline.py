from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import PipelineConfig
from src.contracts import ExecutiveMetrics
from src.decisioning import DecisionPolicy, decision_threshold, get_policy
from src.feature_engineering import BASE_MODEL_FEATURES
from src.modeling.churn import (
    BUSINESS_FEATURE_NAMES,
    FEATURES,
    MODEL_DISPLAY_NAMES,
    PIPELINE_VISUAL,
    XGBClassifier,
    add_next_purchase_target,
    build_churn_models,
    build_feature_engineering_step,
    build_preprocessor,
    month_to_month_risk_ratio,
    top_feature_drivers,
)
from src.validation import validate_training_dataframe

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except ImportError:  # pragma: no cover
    mlflow = None
    infer_signature = None


@dataclass(frozen=True)
class ModelOutputs:
    scored_df: pd.DataFrame
    metrics: dict[str, object]


def _risk_profile_summary(scored_df: pd.DataFrame, top_n: int = 3) -> list[dict[str, object]]:
    grouped = (
        scored_df.groupby(["Contract", "InternetService"], dropna=False)
        .agg(
            customers=("customerID", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            churn_rate=("Churn", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
        )
        .reset_index()
        .sort_values("avg_churn_probability", ascending=False)
        .head(top_n)
    )
    return [
        {
            "contract": str(row["Contract"]),
            "internet_service": str(row["InternetService"]),
            "customers": int(row["customers"]),
            "avg_churn_probability": float(row["avg_churn_probability"]),
            "churn_rate": float(row["churn_rate"]),
            "avg_monthly_charges": float(row["avg_monthly_charges"]),
        }
        for _, row in grouped.iterrows()
    ]


def _build_metrics_payload(
    y_test: pd.Series,
    churn_pred: pd.Series,
    churn_prob: pd.Series,
    y_np_test: pd.Series,
    np_pred_test: pd.Series,
    comparison_rows: list[dict[str, float | str]],
    top_drivers: list[tuple[str, float]],
    selected_model_name: str,
    risk_ratio: float,
    decision_threshold: float,
    policy: DecisionPolicy,
    scored_df: pd.DataFrame,
) -> dict[str, object]:
    sorted_comparison = sorted(comparison_rows, key=lambda row: float(row["roc_auc"]), reverse=True)
    baseline_auc = float(
        next(
            row["roc_auc"]
            for row in sorted_comparison
            if row["model"] == MODEL_DISPLAY_NAMES["Logistic"]
        )
    )
    precision = float(precision_score(y_test, churn_pred, zero_division=0))
    recall = float(recall_score(y_test, churn_pred, zero_division=0))
    f1 = float(f1_score(y_test, churn_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, churn_prob))
    tn, fp, fn, tp = confusion_matrix(y_test, churn_pred).ravel()

    metrics = ExecutiveMetrics(
        churn_precision=precision,
        churn_recall=recall,
        churn_f1=f1,
        churn_roc_auc=roc_auc,
        next_purchase_mae=float(mean_absolute_error(y_np_test, np_pred_test)),
        baseline_model={"name": MODEL_DISPLAY_NAMES["Logistic"], "roc_auc": baseline_auc},
        model_comparison=[
            {"model": str(row["model"]), "roc_auc": float(row["roc_auc"])}
            for row in sorted_comparison
        ],
        selected_churn_model=selected_model_name,
        feature_importance=[
            {"feature": feature, "importance": float(importance)}
            for feature, importance in top_drivers
        ],
        top_drivers_of_churn=[
            BUSINESS_FEATURE_NAMES.get(feature, feature) for feature, _ in top_drivers
        ],
        key_insights=[
            f"Customers with month-to-month contracts show {risk_ratio:.1f}x higher churn risk."
        ],
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        decision_threshold=decision_threshold,
        decision_policy={
            "name": policy.name,
            "fp_cost": policy.fp_cost,
            "fn_cost": policy.fn_cost,
            "description": policy.description,
        },
        cost_interpretation={
            "false_positive": (
                "Contato de retencao aplicado a cliente sem risco real. "
                "Aumenta custo de campanha e ocupa capacidade operacional."
            ),
            "false_negative": (
                "Cliente com risco real nao acionado. "
                "Tende a gerar perda de receita e piora no churn evitavel."
            ),
            "business_tradeoff": (
                "Reducao do threshold aumenta recall e cobertura da carteira, "
                "mas tambem eleva custo operacional com mais falsos positivos."
            ),
        },
        risk_profiles=_risk_profile_summary(scored_df),
        pipeline_visual=PIPELINE_VISUAL,
        model_comparison_note=(
            "XGBoost unavailable; using GradientBoosting fallback."
            if XGBClassifier is None
            else None
        ),
    )
    return metrics.to_dict()


def train_models_and_score(config: PipelineConfig, silver_df: pd.DataFrame) -> ModelOutputs:
    validate_training_dataframe(silver_df)
    X = silver_df[BASE_MODEL_FEATURES].copy()
    y_churn = silver_df["Churn"].astype(int)
    y_next_purchase = add_next_purchase_target(silver_df)
    policy = get_policy(config.decision_policy)

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
    churn_prob = champion_model.predict_proba(X_test)[:, 1]
    classification_threshold = decision_threshold(policy)
    churn_pred = (churn_prob >= classification_threshold).astype(int)

    next_purchase_model = Pipeline(
        steps=[
            ("features", build_feature_engineering_step()),
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

    top_drivers = top_feature_drivers(champion_model, top_n=5)
    risk_ratio = month_to_month_risk_ratio(silver_df)
    metrics = _build_metrics_payload(
        y_test=y_test,
        churn_pred=pd.Series(churn_pred),
        churn_prob=pd.Series(churn_prob),
        y_np_test=y_np_test,
        np_pred_test=pd.Series(np_pred_test),
        comparison_rows=comparison_rows,
        top_drivers=top_drivers,
        selected_model_name=model_aliases[best_key],
        risk_ratio=risk_ratio,
        decision_threshold=classification_threshold,
        policy=policy,
        scored_df=scored,
    )

    logger.info(
        "model_selected algorithm=%s precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        model_aliases[best_key],
        metrics["churn_precision"],
        metrics["churn_recall"],
        metrics["churn_f1"],
        metrics["churn_roc_auc"],
    )

    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.model_registry_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, config.churn_model_path)
    joblib.dump(next_purchase_model, config.next_purchase_model_path)
    joblib.dump(champion_model, config.versioned_model_path)
    joblib.dump(
        {
            "model": champion_model,
            "next_purchase_model": next_purchase_model,
            "selected_model_name": model_aliases[best_key],
            "decision_threshold": classification_threshold,
            "decision_policy": policy.name,
            "base_feature_names": BASE_MODEL_FEATURES,
            "feature_names": FEATURES,
        },
        config.enterprise_bundle_path,
    )

    metadata = {
        "model_version": "v1.0.0",
        "model_file": str(config.versioned_model_path.as_posix()),
        "bundle_file": str(config.enterprise_bundle_path.as_posix()),
        "algorithm": model_aliases[best_key],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "base_features": BASE_MODEL_FEATURES,
        "features": FEATURES,
        "decision_policy": {
            "name": policy.name,
            "fp_cost": policy.fp_cost,
            "fn_cost": policy.fn_cost,
            "description": policy.description,
        },
        "decision_threshold": classification_threshold,
        "threshold_strategy": {"high_ltv": 0.65, "low_ltv": 0.80},
        "metrics": {
            "churn_precision": metrics["churn_precision"],
            "churn_recall": metrics["churn_recall"],
            "churn_f1": metrics["churn_f1"],
            "churn_roc_auc": metrics["churn_roc_auc"],
            "next_purchase_mae": metrics["next_purchase_mae"],
        },
    }
    with open(config.model_metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    if mlflow is not None and config.mlflow_tracking_uri.lower() != "disabled":
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        with mlflow.start_run(run_name="churn-enterprise-pipeline"):
            mlflow.log_param("seed", config.seed)
            mlflow.log_param("test_size", config.test_size)
            mlflow.log_param("decision_policy", policy.name)
            mlflow.log_param("decision_threshold", classification_threshold)
            mlflow.log_metric("churn_precision", metrics["churn_precision"])
            mlflow.log_metric("churn_recall", metrics["churn_recall"])
            mlflow.log_metric("churn_f1", metrics["churn_f1"])
            mlflow.log_metric("churn_roc_auc", metrics["churn_roc_auc"])
            mlflow.log_metric("next_purchase_mae", metrics["next_purchase_mae"])

            churn_input_example = X_train.head(5).copy()
            next_purchase_input_example = X_np_train.head(5).copy()

            if infer_signature is not None:
                churn_signature = infer_signature(
                    churn_input_example, champion_model.predict_proba(churn_input_example)
                )
                next_purchase_signature = infer_signature(
                    next_purchase_input_example,
                    next_purchase_model.predict(next_purchase_input_example),
                )
            else:  # pragma: no cover
                churn_signature = None
                next_purchase_signature = None

            mlflow.sklearn.log_model(
                champion_model,
                artifact_path="churn_model",
                input_example=churn_input_example,
                signature=churn_signature,
            )
            mlflow.sklearn.log_model(
                next_purchase_model,
                artifact_path="next_purchase_model",
                input_example=next_purchase_input_example,
                signature=next_purchase_signature,
            )

    return ModelOutputs(scored_df=scored, metrics=metrics)

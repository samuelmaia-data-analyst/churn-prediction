from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


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
MODEL_DISPLAY_NAMES = {
    "Logistic": "Logistic Regression",
    "RandomForest": "RandomForest",
    "XGBoost": "XGBoost",
}
BUSINESS_FEATURE_NAMES = {
    "Contract": "Contract type",
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
}
PIPELINE_VISUAL = "Raw -> Bronze -> Silver -> Gold"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def add_next_purchase_target(df: pd.DataFrame) -> pd.Series:
    contract_factor = np.where(df["Contract"].eq("Month-to-month"), 1.04, 1.015)
    service_factor = np.where(df["InternetService"].eq("Fiber optic"), 1.025, 1.0)
    loyalty_factor = np.where(df["tenure"] >= 24, 1.01, 1.0)
    return df["MonthlyCharges"] * contract_factor * service_factor * loyalty_factor


def build_churn_models(seed: int) -> tuple[dict[str, Pipeline], dict[str, str]]:
    models = {
        "Logistic": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("clf", LogisticRegression(max_iter=1500, random_state=seed)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("clf", RandomForestClassifier(n_estimators=250, random_state=seed)),
            ]
        ),
    }
    aliases = {
        "Logistic": MODEL_DISPLAY_NAMES["Logistic"],
        "RandomForest": MODEL_DISPLAY_NAMES["RandomForest"],
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=260,
                        max_depth=5,
                        learning_rate=0.06,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=seed,
                    ),
                ),
            ]
        )
        aliases["XGBoost"] = MODEL_DISPLAY_NAMES["XGBoost"]
    else:
        models["XGBoost"] = Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("clf", GradientBoostingClassifier(random_state=seed)),
            ]
        )
        aliases["XGBoost"] = "XGBoost (fallback: GradientBoosting)"

    return models, aliases


def top_feature_drivers(model: Pipeline, top_n: int = 3) -> list[tuple[str, float]]:
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    feature_names = prep.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        raw_importance = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        raw_importance = np.asarray(np.abs(clf.coef_)).ravel()
    else:
        return []

    feature_scores: dict[str, float] = {}
    for transformed_name, score in zip(feature_names, raw_importance):
        cleaned = transformed_name.split("__", 1)[-1]
        base_feature = cleaned.split("_", 1)[0]
        feature_scores[base_feature] = feature_scores.get(base_feature, 0.0) + float(score)

    ranked = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def month_to_month_risk_ratio(df: pd.DataFrame) -> float:
    month_mask = df["Contract"].eq("Month-to-month")
    other_mask = ~month_mask
    month_rate = float(df.loc[month_mask, "Churn"].mean()) if month_mask.any() else 0.0
    other_rate = float(df.loc[other_mask, "Churn"].mean()) if other_mask.any() else 0.0
    if other_rate <= 0:
        return 0.0
    return month_rate / other_rate

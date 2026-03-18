from __future__ import annotations

import numpy as np
import pandas as pd

BASE_MODEL_FEATURES = [
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

ENGINEERED_NUMERIC_FEATURES = [
    "charges_per_tenure",
    "service_count",
    "support_services_count",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "is_month_to_month",
    "is_fiber_customer",
    "tenure_band",
]

MODEL_FEATURES = BASE_MODEL_FEATURES + ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    *ENGINEERED_NUMERIC_FEATURES,
]
CATEGORICAL_FEATURES = [feature for feature in MODEL_FEATURES if feature not in NUMERIC_FEATURES]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    tenure_safe = featured["tenure"].replace(0, 1)
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    support_columns = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

    featured["charges_per_tenure"] = featured["TotalCharges"] / tenure_safe
    featured["service_count"] = featured[service_columns].isin(["Yes"]).sum(axis=1) + featured[
        "InternetService"
    ].isin(["DSL", "Fiber optic"]).astype(int)
    featured["support_services_count"] = featured[support_columns].isin(["Yes"]).sum(axis=1)
    featured["is_month_to_month"] = np.where(featured["Contract"].eq("Month-to-month"), "Yes", "No")
    featured["is_fiber_customer"] = np.where(
        featured["InternetService"].eq("Fiber optic"), "Yes", "No"
    )
    featured["tenure_band"] = pd.cut(
        featured["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12m", "13-24m", "25-48m", "48m+"],
    ).astype(str)

    return featured

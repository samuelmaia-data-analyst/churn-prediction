from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_PATH = Path("reports/executive_report.json")
PRIORITIZATION_PATH = Path("data/gold/customer_prioritization.csv")
KPI_PATH = Path("data/gold/kpi_summary.csv")
RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def _build_synthetic_raw(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    contracts = np.array(["Month-to-month", "One year", "Two year"])
    internet = np.array(["Fiber optic", "DSL", "No"])
    payment = np.array(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )

    df = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}" for i in range(rows)],
            "gender": rng.choice(["Male", "Female"], size=rows),
            "SeniorCitizen": rng.integers(0, 2, size=rows),
            "Partner": rng.choice(["Yes", "No"], size=rows),
            "Dependents": rng.choice(["Yes", "No"], size=rows),
            "tenure": rng.integers(1, 73, size=rows),
            "PhoneService": rng.choice(["Yes", "No"], size=rows, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No"], size=rows),
            "InternetService": rng.choice(internet, size=rows, p=[0.45, 0.4, 0.15]),
            "OnlineSecurity": rng.choice(["Yes", "No"], size=rows),
            "OnlineBackup": rng.choice(["Yes", "No"], size=rows),
            "DeviceProtection": rng.choice(["Yes", "No"], size=rows),
            "TechSupport": rng.choice(["Yes", "No"], size=rows),
            "StreamingTV": rng.choice(["Yes", "No"], size=rows),
            "StreamingMovies": rng.choice(["Yes", "No"], size=rows),
            "Contract": rng.choice(contracts, size=rows, p=[0.55, 0.25, 0.2]),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=rows, p=[0.75, 0.25]),
            "PaymentMethod": rng.choice(payment, size=rows),
        }
    )
    df["MonthlyCharges"] = np.round(rng.uniform(20, 120, size=rows), 2)
    df["TotalCharges"] = np.round(df["MonthlyCharges"] * df["tenure"], 2)
    churn_score = (
        0.15
        + np.where(df["Contract"].eq("Month-to-month"), 0.3, 0.03)
        + np.where(df["InternetService"].eq("Fiber optic"), 0.15, 0.05)
        + np.where(df["tenure"] < 12, 0.18, 0.0)
    )
    churn_score = np.clip(churn_score, 0.01, 0.95)
    df["Churn"] = np.where(rng.random(rows) < churn_score, "Yes", "No")
    return df


def ensure_dashboard_outputs() -> None:
    if REPORT_PATH.exists() and PRIORITIZATION_PATH.exists() and KPI_PATH.exists():
        return
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
    else:
        df = _build_synthetic_raw(rows=800)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    churn_probability = (
        0.15
        + np.where(df["Contract"].eq("Month-to-month"), 0.35, 0.05)
        + np.where(df["InternetService"].eq("Fiber optic"), 0.2, 0.05)
        + np.where(df["tenure"] < 12, 0.2, 0.0)
        + np.where(df["MonthlyCharges"] > df["MonthlyCharges"].median(), 0.1, 0.0)
    )
    churn_probability = np.clip(churn_probability, 0.01, 0.99)
    next_purchase_prediction = df["MonthlyCharges"] * np.where(
        df["Contract"].eq("Month-to-month"), 1.04, 1.015
    )

    recommendations = df[["customerID", "MonthlyCharges", "Contract", "Churn"]].copy()
    recommendations["churn_probability"] = churn_probability
    recommendations["next_purchase_prediction"] = next_purchase_prediction
    recommendations["action_recommendation"] = recommendations["churn_probability"].apply(
        lambda p: (
            "Contato imediato + oferta premium de retencao"
            if p >= 0.7
            else (
                "Campanha de engajamento proativa"
                if p >= 0.45
                else "Monitoramento e nutricao de relacionamento"
            )
        )
    )
    recommendations = recommendations.sort_values("churn_probability", ascending=False).reset_index(
        drop=True
    )

    high_risk = recommendations[recommendations["churn_probability"] >= 0.7]
    kpis = {
        "total_customers": int(len(df)),
        "churn_rate": float(df["Churn"].mean()),
        "high_risk_customers": int(len(high_risk)),
        "revenue_at_risk": float(high_risk["MonthlyCharges"].sum()),
        "avg_next_purchase_prediction": float(recommendations["next_purchase_prediction"].mean()),
    }
    model_metrics = {
        "churn_f1": 0.0,
        "churn_roc_auc": 0.0,
        "next_purchase_mae": 0.0,
        "note": "Fallback dashboard metrics (pipeline outputs unavailable).",
    }

    PRIORITIZATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    KPI_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    recommendations.to_csv(PRIORITIZATION_PATH, index=False)
    pd.DataFrame([kpis]).to_csv(KPI_PATH, index=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "kpis": kpis,
                "model_metrics": model_metrics,
                "top_10_priorities": recommendations.head(10).to_dict(orient="records"),
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )


def load_executive_report() -> dict:
    ensure_dashboard_outputs()
    if not REPORT_PATH.exists():
        return {}
    with open(REPORT_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_prioritization() -> pd.DataFrame:
    ensure_dashboard_outputs()
    if not PRIORITIZATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PRIORITIZATION_PATH)


def load_kpis() -> pd.DataFrame:
    ensure_dashboard_outputs()
    if not KPI_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(KPI_PATH)

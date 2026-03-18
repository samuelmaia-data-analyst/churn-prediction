from __future__ import annotations

from fastapi.testclient import TestClient

import apps.api as api_module
from src.modeling.predictor import PredictionResult


def test_health_endpoint_exposes_readiness_metadata() -> None:
    client = TestClient(api_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"status", "ready", "bundle_path", "bundle_exists"}
    assert payload["status"] in {"healthy", "unhealthy"}


def test_predict_endpoint_returns_prediction_payload(monkeypatch) -> None:
    client = TestClient(api_module.app)

    monkeypatch.setattr(api_module.predictor, "model", object())
    monkeypatch.setattr(
        api_module.predictor,
        "predict_from_dict",
        lambda _payload: PredictionResult(churn="Nao", probability=0.23, risk_level="Baixo"),
    )

    response = client.post(
        "/predict",
        json={
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
        },
    )

    assert response.status_code == 200
    assert response.json() == {"churn": "Nao", "probability": 0.23, "risk_level": "Baixo"}

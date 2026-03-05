from __future__ import annotations

from pprint import pprint

from src.modeling.predictor import ChurnPredictor


def predict_single_customer(customer_data: dict) -> dict:
    predictor = ChurnPredictor()
    result = predictor.predict_from_dict(customer_data)
    return {
        "churn": result.churn,
        "probabilidade": result.probability,
        "risco": result.risk_level,
    }


def main() -> None:
    customer_example = {
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
    }

    pprint(predict_single_customer(customer_example))


if __name__ == "__main__":
    main()

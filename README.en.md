# Churn Prediction

[![Status](https://img.shields.io/badge/status-in_development-yellow)](#status)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](#requirements)
[![Streamlit](https://img.shields.io/badge/interface-streamlit-red)](#running-the-dashboard)
[![FastAPI](https://img.shields.io/badge/api-fastapi-009688)](#running-the-api)

Language: [Português](README.md) | **English**

Customer churn prediction system with an end-to-end ML pipeline in Python, a Streamlit dashboard, and a FastAPI inference API.

## Table of Contents

- [Executive Overview](#executive-overview)
- [Status](#status)
- [Functional Scope](#functional-scope)
- [Solution Architecture](#solution-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Local Setup](#local-setup)
- [Execution Flow](#execution-flow)
- [Running the Dashboard](#running-the-dashboard)
- [Running the API](#running-the-api)
- [API Usage Example](#api-usage-example)
- [Current Model Metrics](#current-model-metrics)
- [Tests](#tests)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Executive Overview

This project addresses a recurring-revenue challenge: identifying customers with high churn risk to prioritize retention actions. The solution includes:

- supervised ML training pipeline;
- automatic best-model selection based on `F1`;
- interactive churn analytics dashboard;
- REST API for online single-customer inference.

## Status

In development.

## Functional Scope

- Binary churn classification (`Yes`/`No`).
- Model training and comparison (`LogisticRegression`, `RandomForest`, `GradientBoosting`).
- Model artifact persistence in `models/`.
- Inference through CLI script, dashboard, and HTTP endpoint.

## Solution Architecture

![Project Architecture](assets/architecture.png)

Main flow:

```text
CSV Dataset -> Data + Feature Pipeline -> Model Training -> Artifacts
                                                   |
                                                   +-> Streamlit Dashboard
                                                   +-> FastAPI Endpoint
                                                   +-> CLI Prediction
```

Best-model selection rule in code: highest `F1` score on test set.

## Tech Stack

- Language: `Python`
- Data: `pandas`, `numpy`
- ML: `scikit-learn`
- Persistence: `joblib`
- Dashboard: `Streamlit`, `Plotly`
- API: `FastAPI`, `Pydantic`, `Uvicorn`

## Repository Structure

```text
churn-prediction/
|-- api.py
|-- app.py
|-- main.py
|-- predict_customer.py
|-- save_processed_data.py
|-- config.yaml
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|   |-- data/
|   |-- features/
|   |-- models/
|   `-- visualization/
|-- tests/
`-- assets/
```

## Requirements

- Python 3.12+
- `pip`

Note: `requirements.txt` references compatibility with Python 3.13.

## Local Setup

```bash
git clone <repository-url>
cd churn-prediction
python -m venv .venv
```

Activate virtual environment:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution Flow

1. Train models and persist artifacts:

```bash
python main.py
```

2. Optional: generate processed dataset for analytics:

```bash
python save_processed_data.py
```

3. Start consumption interfaces (dashboard/API).

## Running the Dashboard

```bash
streamlit run app.py
```

## Running the API

```bash
uvicorn api:app --reload
```

Available endpoints:

- `GET /` returns API base status.
- `GET /health` validates model and preprocessor loading.
- `POST /predict` returns churn prediction, probability, and risk level.

## API Usage Example

`POST /predict` payload:

```json
{
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
  "TotalCharges": 786.0
}
```

Example response:

```json
{
  "churn": "Yes",
  "probability": 0.73,
  "risk_level": "High"
}
```

## Current Model Metrics

Reported metrics for the currently saved model (`models/LogisticRegression.joblib`) with `test_size=0.2` and `random_state=42`:

- Accuracy: `0.8055`
- Precision: `0.6572`
- Recall: `0.5588`
- F1-score: `0.6040`
- ROC-AUC: `0.8420`

Metrics may vary after retraining.

## Tests

A test structure exists in `tests/`, but current files are empty.

Default command:

```bash
pytest -q
```

## Current Limitations

- No implemented automated tests yet.
- No formal experiment/metric versioning yet.
- Project license not defined yet.
- Some project files still have inconsistent text encoding.

## Roadmap

- Add unit and integration tests.
- Version metrics and artifacts per experiment.
- Add data/model drift monitoring.
- Evolve API to batch scoring.
- Integrate pipeline with CI/CD.
- Evaluate additional models (e.g., XGBoost, LightGBM).

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "feat: my feature"`.
4. Open a Pull Request.

## License

Pending. Recommended to add a `LICENSE` file (e.g., MIT).

## Contact

- Samuel de Andrade Maia
- GitHub: https://github.com/samuelmaia-data-analyst
- LinkedIn: https://linkedin.com/in/samuelmaia-data-analyst

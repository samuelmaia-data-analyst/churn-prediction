# Churn Prediction Platform

[![Status](https://img.shields.io/badge/status-in_development-yellow)](#roadmap)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](#technology-stack)
[![Machine Learning](https://img.shields.io/badge/machine_learning-scikit--learn-orange)](#model-performance)
[![API](https://img.shields.io/badge/api-fastapi-009688)](#api-contract)
[![Dashboard](https://img.shields.io/badge/dashboard-streamlit-red)](#demo)

Language: **English** | [Portuguese (PT-BR)](README.md)

Production-oriented churn prediction project with a Machine Learning pipeline, FastAPI inference API, and Streamlit dashboard for retention decision support.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Business Context](#business-context)
- [Key Results](#key-results)
- [Solution Architecture](#solution-architecture)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Demo](#demo)
- [API Contract](#api-contract)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Engineering Decisions](#engineering-decisions)
- [Testing and Quality](#testing-and-quality)
- [Roadmap](#roadmap)
- [ATS Keywords](#ats-keywords)
- [Contact](#contact)
- [License](#license)

## Executive Summary

- End-to-end **Customer Churn Prediction** solution built with Python and scikit-learn.
- Deployable inference layer using **FastAPI** and interactive analytics using **Streamlit + Plotly**.
- Persistent preprocessing pipeline for consistent training-serving behavior.
- Current best saved model shows strong ranking quality (**ROC-AUC 0.8420**).

## Business Context

### Problem
Recurring-revenue companies lose margin when churn risk is detected too late.

### Solution
A supervised binary classification pipeline predicts customer-level churn probability and exposes scores through API and dashboard.

### Expected Outcome
Enables teams to prioritize high-risk customers, optimize retention budget, and protect revenue.

## Key Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1-score | 0.6040 |
| ROC-AUC | 0.8420 |

Primary model artifact: `models/LogisticRegression.joblib`

## Solution Architecture

![Project Architecture](assets/architecture.png)

```text
Raw CSV Data
  -> Cleaning and split
  -> Feature engineering + preprocessing
  -> Training and model selection
  -> Persisted artifacts (model + preprocessor)
  -> Consumption via API / Dashboard / CLI
```

## Technology Stack

- **Language:** Python
- **Data & ML:** pandas, numpy, scikit-learn
- **Persistence:** joblib
- **API:** FastAPI, Pydantic, Uvicorn
- **Dashboard:** Streamlit, Plotly

## Features

- End-to-end churn modeling pipeline.
- Interactive dashboard with filters and individual prediction.
- REST endpoint for real-time scoring.
- Shared preprocessor in app and API to prevent train-serving skew.

## Demo

| API Demo | Dashboard Demo |
|---|---|
| ![API REST Demo](assets/api-demo.gif) | ![Dashboard Demo](assets/dashboard-demo.gif) |

## API Contract

### Health
- `GET /health`

### Prediction
- `POST /predict`

Sample request:

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

Sample response (current implementation):

```json
{
  "churn": "Sim",
  "probability": 0.73,
  "risk_level": "Alto"
}
```

## Quick Start

```bash
git clone <repository-url>
cd churn-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
uvicorn api:app --reload
# in another terminal: streamlit run app.py
```

## Repository Structure

```text
churn-prediction/
|-- app.py
|-- api.py
|-- main.py
|-- predict_customer.py
|-- config.yaml
|-- requirements.txt
|-- data/
|-- models/
|-- src/
|   |-- data/
|   |-- features/
|   `-- models/
|-- tests/
`-- assets/
```

## Engineering Decisions

- Model selection by **F1-score** to balance precision and recall.
- Persistent `preprocessor.joblib` for feature consistency.
- API and dashboard decoupled, consuming shared artifacts.

## Testing and Quality

Current state:
- test scaffolding exists in `tests/`, but test suites are not implemented yet.

Next steps:
- unit tests for preprocessing;
- API contract tests;
- model input schema validation.

## Roadmap

- Automated test coverage (unit + integration).
- Experiment and metric traceability.
- Data/model drift monitoring.
- API batch scoring.
- CI/CD validation and release flow.

## ATS Keywords

`Python` `Machine Learning` `Churn Prediction` `scikit-learn` `FastAPI` `Streamlit` `Model Deployment` `REST API` `Data Science` `MLOps` `Feature Engineering` `Binary Classification` `Model Evaluation` `ROC-AUC`

## Contact

**Samuel de Andrade Maia**
- GitHub: https://github.com/samuelmaia-data-analyst
- LinkedIn: https://linkedin.com/in/samuelmaia-data-analyst

## License

License is not defined yet. Recommended: MIT.

# Churn Prediction Pipeline (Enterprise)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/streamlit-online-brightgreen)](https://data-senior-analytics.streamlit.app/)

Language: **English** | [Portuguese (PT-BR)](README.md)

Production-style analytics and ML project using the Kaggle Telco Churn dataset.

- `raw -> bronze -> silver -> gold`
- `churn prediction` model
- `next purchase prediction` model
- executive reporting for business consumption
- orchestration with `Prefect`
- data quality checks with `Pandera`
- model traceability with `MLflow`

## Table of Contents
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Public Streamlit App](#public-streamlit-app)
- [Data Outputs](#data-outputs)
- [Setup](#setup)
- [Run Pipeline](#run-pipeline)
- [Prefect Orchestration](#prefect-orchestration)
- [Quality](#quality)
- [Executive Dashboard Pages](#executive-dashboard-pages)
- [Dataset](#dataset)

## Highlights
- Layered data architecture: `raw -> bronze -> silver -> gold`
- Star schema in gold (`fact + dimensions`)
- Churn prediction + next purchase prediction
- Executive outputs: `executive_report.json`, KPI CSV, prioritization CSV
- Orchestration with `Prefect`
- Data quality contracts with `Pandera`
- Model tracking with `MLflow`
- Multi-page Streamlit executive dashboard

## Architecture
- Detailed architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Domain-oriented organization with contracts in `src/contracts` and modeling in `src/modeling`

## Churn Modeling

### 1) Baseline model
```
Baseline Model
Logistic Regression
ROC-AUC: 0.842
```

### 2) Model comparison
Last run (`2026-03-05`) in `reports/executive_report.json -> model_metrics.model_comparison`:

| Model | ROC-AUC |
|---|---:|
| Logistic | 0.842 |
| RandomForest | 0.818 |
| XGBoost* | 0.843 |
`*` fallback to `GradientBoosting` when `xgboost` is unavailable.

### 3) Feature importance
Top drivers for business communication:

```
Top Drivers of Churn

• Contract type
• Tenure
• Monthly charges
```

### 4) Business insight
Executive insights in `model_metrics.key_insights`:

```
Key Insights

Customers with month-to-month contracts
show 3x+ higher churn risk.
```
In the last run, the observed value was `6.3x`.

### 5) Pipeline visual
Data and analytics flow:

```mermaid
flowchart LR
    A[Raw] --> B[Bronze]
    B --> C[Silver]
    C --> D[Gold]
```
## Public Streamlit App
https://data-senior-analytics.streamlit.app/

## Data Outputs
- `data/bronze/customer_churn_bronze.csv`
- `data/silver/customer_churn_silver.csv`
- `data/gold/dim_customer.csv`
- `data/gold/dim_contract.csv`
- `data/gold/dim_service.csv`
- `data/gold/fact_customer_churn.csv`
- `data/gold/kpi_summary.csv`
- `data/gold/customer_prioritization.csv`
- `reports/executive_report.json`
- `reports/model_card.md`
- `reports/executive_brief.md`

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline
```bash
python main.py --seed 42 --data-dir data --log-level INFO
```

## Prefect Orchestration
Deployment is defined in `prefect.yaml` with daily schedule (`07:00 UTC`).

```bash
prefect server start
prefect work-pool create --type process default-agent-pool
prefect worker start --pool default-agent-pool
prefect deploy --all
prefect deployment run "enterprise-churn-pipeline/daily-enterprise-run"
```

## Quality
- `pre-commit` with `black`, `ruff`, `isort`
- CI checks:
  - `ruff check main.py src tests pages`
  - `black --check main.py src tests pages`
  - `pytest -q`

## Artifact versioning
- `logs/` and `mlruns/` should not be pushed to Git.
- version only code, configuration, and documentation.

## Executive Dashboard Pages
- Executive Overview
- Risk and Growth
- Prioritization
- Simulation

Includes direct download of:
- `executive_report.json`
- `customer_prioritization.csv`

Dashboard bootstrap behavior:
- if `reports/` and `data/gold/` are missing and `data/raw` exists, the app generates artifacts through the real pipeline;
- if pipeline execution fails or `data/raw` is unavailable, the app uses a synthetic fallback to avoid empty pages.

## Dataset
Source: Kaggle - Telco Customer Churn  
Official source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Expected file path: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

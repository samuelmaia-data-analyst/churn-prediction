# Churn Prediction Pipeline (Enterprise)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/streamlit-online-brightgreen)](https://data-senior-analytics.streamlit.app/)

Language: **English** | [Portuguese (PT-BR)](README.md)

Production-style analytics and ML project using the Kaggle Telco Churn dataset.

## Highlights
- Layered data architecture: `raw -> bronze -> silver -> gold`
- Star schema in gold (`fact + dimensions`)
- Churn prediction + next purchase prediction
- Executive outputs: `executive_report.json`, KPI CSV, prioritization CSV
- Orchestration with `Prefect`
- Data quality contracts with `Pandera`
- Model tracking with `MLflow`
- Multi-page Streamlit executive dashboard

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

## Executive Dashboard Pages
- Executive Overview
- Risk and Growth
- Prioritization
- Simulation

Includes direct download of:
- `executive_report.json`
- `customer_prioritization.csv`

## Dataset
Source: Kaggle - Telco Customer Churn  
Expected file path: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`


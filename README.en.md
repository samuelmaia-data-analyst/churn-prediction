# Churn Prediction Pipeline (Enterprise)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/streamlit-online-brightgreen)](https://data-senior-analytics.streamlit.app/)
[![Official Release](https://img.shields.io/badge/release-v1.0.0-success)](https://github.com/samuelmaia-data-analyst/churn-prediction/releases/latest)

Language: **English** | [Portuguese (PT-BR)](README.md)

Production-style analytics and ML project using the Kaggle Telco Churn dataset.

- `raw -> bronze -> silver -> gold`
- `churn prediction` model
- `next purchase prediction` model
- executive reporting for business consumption
- orchestration with `Prefect`
- data quality checks with `Pandera`
- model traceability with `MLflow`

## Official Public Release
- Public release with notes: https://github.com/samuelmaia-data-analyst/churn-prediction/releases/latest
- Local notes for current version: [RELEASE_NOTES_v1.0.0.md](RELEASE_NOTES_v1.0.0.md)

## Business Outcome Simulation (Executive Snapshot)
Goal: convert technical scoring into defendable financial outcome for executive steering.

- Protected value (prioritized portfolio): `~$68k` per cycle
- Expected net impact (base scenario): `+$16k` per cycle after action cost
- Adoption scenario: `70%` playbook coverage with `<24h` first-touch SLA
- Executive interpretation: capturing `~24%` of value-at-risk already covers operational cost in the base scenario

## Action Playbook (Executive Summary)
Direct translation from risk + value into commercial action:

| Segment | Risk | Recommended action | Expected outcome |
|---|---|---|---|
| High LTV | High | Human retention call within 24h | Highest protected value per customer |
| Low LTV | High | Retention email offer | Controlled CAC with strong scale |
| High LTV | Medium | Proactive loyalty outreach | Incremental churn reduction |
| Low LTV | Medium | Automated nurture journey | Efficient conversion uplift |

## Table of Contents
- [Official Public Release](#official-public-release)
- [Business Outcome Simulation (Executive Snapshot)](#business-outcome-simulation-executive-snapshot)
- [Action Playbook (Executive Summary)](#action-playbook-executive-summary)
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
Last run (`2026-03-05`) in `artifacts/reports/executive_report.json -> model_metrics.model_comparison`:

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

- Contract type
- Tenure
- Monthly charges
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
- `artifacts/reports/executive_report.json`
- `artifacts/reports/model_card.md`
- `artifacts/reports/executive_brief.md`

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline
```bash
python -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO
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
  - `ruff check app.py api.py main.py predict_customer.py save_processed_data.py apps src tests pages`
  - `black --check app.py api.py main.py predict_customer.py save_processed_data.py apps src tests pages`
  - `pytest -q`

## Artifact versioning
- `artifacts/` and `mlruns/` should not be pushed to Git.
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
- if `artifacts/reports/` and `data/gold/` are missing and `data/raw` exists, the app generates artifacts through the real pipeline;
- if pipeline execution fails or `data/raw` is unavailable, the app uses a synthetic fallback to avoid empty pages.

## Cost-sensitive Threshold Strategy
- global baseline (`balanced`): `0.50`
- value-sensitive thresholds:
  - `High LTV`: `0.65`
  - `Low LTV`: `0.80`

## Actionable Playbook
| Segment | Risk | Action | Expected ROI |
|---|---|---|---|
| High LTV | High | Call retention | +$200/customer |
| Low LTV | High | Retention offer by email | +$90/customer |
| High LTV | Medium | Proactive loyalty outreach | +$80/customer |
| Low LTV | Medium | Automated nurture journey | +$35/customer |

Outputs:
- `data/gold/action_playbook.csv`
- `artifacts/reports/action_playbook.md`

## Drift Monitoring
- runtime drift checks: `PSI` and `KS`
- standalone script: `monitoring/drift_detection.py`

```bash
python monitoring/drift_detection.py --baseline reports/drift_reference.csv --current data/gold/customer_prioritization.csv --output reports/drift_alert.json
```

## Release
- Version: `v1.0.0`
- Model versioning artifacts:
  - `models/model_v1.pkl`
  - `models/model_metadata.json`

## Dataset
Source: Kaggle - Telco Customer Churn  
Official source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Expected file path: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`


# Churn Prediction Pipeline (Senior)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/streamlit-online-brightgreen)](https://data-senior-analytics.streamlit.app/)

Pipeline de analytics e ML com dataset Kaggle (Telco Customer Churn), estruturado em camadas:

- `raw -> bronze -> silver -> gold`
- modelo de `churn prediction`
- modelo de `next purchase prediction`
- reporting executivo para consumo de negocio
- orquestracao com `Prefect`
- data quality checks com `Pandera`
- rastreabilidade de modelos com `MLflow`

## Sumario
- [Streamlit (publico)](#streamlit-publico)
- [Arquitetura](#arquitetura)
- [Setup](#setup)
- [Execucao do pipeline](#execucao-do-pipeline)
- [Orquestracao com Prefect](#orquestracao-com-prefect)
- [Qualidade](#qualidade)
- [Dashboard Executivo Multipagina](#dashboard-executivo-multipagina)
- [Dados](#dados)

## Streamlit (publico)

https://data-senior-analytics.streamlit.app/

## Arquitetura

### Camadas de dados
- `raw`: fonte original do Kaggle
- `bronze`: copia com metadados de ingestao
- `silver`: dados limpos, tipados e validados
- `gold`: star schema + KPIs + priorizacao de clientes

### Star schema (gold)
- `dim_customer.csv`
- `dim_contract.csv`
- `dim_service.csv`
- `fact_customer_churn.csv`

### Outputs de negocio
- `reports/executive_report.json`
- `data/gold/kpi_summary.csv`
- `data/gold/customer_prioritization.csv`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execucao do pipeline

```bash
python main.py --seed 42 --data-dir data --log-level INFO
```

## Orquestracao com Prefect

Deploy configurado em [prefect.yaml](prefect.yaml) com agenda diaria (`07:00 UTC`).

```bash
# 1) iniciar API local do Prefect (opcional, para UI local)
prefect server start

# 2) criar pool e iniciar worker
prefect work-pool create --type process default-agent-pool
prefect worker start --pool default-agent-pool

# 3) registrar deployment
prefect deploy --all

# 4) disparar execucao manual
prefect deployment run "enterprise-churn-pipeline/daily-enterprise-run"
```

### Tracking de ML
- MLflow local em `./mlruns`
- modelos versionados por execucao no run do pipeline

### Logging estruturado
- logs JSON em `logs/pipeline.log`
- cada execucao tem `run_id`

## Qualidade

- `pre-commit` com `black`, `ruff`, `isort`
- `pytest` para contratos do pipeline e outputs
- CI executando:
  - `ruff check main.py src tests pages`
  - `black --check main.py src tests pages`
  - `pytest -q`

## Dashboard Executivo Multipagina
- `Executive Overview`
- `Risk and Growth`
- `Prioritization`
- `Simulation`

Com download direto de:
- `executive_report.json`
- `customer_prioritization.csv`

## Dados

Dataset utilizado: Kaggle - Telco Customer Churn  
Fonte oficial: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Arquivo esperado em: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

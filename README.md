# Churn Prediction Pipeline (Enterprise)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/streamlit-online-brightgreen)](https://data-senior-analytics.streamlit.app/)

Idioma: **Português (PT-BR)** | [English](README.en.md)

Pipeline de analytics e ML com dataset Kaggle (Telco Customer Churn), estruturado em camadas:

- `raw -> bronze -> silver -> gold`
- modelo de `churn prediction`
- modelo de `next purchase prediction`
- reporting executivo para consumo de negocio
- orquestracao com `Prefect`
- data quality checks com `Pandera`
- rastreabilidade de modelos com `MLflow`

## Sumario
- [Destaques](#destaques)
- [Streamlit (publico)](#streamlit-publico)
- [Saidas de dados](#saidas-de-dados)
- [Setup](#setup)
- [Execucao do pipeline](#execucao-do-pipeline)
- [Orquestracao com Prefect](#orquestracao-com-prefect)
- [Qualidade](#qualidade)
- [Versionamento de artefatos](#versionamento-de-artefatos)
- [Dashboard Executivo Multipagina](#dashboard-executivo-multipagina)
- [Dados](#dados)

## Destaques
- Arquitetura em camadas: `raw -> bronze -> silver -> gold`
- Star schema no gold (`fato + dimensoes`)
- Predicao de churn + predicao de proxima compra
- Saidas executivas: `executive_report.json`, KPI CSV e priorizacao CSV
- Orquestracao com `Prefect`
- Contratos de qualidade de dados com `Pandera`
- Rastreabilidade de modelos com `MLflow`
- Dashboard executivo multipagina em Streamlit

## Streamlit (publico)

https://data-senior-analytics.streamlit.app/

## Saidas de dados
- `data/bronze/customer_churn_bronze.csv`
- `data/silver/customer_churn_silver.csv`
- `data/gold/dim_customer.csv`
- `data/gold/dim_contract.csv`
- `data/gold/dim_service.csv`
- `data/gold/fact_customer_churn.csv`
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

## Versionamento de artefatos
- `logs/` e `mlruns/` nao devem ser enviados para o Git.
- versione apenas codigo, configuracoes e documentacao.

## Dashboard Executivo Multipagina
- `Executive Overview`
- `Risk and Growth`
- `Prioritization`
- `Simulation`

Com download direto de:
- `executive_report.json`
- `customer_prioritization.csv`

Comportamento de bootstrap do dashboard:
- se `reports/` e `data/gold/` nao existirem e houver `data/raw`, o app gera os artefatos via pipeline real;
- se o pipeline falhar ou nao houver `data/raw`, o app gera fallback sintetico para nao ficar vazio.

## Dados

Dataset utilizado: Kaggle - Telco Customer Churn  
Fonte oficial: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Arquivo esperado em: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

# Churn Prediction Pipeline (Enterprise)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/streamlit-online-brightgreen)](https://data-senior-analytics.streamlit.app/)

Idioma: **Portugues (PT-BR)** | [English](README.en.md)

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
- [Arquitetura](#arquitetura)
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

## Arquitetura
- Visao detalhada em [ARCHITECTURE.md](ARCHITECTURE.md)
- Organizacao por dominio com contratos em `src/contracts` e modelagem em `src/modeling`

## Modelagem de Churn

### 1) Baseline model
```
Baseline Model
Logistic Regression
ROC-AUC: 0.842
```

### 2) Comparacao de modelos
Ultimo run (`2026-03-05`) em `reports/executive_report.json -> model_metrics.model_comparison`:

| Model | ROC-AUC |
|---|---:|
| Logistic | 0.842 |
| RandomForest | 0.818 |
| XGBoost* | 0.843 |
`*` fallback para `GradientBoosting` quando `xgboost` nao esta instalado.

### 3) Feature importance
Top drivers de churn para narrativa de negocio:

```
Top Drivers of Churn

• Contract type
• Tenure
• Monthly charges
```

### 4) Business insight
Insights executivos em `model_metrics.key_insights`:

```
Key Insights

Customers with month-to-month contracts
show 3x+ higher churn risk.
```
No ultimo run, o valor observado foi `6.3x`.

### 5) Pipeline visual
Pipeline de dados e consumo de modelos:

```mermaid
flowchart LR
    A[Raw] --> B[Bronze]
    B --> C[Silver]
    C --> D[Gold]
```
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

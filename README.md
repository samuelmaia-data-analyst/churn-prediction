# Churn Prediction Pipeline (Enterprise)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#setup)
[![Streamlit](https://img.shields.io/badge/streamlit-online-brightgreen)](https://data-senior-analytics.streamlit.app/)

Idioma: **Português (PT-BR)** | [English](README.en.md)

Pipeline de analytics e ML com dataset Kaggle (Telco Customer Churn), estruturado em camadas:

- `raw -> bronze -> silver -> gold`
- modelo de `churn prediction`
- modelo de `next purchase prediction`
- reporting executivo para consumo de negócio
- orquestração com `Prefect`
- data quality checks com `Pandera`
- rastreabilidade de modelos com `MLflow`

## Sumário
- [Destaques](#destaques)
- [Arquitetura](#arquitetura)
- [Streamlit (público)](#streamlit-publico)
- [Saídas de dados](#saidas-de-dados)
- [Setup](#setup)
- [Execução do pipeline](#execucao-do-pipeline)
- [Orquestração com Prefect](#orquestracao-com-prefect)
- [Qualidade](#qualidade)
- [Versionamento de artefatos](#versionamento-de-artefatos)
- [Dashboard Executivo Multipágina](#dashboard-executivo-multipagina)
- [Dados](#dados)

## Destaques
- Arquitetura em camadas: `raw -> bronze -> silver -> gold`
- Star schema no gold (`fato + dimensões`)
- Predição de churn + predição de próxima compra
- Saídas executivas: `executive_report.json`, KPI CSV e priorização CSV
- Orquestração com `Prefect`
- Contratos de qualidade de dados com `Pandera`
- Rastreabilidade de modelos com `MLflow`
- Dashboard executivo multipágina em Streamlit

## Arquitetura
- Visão detalhada em [ARCHITECTURE.md](ARCHITECTURE.md)
- Organização por domínio com contratos em `src/contracts` e modelagem em `src/modeling`

## Modelagem de Churn

### 1) Baseline model
```
Baseline Model
Logistic Regression
ROC-AUC: 0.842
```

### 2) Comparação de modelos
Último run (`2026-03-05`) em `reports/executive_report.json -> model_metrics.model_comparison`:

| Model | ROC-AUC |
|---|---:|
| Logistic | 0.842 |
| RandomForest | 0.818 |
| XGBoost* | 0.843 |
`*` fallback para `GradientBoosting` quando `xgboost` não está instalado.

### 3) Feature importance
Top drivers de churn para narrativa de negócio:

```
Top Drivers of Churn

- Contract type
- Tenure
- Monthly charges
```

### 4) Business insight
Insights executivos em `model_metrics.key_insights`:

```
Key Insights

Customers with month-to-month contracts
show 3x+ higher churn risk.
```
No último run, o valor observado foi `6.3x`.

### 5) Executive flowcharts (Mermaid)
Duas visoes complementares: `Board View` (sintese de decisao) e `Operator View` (execucao com owners e SLAs).

Board View:

```mermaid
flowchart LR
    A[Strategic Targets\nRevenue Retention Margin] --> B[Decision Intelligence\nRisk Value Prioritization]
    B --> C[Capital Allocation\nApprove Hold Reject]
    C --> D[Commercial Execution\nRetention Upsell Cross-sell]
    D --> E[Value Realization\nROI Payback SLA]
    E --> F[Operating Review\nWeekly Governance]
    F -. Rebalance budget and capacity .-> C
    F -. Model feedback loop .-> B
```

Operator View:

```mermaid
flowchart LR
    subgraph O1[Data and ML Factory]
        A1[Ingestion and Standardization\nOwner Data Engineering\nSLA Daily 07:00 UTC] --> A2[Quality and Contract Gates\nOwner Data Governance\nSLA <2 percent failed checks]
        A2 --> A3[Scoring and Value at Risk\nOwner Data Science\nSLA AUC >= 0.82]
    end

    subgraph O2[Portfolio Governance]
        A3 --> B1[Prioritization Engine\nOwner RevOps\nSLA Top 10 published by 09:00]
        B1 --> B2{Investment Committee Gate\nOwner CCO CFO}
    end

    subgraph O3[Field Execution]
        B2 --> C1[Playbook Activation\nOwner Sales and CS\nSLA First touch <24h]
        C1 --> C2[Customer Outcomes\nSave Upsell Cross-sell]
        C2 --> C3[Outcome Ledger\nOwner Finance\nSLA Weekly close]
    end

    C3 --> D1[Executive Cockpit\nOwner Strategy Office]
    D1 --> D2[Weekly Operating Review]
    D2 -. Drift incidents and policy breaches .-> D3[Model Risk Monitoring\nOwner MRM]
    D3 -. Controlled retrain and recalibration .-> A3
    D2 -. Reallocate budget and capacity .-> B2
```

## Streamlit (público)

https://data-senior-analytics.streamlit.app/

## Saídas de dados
- `data/bronze/customer_churn_bronze.csv`
- `data/silver/customer_churn_silver.csv`
- `data/gold/dim_customer.csv`
- `data/gold/dim_contract.csv`
- `data/gold/dim_service.csv`
- `data/gold/fact_customer_churn.csv`
- `reports/executive_report.json`
- `reports/model_card.md`
- `reports/executive_brief.md`
- `data/gold/kpi_summary.csv`
- `data/gold/customer_prioritization.csv`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execução do pipeline

```bash
python main.py --seed 42 --data-dir data --log-level INFO
```

## Orquestração com Prefect

Deploy configurado em [prefect.yaml](prefect.yaml) com agenda diária (`07:00 UTC`).

```bash
# 1) iniciar API local do Prefect (opcional, para UI local)
prefect server start

# 2) criar pool e iniciar worker
prefect work-pool create --type process default-agent-pool
prefect worker start --pool default-agent-pool

# 3) registrar deployment
prefect deploy --all

# 4) disparar execução manual
prefect deployment run "enterprise-churn-pipeline/daily-enterprise-run"
```

### Tracking de ML
- MLflow local em `./mlruns`
- modelos versionados por execução no run do pipeline

### Logging estruturado
- logs JSON em `logs/pipeline.log`
- cada execução tem `run_id`

## Qualidade

- `pre-commit` com `black`, `ruff`, `isort`
- `pytest` para contratos do pipeline e outputs
- CI executando:
  - `ruff check main.py src tests pages`
  - `black --check main.py src tests pages`
  - `pytest -q`

## Versionamento de artefatos
- `logs/` e `mlruns/` não devem ser enviados para o Git.
- versione apenas código, configurações e documentação.

## Dashboard Executivo Multipágina
- `Executive Overview`
- `Risk and Growth`
- `Prioritization`
- `Simulation`

Com download direto de:
- `executive_report.json`
- `customer_prioritization.csv`

Comportamento de bootstrap do dashboard:
- se `reports/` e `data/gold/` não existirem e houver `data/raw`, o app gera os artefatos via pipeline real;
- se o pipeline falhar ou não houver `data/raw`, o app gera fallback sintético para não ficar vazio.

## Dados

Dataset utilizado: Kaggle - Telco Customer Churn  
Fonte oficial: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Arquivo esperado em: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

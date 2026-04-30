# Churn Prediction Data Product

Idioma: [EN principal](README.md) | [PT-BR](README.pt-BR.md) | **PT-PT**

O inglês é o idioma canónico da documentação. PT-BR e PT-PT são referências localizadas.

Projeto de dados e machine learning para churn com foco em operação real: pipeline em camadas, artefactos reprocessáveis, observabilidade e consumo por API/dashboard.

## Porque Este Repositório Existe

A maioria dos projetos de churn de portefólio para no treino. Este vai além:

- trata churn como sistema de dados, não só classificação
- separa ingestão, transformação, modelação, reporting e consumo
- persiste artefactos reutilizáveis em vez de depender de estado de notebook
- traduz score em decisão operacional

Perguntas que o repositório responde:

1. Que clientes têm maior risco de churn?
2. Que clientes priorizar primeiro?
3. Como ajustar threshold quando o custo de campanha muda?
4. Que artefactos o negócio precisa para agir?

## Valor de Negócio

Saídas orientadas a equipas de retenção/RevOps:

- lista priorizada de clientes
- resumo executivo de KPIs
- playbook de ação
- model card
- monitorização de drift
- consumo por API e dashboard

## Arquitetura

Fluxo canónico:

`raw -> bronze -> silver -> gold -> model artifacts -> reporting -> dashboard/api`

Caminhos canónicos:

- `src/runtime/`: configuração e logging
- `src/pipelines/`: ingestão, transformação, validação, monitorização e reporting
- `src/modeling/`: treino, inferência e contrato do predictor
- `src/compat/`: camada explícita de compatibilidade
- `scripts/`: rotinas operacionais
- `analytics_dbt/`: camada opcional dbt para marts analíticos

Compat wrappers existem em `src/*.py` e alguns entrypoints de raiz, mas não são o caminho preferencial para código novo.

Caminhos de UI:

- `app.py`: entrypoint canónico para deploy Streamlit
- `apps/streamlit_app.py`: control room e score individual
- `apps/dashboard_ui.py`: shell visual e componentes partilhados
- `apps/dashboard_runtime.py`: carga de artefactos e status de runtime
- `apps/pages/`: páginas executiva, risco, priorização e simulação
- `pages/`: wrappers de compatibilidade multipage

Ver também:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DASHBOARD.md](docs/DASHBOARD.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Stack Técnica

- Python 3.12
- Pandas / NumPy
- scikit-learn
- Pandera
- MLflow (opcional)
- Streamlit
- FastAPI
- Pytest, Ruff, Black, Isort, MyPy, Pre-commit
- GitHub Actions

## Decisões Técnicas

- modelo em camadas (`raw -> bronze -> silver -> gold`) para reprocessamento explícito
- persistência atómica de JSON/CSV/Markdown
- padrão artifact-first para dashboard e API
- compatibilidade preservada, mas isolada
- retry com limite e fail-fast para erros não-transientes de contrato

## Mapa do Repositório

```text
.
|-- .github/                  # CI, templates e ownership
|-- analytics_dbt/            # projeto dbt opcional
|-- apps/                     # Streamlit/FastAPI e helpers
|-- artifacts/                # metadados e artefactos gerados
|-- data/                     # raw/bronze/silver/gold
|-- deploy/                   # perfis de ambiente para contentores
|-- docs/                     # arquitetura, operação e convenções
|-- models/                   # registry local de modelos
|-- notebooks/                # exploração apenas
|-- pages/                    # wrappers Streamlit multipage
|-- scripts/                  # utilitários operacionais
|-- src/                      # implementação canónica
|-- tests/                    # testes automatizados
`-- README.md                 # entrypoint principal
```

## Quickstart

### 1. Preparar ambiente

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env
```

### 2. Adicionar dataset

Ficheiro esperado:
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Fonte:
- Kaggle `blastchar/telco-customer-churn`

### 3. Executar pipeline canónico

```bash
.venv\Scripts\python.exe -m src.cli.pipeline --data-dir data --log-level INFO --decision-policy balanceada --environment dev
```

### 4. Executar quality gates

```bash
make test
make lint
make typecheck
```

## Runtime e Fiabilidade

- configuração por `.env` e variáveis de ambiente
- logging estruturado com `run_id`
- metadados de execução em `artifacts/metadata/`
- lineage em `artifacts/metadata/lineage_run_<run_id>.json`
- governança/LGPD em `artifacts/metadata/governance_run_<run_id>.json`
- validação de schema antes do treino
- camadas bronze/silver/gold regeneráveis
- drift monitorizado com PSI e KS
- EDA operacional em `data/gold/eda_profile.json` e `artifacts/reports/eda_report.md`
- vista pseudonimizada: `data/gold/customer_prioritization_public.csv`

Perfis contentorizados (`docker-compose.yml`):

- `dev`: API + dashboard
- `pipeline`: execução batch
- `prod`: API com perfil de produção

## Qualidade de Engenharia

Ferramentas:
- `pytest`, `ruff`, `black`, `isort`, `mypy`, `pre-commit`, `GitHub Actions`

Dependências:
- `requirements-runtime.txt`: runtime
- `requirements-dev.txt`: lint/test/static analysis

## Trade-offs

- artefactos em filesystem local
- monitorização leve (sem stack completa de observabilidade)
- orquestração externa opcional (Prefect), não obrigatória
- wrappers legados preservados por compatibilidade

## Roadmap

- reduzir superfície legada em `src/models`
- reforçar contratos de saída
- ampliar exemplos de deploy e storage remoto
- evoluir contrato versionado da API de inferência

## Documentação

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/README.md](docs/README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DASHBOARD.md](docs/DASHBOARD.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)
- [docs/ANALYTICS_ENGINEERING.md](docs/ANALYTICS_ENGINEERING.md)
- [docs/GOVERNANCE.md](docs/GOVERNANCE.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Padrão de Contribuição

Mudanças devem aumentar clareza, fiabilidade e manutenção sustentável. O repositório deve manter-se legível, testável e defensável em review técnico.

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)


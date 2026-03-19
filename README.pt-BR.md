# Churn Prediction Data Product

Idioma: [EN principal](README.md) | **PT-BR** | [PT-PT](README.pt-PT.md)

O inglês é o idioma canônico da documentação do repositório. As versões PT-BR e PT-PT são mantidas como referências localizadas.

Pipeline de dados e machine learning para churn de clientes, com foco em operação analítica, confiabilidade de execução e tradução do score em ações de retenção.

Este repositório foi estruturado para se comportar como um produto de dados pequeno, mas crível:

- pipeline em camadas com responsabilidades explícitas
- artefatos reprocessáveis e observáveis
- saídas orientadas a negócio, não apenas métricas de modelo
- contratos testados e automação de qualidade

## Por Que Este Projeto Existe

Em muitos portfólios, churn aparece como notebook e gráfico. Aqui o objetivo é diferente: mostrar como um caso aplicado de ML pode ser implementado como sistema de dados sustentável.

O projeto responde a quatro perguntas práticas:

1. Quais clientes apresentam maior risco de churn?
2. Quais clientes devem ser priorizados primeiro?
3. Como o threshold muda quando o custo operacional da campanha muda?
4. Quais artefatos o time de negócio precisa para agir sobre o score?

## O Que o Repositório Entrega

- ingestão `raw -> bronze`
- tratamento e validação `bronze -> silver`
- camada analítica `silver -> gold`
- treino de modelos com comparação de candidatos
- thresholding sensível a custo
- persistência de artefatos de inferência
- relatórios executivos e playbook de ação
- monitoramento leve de drift
- testes automatizados e CI

## Quickstart

### 1. Preparar o ambiente

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

Para desenvolvimento local, testes e lint:

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env
```

### 2. Adicionar o dataset

Arquivo esperado:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Fonte do dataset:

- Kaggle: `blastchar/telco-customer-churn`
- arquivo utilizado neste projeto: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. Executar o pipeline

```bash
.venv\Scripts\python.exe -m src.cli.pipeline --data-dir data --log-level INFO --decision-policy balanceada --environment dev
```

### 4. Validar os gates de qualidade

```bash
make test
make lint
make typecheck
```

## Estrutura do Repositório

```text
.
|-- .github/                  # CI, templates e padrões de colaboração
|-- apps/                     # apps principais (Streamlit/FastAPI)
|-- assets/                   # imagens e mídia do repositório
|-- data/                     # datasets e camadas locais de pipeline
|-- docs/                     # arquitetura, operação e convenções
|-- notebooks/                # exploração isolada do caminho produtivo
|-- pages/                    # páginas do Streamlit
|-- src/                      # código canônico do produto de dados
|-- tests/                    # contratos e regressão
|-- .env.example              # configuração base por ambiente
|-- Makefile                  # comandos padrão
|-- pyproject.toml            # configuração de ferramentas
`-- README.md                 # entrada principal do repositório
```

Mapa detalhado:

- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Arquitetura

Fluxo principal:

`raw -> bronze -> silver -> gold -> model artifacts -> reporting -> dashboard/api`

Componentes centrais:

- `src/runtime/config.py`: configuração de runtime, paths e ambiente
- `src/pipelines/ingestion.py`: leitura do raw e camada bronze
- `src/pipelines/transformation.py`: limpeza e silver layer
- `src/pipelines/feature_engineering.py`: features de negócio reutilizáveis
- `src/modeling/pipeline.py`: treino, score, persistência e metadata
- `src/pipelines/reporting.py`: executive report, model card e action playbook
- `src/pipelines/monitoring.py`: baseline e alerta de drift com PSI/KS
- `src/cli/pipeline.py`: orquestração ponta a ponta

Pastas como `src/data`, `src/features`, `src/models` e os wrappers em `src/*.py` existem apenas como camada de compatibilidade. A implementação canônica está em `src/runtime/`, `src/pipelines/`, `src/modeling/` e `src/compat/`.

Arquitetura detalhada:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Operação e Confiabilidade

O projeto implementa controles proporcionais ao escopo:

- configuração por `.env` e variáveis de ambiente
- isolamento de artefatos por contexto de execução
- logging estruturado com `run_id`
- persistência atômica de CSV, JSON e Markdown
- metadados de run em `artifacts/metadata/`
- manifests versionados para `gold` e model registry
- validação de schema antes do treino
- monitoramento simples de drift
- reprocessamento local sem estado manual oculto
- deployment opcional com Prefect mantido apenas como exemplo de agendamento, não como caminho canônico

Guia operacional:

- [docs/OPERATIONS.md](docs/OPERATIONS.md)

## Comandos Principais

```bash
make install
make train
make train-cheap
make train-expensive
make test
make lint
make typecheck
make predict
```

## Qualidade de Engenharia

Ferramentas usadas:

- `pytest`
- `ruff`
- `black`
- `isort`
- `mypy`
- `pre-commit`
- `GitHub Actions`

Separação de dependências:

- `requirements-runtime.txt`: pipeline, API, dashboard e dependências de execução
- `requirements-dev.txt`: lint, testes, hooks e análise estática

Os gates de qualidade estão definidos em:

- [.github/workflows/ci.yml](.github/workflows/ci.yml)
- [.pre-commit-config.yaml](.pre-commit-config.yaml)

## Documentação Complementar

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DASHBOARD.md](docs/DASHBOARD.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Roadmap Realista

- reduzir ou eliminar a camada legada restante em `src/models`
- adicionar validações de contrato mais estritas para outputs gold e reporting
- plugar armazenamento remoto para artefatos e tracking
- expor inferência e metadata por meio de um contrato de API versionado

## Colaboração

Pull requests, bugs e melhorias devem seguir:

- [CONTRIBUTING.md](CONTRIBUTING.md)

Clareza e reprodutibilidade importam mais do que complexidade artificial. O objetivo é manter o repositório legível, testável e defensável em review técnico.

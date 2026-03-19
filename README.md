# Churn Prediction Data Product

Language: **EN** | [PT-BR](README.pt-BR.md) | [PT-PT](README.pt-PT.md)

Production-minded data and machine learning pipeline for customer churn, designed to show execution reliability, clear system boundaries, and business-oriented outputs rather than model experimentation alone.

This repository is structured to behave like a small but credible data product:

- layered pipeline with explicit responsibilities
- reprocessable and observable artifacts
- business-facing outputs, not only model metrics
- tested contracts and quality automation

## Why This Project Exists

In many portfolios, churn appears as a notebook plus a chart. The goal here is different: show how an applied ML use case can be implemented as a maintainable data system.

The project answers four practical questions:

1. Which customers have the highest churn risk?
2. Which customers should be prioritized first?
3. How should the decision threshold change when campaign cost changes?
4. Which artifacts does the business team need in order to act on the score?

## What The Repository Delivers

- ingestion `raw -> bronze`
- cleaning and validation `bronze -> silver`
- analytical layer `silver -> gold`
- model training with candidate comparison
- cost-sensitive thresholding
- inference artifact persistence
- executive reporting and action playbook
- lightweight drift monitoring
- automated tests and CI

## Quickstart

### 1. Prepare the environment

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

For local development, tests, and linting:

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env
```

### 2. Add the dataset

Expected file:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Dataset source:

- Kaggle: `blastchar/telco-customer-churn`
- file used in this project: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. Run the pipeline

```bash
.venv\Scripts\python.exe -m src.cli.pipeline --data-dir data --log-level INFO --decision-policy balanceada --environment dev
```

### 4. Validate quality gates

```bash
make test
make lint
make typecheck
```

## Repository Structure

```text
.
|-- .github/                  # CI, templates, and collaboration standards
|-- apps/                     # primary apps (Streamlit/FastAPI)
|-- assets/                   # repository images and media
|-- data/                     # datasets and local pipeline layers
|-- docs/                     # architecture, operations, and conventions
|-- notebooks/                # exploration isolated from the production path
|-- pages/                    # Streamlit pages
|-- src/                      # canonical data product code
|-- tests/                    # contracts and regression
|-- .env.example              # base environment configuration
|-- Makefile                  # standard commands
|-- pyproject.toml            # tool configuration
`-- README.md                 # main repository entry point
```

Detailed map:

- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Architecture

Main flow:

`raw -> bronze -> silver -> gold -> model artifacts -> reporting -> dashboard/api`

Core components:

- `src/runtime/config.py`: runtime configuration, paths, and environment
- `src/pipelines/ingestion.py`: raw loading and bronze layer
- `src/pipelines/transformation.py`: cleaning and silver layer
- `src/pipelines/feature_engineering.py`: reusable business features
- `src/modeling/pipeline.py`: training, scoring, persistence, and metadata
- `src/pipelines/reporting.py`: executive report, model card, and action playbook
- `src/pipelines/monitoring.py`: baseline and drift alerting with PSI/KS
- `src/cli/pipeline.py`: end-to-end orchestration

Folders such as `src/data`, `src/features`, `src/models`, and wrappers under `src/*.py` exist only as compatibility layers. The canonical implementation lives under `src/runtime/`, `src/pipelines/`, `src/modeling/`, and `src/compat/`.

Detailed architecture:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Operations and Reliability

The project implements controls proportional to its scope:

- configuration through `.env` and environment variables
- artifact isolation by execution context
- structured logging with `run_id`
- atomic persistence for CSV, JSON, and Markdown
- run metadata under `artifacts/metadata/`
- versioned manifests for `gold` and model registry
- schema validation before training
- simple drift monitoring
- local reprocessing without hidden manual state

Operational guide:

- [docs/OPERATIONS.md](docs/OPERATIONS.md)

## Main Commands

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

## Engineering Quality

Tools in use:

- `pytest`
- `ruff`
- `black`
- `isort`
- `mypy`
- `pre-commit`
- `GitHub Actions`

Dependency split:

- `requirements-runtime.txt`: pipeline, API, dashboard, and runtime dependencies
- `requirements-dev.txt`: lint, tests, hooks, and static analysis

Quality gates are defined in:

- [.github/workflows/ci.yml](.github/workflows/ci.yml)
- [.pre-commit-config.yaml](.pre-commit-config.yaml)

## Supporting Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Realistic Roadmap

- reduce or remove the remaining legacy layer in `src/models`
- add stricter contract validation for gold and reporting outputs
- plug remote storage for artifacts and tracking
- expose inference and metadata through a versioned API contract

## Collaboration

Pull requests, bugs, and enhancements should follow:

- [CONTRIBUTING.md](CONTRIBUTING.md)

Clarity and reproducibility matter more than artificial complexity. The goal is to keep the repository readable, testable, and defensible in technical review.

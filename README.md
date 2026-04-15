# Churn Prediction Data Product

Language: **EN (primary)** | [PT-BR](README.pt-BR.md) | [PT-PT](README.pt-PT.md)

English is the canonical documentation language for the repository. PT-BR and PT-PT versions are maintained as localized references.

A production-minded data and machine learning project for customer churn, built to demonstrate how a small portfolio repository can behave like a credible data product instead of a notebook-based case study.

## Why This Repository Exists

Most churn portfolio projects stop at model training. This one is intentionally broader:

- it treats churn as a data system, not only as a classification problem
- it separates ingestion, transformation, modeling, reporting, and consumption
- it persists reusable artifacts instead of relying on notebook state
- it translates scores into business decisions and operator-facing outputs

The repository is designed to answer four practical questions:

1. Which customers have the highest churn risk?
2. Which customers should be prioritized first?
3. How should the decision threshold change when campaign cost changes?
4. Which artifacts does the business team need to act on the score?

## Business Value

The project produces outputs that are closer to what a retention or RevOps team would actually consume:

- prioritized customer list
- executive KPI summary
- action playbook
- model card
- drift monitoring output
- API and dashboard consumption paths

This matters because model quality alone is not enough. In practice, teams need traceability, repeatability, and a clear path from score to action.

The Streamlit experience is intentionally product-shaped rather than notebook-shaped: a shared layout shell, artifact-backed status banners, tabbed exploration, and dedicated pages for executive review, prioritization, and simulation.

## Architecture

Canonical flow:

`raw -> bronze -> silver -> gold -> model artifacts -> reporting -> dashboard/api`

Canonical implementation paths:

- `src/runtime/`: runtime configuration, environment handling, logging
- `src/pipelines/`: ingestion, transformation, validation, monitoring, reporting
- `src/modeling/`: training, inference, predictor contract
- `src/compat/`: explicit backward compatibility layer
- `scripts/`: operational scripts outside the core application packages
- `analytics_dbt/`: optional dbt layer for analytical marts

Compatibility wrappers still exist under `src/*.py` and some root entrypoints, but they are not the preferred path for new code.

UI-specific paths:

- `apps/streamlit_app.py`: control room and customer-level scoring entrypoint
- `apps/dashboard_ui.py`: shared dashboard layout shell, styling, and UI helpers
- `apps/dashboard_runtime.py`: shared dashboard asset loading and status helpers
- `apps/pages/`: executive, risk, prioritization, and simulation views
- `pages/`: Streamlit compatibility wrappers

See:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DASHBOARD.md](docs/DASHBOARD.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Tech Stack

- Python 3.12 (package target)
- Pandas / NumPy for batch data processing
- scikit-learn for model training and inference pipelines
- Pandera for dataframe contracts and schema checks
- MLflow for optional experiment tracking
- Streamlit for business-facing analytical consumption
- FastAPI for inference API surface
- Pytest, Ruff, Black, Isort, MyPy, Pre-commit for quality gates
- GitHub Actions for CI validation and package build

## Technical Decisions

- Layered local data model (`raw -> bronze -> silver -> gold`) to keep reruns explicit and reproducible.
- Atomic persistence for JSON/CSV/Markdown outputs to avoid partial writes in runtime artifacts.
- Artifact-first consumption pattern (dashboard and API consume persisted outputs, not notebook state).
- Compatibility wrappers are retained, but canonical implementation remains under `src/runtime`, `src/pipelines`, and `src/modeling`.
- Pipeline retries are bounded and now fail fast for non-retryable data-contract errors.

## Repository Map

```text
.
|-- .github/                  # CI, issue templates, PR template, ownership
|-- apps/                     # Streamlit, FastAPI, and shared dashboard helpers
|-- artifacts/                # execution metadata and generated runtime artifacts
|-- analytics_dbt/            # optional dbt project for analytical marts
|-- data/                     # raw/bronze/silver/gold local layers
|-- docs/                     # architecture, operations, repository conventions
|-- models/                   # local model registry and generated bundles
|-- notebooks/                # exploratory work only
|-- pages/                    # Streamlit compatibility wrappers
|-- scripts/                  # operational scripts and utilities
|-- src/                      # canonical implementation
|-- tests/                    # automated test suite
|-- requirements-runtime.txt  # runtime dependencies
|-- requirements-dev.txt      # dev, lint, and test dependencies
`-- README.md                 # main entry point
```

## Quickstart

### 1. Create the environment

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env
```

### 2. Add the dataset

Expected file:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Dataset source:

- Kaggle: `blastchar/telco-customer-churn`
- file used in this repository: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. Run the canonical pipeline

```bash
.venv\Scripts\python.exe -m src.cli.pipeline --data-dir data --log-level INFO --decision-policy balanceada --environment dev
```

### 4. Run quality gates

```bash
make test
make lint
make typecheck
```

## Runtime and Reliability

The repository implements controls proportional to its scope:

- configuration through `.env` and environment variables
- environment-aware runtime resolution
- structured logging with `run_id`
- execution metadata persisted under `artifacts/metadata/`
- lineage manifest persisted under `artifacts/metadata/lineage_run_<run_id>.json`
- governance manifest persisted under `artifacts/metadata/governance_run_<run_id>.json`
- atomic persistence for CSV, JSON, and Markdown
- schema validation before training
- regenerable bronze, silver, and gold layers
- retry in the canonical pipeline path
- drift monitoring using PSI and KS
- dashboard artifact status with explicit fallback signaling
- multipage dashboard with a shared visual shell and runtime-aware status
- EDA profile artifacts (`eda_profile.json` + `eda_report.md`) generated from silver layer
- LGPD-oriented pseudonymized output (`customer_prioritization_public.csv`)

Containerized runtime profiles are available through `docker-compose.yml`:

- `dev`: API + dashboard
- `pipeline`: one-shot pipeline execution
- `prod`: API with production environment profile

The default execution path is local and explicit by design. Prefect is kept only as an optional scheduling example, not as the canonical runtime dependency.

## Engineering Quality

Quality tooling:

- `pytest`
- `ruff`
- `black`
- `isort`
- `mypy`
- `pre-commit`
- `GitHub Actions`

Dependency split:

- `requirements-runtime.txt`: dashboard, API, pipeline, and runtime dependencies
- `requirements-dev.txt`: test, lint, hooks, and static analysis dependencies

## Trade-offs

This repository is intentionally production-minded, but it is not pretending to be a full platform.

Current trade-offs:

- local filesystem artifacts instead of remote storage
- lightweight drift monitoring instead of full observability stack
- optional scheduling example instead of a required orchestrator
- compatibility wrappers preserved to avoid breaking legacy entrypoints

These are conscious scope choices, not hidden gaps.

## Roadmap

- reduce the remaining legacy surface in `src/models` and root wrappers
- strengthen output contracts for gold and reporting artifacts
- align deployment metadata and runtime files completely
- add remote storage and richer operational deployment examples

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/README.md](docs/README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DASHBOARD.md](docs/DASHBOARD.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Contribution Standard

Changes should increase clarity, reduce maintenance cost, or improve reliability. The repository should stay readable, testable, and defensible in technical review.

# Architecture

## Objective

Model the churn use case as a data product, not only as an ML experiment.

## Design Principles

- one canonical execution path for the pipeline
- explicit responsibilities by layer
- contracts between modeling, reporting, and consumption layers
- backward compatibility isolated from the main implementation path
- artifacts and logs treated as first-class outputs

## Data Flow

```text
raw
  -> bronze
  -> silver
  -> eda
  -> gold
  -> governance
  -> modeling
  -> reporting
  -> analytics_dbt (optional)
  -> dashboard/api
```

## Core Domains

### Runtime and Configuration

- `src/runtime/config.py`
- `src/runtime/logging.py`
- `src/cli/pipeline.py`

Responsibility:

Define environment, `run_id`, resolved paths, execution metadata, and logging behavior.

### Ingestion and Quality Gate

- `src/pipelines/ingestion.py`
- `src/pipelines/transformation.py`
- `src/pipelines/validation.py`

Responsibility:

Guarantee that input data is valid and reproducible before training or reporting.

### Feature Engineering and Modeling

- `src/pipelines/feature_engineering.py`
- `src/modeling/churn.py`
- `src/modeling/pipeline.py`
- `src/modeling/predictor.py`

Responsibility:

Train, evaluate, persist, and serve inference artifacts.

### Analytics Output

- `src/pipelines/warehouse.py`
- `src/pipelines/reporting.py`
- `src/pipelines/eda.py`
- `src/pipelines/governance.py`
- `src/pipelines/dashboard_data.py`

Responsibility:

Produce gold outputs, EDA/governance artifacts, prioritization assets, KPIs, and operator-facing playbooks.

## Key Decisions

### Cost-Sensitive Decision Threshold

The global threshold is not fixed. It is derived from the selected campaign cost policy to reflect the operational context of retention.

### Compatibility Layers Remain Explicit

Packages such as `src/data`, `src/features`, `src/models`, and wrappers under `src/*.py` still exist for compatibility.
The compatibility boundary is explicit in `src/compat/`.
The canonical implementation lives under `src/runtime/`, `src/pipelines/`, and `src/modeling/`.

### Local and Explicit Orchestration

The primary pipeline path uses local orchestration with explicit retry behavior. This keeps the default execution path reproducible and avoids unnecessary coupling to a heavier external orchestrator for a portfolio-scale project.

### Proportional Monitoring

Drift monitoring uses PSI and KS in a lightweight implementation. The goal is to demonstrate sound engineering judgment without pretending this repository is a full production platform.

## Current Limits

- no distributed scheduler
- no external warehouse or lakehouse
- no remote artifact store
- no production deployment of inference serving

These limits are intentional. They should be treated as roadmap items, not hidden gaps.

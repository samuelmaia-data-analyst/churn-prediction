# Repository Structure

## Purpose

The repository is organized to make the canonical implementation path obvious while still preserving a compatibility boundary for older entrypoints.

The structure separates:

- canonical implementation
- operational scripts
- exploration
- documentation
- execution artifacts
- compatibility layers

## Annotated Tree

```text
.
|-- .github/                    # CI, issue templates, PR template, ownership
|-- apps/                       # Streamlit and FastAPI application entrypoints
|-- assets/                     # images, media, and repository visuals
|-- data/
|   |-- raw/                    # source input
|   |-- bronze/                 # regenerable ingestion layer
|   |-- silver/                 # regenerable cleaned layer
|   `-- gold/                   # regenerable analytics outputs
|-- docs/                       # architecture, operations, repository conventions
|-- notebooks/                  # exploratory work only
|-- pages/                      # Streamlit multi-page views
|-- scripts/                    # operational scripts and standalone utilities
|-- src/
|   |-- cli/                    # canonical CLI entrypoints
|   |-- compat/                 # explicit backward compatibility
|   |-- contracts/              # typed contracts and schemas
|   |-- modeling/               # training and inference logic
|   |-- pipelines/              # ingestion, transformation, reporting, monitoring
|   |-- runtime/                # configuration and structured logging
|   |-- utils/                  # small shared helpers
|   `-- *.py                    # compatibility wrappers at package root
|-- tests/                      # automated regression and contract coverage
|-- artifacts/                  # runtime outputs
`-- models/                     # local model registry
```

## Canonical Paths

New code should target:

- `src/runtime/`
- `src/pipelines/`
- `src/modeling/`
- `src/compat/` only when compatibility is the explicit goal
- `scripts/` for operational utilities that are not part of the application packages

## Non-Canonical Paths

The following exist primarily for backward compatibility or convenience:

- root entrypoints such as `app.py`, `api.py`, `main.py`
- wrappers under `src/*.py`
- legacy folders such as `src/data`, `src/features`, `src/models`

These paths should not receive new business logic unless the purpose of the change is compatibility itself.

## Placement Rules

### `src/`

Use for business logic, pipelines, runtime behavior, modeling, persistence, and reusable application code.

### `scripts/`

Use for operational scripts that can run independently of the app packages, such as standalone checks or utility workflows.

### `docs/`

Use for architecture, operations, collaboration standards, and repository conventions.

### `notebooks/`

Use for exploratory analysis only. Notebooks must not become dependencies of the canonical pipeline path.

### `tests/`

Use for regression, contract, runtime, and compatibility coverage.

## Why This Matters

A senior repository should make it easy to answer three questions immediately:

1. Where is the real implementation?
2. What is compatibility only?
3. Which entrypoint is the preferred one for execution and review?

This structure is intended to answer those questions without requiring guesswork.

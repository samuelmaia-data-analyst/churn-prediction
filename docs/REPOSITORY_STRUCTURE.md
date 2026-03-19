# Repository Structure

## Overview

The repository is organized to separate:

- canonical implementation
- exploration
- automation
- documentation
- execution artifacts
- operational scripts

## Annotated Tree

```text
.
|-- .github/
|   |-- workflows/               # CI
|   `-- ISSUE_TEMPLATE/          # issue templates
|-- apps/                        # application entry points
|-- assets/                      # images and demos
|-- data/
|   |-- raw/                     # expected input
|   |-- bronze/                  # regenerable layer
|   |-- silver/                  # regenerable layer
|   `-- gold/                    # regenerable layer
|-- docs/                        # architecture and operations documentation
|-- notebooks/                   # exploration and analysis
|-- pages/                       # Streamlit pages
|-- scripts/                     # operational scripts and batch utilities
|-- src/
|   |-- cli/                     # command-line interfaces
|   |-- compat/                  # explicit backward compatibility
|   |-- contracts/               # typed contracts
|   |-- modeling/                # training and inference
|   |-- pipelines/               # canonical pipeline and business logic
|   |-- runtime/                 # configuration and logging
|   |-- utils/                   # small shared helpers
|   `-- *.py                     # compatibility wrappers at package root
|-- tests/                       # automated tests
|-- artifacts/                   # execution outputs
`-- models/                      # local model registry
```

## Conventions

### What belongs in `src/`

Business logic, pipelines, modeling, persistence, and reusable application code.

### What does not belong in `src/`

Exploration, screenshots, raw data, GitHub templates, and ad-hoc notes.

### What belongs in `docs/`

Architecture, operations, repository conventions, and collaboration guidance.

### What belongs in `notebooks/`

EDA, experiments, and exploratory validation. Nothing in a notebook should be a dependency of the canonical pipeline path.

## Canonical Structure

Without breaking compatibility, the canonical structure is now:

```text
src/
  compat/
  contracts/
  pipelines/
  modeling/
  runtime/
  utils/
```

Wrappers at the root of `src/` still exist for backward compatibility. New code should target `src/runtime/`, `src/pipelines/`, `src/modeling/`, and `src/compat/`.

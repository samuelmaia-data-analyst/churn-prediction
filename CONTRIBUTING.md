# Contributing

This repository follows a contribution standard oriented toward engineering quality, operational clarity, and maintainability.

The goal is not to maximize process. The goal is to keep changes reviewable, reproducible, and aligned with the canonical architecture.

## Core Principles

- prefer clarity over cleverness
- treat the repository as a data product, not as a notebook collection
- avoid adding logic to compatibility wrappers unless there is a strong reason
- preserve idempotency, reprocessing, and artifact traceability
- document trade-offs when the decision is not obvious

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env
.venv\Scripts\python.exe -m pre_commit install
```

Dataset expected under:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Where Changes Should Go

- `src/runtime/`: configuration and logging
- `src/pipelines/`: ingestion, transformation, validation, reporting, monitoring
- `src/modeling/`: training and inference logic
- `src/compat/`: explicit backward compatibility only
- `scripts/`: operational scripts outside the application packages
- `tests/`: regression, contract, and runtime coverage
- `docs/`: architecture, operations, and repository conventions

Avoid introducing new canonical logic into:

- `src/*.py` compatibility wrappers
- root entrypoints that exist only for backward compatibility
- notebooks

## Change Types

When opening a contribution, be explicit about the category:

- bug fix
- pipeline reliability
- modeling improvement
- reporting or analytics output change
- runtime/configuration improvement
- documentation or collaboration workflow improvement

## Quality Expectations

Before opening a pull request:

```bash
make test
make lint
make typecheck
```

Add or update tests when the change affects:

- runtime behavior
- pipeline contracts
- artifact generation
- API or dashboard consumption
- compatibility wrappers
- dashboard fallback behavior or artifact status logic
- dashboard copy or page-level decision framing

## Pull Request Standard

Every PR should answer:

1. What real problem is being solved?
2. What technical behavior changes?
3. How was the change validated?
4. What risks or trade-offs remain?

PRs are likely to be rejected when they:

- increase complexity without a clear payoff
- mix large refactors and functional changes without isolating risk
- add hidden configuration or hardcoded environment assumptions
- weaken the reprocessable pipeline path
- change contracts without updating tests and docs

## Commit Guidance

Conventional Commits are not mandatory, but commit messages should be specific and auditable.

Good examples:

- `align automation dependencies and operational scripts`
- `restructure canonical runtime and pipeline packages`
- `fix reporting policy mismatch with training threshold`

## Documentation Policy

Update documentation when the change affects:

- setup
- runtime behavior
- architecture
- repository structure
- contribution flow

Primary docs to keep in sync:

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DASHBOARD.md](docs/DASHBOARD.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Issue and PR Workflow

- use issue templates for defects, feature proposals, and data quality incidents
- open a discussion issue before large structural changes when possible
- keep PRs scoped enough that validation and rollback reasoning remain clear

## Review Standard

The repository is intended to look credible to senior engineers and hiring reviewers. Changes should therefore improve at least one of:

- clarity
- maintainability
- reliability
- testability
- operational realism

If a change does not improve one of those dimensions, it probably does not belong here.

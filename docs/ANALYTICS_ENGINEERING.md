# Analytics Engineering (dbt)

## Purpose

This repository now includes an optional dbt layer under `analytics_dbt/` to model analytical marts from the silver output.

The intent is to separate:

- data generation and ML lifecycle in Python (`src/`)
- analytical semantic modeling in SQL (`analytics_dbt/models/`)

## Project Layout

- `analytics_dbt/dbt_project.yml`
- `analytics_dbt/profiles.yml.example`
- `analytics_dbt/models/staging/stg_customer_churn.sql`
- `analytics_dbt/models/marts/fct_contract_risk.sql`
- `analytics_dbt/models/schema.yml`

## Running Locally

1. Install dbt duckdb adapter:

```bash
pip install dbt-duckdb
```

2. Copy profile template:

```bash
copy analytics_dbt\profiles.yml.example analytics_dbt\profiles.yml
```

3. Build models:

```bash
dbt --project-dir analytics_dbt --profiles-dir analytics_dbt run
dbt --project-dir analytics_dbt --profiles-dir analytics_dbt test
```

## Design Notes

- The staging model reads directly from `data/silver/customer_churn_silver.csv`.
- Marts are declarative SQL models with schema tests for critical fields.
- This keeps the analytics contract explicit and reviewable in pull requests.

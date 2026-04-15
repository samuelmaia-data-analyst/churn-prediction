# Governance and LGPD

## Scope

This repository applies proportional data governance controls for a portfolio-grade churn system:

- dataset classification (identifier, sensitive, behavioral)
- pseudonymized analytical export for downstream consumers
- retention metadata and privacy mode in runtime configuration
- auditable governance manifest per run

## Runtime Controls

Environment variables:

- `CHURN_LGPD_MODE`: `standard` or `strict`
- `CHURN_GOV_RETENTION_DAYS`: retention policy metadata
- `CHURN_LGPD_SALT`: tokenization salt for pseudonymized output

## Artifacts

- `data/gold/customer_prioritization.csv`: restricted analytical view
- `data/gold/customer_prioritization_public.csv`: pseudonymized view
- `artifacts/metadata/governance_run_<run_id>.json`
- `artifacts/metadata/latest_governance.json`

## Operating Notes

- `strict` mode removes direct identifier `customerID` from public prioritization output.
- governance manifest does not replace legal review; it makes privacy controls explicit for technical review.

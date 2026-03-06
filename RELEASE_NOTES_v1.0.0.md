# Release Notes - v1.0.0 (Official)

Date: 2026-03-06

## Executive Summary
Enterprise-grade churn analytics release with:
- end-to-end pipeline (`raw -> bronze -> silver -> gold`)
- churn and next-purchase prediction
- decision layer (prioritization + playbook)
- executive dashboard and drift monitoring

## Business Outcome Simulation
- Protected value (prioritized portfolio): `~$68k` per cycle
- Expected net impact (base scenario): `+$16k` per cycle
- Adoption assumption: `70%` playbook coverage with `<24h` first touch

## What is included
- Data contracts and quality gates (`Pandera`)
- Model comparison and explainability outputs
- Cost-sensitive threshold policy by LTV segment
- Action playbook output (`data/gold/action_playbook.csv`)
- Executive outputs (`executive_report.json`, `executive_brief.md`)
- Drift monitoring (`PSI`, `KS`) with runtime alert file

## Operating Model
- Daily orchestration with `Prefect` (`07:00 UTC`)
- Model lineage and tracking with `MLflow`
- CI quality gates (`ruff`, `black`, `pytest`)

## Artifacts
- Model: `models/model_v1.pkl`
- Model metadata: `models/model_metadata.json`
- Executive report: `artifacts/reports/executive_report.json`
- Playbook report: `artifacts/reports/action_playbook.md`

## Compatibility and Notes
- If `xgboost` is unavailable, the pipeline falls back to `GradientBoosting`.
- Streamlit app includes safe bootstrap and synthetic fallback for empty environments.

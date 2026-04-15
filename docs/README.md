# Docs Index

Supporting documentation for the canonical architecture, runtime model, and collaboration standard.

Use this folder to understand how the repository is meant to behave in technical review and in day-to-day contribution.

## Reading Order

1. [ARCHITECTURE.md](ARCHITECTURE.md)
2. [OPERATIONS.md](OPERATIONS.md)
3. [DASHBOARD.md](DASHBOARD.md)
4. [ANALYTICS_ENGINEERING.md](ANALYTICS_ENGINEERING.md)
5. [GOVERNANCE.md](GOVERNANCE.md)
6. [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md)
7. [releases/RELEASE_NOTES_v1.0.0.md](releases/RELEASE_NOTES_v1.0.0.md)

## Document Roles

- `ARCHITECTURE.md`: canonical boundaries between runtime, pipelines, modeling, compatibility, and outputs
- `OPERATIONS.md`: local setup, execution path, quality gates, and operational expectations
- `DASHBOARD.md`: Streamlit consumption layer, artifact sources, fallback semantics, and product intent
- `ANALYTICS_ENGINEERING.md`: dbt-layer conventions, local profile, and model execution
- `GOVERNANCE.md`: LGPD controls, pseudonymization, and governance artifacts
- `REPOSITORY_STRUCTURE.md`: ownership of folders, scripts, compatibility wrappers, and root-level conventions
- `releases/`: release notes and change summaries

## Maintenance Rule

If a change affects setup, runtime behavior, dashboard consumption, or canonical package boundaries, update the corresponding document in the same pull request.

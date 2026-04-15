# Operations

## Configuration

The project supports configuration through `.env` and environment variables.

Primary variables:

- `CHURN_ENV`
- `CHURN_DATA_DIR`
- `CHURN_ARTIFACTS_DIR`
- `CHURN_MODEL_REGISTRY_DIR`
- `CHURN_SEED`
- `CHURN_LOG_LEVEL`
- `CHURN_DECISION_POLICY`
- `CHURN_MLFLOW_TRACKING_URI`
- `CHURN_LGPD_MODE` (`standard` or `strict`)
- `CHURN_GOV_RETENTION_DAYS`
- `CHURN_LGPD_SALT`

## Local Execution

### Main pipeline

```bash
.venv\Scripts\python.exe -m src.cli.pipeline --data-dir data --log-level INFO --decision-policy balanceada --environment dev
```

This is the canonical execution path for local runs and for repository evaluation.

### Standalone drift detection

```bash
.venv\Scripts\python.exe -m scripts.drift_detection --baseline data/baseline.csv --current data/current.csv
```

### Optional Prefect deployment

`prefect.yaml` is kept as an optional scheduling example, not as the primary execution path.

### Dependency installation

```bash
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### Containerized profiles

Dev profile (API + dashboard):

```bash
docker compose --profile dev up --build
```

Pipeline profile:

```bash
docker compose --profile pipeline run --rm pipeline
```

Prod profile (API only):

```bash
docker compose --profile prod up --build
```

### Tests

```bash
make test
```

### Lint and type check

```bash
make lint
make typecheck
```

## Relevant Artifacts

### Logs

- `artifacts/logs/pipeline.log`
- `artifacts/logs/pipeline_<run_id>.log`

### Execution metadata

- `artifacts/metadata/latest_run.json`
- `artifacts/metadata/pipeline_run_<run_id>.json`
- `artifacts/metadata/latest_lineage.json`
- `artifacts/metadata/lineage_run_<run_id>.json`
- `artifacts/metadata/latest_governance.json`
- `artifacts/metadata/governance_run_<run_id>.json`
- metadata now includes input fingerprint (`raw_sha256`) and per-stage telemetry (`stages`)

### Reports

- `artifacts/reports/executive_report.json`
- `artifacts/reports/model_card.md`
- `artifacts/reports/executive_brief.md`
- `artifacts/reports/action_playbook.md`
- `artifacts/reports/data_quality_report.json`
- `artifacts/reports/eda_report.md`

### Models

- `artifacts/models/enterprise_churn_model.joblib`
- `artifacts/models/enterprise_churn_bundle.joblib`
- `models/registry_manifest.json`
- `models/README.md`

### Gold layer manifest

- `data/gold/_manifest.json`
- `data/gold/eda_profile.json`
- `data/gold/customer_prioritization_public.csv` (pseudonymized view)

## Reprocessing

The pipeline is structured to allow local reruns without hidden in-memory state.

Practical rules:

- treat `data/raw/` as input
- treat `data/bronze`, `data/silver`, and `data/gold` as regenerable layers
- treat `artifacts/` and `models/` as execution outputs

Retry policy notes:

- transient task failures are retried with backoff
- non-retryable failures (for example invalid input schema) fail fast

## Common Incidents

### Dataset missing

Symptom:

`FileNotFoundError` during ingestion.

Action:

Place the expected CSV file under `data/raw/`.

### Environment missing dependencies

Symptom:

import failures for `joblib`, `mlflow`, or `mypy`.

Action:

Reinstall runtime and development dependencies with:

```bash
pip install -r requirements-runtime.txt
pip install -r requirements-dev.txt
```

### Inference bundle missing

Symptom:

The API or dashboard starts without readiness or fails to predict.

Action:

Run the main pipeline again to rebuild the versioned inference artifacts.

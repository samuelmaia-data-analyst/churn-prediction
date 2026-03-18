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

## Local Execution

### Main pipeline

```bash
.venv\Scripts\python.exe -m src.cli.pipeline --data-dir data --log-level INFO --decision-policy balanceada --environment dev
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

### Reports

- `artifacts/reports/executive_report.json`
- `artifacts/reports/model_card.md`
- `artifacts/reports/executive_brief.md`
- `artifacts/reports/action_playbook.md`
- `artifacts/reports/data_quality_report.json`

### Models

- `artifacts/models/enterprise_churn_model.joblib`
- `artifacts/models/enterprise_churn_bundle.joblib`
- `models/registry_manifest.json`
- `models/README.md`

### Gold layer manifest

- `data/gold/_manifest.json`

## Reprocessing

The pipeline is structured to allow local reruns without hidden in-memory state.

Practical rules:

- treat `data/raw/` as input
- treat `data/bronze`, `data/silver`, and `data/gold` as regenerable layers
- treat `artifacts/` and `models/` as execution outputs

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

Reinstall dependencies with `pip install -r requirements.txt`.

### Inference bundle missing

Symptom:

The API or dashboard starts without readiness or fails to predict.

Action:

Run the main pipeline again to rebuild the versioned inference artifacts.

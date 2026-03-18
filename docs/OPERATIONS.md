# Operations

## Configuração

O projeto suporta configuração por `.env` e variáveis de ambiente.

Variáveis principais:

- `CHURN_ENV`
- `CHURN_DATA_DIR`
- `CHURN_ARTIFACTS_DIR`
- `CHURN_MODEL_REGISTRY_DIR`
- `CHURN_SEED`
- `CHURN_LOG_LEVEL`
- `CHURN_DECISION_POLICY`
- `CHURN_MLFLOW_TRACKING_URI`

## Execução Local

### Pipeline principal

```bash
.venv\Scripts\python.exe -m src.cli.pipeline --data-dir data --log-level INFO --decision-policy balanceada --environment dev
```

### Testes

```bash
make test
```

### Lint e type check

```bash
make lint
make typecheck
```

## Artefatos Relevantes

### Logs

- `artifacts/logs/pipeline.log`
- `artifacts/logs/pipeline_<run_id>.log`

### Metadata de execução

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

## Reprocessamento

O pipeline foi estruturado para permitir rerun local sem dependência de estado manual em memória.

Regras práticas:

- trate `data/raw/` como entrada
- trate `data/bronze`, `data/silver`, `data/gold` como camadas regeneráveis
- trate `artifacts/` e `models/` como outputs de execução

## Incidentes Comuns

### Dataset ausente

Sintoma:

`FileNotFoundError` na ingestão.

Ação:

adicionar o CSV esperado em `data/raw/`.

### Ambiente sem dependências

Sintoma:

falha em import de `joblib`, `mlflow` ou `mypy`.

Ação:

reinstalar dependências com `pip install -r requirements.txt`.

### Bundle de inferência ausente

Sintoma:

API ou dashboard iniciam sem readiness ou falham ao prever.

Ação:

executar novamente o pipeline principal para reconstruir os artefatos versionados de inferência.

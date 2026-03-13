# Churn Prediction for Customer Retention

Language: **International** | [PT-BR](README.pt-BR.md) | [EURO](README.euro.md)

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-2ea44f)](https://data-senior-analytics.streamlit.app/)

Professional analytics and machine learning case focused on customer retention. The project predicts churn, prioritizes customers by risk and value, and translates model outputs into retention actions that business teams can execute.

## Business Problem

Subscription businesses lose revenue when they react only after cancellation. A churn model becomes valuable when it helps answer:

- which customers are most likely to leave
- which customers should be prioritized first
- what trade-off exists between campaign cost and missed churners
- which retention levers are supported by the model drivers

## Canonical Architecture

The project uses one primary ML path:

- `src/ingestion.py`: raw ingestion and bronze layer
- `src/transformation.py`: silver preparation and schema checks
- `src/feature_engineering.py`: reusable business features
- `src/modeling/pipeline.py`: model training, evaluation, persistence, MLflow
- `src/modeling/predictor.py`: inference from saved artifacts
- `src/reporting.py`: executive outputs, model card, action playbook
- `tests/`: preprocessing, feature engineering, training and inference checks

Legacy folders under `src/data`, `src/features`, and `src/models` remain only as compatibility wrappers.

## Modeling Approach

The training pipeline:

1. validates schema and training readiness
2. engineers business features such as `charges_per_tenure`, `service_count`, `is_month_to_month`
3. compares `Logistic Regression`, `Random Forest`, and `XGBoost` fallback
4. selects the champion model by `ROC-AUC`
5. evaluates with:
   - `Precision`
   - `Recall`
   - `F1-score`
   - `ROC-AUC`
   - confusion matrix
   - risk profile summary
6. persists:
   - champion model
   - inference bundle
   - metadata
   - executive report artifacts

## Cost-Sensitive Thresholding

The global classification threshold is derived from business error cost:

`threshold = FP_cost / (FP_cost + FN_cost)`

Available policies:

- `campanha_cara`
- `balanceada`
- `campanha_barata`

Business interpretation:

- expensive campaign: higher threshold, higher precision
- cheaper campaign: lower threshold, higher recall

## Standard Commands

```bash
make install
make train
make train-cheap
make train-expensive
make test
make lint
make predict
```

Equivalent training command:

```bash
python -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy balanceada
```

## Artifacts

Main generated artifacts:

- `artifacts/models/enterprise_churn_model.joblib`
- `artifacts/models/enterprise_churn_bundle.joblib`
- `models/model_v1.pkl`
- `models/model_metadata.json`
- `artifacts/reports/executive_report.json`
- `artifacts/reports/model_card.md`
- `artifacts/reports/executive_brief.md`
- `artifacts/reports/action_playbook.md`

## Model Interpretation

The project exposes both feature importance and business interpretation:

- contract structure and tenure are relevant churn drivers
- high-risk profiles are surfaced by contract and internet service mix
- false positives and false negatives are explained in operational and financial terms

This supports retention actions such as:

- migrating fragile contracts to more stable plans
- intervening on price-sensitive customers
- improving onboarding and support for low-tenure customers

## Validation

Validated locally with:

```bash
.venv\Scripts\python.exe -m pytest -q tests/test_data.py tests/test_features.py tests/test_models.py tests/test_enterprise_pipeline.py tests/test_reporting_contract.py
```

## Dataset

Source: Kaggle Telco Customer Churn  
Expected file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

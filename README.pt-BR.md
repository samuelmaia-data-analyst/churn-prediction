# Churn Prediction para Retencao de Clientes

Idioma: [International](README.md) | **PT-BR** | [EURO](README.euro.md)

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-2ea44f)](https://data-senior-analytics.streamlit.app/)

Case profissional de analytics e machine learning focado em retencao de clientes. O projeto preve churn, prioriza clientes por risco e valor e converte os outputs do modelo em acoes que times de negocio conseguem executar.

## Problema de Negocio

Empresas de assinatura perdem receita quando reagem apenas apos o cancelamento. Um modelo de churn gera valor quando ajuda a responder:

- quais clientes tem maior probabilidade de sair
- quais clientes devem ser priorizados primeiro
- qual o trade-off entre custo de campanha e clientes perdidos
- quais alavancas de retencao sao sustentadas pelos drivers do modelo

## Arquitetura Canonica

O projeto usa uma trilha principal de ML:

- `src/ingestion.py`: ingestao do dado bruto e camada bronze
- `src/transformation.py`: preparacao silver e checagens de schema
- `src/feature_engineering.py`: features de negocio reutilizaveis
- `src/modeling/pipeline.py`: treino, avaliacao, persistencia e MLflow
- `src/modeling/predictor.py`: inferencia a partir de artefatos salvos
- `src/reporting.py`: outputs executivos, model card e playbook
- `tests/`: testes de preprocessamento, features, treino e inferencia

Pastas legadas em `src/data`, `src/features` e `src/models` permanecem apenas como compatibilidade.

## Abordagem de Modelagem

A pipeline de treino:

1. valida schema e prontidao para treino
2. cria features como `charges_per_tenure`, `service_count` e `is_month_to_month`
3. compara `Logistic Regression`, `Random Forest` e fallback de `XGBoost`
4. seleciona o modelo campeao por `ROC-AUC`
5. avalia com:
   - `Precision`
   - `Recall`
   - `F1-score`
   - `ROC-AUC`
   - matriz de confusao
   - resumo de perfis de maior risco
6. persiste:
   - modelo campeao
   - bundle de inferencia
   - metadata
   - artefatos executivos

## Threshold Sensivel a Custo

O threshold global de classificacao e derivado do custo de erro de negocio:

`threshold = FP_cost / (FP_cost + FN_cost)`

Politicas disponiveis:

- `campanha_cara`
- `balanceada`
- `campanha_barata`

Leitura de negocio:

- campanha cara: threshold mais alto, maior precision
- campanha barata: threshold mais baixo, maior recall

## Comandos Padrao

```bash
make install
make train
make train-cheap
make train-expensive
make test
make lint
make predict
```

Comando equivalente de treino:

```bash
python -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy balanceada
```

## Artefatos

Principais artefatos gerados:

- `artifacts/models/enterprise_churn_model.joblib`
- `artifacts/models/enterprise_churn_bundle.joblib`
- `models/model_v1.pkl`
- `models/model_metadata.json`
- `artifacts/reports/executive_report.json`
- `artifacts/reports/model_card.md`
- `artifacts/reports/executive_brief.md`
- `artifacts/reports/action_playbook.md`

## Interpretacao do Modelo

O projeto expoe importancia de features e interpretacao de negocio:

- estrutura contratual e tenure sao drivers relevantes
- perfis de maior risco aparecem por combinacao de contrato e servico de internet
- falsos positivos e falsos negativos sao explicados em termos operacionais e financeiros

Isso suporta acoes como:

- migracao de contratos frageis para planos mais estaveis
- intervencao em clientes sensiveis a preco
- reforco de onboarding e suporte para clientes com baixo tenure

## Validacao

Validado localmente com:

```bash
.venv\Scripts\python.exe -m pytest -q tests/test_data.py tests/test_features.py tests/test_models.py tests/test_enterprise_pipeline.py tests/test_reporting_contract.py
```

## Dataset

Fonte: Kaggle Telco Customer Churn  
Arquivo esperado: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

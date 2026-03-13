# Churn Prediction para Retencao de Clientes

Idioma: [International](README.md) | [PT-BR](README.pt-BR.md) | **EURO**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-2ea44f)](https://data-senior-analytics.streamlit.app/)

Caso profissional de analytics e machine learning focado na retencao de clientes. O projeto preve churn, prioriza clientes por risco e valor e converte os outputs do modelo em acoes que equipas de negocio conseguem executar.

## Problema de Negocio

Empresas de subscricao perdem receita quando reagem apenas apos o cancelamento. Um modelo de churn gera valor quando ajuda a responder:

- que clientes tem maior probabilidade de sair
- que clientes devem ser priorizados primeiro
- qual e o trade-off entre custo de campanha e clientes perdidos
- que alavancas de retencao sao suportadas pelos drivers do modelo

## Arquitetura Canonica

O projeto utiliza uma trilha principal de ML:

- `src/ingestion.py`: ingestao de dados brutos e camada bronze
- `src/transformation.py`: preparacao silver e verificacoes de schema
- `src/feature_engineering.py`: features de negocio reutilizaveis
- `src/modeling/pipeline.py`: treino, avaliacao, persistencia e MLflow
- `src/modeling/predictor.py`: inferencia a partir de artefactos guardados
- `src/reporting.py`: outputs executivos, model card e playbook
- `tests/`: testes de preprocessamento, features, treino e inferencia

Pastas antigas em `src/data`, `src/features` e `src/models` mantem-se apenas por compatibilidade.

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
   - resumo de perfis com maior risco
6. persiste:
   - modelo campeao
   - bundle de inferencia
   - metadata
   - artefactos executivos

## Threshold Sensivel ao Custo

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

## Artefactos

Principais artefactos gerados:

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
- perfis de maior risco aparecem pela combinacao de contrato e servico de internet
- falsos positivos e falsos negativos sao explicados em termos operacionais e financeiros

Isto suporta acoes como:

- migracao de contratos frageis para planos mais estaveis
- intervencao em clientes sensiveis ao preco
- reforco de onboarding e suporte para clientes com baixo tenure

## Validacao

Validado localmente com:

```bash
.venv\Scripts\python.exe -m pytest -q tests/test_data.py tests/test_features.py tests/test_models.py tests/test_enterprise_pipeline.py tests/test_reporting_contract.py
```

## Dataset

Fonte: Kaggle Telco Customer Churn  
Ficheiro esperado: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

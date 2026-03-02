# Plataforma de Predicao de Churn

[![Status](https://img.shields.io/badge/status-em_desenvolvimento-yellow)](#roadmap)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](#stack-tecnologica)
[![Machine Learning](https://img.shields.io/badge/machine_learning-scikit--learn-orange)](#performance-do-modelo)
[![API](https://img.shields.io/badge/api-fastapi-009688)](#contrato-da-api)
[![Dashboard](https://img.shields.io/badge/dashboard-streamlit-red)](#demonstracao)

Idioma: **PT-BR** | [English](README.en.md)

Projeto de previsao de churn com pipeline de Machine Learning, API FastAPI e dashboard Streamlit para apoiar decisoes de retencao de clientes.

## Resumo Executivo

- Solucao end-to-end de **Customer Churn Prediction** com Python e scikit-learn.
- Camada de inferencia deployavel com **FastAPI** e analise visual com **Streamlit + Plotly**.
- Pipeline de preprocessamento persistido para inferencia consistente entre treino e producao.
- Modelo salvo atual com alta capacidade de ranking (**ROC-AUC 0.8420**).

## Contexto de Negocio

### Problema
Empresas com receita recorrente perdem margem quando o churn e identificado tardiamente.

### Solucao
Pipeline supervisionado de classificacao binaria para estimar probabilidade de churn por cliente, com exposicao via API e dashboard.

### Resultado Esperado
Permite priorizar clientes de alto risco, otimizar budget de retencao e proteger receita.

## Resultados-Chave

| Metrica | Valor |
|---|---:|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1-score | 0.6040 |
| ROC-AUC | 0.8420 |

Artefato principal: `models/LogisticRegression.joblib`

## Arquitetura da Solucao

![Arquitetura do Projeto](assets/architecture.png)

```text
Dados CSV
  -> Limpeza e split
  -> Engenharia de features + preprocessamento
  -> Treino e selecao do modelo
  -> Persistencia de artefatos (modelo + preprocessor)
  -> Consumo via API / Dashboard / CLI
```

## Stack Tecnologica

- **Linguagem:** Python
- **Dados e ML:** pandas, numpy, scikit-learn
- **Persistencia:** joblib
- **API:** FastAPI, Pydantic, Uvicorn
- **Dashboard:** Streamlit, Plotly

## Funcionalidades

- Pipeline completo de modelagem de churn.
- Dashboard interativo com filtros e predicao individual.
- Endpoint REST para scoring em tempo real.
- Reuso do mesmo preprocessor no app e na API (evita train-serving skew).

## Demonstracao

| API Demo | Dashboard Demo |
|---|---|
| ![API REST Demo](assets/api-demo.gif) | ![Dashboard Demo](assets/dashboard-demo.gif) |

## Contrato da API

### Health
- `GET /health`

### Predicao
- `POST /predict`

Exemplo de request:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0
}
```

Exemplo de response (implementacao atual):

```json
{
  "churn": "Sim",
  "probability": 0.73,
  "risk_level": "Alto"
}
```

## Setup Rapido

```bash
git clone <url-do-repositorio>
cd churn-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
uvicorn api:app --reload
# em outro terminal: streamlit run app.py
```

## Estrutura do Repositorio

```text
churn-prediction/
|-- app.py
|-- api.py
|-- main.py
|-- predict_customer.py
|-- config.yaml
|-- requirements.txt
|-- data/
|-- models/
|-- src/
|   |-- data/
|   |-- features/
|   `-- models/
|-- tests/
`-- assets/
```

## Decisoes de Engenharia

- Selecao de modelo por **F1-score** para balancear precision e recall.
- Persistencia de `preprocessor.joblib` para consistencia de features.
- API e dashboard desacoplados, consumindo os mesmos artefatos.

## Qualidade e Testes

Estado atual:
- estrutura de testes existe em `tests/`, mas suites ainda nao implementadas.

Proximos passos:
- testes unitarios de preprocessamento;
- testes de contrato da API;
- validacao de schema de entrada do modelo.

## Roadmap

- Cobertura automatizada de testes (unit + integration).
- Rastreabilidade de experimentos e metricas.
- Monitoramento de drift de dados/modelo.
- Batch scoring na API.
- CI/CD para validacao e release.

## Palavras-chave ATS

`Python` `Machine Learning` `Churn Prediction` `scikit-learn` `FastAPI` `Streamlit` `Model Deployment` `REST API` `Data Science` `MLOps` `Feature Engineering` `Binary Classification` `Model Evaluation` `ROC-AUC`

## Contato

**Samuel de Andrade Maia**
- GitHub: https://github.com/samuelmaia-data-analyst
- LinkedIn: https://linkedin.com/in/samuelmaia-data-analyst

## Licenca

Licenca ainda nao definida. Recomendado: MIT.

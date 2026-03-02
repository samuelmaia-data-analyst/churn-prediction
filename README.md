# Churn Prediction

[![Status](https://img.shields.io/badge/status-em_desenvolvimento-yellow)](#status)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](#requisitos)
[![Streamlit](https://img.shields.io/badge/interface-streamlit-red)](#executando-o-dashboard)
[![FastAPI](https://img.shields.io/badge/api-fastapi-009688)](#executando-a-api)

Idioma: **Portugues** | [English](README.en.md)

Sistema de predicao de churn com pipeline de ML, dashboard em Streamlit e API de inferencia em FastAPI.

## Sumario

- [Visao Executiva](#visao-executiva)
- [Impacto de Negocio](#impacto-de-negocio)
- [Status](#status)
- [Escopo Funcional](#escopo-funcional)
- [Arquitetura da Solucao](#arquitetura-da-solucao)
- [Demonstracao](#demonstracao)
- [Stack Tecnica](#stack-tecnica)
- [Estrutura do Repositorio](#estrutura-do-repositorio)
- [Requisitos](#requisitos)
- [Setup Local](#setup-local)
- [Fluxo de Execucao](#fluxo-de-execucao)
- [Executando o Dashboard](#executando-o-dashboard)
- [Executando a API](#executando-a-api)
- [Exemplo de Uso da API](#exemplo-de-uso-da-api)
- [Metricas do Modelo Atual](#metricas-do-modelo-atual)
- [Testes](#testes)
- [Limitacoes Atuais](#limitacoes-atuais)
- [Roadmap](#roadmap)
- [Contribuicao](#contribuicao)
- [Licenca](#licenca)
- [Contato](#contato)

## Visao Executiva

Este projeto resolve um problema de receita recorrente: identificar clientes com maior risco de cancelamento para priorizar acoes de retencao. A solucao entrega:

- pipeline supervisionado de treino e avaliacao;
- selecao automatica do melhor modelo por `F1`;
- dashboard analitico para exploracao de churn;
- API REST para inferencia por cliente.

## Impacto de Negocio

KPIs que a solucao suporta:

- reducao de churn em cohorts de maior risco;
- aumento de efetividade de campanhas de retencao;
- priorizacao operacional por probabilidade de churn;
- melhor previsibilidade de receita recorrente.

Sugestao de acompanhamento executivo:

- Churn Rate mensal (%);
- Retencao apos acao (%);
- Lift de retencao (grupo tratado vs controle);
- Receita preservada por campanha.

## Status

Em desenvolvimento.

## Escopo Funcional

- Classificacao binaria de churn (`Yes`/`No`).
- Treino e comparacao de modelos (`LogisticRegression`, `RandomForest`, `GradientBoosting`).
- Persistencia de artefatos em `models/`.
- Inferencia por script local, dashboard e endpoint HTTP.

## Arquitetura da Solucao

![Arquitetura do Projeto](assets/architecture.png)

Fluxo principal:

```text
CSV Dataset -> Data + Feature Pipeline -> Model Training -> Artifacts
                                                   |
                                                   +-> Streamlit Dashboard
                                                   +-> FastAPI Endpoint
                                                   +-> CLI Prediction
```

Regra de selecao do melhor modelo no codigo: maior `F1` no conjunto de teste.

## Demonstracao

| API Demo | Dashboard Demo |
|---|---|
| ![API REST Demo](assets/api-demo.gif) | ![Dashboard Demo](assets/dashboard-demo.gif) |
| API REST com FastAPI - documentacao interativa automatica | Dashboard interativo - visualizacao de metricas e predicoes |

## Stack Tecnica

- Linguagem: `Python`
- Dados: `pandas`, `numpy`
- ML: `scikit-learn`
- Persistencia: `joblib`
- Dashboard: `Streamlit`, `Plotly`
- API: `FastAPI`, `Pydantic`, `Uvicorn`

## Estrutura do Repositorio

```text
churn-prediction/
|-- api.py
|-- app.py
|-- main.py
|-- predict_customer.py
|-- save_processed_data.py
|-- config.yaml
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|   |-- data/
|   |-- features/
|   |-- models/
|   `-- visualization/
|-- tests/
`-- assets/
```

## Requisitos

- Python 3.12+
- `pip`

Observacao: `requirements.txt` menciona compatibilidade com Python 3.13.

## Setup Local

```bash
git clone <url-do-repositorio>
cd churn-prediction
python -m venv .venv
```

Ativar ambiente virtual:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Fluxo de Execucao

1. Treinar modelos e salvar artefatos:

```bash
python main.py
```

2. Opcional: gerar base processada para consumo analitico:

```bash
python save_processed_data.py
```

3. Subir interfaces de consumo (dashboard/API).

## Executando o Dashboard

```bash
streamlit run app.py
```

## Executando a API

```bash
uvicorn api:app --reload
```

Endpoints disponiveis:

- `GET /` retorna status basico da API.
- `GET /health` valida carga de modelo e pre-processador.
- `POST /predict` retorna predicao, probabilidade e nivel de risco.

## Exemplo de Uso da API

Payload para `POST /predict`:

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

Resposta esperada (API atual):

```json
{
  "churn": "Sim",
  "probability": 0.73,
  "risk_level": "Alto"
}
```

## Metricas do Modelo Atual

Metricas reportadas para o modelo salvo atual (`models/LogisticRegression.joblib`), com `test_size=0.2` e `random_state=42`:

- Accuracy: `0.8055`
- Precision: `0.6572`
- Recall: `0.5588`
- F1-score: `0.6040`
- ROC-AUC: `0.8420`

As metricas podem variar apos retreino.

## Testes

A estrutura de testes existe em `tests/`, mas os arquivos atuais estao vazios.

Comando padrao:

```bash
pytest -q
```

## Limitacoes Atuais

- Testes automatizados ainda nao implementados.
- Versionamento formal de experimentos/metricas ainda nao implementado.
- Licenca do projeto ainda nao definida.
- Alguns arquivos do projeto ainda possuem problemas de encoding.

## Roadmap

- Implementar testes unitarios e de integracao.
- Versionar metricas e artefatos por experimento.
- Adicionar monitoramento de drift de dados/modelo.
- Evoluir API para batch scoring.
- Integrar pipeline com CI/CD.
- Avaliar modelos adicionais (ex.: XGBoost, LightGBM).

## Contribuicao

1. Faca um fork do projeto.
2. Crie uma branch de feature: `git checkout -b feature/minha-feature`.
3. Commit suas mudancas: `git commit -m "feat: minha feature"`.
4. Abra um Pull Request.

## Licenca

Pendente. Recomenda-se adicionar `LICENSE` (ex.: MIT).

## Contato

- Samuel de Andrade Maia
- GitHub: https://github.com/samuelmaia-data-analyst
- LinkedIn: https://linkedin.com/in/samuelmaia-data-analyst

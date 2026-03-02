# Churn Prediction

[![Status](https://img.shields.io/badge/status-em_desenvolvimento-yellow)](#status)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](#requisitos)
[![Streamlit](https://img.shields.io/badge/interface-streamlit-red)](#executando-o-dashboard)
[![FastAPI](https://img.shields.io/badge/api-fastapi-009688)](#executando-a-api)

Idioma: **Português** | [English](README.en.md)

Sistema de predição de churn de clientes com pipeline de ML em Python, dashboard em Streamlit e API de inferência com FastAPI.

## Sumário

- [Visão Executiva](#visão-executiva)
- [Status](#status)
- [Escopo Funcional](#escopo-funcional)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Stack Técnica](#stack-técnica)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Requisitos](#requisitos)
- [Setup Local](#setup-local)
- [Fluxo de Execução](#fluxo-de-execução)
- [Executando o Dashboard](#executando-o-dashboard)
- [Executando a API](#executando-a-api)
- [Exemplo de Uso da API](#exemplo-de-uso-da-api)
- [Métricas do Modelo Atual](#métricas-do-modelo-atual)
- [Testes](#testes)
- [Limitações Atuais](#limitações-atuais)
- [Roadmap](#roadmap)
- [Contribuição](#contribuição)
- [Licença](#licença)
- [Contato](#contato)

## Visão Executiva

Este projeto resolve um problema clássico de receita recorrente: identificar clientes com maior risco de cancelamento para priorizar ações de retenção. A solução inclui:

- pipeline de dados e treino de modelos supervisionados;
- seleção automática do melhor modelo por `F1`;
- dashboard interativo para análise de churn;
- API REST para inferência online por cliente.

## Status

Em desenvolvimento.

## Escopo Funcional

- Classificação binária de churn (`Yes`/`No`).
- Treino e comparação de modelos (`LogisticRegression`, `RandomForest`, `GradientBoosting`).
- Persistência de artefatos em `models/`.
- Inferência por script local, dashboard e endpoint HTTP.

## Arquitetura da Solução

![Arquitetura do Projeto](assets/architecture.png)

Fluxo principal:

```text
Dataset CSV -> Data + Feature Pipeline -> Model Training -> Artifacts
                                                    |
                                                    +-> Streamlit Dashboard
                                                    +-> FastAPI Endpoint
                                                    +-> CLI Prediction
```

Regra de seleção do melhor modelo no código: maior `F1` no conjunto de teste.

## Stack Técnica

- Linguagem: `Python`
- Dados: `pandas`, `numpy`
- ML: `scikit-learn`
- Persistência: `joblib`
- Dashboard: `Streamlit`, `Plotly`
- API: `FastAPI`, `Pydantic`, `Uvicorn`

## Estrutura do Repositório

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

Observação: o arquivo `requirements.txt` menciona compatibilidade com Python 3.13.

## Setup Local

```bash
git clone <url-do-repositorio>
cd churn-prediction
python -m venv .venv
```

Ativação do ambiente virtual:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Instalação de dependências:

```bash
pip install -r requirements.txt
```

## Fluxo de Execução

1. Treinar modelos e salvar artefatos:

```bash
python main.py
```

2. Opcional: gerar base processada para consumo analítico:

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

Endpoints disponíveis:

- `GET /` retorna status básico da API.
- `GET /health` valida carga de modelo e pré-processador.
- `POST /predict` retorna predição, probabilidade e nível de risco.

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

Exemplo de resposta:

```json
{
  "churn": "Sim",
  "probability": 0.73,
  "risk_level": "Alto"
}
```

## Métricas do Modelo Atual

Métricas reportadas para o modelo salvo atual (`models/LogisticRegression.joblib`), com `test_size=0.2` e `random_state=42`:

- Accuracy: `0.8055`
- Precision: `0.6572`
- Recall: `0.5588`
- F1-score: `0.6040`
- ROC-AUC: `0.8420`

As métricas podem variar conforme novos treinos e dados.

## Testes

Existe estrutura de testes em `tests/`, mas os arquivos atuais estão vazios.

Comando padrão:

```bash
pytest -q
```

## Limitações Atuais

- Ausência de testes automatizados implementados.
- Ausência de versionamento formal de experimentos e métricas.
- Licença do projeto ainda não definida.
- Alguns arquivos do projeto apresentam textos com encoding inconsistente.

## Roadmap

- Implementar testes unitários e de integração.
- Versionar métricas e artefatos por experimento.
- Adicionar monitoramento de drift de dados/modelo.
- Evoluir API para batch scoring.
- Integrar pipeline com CI/CD.
- Avaliar modelos adicionais (ex.: XGBoost, LightGBM).

## Contribuição

1. Faça um fork do projeto.
2. Crie uma branch de feature: `git checkout -b feature/minha-feature`.
3. Commit suas mudanças: `git commit -m "feat: minha feature"`.
4. Abra um Pull Request.

## Licença

Pendente. Recomenda-se adicionar `LICENSE` (ex.: MIT).

## Contato

- Samuel de Andrade Maia
- GitHub: https://github.com/samuelmaia-data-analyst
- LinkedIn: https://linkedin.com/in/samuelmaia-data-analyst

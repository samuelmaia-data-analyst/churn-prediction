# Arquitetura do Projeto (Nível Sênior)

## Objetivo
Padronizar o projeto para operação enterprise com:
- separação clara por domínio,
- contratos estáveis entre camadas,
- baixa fricção para evolução de modelos e dashboard.

## Estrutura por Domínio
```
src/
  config.py                 # configuração central do pipeline
  ingestion.py              # raw -> bronze
  transformation.py         # bronze -> silver
  warehouse.py              # silver -> gold (star schema)
  reporting.py              # outputs executivos e persistência
  dashboard_data.py         # bootstrap/fallback para Streamlit
  contracts/
    executive_metrics.py    # contrato tipado de model_metrics
    report_schema.py        # contrato tipado de executive_report.json
  modeling/
    churn.py                # features, pré-processamento e modelos de churn
    pipeline.py             # orquestra treino e score de modelos
  ml.py                     # façade pública para compatibilidade
main.py                     # flow principal (Prefect)
pages/                      # Streamlit multipágina
tests/                      # contratos e regressão
```

## Princípios de Design
1. Contrato primeiro:
`model_metrics` deve ser gerado por um contrato explícito (`ExecutiveMetrics`) e nunca por dicionários ad-hoc.

2. Camadas desacopladas:
Modelagem (`src/modeling`) não conhece dashboard; dashboard consome apenas artefatos persistidos.

3. Compatibilidade retroativa:
`src/ml.py` permanece como API estável para não quebrar imports existentes.

4. Fallback controlado:
Se `xgboost` não estiver disponível, usar fallback explícito com nota em `model_comparison_note`.

5. Operação observável:
Métricas principais continuam registradas em MLflow e `executive_report.json`.

## Contratos Críticos
- `reports/executive_report.json`
  - `kpis`
  - `model_metrics`
  - `top_10_priorities`
- `data/gold/customer_prioritization.csv`
- `data/gold/kpi_summary.csv`

## Fluxo de Dados
`Raw -> Bronze -> Silver -> Gold -> Modeling -> Reporting -> Dashboard/API`

## Evolução Recomendada
1. Criar testes unitários dedicados em `tests/test_modeling_pipeline.py`.
2. Adicionar validação de contrato JSON no CI.

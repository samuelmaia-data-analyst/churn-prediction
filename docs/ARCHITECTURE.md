# Architecture

## Objetivo

Organizar o problema de churn como produto de dados, não apenas experimento de ML.

## Princípios

- uma trilha canônica de pipeline
- camadas com responsabilidade explícita
- contratos entre modelagem e reporting
- compatibilidade retroativa isolada
- artefatos e logs tratados como outputs de primeira classe

## Fluxo de Dados

```text
raw
  -> bronze
  -> silver
  -> gold
  -> modeling
  -> reporting
  -> dashboard/api
```

## Domínios Principais

### Runtime e Configuração

- `src/config.py`
- `src/logging_utils.py`
- `src/cli/pipeline.py`

Responsabilidade:

definir ambiente, `run_id`, paths resolvidos, metadata e observabilidade.

### Ingestion e Quality Gate

- `src/ingestion.py`
- `src/transformation.py`
- `src/validation.py`

Responsabilidade:

garantir que dados de entrada sejam válidos e reproduzíveis antes de modelagem.

### Feature e Modeling

- `src/feature_engineering.py`
- `src/modeling/churn.py`
- `src/modeling/pipeline.py`
- `src/modeling/predictor.py`

Responsabilidade:

treinar, avaliar, persistir e servir artefatos de inferência.

### Analytics Output

- `src/warehouse.py`
- `src/reporting.py`
- `src/dashboard_data.py`

Responsabilidade:

entregar gold layer, priorização, KPIs, model card e playbook operacional.

## Decisões Importantes

### Threshold por política de custo

O threshold global não é fixo. Ele é derivado da política de custo para refletir contexto operacional.

### Wrappers legados mantidos

Pastas como `src/data`, `src/features` e `src/models` ainda existem por compatibilidade.
O isolamento da compatibilidade agora fica explicitado em `src/compat/`.
O caminho canônico está em `src/` raiz e `src/modeling/`.

### Orquestração local e explícita

O pipeline principal usa execução local com retry explícito. Isso reduz ruído operacional, evita acoplamento desnecessário com orquestrador externo e deixa o caminho padrão mais reproduzível para review técnico.

### Monitoramento proporcional

O drift monitoring implementa PSI/KS simples porque o objetivo é demonstrar critério de engenharia sem simular uma plataforma completa.

## Limites Atuais

- sem scheduler distribuído real
- sem lakehouse ou warehouse externo
- sem artifact store remoto
- sem deploy de inferência em produção

Esses limites são conscientes e devem ser tratados como roadmap, não como lacuna escondida.

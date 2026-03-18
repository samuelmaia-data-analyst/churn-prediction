# Repository Structure

## Visão Geral

O repositório está organizado para separar:

- código canônico
- exploração
- automação
- documentação
- artefatos de execução

## Árvore Comentada

```text
.
|-- .github/
|   |-- workflows/               # CI
|   `-- ISSUE_TEMPLATE/          # templates de issue
|-- apps/                        # entrypoints de aplicação
|-- assets/                      # imagens e demos
|-- data/
|   |-- raw/                     # input esperado
|   |-- bronze/                  # camada regenerável
|   |-- silver/                  # camada regenerável
|   `-- gold/                    # camada regenerável
|-- docs/                        # documentação de arquitetura e operação
|-- notebooks/                   # exploração e análise
|-- pages/                       # páginas Streamlit
|-- src/
|   |-- cli/                     # interfaces de linha de comando
|   |-- compat/                  # backward compatibility explícita
|   |-- contracts/               # contratos tipados
|   |-- modeling/                # treino e inferência
|   |-- utils/                   # helpers pequenos e compartilhados
|   `-- ...                      # pipeline, reporting, monitoring
|-- tests/                       # testes automatizados
|-- artifacts/                   # outputs de execução
`-- models/                      # model registry local
```

## Convenções

### O que deve ir em `src/`

código de negócio, pipeline, modelagem e persistência.

### O que não deve ir em `src/`

exploração, screenshots, dados brutos, templates de GitHub e anotações ad-hoc.

### O que deve ir em `docs/`

arquitetura, operação, convenções e guias de colaboração.

### O que deve ir em `notebooks/`

EDA, experimentos e validações exploratórias. Nada em notebook deve ser dependência do caminho canônico do pipeline.

## Estrutura Alvo de Evolução

Sem quebrar compatibilidade, a evolução desejada é:

```text
src/
  config/
  compat/
  ingestion/
  transformations/
  validation/
  pipelines/
  modeling/
  reporting/
  monitoring/
  utils/
```

O repositório ainda não foi movido integralmente para esse layout porque a prioridade atual foi elevar confiabilidade e colaboração sem introduzir refactor de alto risco.

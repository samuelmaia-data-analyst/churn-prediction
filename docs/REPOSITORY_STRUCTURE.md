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
|   |-- pipelines/               # pipeline canônico e business logic
|   |-- runtime/                 # configuração e logging
|   |-- utils/                   # helpers pequenos e compartilhados
|   `-- *.py                     # wrappers de compatibilidade na raiz
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

## Estrutura Canônica Atual

Sem quebrar compatibilidade, a estrutura canônica agora é:

```text
src/
  compat/
  contracts/
  pipelines/
  modeling/
  runtime/
  utils/
```

Wrappers na raiz de `src/` continuam existindo para backward compatibility. Código novo deve apontar para `src/runtime/`, `src/pipelines/`, `src/modeling/` e `src/compat/`.

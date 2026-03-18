# Contributing

Este repositório segue um padrão de contribuição orientado a qualidade de engenharia. O objetivo é manter o projeto consistente, auditável e fácil de revisar.

## Princípios

- prefira clareza sobre cleverness
- trate o repositório como produto de dados, não como notebook
- evite acoplamento entre pipeline, dashboard e lógica de negócio
- preserve reprocessamento e idempotência
- documente trade-offs quando a decisão não for óbvia

## Como Começar

### Setup local

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe -m pre_commit install
```

### Dataset

Coloque o arquivo abaixo em `data/raw/`:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Estratégia de Mudança

Ao abrir uma contribuição, deixe claro qual categoria ela ataca:

- correção de bug
- confiabilidade do pipeline
- melhoria de modelagem
- melhoria de observabilidade
- documentação estrutural
- melhoria de experiência de contribuição

## Padrões de Código

- use o caminho canônico em `src/`
- evite adicionar nova lógica em wrappers legados sem justificativa
- mantenha type hints em caminhos críticos
- qualquer persistência nova deve ser segura e previsível
- logs devem carregar contexto útil de execução

## Testes Esperados

Antes de abrir PR:

```bash
make test
make lint
make typecheck
```

Se a mudança alterar pipeline, modelagem, reporting ou runtime config, adicione ou atualize testes.

## Pull Requests

Toda PR deve responder:

1. Qual problema real está sendo resolvido?
2. Qual impacto técnico a mudança tem?
3. Como isso foi validado?
4. Quais riscos ou trade-offs permanecem?

## Requisitos para Aprovação

Uma contribuição tende a ser rejeitada quando:

- aumenta complexidade sem ganho claro
- mistura refactor com mudança funcional sem explicitar risco
- introduz path hardcoded ou config escondida
- quebra o caminho reprocessável do pipeline
- altera contratos sem atualizar testes e documentação

## Convenção de Commits

Não é obrigatório usar Conventional Commits, mas a mensagem deve ser objetiva e auditável.

Exemplos bons:

- `improve pipeline runtime metadata and environment config`
- `add issue templates and contributing guide`
- `fix reporting policy mismatch with training threshold`

## Onde Colocar Cada Tipo de Mudança

- `src/`: lógica de produto e pipeline
- `tests/`: contratos e regressão
- `docs/`: arquitetura, operação e convenções
- `.github/`: automação e colaboração
- `notebooks/`: análise exploratória, não lógica canônica

## Documentação

Se a mudança afeta arquitetura, execução, setup ou fluxo de contribuição, atualize pelo menos um destes arquivos:

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Discussão de Escopo

Se a mudança for grande, prefira abrir uma issue antes da PR. Isso reduz retrabalho e evita arquitetura improvisada durante o review.

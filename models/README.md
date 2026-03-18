# Model Registry

Este diretorio e reservado para artefatos de modelo gerados em runtime.

Politica do repositorio:

- nao commitar modelos treinados manualmente
- nao commitar metadata de runs locais
- usar este diretorio apenas como target local do pipeline

Arquivos esperados em execucao:

- `model_v1.pkl`
- `model_metadata.json`
- `registry_manifest.json`

Para avaliacao tecnica, o importante e o pipeline reprodutivel que gera esses artefatos, nao o binario em si.

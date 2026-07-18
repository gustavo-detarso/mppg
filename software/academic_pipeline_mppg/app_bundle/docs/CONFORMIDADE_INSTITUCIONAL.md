# Conformidade institucional

A camada de conformidade institucional permite validar se os artefatos gerados
estão aderentes ao perfil definido em `[instituicao]`.

## Ativar perfil

```toml
[instituicao]
perfil = "fgv"
```

## Explicar perfil

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --explain-profile fgv
```

## Validar artefatos gerados

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --check-institution-compliance
```

Por padrão o comando procura, na pasta de saída do prefixo, os arquivos:

```text
<prefixo>.org
<prefixo>.bib
<prefixo>.docx
<prefixo>.pdf
```

Também é possível informar caminhos explicitamente:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --check-institution-compliance \
  --org caminho/documento.org \
  --bib caminho/documento.bib \
  --docx caminho/documento.docx \
  --pdf caminho/documento.pdf
```

## Prompt lock

Para gerar apenas a auditoria dos prompts/diretivas carregados:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --write-prompt-lock
```

A execução completa gera automaticamente:

```text
<prefixo>.prompt_lock.json
<prefixo>.prompt_lock.md
```

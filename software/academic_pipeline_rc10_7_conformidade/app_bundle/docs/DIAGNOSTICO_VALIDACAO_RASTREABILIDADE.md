# Diagnóstico, validação e rastreabilidade — rc10.3

## 1. Diagnosticar ambiente

Sem TOML:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --doctor
```

Com TOML:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --doctor
```

O diagnóstico verifica:

- módulos Python;
- Emacs, LaTeX, Biber, Pandoc;
- `academic-writing.el`;
- `fgv.png`;
- `.sty` FGV;
- `biblatex-apa` ou `biblatex-abnt` conforme TOML;
- `OPENAI_API_KEY`, quando a execução exigir IA;
- permissão de escrita no diretório de saída.

## 2. Validar TOML antes de rodar

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --check-config
```

Esse comando não chama a IA. Ele evita descobrir no final erros como:

- `input_zip` inexistente;
- `doi_manifest.csv` ausente;
- `program_name` duplicando o curso na capa;
- `reference_docx` inexistente;
- `pdf_engine` inválido;
- caminho LaTeX errado.

## 3. Recompilar sem regenerar

Após ajustes manuais em `.org` ou `.bib`, não rode o pipeline completo. Use:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --recompile \
  --org app_bundle/output/documento/meu_paper/meu_paper.org
```

Use `--no-clean` se não quiser remover auxiliares antes da recompilação.

## 4. Arquivos de rastreabilidade

Após a execução, confira:

```text
<prefixo>.run_report.json
<prefixo>.outputs.txt
<prefixo>.rc10_report.json
```

O `run_report.json` é o principal. Ele registra:

- versão do pipeline;
- data/hora;
- TOML usado;
- hash SHA-256 do ZIP de entrada;
- hash SHA-256 do DOI manifest;
- modelo OpenAI;
- estilo bibliográfico;
- PDF engine;
- todos os artefatos gerados;
- avisos.

## 5. Fluxo recomendado

```bash
# 1. Diagnóstico geral
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --doctor

# 2. Validação do TOML
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config app_bundle/projetos/meu_paper/paper_config.toml --check-config

# 3. Execução completa
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config app_bundle/projetos/meu_paper/paper_config.toml

# 4. Se fizer ajuste manual, recompilar apenas
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config app_bundle/projetos/meu_paper/paper_config.toml --recompile --org app_bundle/output/documento/meu_paper/meu_paper.org
```

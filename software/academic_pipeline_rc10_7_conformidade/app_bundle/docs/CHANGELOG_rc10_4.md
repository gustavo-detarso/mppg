# CHANGELOG rc10.4 — Usabilidade, controle bibliográfico e qualidade textual

Esta versão acrescenta quatro melhorias operacionais sobre a rc10.3:

## 1. `--init-project`

Cria automaticamente a estrutura de um novo projeto em `app_bundle/projetos/<nome>` com:

- `documentos-base.zip` placeholder;
- `orientacoes.zip` placeholder;
- `doi_manifest.csv` vazio;
- `paper_config.toml` já ajustado para o projeto;
- `README_PROJETO.md` com comandos de uso.

Exemplo:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --init-project paper_capacidades_estatais \
  --project-type paper
```

## 2. `--make-doi-manifest`

Gera `doi_manifest.csv` automaticamente a partir de um ZIP ou diretório de documentos.

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --make-doi-manifest \
  --input-zip app_bundle/projetos/meu_paper/documentos-base.zip \
  --output app_bundle/projetos/meu_paper/doi_manifest.csv
```

## 3. `--inspect-bib`

Inspeciona um `.bib` e produz relatórios `.md` e `.json` com:

- duplicatas prováveis;
- DOI malformado;
- título com HTML/XML;
- autor/ano/título ausente;
- artigo sem periódico;
- páginas ausentes;
- entradas `misc` suspeitas;
- notas/autores genéricos.

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --inspect-bib output/documento/meu_paper/meu_paper.bib
```

## 4. `quality_report.md`

Toda execução completa agora gera também:

- `<prefixo>.quality_report.md`;
- `<prefixo>.quality_report.json`.

O relatório mostra:

- total de palavras;
- palavras por seção;
- citações únicas;
- referências previstas não citadas;
- alertas de seção curta;
- ausência de conclusão;
- problemas remanescentes no ORG, quando detectados.

Também pode ser executado manualmente:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --quality-report \
  --document-json output/documento/meu_paper/meu_paper.document.json \
  --org output/documento/meu_paper/meu_paper.org \
  --bib output/documento/meu_paper/meu_paper.bib
```

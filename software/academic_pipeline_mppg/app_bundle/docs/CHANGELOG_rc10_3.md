# CHANGELOG rc10.3 — Diagnóstico, validação e rastreabilidade

## Novos comandos

### `--doctor`
Diagnostica ambiente local antes de executar o pipeline:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --doctor
```

Também pode ser usado com TOML para checar caminhos e estilo bibliográfico específicos:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config caminho.toml --doctor
```

### `--check-config`
Valida preventivamente o TOML, sem chamar a IA:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config caminho.toml --check-config
```

Gera `<prefixo>.check_config_report.json` no diretório de saída.

### `--recompile`
Recompila um `.org` existente sem regenerar conteúdo nem chamar IA:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config caminho.toml \
  --recompile \
  --org output/documento/meu_paper/meu_paper.org
```

Gera `<prefixo>.run_report.json` e `<prefixo>.outputs.txt` ao lado do `.org`.

## Rastreabilidade

A execução completa agora gera:

- `<prefixo>.run_report.json`: versão do pipeline, config, hashes de entrada, modelo, estilo bibliográfico, saídas e avisos;
- `<prefixo>.outputs.txt`: manifesto simples dos artefatos gerados;
- `<prefixo>.doctor_report.json`, quando `--doctor` é executado com `--config`;
- `<prefixo>.check_config_report.json`, quando `--check-config` é executado.

## Validação adicional

- `--check-config` valida caminhos de `input_zip`, `input_dir`, `doi_manifest.csv`, LaTeX, DOCX, CSL e PRISMA.
- A execução completa roda `check_config` antes de iniciar tarefas longas.
- O DOCX gerado passa por validação mínima: existência, tamanho, título, referências e quantidade de parágrafos.
- O `--doctor` verifica comandos externos, módulos Python, arquivos FGV locais, BibLaTeX APA/ABNT e permissão de escrita em output.

## Observação

Os arquivos locais `app_bundle/misc/academic-writing.el` e `app_bundle/misc/fgv.png` continuam fora do pacote e devem ser inseridos manualmente pelo usuário.

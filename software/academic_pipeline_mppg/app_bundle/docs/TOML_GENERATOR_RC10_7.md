# Gerador interativo de TOML — academic_pipeline rc10.7

A rc10.7 inclui um gerador interativo completo para criar arquivos TOML sem editar manualmente todas as seções do pipeline.

## Comando principal

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --init-toml
```

Também é possível chamar o script diretamente:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py
```

## Listar perfis disponíveis

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --list-toml-profiles
```

Ou:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py --list-profiles
```

## Iniciar direto em um preset

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --init-toml \
  --toml-profile atividade_local_fgv
```

## Perfis disponíveis

- `atividade_local_fgv`: atividade acadêmica com Ficha Técnica FGV a partir de corpus local.
- `paper_local_fgv`: paper acadêmico FGV a partir de corpus local.
- `paper_prisma_fgv`: paper + relatório PRISMA como saída própria.
- `dissertacao_local_fgv`: dissertação FGV a partir de corpus local.
- `dissertacao_prisma_fgv`: dissertação + relatório PRISMA.
- `relatorio_prisma_fgv`: apenas relatório PRISMA.
- `somente_renderizar_fgv`: renderiza um `document.json` existente em ORG/PDF/DOCX sem chamar IA para recriar conteúdo.
- modo avançado por componentes: permite combinar tipo de documento, corpus local, relatório PRISMA e saídas.

## Organização do TOML gerado

O arquivo é salvo, por padrão, em:

```text
app_bundle/projetos/<nome_do_projeto>/<tipo>_config.toml
```

Ele inclui as principais seções da rc10.7:

```toml
[projeto]
[instituicao]
[openai]
[pipeline]
[saida]
[documentos_locais]
[pesquisa]
[atividade]
[documento]
[bibliografia]
[relatorio_pesquisa]
[docx]
[latex]
[prompts]
[mapa_mental]
[conformidade]
[qualidade]
[controle]
```

## Fluxo recomendado após gerar o TOML

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/<projeto>/<config>.toml \
  --show-prompts

pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/<projeto>/<config>.toml \
  --check-config

pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/<projeto>/<config>.toml
```

Para renderização sem recriar o JSON canônico:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/<projeto>/render_config.toml \
  --somente-renderizar \
  --document-json app_bundle/output/documento/<projeto>/<projeto>.document.json
```

## Observação sobre PRISMA

Na rc10.7, o relatório PRISMA pode ser gerado como saída própria a partir de corpus local, de um `prisma_report.json` existente ou de uma pasta de pesquisa/triagem já produzida. O gerador deixa os campos preparados para esses cenários.

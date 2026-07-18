# AP-003A — Inventário e mapa estrutural do orquestrador

> Documento gerado por análise AST. Nenhum módulo produtivo foi alterado nesta subfase.

## Identificação do baseline

- Branch: `ap-refactor/03-orchestrator-decomposition`
- HEAD: `56b33739518026f379e076bdfdf06e781268c358`
- Base remota: `origin/refactor/academic-pipeline` em `56b33739518026f379e076bdfdf06e781268c358`
- Orquestrador: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`
- SHA-256: `e0a1b4b80f3cae45c99316223430d2bb6360167ef0b220974cdb0e9b735b87cc`
- Linhas físicas: 1968

## Contratos estruturais preservados até a AP-003F

- Definições `main()` de nível superior: **2**
- Definições de nível superior do wrapper `_original_main_before_prisma_artigo_generico_wrapper` no orquestrador: **0**
- Ocorrências produtivas do símbolo histórico: **3**
- A AP-003A registra a forma real do wrapper sem exigir que ele seja uma função no arquivo canônico.
- Guardas de execução direta: **1**

## Entrypoints e empacotamento

- `academic_pipeline/__main__.py`: **presente**
- Projeto: `academic-pipeline-mppg`
- Versão: `0.1.0`
- Scripts: `{"academic-pipeline": "academic_pipeline.cli:main"}`

## Funções de nível superior

| Intervalo | Função | Argumentos | Grupos candidatos | Chamadas distintas |
|---:|---|---|---|---:|
| 272-274 | `stage` | `message` | nao-classificado | 1 |
| 277-283 | `_json_or_none` | `value` | nao-classificado | 1 |
| 286-292 | `load_config` | `path` | nao-classificado | 5 |
| 295-300 | `make_client` | `model_override` | entrypoint-e-fluxo-principal | 4 |
| 303-305 | `_section` | `cfg, name` | nao-classificado | 2 |
| 308-317 | `output_paths` | `cfg` | orquestracao-documental | 12 |
| 320-337 | `research_output_paths` | `cfg` | orquestracao-documental, prisma-e-artigo-generico | 12 |
| 340-350 | `work_cache_paths` | `cfg, prefix` | nao-classificado | 11 |
| 353-386 | `apply_cli_path_overrides` | `cfg, args` | despacho-de-comandos, entrypoint-e-fluxo-principal | 3 |
| 389-390 | `load_existing_document_json` | `path` | orquestracao-documental | 2 |
| 393-420 | `resolve_bib_for_existing_document` | `document, document_json_path, out_dir, prefix` | orquestracao-documental | 21 |
| 423-424 | `_openai_model_from_cfg` | `cfg` | nao-classificado | 5 |
| 427-433 | `_load_optional_config` | `path` | nao-classificado | 6 |
| 436-442 | `_resolve_latex_paths_for_recompile` | `args, cfg` | orquestracao-documental | 8 |
| 445-471 | `run_recompile` | `args, cfg` | orquestracao-documental | 18 |
| 475-532 | `render_external_prisma_outputs` | `cfg, out_dir, prefix, prisma_payload, phase` | orquestracao-documental, prisma-e-artigo-generico | 15 |
| 535-648 | `render_additional_language_versions` | `client, model, cfg, document, bib_path, bib_keys, out_dir, prefix, doc_cfg, latex_cfg, config_dir, abstract_bundle` | orquestracao-documental | 32 |
| 653-654 | `_prisma_curadoria_default_config` | `` | prisma-e-artigo-generico | 0 |
| 657-658 | `_prisma_curadoria_default_out_dir` | `` | prisma-e-artigo-generico | 0 |
| 661-662 | `_prisma_curadoria_default_prompt` | `` | prisma-e-artigo-generico | 0 |
| 665-666 | `_prisma_curadoria_script_path` | `` | prisma-e-artigo-generico | 0 |
| 669-670 | `_prisma_curadoria_arg` | `args, name, default` | prisma-e-artigo-generico | 1 |
| 673-678 | `_prisma_curadoria_config_from_args` | `args` | prisma-e-artigo-generico | 2 |
| 681-682 | `_prisma_curadoria_out_from_args` | `args` | prisma-e-artigo-generico | 2 |
| 685-686 | `_prisma_curadoria_prompt_from_args` | `args` | prisma-e-artigo-generico | 2 |
| 689-696 | `_prisma_curadoria_input_from_args` | `args, default_xlsx` | prisma-e-artigo-generico | 4 |
| 699-710 | `_prisma_curadoria_run_command` | `cmd` | despacho-de-comandos, prisma-e-artigo-generico | 3 |
| 713-759 | `_prisma_curadoria_build_cmd` | `args, usar_ia, reexportar_xlsx` | prisma-e-artigo-generico | 10 |
| 762-764 | `_prisma_curadoria_run_ia` | `args, usar_ia` | despacho-de-comandos, prisma-e-artigo-generico | 2 |
| 767-769 | `_prisma_curadoria_reexportar_xlsx` | `args` | despacho-de-comandos, prisma-e-artigo-generico | 2 |
| 772-786 | `_prisma_curadoria_pipeline_supports_flag` | `flag` | prisma-e-artigo-generico | 1 |
| 789-809 | `_prisma_curadoria_importar_no_pipeline` | `args` | despacho-de-comandos, prisma-e-artigo-generico | 8 |
| 812-819 | `_prisma_curadoria_fluxo_completo` | `args` | prisma-e-artigo-generico | 4 |
| 822-841 | `_prisma_curadoria_mostrar_caminhos` | `args` | prisma-e-artigo-generico | 6 |
| 844-880 | `_prisma_curadoria_menu` | `args` | prisma-e-artigo-generico | 8 |
| 883-895 | `_prisma_curadoria_dispatch` | `args` | despacho-de-comandos, prisma-e-artigo-generico | 7 |
| 899-1682 | `main` | `` | parser-e-argumentos, despacho-de-comandos, orquestracao-documental, prisma-e-artigo-generico, entrypoint-e-fluxo-principal | 126 |
| 1691-1706 | `_refs_v6_disabled` | `cfg` | orquestracao-documental | 6 |
| 1709-1747 | `_refs_v6_apply_runtime_policy` | `cfg` | nao-classificado | 6 |
| 1752-1753 | `load_config` | `path` | nao-classificado | 2 |
| 1759-1770 | `build_bibliography` | `cfg, docs, out_dir, prefix, client, model` | nao-classificado | 4 |
| 1773-1784 | `_refs_v6_clear_document_bibliography` | `document` | orquestracao-documental | 1 |
| 1787-1815 | `_refs_v6_strip_org` | `text` | orquestracao-documental | 2 |
| 1821-1847 | `render_org_latex` | `document, org_path, bib_filename, cfg, bib_keys` | orquestracao-documental | 6 |
| 1860-1866 | `_prisma_artigo_generico_get_arg` | `argv, name` | prisma-e-artigo-generico | 4 |
| 1868-1881 | `_prisma_artigo_generico_strip` | `argv` | prisma-e-artigo-generico | 4 |
| 1883-1891 | `_prisma_artigo_generico_out_dir` | `argv` | prisma-e-artigo-generico | 4 |
| 1893-1912 | `_prisma_artigo_generico_run_export` | `argv, silent` | prisma-e-artigo-generico | 10 |
| 1914-1937 | `_prisma_artigo_generico_run_freeze` | `argv, silent` | prisma-e-artigo-generico | 12 |
| 1941-1963 | `main` | `*args, **kwargs` | prisma-e-artigo-generico, entrypoint-e-fluxo-principal | 5 |

## Superfície de argumentos

Chamadas relacionadas a `argparse`: **63**.

## Leitura para as próximas subfases

- **AP-003B:** confirmar e extrair parser e argumentos.
- **AP-003C:** confirmar a tabela comando → handler antes da extração do despacho.
- **AP-003D:** isolar a orquestração documental.
- **AP-003E:** isolar PRISMA e artigo genérico após confirmar a forma real do wrapper histórico.
- **AP-003F:** atualizar deliberadamente as travas dos dois `main()` e unificar o fluxo.

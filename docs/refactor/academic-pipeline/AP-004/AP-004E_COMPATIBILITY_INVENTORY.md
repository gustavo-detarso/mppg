# AP-004E — Inventário de superfícies de compatibilidade

> Inventário preparatório, reproduzível e sem alteração de código produtivo.

## Baseline validada

- Branch: `ap-refactor/03-orchestrator-decomposition`
- Commit local: `389f0ae526d12327a58ce23937225cf05b032566`
- Commit remoto: `389f0ae526d12327a58ce23937225cf05b032566`
- Divergência: `0 0`
- Assunto: `refactor(academic-pipeline): consolidar marcadores de versão da AP-004D`
- Fingerprint do inventário: `cee4120c2602bb12e78fe7d41cf22fc261b8a64647c2c2b9d6e256903d5574e3`

## Gate vigente

```text
[BLOQUEIO] Não criar nem executar aplicador produtivo.
[BLOQUEIO] Não alterar código produtivo.
[BLOQUEIO] Não criar commit.
[BLOQUEIO] Não realizar push.
[BLOQUEIO] Não integrar na branch refactor/academic-pipeline.
```

## Resumo

- Arquivos lidos: **272**
- Arquivos Python analisados por AST: **83**
- Itens inventariados: **64**
- Itens para decisão manual: **0**
- Candidatos preparatórios à remoção: **0**
- Itens bloqueados por colisão: **0**
- Erros de sintaxe encontrados: **0**

## Método

O inventário combina AST, tokenização de comentários, resolução de imports e `__all__`, entrypoints de empacotamento, fachadas de módulos, aliases, wrappers simples, registries, resolução dinâmica, metadados executáveis e busca separada de consumidores produtivos, testes, documentação e artefatos históricos. A ausência de consumidor interno não é tratada como prova suficiente para remoção de superfície pública ou distribuída.

## Contagens por classificação

| Classificação | Quantidade |
|---|---:|
| alias canônico necessário | 4 |
| bridge de importação necessária | 2 |
| compatibilidade interna necessária | 40 |
| compatibilidade ligada aos três xfail | 4 |
| compatibilidade protegida por decisão da AP-004B | 5 |
| compatibilidade pública durável | 6 |
| entrypoint público preservado | 2 |
| reexport necessário | 6 |
| wrapper histórico congelado | 2 |

## Itens inventariados

| ID | Superfície atual | Canônica/destino | Arquivo:linha | Tipo | Consumidores I/T/D/H | Risco | Decisão proposta | Classificação |
|---|---|---|---|---|---:|---|---|---|
| `AP004E-764b3462bb48` | `cli_main` | `.cli:main` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__init__.py:16` | reexport | 1/0/1/0 | alto | preservar | reexport necessário |
| `AP004E-9bb395bcbaa2` | `academic_pipeline` | `academic_pipeline:main` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__main__.py:1` | entrypoint | 0/4/43/1 | crítico | preservar sem alteração | compatibilidade pública durável; entrypoint público preservado |
| `AP004E-e72e9bb23f1e` | `main` | `run_legacy` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/cli.py:10` | function wrapper | 3/0/7/0 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-40f450199df1` | `load_config_impl` | `runtime['_refs_apply_runtime_policy']` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/document_orchestration.py:11` | function wrapper | 2/1/17/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-a1f507c8ca1e` | `load_existing_document_json_impl` | `runtime['AcademicDocument'].model_validate_json` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/document_orchestration.py:59` | function wrapper | 2/1/14/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-2d5ff25925a0` | `main` | `module` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/legacy.py:76` | getattr | 1/0/0/0 | alto | preservar | bridge de importação necessária; compatibilidade interna necessária |
| `AP004E-36edca88a8f6` | `stage_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:37` | function wrapper | 3/1/11/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-0d3fc308271c` | `_json_or_none_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:48` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-f8594e08fa3d` | `make_client_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:58` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-e0be0597d7fe` | `_section_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:65` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-ff473436589c` | `research_output_paths_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:87` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-7a1f70069b96` | `render_external_prisma_outputs_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:123` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-1847360516ee` | `_prisma_curadoria_default_config_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:129` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-edc8203917be` | `_prisma_curadoria_default_out_dir_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:135` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-1b290f50b5e3` | `_prisma_curadoria_default_prompt_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:141` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-1ecfd96dfa69` | `_prisma_curadoria_script_path_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:147` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-0a600a9d1a44` | `_prisma_curadoria_arg_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:153` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-fe333ce8096a` | `_prisma_curadoria_config_from_args_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:159` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-34041cff6510` | `_prisma_curadoria_out_from_args_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:165` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-ebc42e1d4c6c` | `_prisma_curadoria_prompt_from_args_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:171` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-90e1169791e3` | `_prisma_curadoria_input_from_args_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:183` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-760b7614a4e3` | `_prisma_curadoria_run_command_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:199` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-b0edfe05c4e4` | `_prisma_curadoria_build_cmd_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:232` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-9094555eaafc` | `_prisma_curadoria_run_ia_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:239` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-160e1b08feaf` | `_prisma_curadoria_reexportar_xlsx_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:246` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-648d0683722c` | `_prisma_curadoria_pipeline_supports_flag_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:258` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-187e66036dd3` | `_prisma_curadoria_importar_no_pipeline_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:279` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-656eec3d1374` | `_prisma_curadoria_fluxo_completo_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:288` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-6151cb3f2843` | `_prisma_curadoria_mostrar_caminhos_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:311` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-c9c406150ca6` | `_prisma_curadoria_menu_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:349` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-f04bcf304fe1` | `_prisma_curadoria_dispatch_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:366` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-389bb46f68e6` | `_prisma_artigo_generico_get_arg_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:377` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-317f17a716c7` | `_prisma_artigo_generico_strip_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:400` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-bba253a35116` | `_prisma_artigo_generico_out_dir_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:415` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-b81340a16e30` | `_prisma_artigo_generico_run_export_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:444` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-3cddb3d56457` | `_prisma_artigo_generico_run_freeze_impl_001` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:479` | function wrapper | 3/1/9/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-301f38a187b2` | `run_prisma_generic_entrypoint` | `_invoke_with_runtime` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:713` | function wrapper | 3/1/17/1 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-999329cd7564` | `WorkflowState._normalize` | `WorkflowState._normalize` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | protected symbol | 5/12/90/1 | crítico | preservar | compatibilidade ligada aos três xfail |
| `AP004E-6c490c6d9270` | `_ap003d_impl__refs_v6_strip_org` | `_ap003d_impl__refs_v6_strip_org` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | protected symbol | 2/8/65/0 | crítico | preservar | compatibilidade ligada aos três xfail |
| `AP004E-897018b51308` | `_ap003f_pipeline_core` | `_ap003f_pipeline_core` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | protected symbol | 2/17/259/2 | crítico | preservar | compatibilidade interna necessária |
| `AP004E-2fe302e866ad` | `_refs_v6_strip_org` | `_refs_v6_strip_org` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | protected symbol | 1/17/177/3 | crítico | preservar | compatibilidade ligada aos três xfail |
| `AP004E-12ef4a00117c` | `extract_org_abstracts` | `extract_org_abstracts` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | protected symbol | 3/11/83/1 | crítico | preservar | compatibilidade ligada aos três xfail |
| `AP004E-ab00608841d3` | `academic_pipeline_rc10.py` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/pipeline_orchestrator.py` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1` | module facade | 31/52/6523/42 | crítico | preservar | alias canônico necessário; compatibilidade pública durável |
| `AP004E-efbe86b407de` | `_refs_original_load_config` | `load_config` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1263` | assignment alias | 1/0/1/0 | crítico | preservar | alias canônico necessário; compatibilidade pública durável |
| `AP004E-e123da0779ec` | `_refs_original_build_bibliography` | `build_bibliography` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1271` | redirect import | 1/0/1/0 | crítico | preservar | alias canônico necessário; compatibilidade pública durável |
| `AP004E-299b7d91c4ee` | `_refs_original_render_org_latex` | `render_org_latex` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1289` | redirect import | 2/0/1/0 | crítico | preservar | alias canônico necessário; compatibilidade pública durável |
| `AP004E-c3f6df07093a` | `_collect_outputs_and_options_original` | `collect_outputs_and_options` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4907` | assignment alias | 3/0/1/0 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-936e788786e4` | `_render_toml_original` | `render_toml` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4958` | assignment alias | 1/0/1/0 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-054764be4586` | `_original_ensure_reference_policy` | `_WizInputController._ensure_reference_policy` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4976` | assignment alias | 1/0/1/0 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-5fa6e68ff3fc` | `_wiz_disable_references_original` | `_wiz_disable_references` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4993` | assignment alias | 1/0/2/0 | alto | preservar ou migrar consumidores antes | compatibilidade interna necessária |
| `AP004E-c0a8a6350d64` | `app_bundle.scripts.pipeline.article_workflow` | `.state:STAGES, .state:StageRecord, .state:WorkflowState, .validators:ArticleWorkflow, .validators:StageValidation` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:1` | module facade | 0/5/75/0 | alto | preservar | bridge de importação necessária |
| `AP004E-07fda2a9edec` | `STAGES` | `.state:STAGES` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2` | reexport | 9/0/5/0 | alto | preservar | reexport necessário |
| `AP004E-ceadf33fae1b` | `StageRecord` | `.state:StageRecord` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2` | reexport | 5/0/4/0 | alto | preservar | reexport necessário |
| `AP004E-0718d5435adb` | `WorkflowState` | `.state:WorkflowState` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2` | reexport | 3/15/54/1 | alto | preservar | reexport necessário |
| `AP004E-3d39381c09de` | `ArticleWorkflow` | `.validators:ArticleWorkflow` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:3` | reexport | 7/15/7/0 | alto | preservar | reexport necessário |
| `AP004E-25545d4eed0d` | `StageValidation` | `.validators:StageValidation` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:3` | reexport | 36/0/3/0 | alto | preservar | reexport necessário |
| `AP004E-8ffd2daf0e48` | `executar_artigo_longo_fulltext_v1_13` | `—` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_13.py:1` | historical frozen file | 0/4/63/0 | alto | preservar congelado | wrapper histórico congelado |
| `AP004E-ee9cdcce1bde` | `executar_artigo_longo_fulltext_v1_14` | `—` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_14.py:1` | historical frozen file | 0/4/63/0 | alto | preservar congelado | wrapper histórico congelado |
| `AP004E-15bc59c372e4` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | `app_bundle/scripts/pipeline/pipeline_orchestrator.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:15` | AP-004B compatibility decision | 31/52/6522/42 | alto | preservar | compatibilidade protegida por decisão da AP-004B |
| `AP004E-fd86eccec8b0` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:16` | AP-004B compatibility decision | 18/4/4993/40 | alto | preservar | compatibilidade protegida por decisão da AP-004B |
| `AP004E-2e67d0de59a4` | `configurar_pretriagem_ia_prisma_v16.py` | `configurar_pretriagem_ia_prisma.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:17` | AP-004B compatibility decision | 17/4/4987/40 | alto | preservar | compatibilidade protegida por decisão da AP-004B |
| `AP004E-c5801476fc13` | `gerar_log_diagnostico_artigo_v1_18.py` | `gerar_log_diagnostico_artigo.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:18` | AP-004B compatibility decision | 17/4/4987/40 | alto | preservar | compatibilidade protegida por decisão da AP-004B |
| `AP004E-80cb0eef7050` | `academic_pipeline_rc10.py` | `pipeline_orchestrator.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:29` | AP-004B compatibility decision | 31/52/6522/42 | alto | preservar | compatibilidade protegida por decisão da AP-004B |
| `AP004E-7e7dc8eface9` | `academic-pipeline` | `academic_pipeline.cli:main` | `software/academic_pipeline_rc10_7_conformidade/pyproject.toml:6` | entrypoint | 0/0/0/0 | crítico | preservar sem alteração | compatibilidade pública durável; entrypoint público preservado |

## Leitura dos consumidores

- **I**: consumidor produtivo interno.
- **T**: consumidor em teste.
- **D**: consumidor documental.
- **H**: consumidor em snapshot, fixture, manifesto ou artefato histórico.

A listagem completa das evidências e referências está em `ap004e_compatibility_inventory.json`.

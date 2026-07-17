# AP-005A — Inventário de consumidores e dependências

> Inventário preparatório, reproduzível e sem alteração de código produtivo.

## Baseline

- Branch de trabalho: `ap-refactor/04-consumer-canonicalization`
- Commit-base: `f45c123bc692b80f4796b701fe71019630dba2f5`
- Branch-base: `refactor/academic-pipeline`
- Assunto: `refactor(academic-pipeline): encerrar e validar a AP-004F`
- Data do commit-base: `2026-07-16T22:33:00-03:00`
- Fingerprint do contrato: `ad563ca0c5c46d99b5d17e966e9db7eabfeb96bfdef3c09c4a5ffe39d7141309`

## Gate vigente

```text
[BLOQUEIO] Não alterar código produtivo.
[BLOQUEIO] Não criar aplicador produtivo.
[BLOQUEIO] Não remover wrappers, aliases, fachadas ou reexports.
[BLOQUEIO] Não criar commit ou realizar push antes da aprovação expressa do inventário.
```

## Resumo

- Superfícies herdadas da AP-004E: **64**
- Superfícies em migração prévia: **38**
- Superfícies congeladas ausentes: **2**
- Arquivos do corpus lidos: **306**
- Arquivos Python analisados por AST: **131**
- Erros de sintaxe: **0**
- Componentes cíclicos: **0**
- Superfícies com resolução dinâmica: **1**
- Superfícies com ambiguidades registradas: **0**
- Candidatos autorizados à remoção: **0**

## Método

- O corpus é lido diretamente dos blobs do commit-base.
- Backups, outputs, ambientes virtuais e diretórios excluídos pela AP-004E não são abertos.
- Imports, nomes carregados, atributos e operações dinâmicas são analisados por AST.
- Referências textuais são usadas apenas como evidência documental, histórica ou de metadados.
- Referências ao destino canônico não são contadas como consumo do nome legado.
- Ausência de consumidor interno não é tratada como prova de remoção.

## Contagens por onda

| Onda | Quantidade |
|---|---:|
| fora de remoção | 13 |
| migração prévia | 38 |
| preservação | 13 |

## Ciclos de importação

- Nenhum componente cíclico observado.

## Superfícies inventariadas

| ID | Superfície | Origem | Onda | I/T/D/H | Dinâmica | Ciclo | Prioridade | Ação proposta |
|---|---|---|---|---:|---|---|---|---|
| `AP004E-054764be4586` | `_original_ensure_reference_policy` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4976` | migração prévia | 1/0/0/3 | não | não | baixa | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-0718d5435adb` | `WorkflowState` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2` | preservação | 0/0/0/8 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-07fda2a9edec` | `STAGES` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2` | preservação | 0/0/0/2 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-0a600a9d1a44` | `_prisma_curadoria_arg_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:153` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-0d3fc308271c` | `_json_or_none_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:48` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-12ef4a00117c` | `extract_org_abstracts` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | fora de remoção | 0/0/0/8 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-15bc59c372e4` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:15` | preservação | 4/1/25/23 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-160e1b08feaf` | `_prisma_curadoria_reexportar_xlsx_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:246` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-1847360516ee` | `_prisma_curadoria_default_config_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:129` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-187e66036dd3` | `_prisma_curadoria_importar_no_pipeline_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:279` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-1b290f50b5e3` | `_prisma_curadoria_default_prompt_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:141` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-1ecfd96dfa69` | `_prisma_curadoria_script_path_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:147` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-25545d4eed0d` | `StageValidation` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:3` | preservação | 0/0/0/2 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-299b7d91c4ee` | `_refs_original_render_org_latex` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1289` | fora de remoção | 0/0/0/3 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-2d5ff25925a0` | `main` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/legacy.py:76` | preservação | 1/0/0/4 | sim | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-2e67d0de59a4` | `configurar_pretriagem_ia_prisma_v16.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:17` | preservação | 0/0/0/12 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-2fe302e866ad` | `_refs_v6_strip_org` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | fora de remoção | 0/0/0/11 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-301f38a187b2` | `run_prisma_generic_entrypoint` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:713` | migração prévia | 2/0/0/5 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-317f17a716c7` | `_prisma_artigo_generico_strip_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:400` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-34041cff6510` | `_prisma_curadoria_out_from_args_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:165` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-36edca88a8f6` | `stage_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:37` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-389bb46f68e6` | `_prisma_artigo_generico_get_arg_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:377` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-3cddb3d56457` | `_prisma_artigo_generico_run_freeze_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:479` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-3d39381c09de` | `ArticleWorkflow` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:3` | preservação | 0/0/0/2 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-40f450199df1` | `load_config_impl` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/document_orchestration.py:11` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-5fa6e68ff3fc` | `_wiz_disable_references_original` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4993` | migração prévia | 1/0/0/2 | não | não | baixa | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-6151cb3f2843` | `_prisma_curadoria_mostrar_caminhos_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:311` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-648d0683722c` | `_prisma_curadoria_pipeline_supports_flag_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:258` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-656eec3d1374` | `_prisma_curadoria_fluxo_completo_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:288` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-6c490c6d9270` | `_ap003d_impl__refs_v6_strip_org` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | fora de remoção | 1/0/0/7 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-760b7614a4e3` | `_prisma_curadoria_run_command_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:199` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-764b3462bb48` | `cli_main` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__init__.py:16` | preservação | 1/0/0/2 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-7a1f70069b96` | `render_external_prisma_outputs_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:123` | migração prévia | 2/0/0/3 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-7e7dc8eface9` | `academic-pipeline` | `software/academic_pipeline_rc10_7_conformidade/pyproject.toml:6` | fora de remoção | 1/0/0/0 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-80cb0eef7050` | `academic_pipeline_rc10.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:29` | preservação | 4/2/25/25 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-897018b51308` | `_ap003f_pipeline_core` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | fora de remoção | 0/0/0/9 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-8ffd2daf0e48` | `executar_artigo_longo_fulltext_v1_13` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_13.py:1` | fora de remoção | 0/0/0/9 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-9094555eaafc` | `_prisma_curadoria_run_ia_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:239` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-90e1169791e3` | `_prisma_curadoria_input_from_args_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:183` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-936e788786e4` | `_render_toml_original` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4958` | migração prévia | 1/0/0/3 | não | não | baixa | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-999329cd7564` | `WorkflowState._normalize` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0` | fora de remoção | 0/0/0/8 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-9bb395bcbaa2` | `academic_pipeline` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__main__.py:1` | fora de remoção | 0/0/0/6 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-a1f507c8ca1e` | `load_existing_document_json_impl` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/document_orchestration.py:59` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-ab00608841d3` | `academic_pipeline_rc10.py` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1` | fora de remoção | 4/2/25/22 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-b0edfe05c4e4` | `_prisma_curadoria_build_cmd_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:232` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-b81340a16e30` | `_prisma_artigo_generico_run_export_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:444` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-bba253a35116` | `_prisma_artigo_generico_out_dir_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:415` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-c0a8a6350d64` | `app_bundle.scripts.pipeline.article_workflow` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:1` | preservação | 0/0/0/5 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-c3f6df07093a` | `_collect_outputs_and_options_original` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4907` | migração prévia | 3/0/0/3 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-c5801476fc13` | `gerar_log_diagnostico_artigo_v1_18.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:18` | preservação | 0/0/0/12 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-c9c406150ca6` | `_prisma_curadoria_menu_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:349` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-ceadf33fae1b` | `StageRecord` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2` | preservação | 0/0/0/2 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-e0be0597d7fe` | `_section_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:65` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-e123da0779ec` | `_refs_original_build_bibliography` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1271` | fora de remoção | 0/0/0/3 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-e72e9bb23f1e` | `main` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/cli.py:10` | migração prévia | 4/0/0/0 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-ebc42e1d4c6c` | `_prisma_curadoria_prompt_from_args_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:171` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-edc8203917be` | `_prisma_curadoria_default_out_dir_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:135` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-ee9cdcce1bde` | `executar_artigo_longo_fulltext_v1_14` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_14.py:1` | fora de remoção | 0/0/0/9 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-efbe86b407de` | `_refs_original_load_config` | `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1263` | fora de remoção | 0/0/0/3 | não | não | não aplicável | preservar sem transformação na AP-005 |
| `AP004E-f04bcf304fe1` | `_prisma_curadoria_dispatch_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:366` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-f8594e08fa3d` | `make_client_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:58` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-fd86eccec8b0` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py` | `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:16` | preservação | 0/0/0/12 | não | não | não aplicável | preservar e formalizar o contrato observado |
| `AP004E-fe333ce8096a` | `_prisma_curadoria_config_from_args_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:159` | migração prévia | 2/0/0/4 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |
| `AP004E-ff473436589c` | `research_output_paths_impl_001` | `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:87` | migração prévia | 2/0/0/3 | não | não | média | migrar consumidores internos para a superfície canônica antes de revisar o wrapper |

## Política de aprovação

Este inventário não autoriza transformação produtiva. Após a auditoria nominal e a correção de falsos positivos, será necessária aprovação expressa antes de qualquer aplicador ou migração.

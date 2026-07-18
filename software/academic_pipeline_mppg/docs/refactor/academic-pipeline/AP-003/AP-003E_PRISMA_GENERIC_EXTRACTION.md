# AP-003E — extração PRISMA e artigo genérico

## Estratégia

A extração foi dirigida pelo inventário AST e pelo fechamento do grafo de chamadas. As implementações selecionadas foram movidas para `academic_pipeline/prisma_generic_orchestration.py`, enquanto as funções históricas permaneceram como wrappers finos.

O corpo do segundo `main()` foi delegado integralmente ao novo módulo. As duas definições de `main()` e o alias histórico continuam preservados para a AP-003F.

## Helpers extraídos

| Ocorrência | Wrapper histórico | Implementação | Origem |
|---|---|---|---:|
| `stage#1` | `stage` | `stage_impl_001` | 272–274 |
| `_json_or_none#1` | `_json_or_none` | `_json_or_none_impl_001` | 277–283 |
| `make_client#1` | `make_client` | `make_client_impl_001` | 295–300 |
| `_section#1` | `_section` | `_section_impl_001` | 303–305 |
| `research_output_paths#1` | `research_output_paths` | `research_output_paths_impl_001` | 314–331 |
| `render_external_prisma_outputs#1` | `render_external_prisma_outputs` | `render_external_prisma_outputs_impl_001` | 392–449 |
| `_prisma_curadoria_default_config#1` | `_prisma_curadoria_default_config` | `_prisma_curadoria_default_config_impl_001` | 465–466 |
| `_prisma_curadoria_default_out_dir#1` | `_prisma_curadoria_default_out_dir` | `_prisma_curadoria_default_out_dir_impl_001` | 469–470 |
| `_prisma_curadoria_default_prompt#1` | `_prisma_curadoria_default_prompt` | `_prisma_curadoria_default_prompt_impl_001` | 473–474 |
| `_prisma_curadoria_script_path#1` | `_prisma_curadoria_script_path` | `_prisma_curadoria_script_path_impl_001` | 477–478 |
| `_prisma_curadoria_arg#1` | `_prisma_curadoria_arg` | `_prisma_curadoria_arg_impl_001` | 481–482 |
| `_prisma_curadoria_config_from_args#1` | `_prisma_curadoria_config_from_args` | `_prisma_curadoria_config_from_args_impl_001` | 485–490 |
| `_prisma_curadoria_out_from_args#1` | `_prisma_curadoria_out_from_args` | `_prisma_curadoria_out_from_args_impl_001` | 493–494 |
| `_prisma_curadoria_prompt_from_args#1` | `_prisma_curadoria_prompt_from_args` | `_prisma_curadoria_prompt_from_args_impl_001` | 497–498 |
| `_prisma_curadoria_input_from_args#1` | `_prisma_curadoria_input_from_args` | `_prisma_curadoria_input_from_args_impl_001` | 501–508 |
| `_prisma_curadoria_run_command#1` | `_prisma_curadoria_run_command` | `_prisma_curadoria_run_command_impl_001` | 511–522 |
| `_prisma_curadoria_build_cmd#1` | `_prisma_curadoria_build_cmd` | `_prisma_curadoria_build_cmd_impl_001` | 525–571 |
| `_prisma_curadoria_run_ia#1` | `_prisma_curadoria_run_ia` | `_prisma_curadoria_run_ia_impl_001` | 574–576 |
| `_prisma_curadoria_reexportar_xlsx#1` | `_prisma_curadoria_reexportar_xlsx` | `_prisma_curadoria_reexportar_xlsx_impl_001` | 579–581 |
| `_prisma_curadoria_pipeline_supports_flag#1` | `_prisma_curadoria_pipeline_supports_flag` | `_prisma_curadoria_pipeline_supports_flag_impl_001` | 584–598 |
| `_prisma_curadoria_importar_no_pipeline#1` | `_prisma_curadoria_importar_no_pipeline` | `_prisma_curadoria_importar_no_pipeline_impl_001` | 601–621 |
| `_prisma_curadoria_fluxo_completo#1` | `_prisma_curadoria_fluxo_completo` | `_prisma_curadoria_fluxo_completo_impl_001` | 624–631 |
| `_prisma_curadoria_mostrar_caminhos#1` | `_prisma_curadoria_mostrar_caminhos` | `_prisma_curadoria_mostrar_caminhos_impl_001` | 634–653 |
| `_prisma_curadoria_menu#1` | `_prisma_curadoria_menu` | `_prisma_curadoria_menu_impl_001` | 656–692 |
| `_prisma_curadoria_dispatch#1` | `_prisma_curadoria_dispatch` | `_prisma_curadoria_dispatch_impl_001` | 695–707 |
| `_prisma_artigo_generico_get_arg#1` | `_prisma_artigo_generico_get_arg` | `_prisma_artigo_generico_get_arg_impl_001` | 1490–1496 |
| `_prisma_artigo_generico_strip#1` | `_prisma_artigo_generico_strip` | `_prisma_artigo_generico_strip_impl_001` | 1498–1511 |
| `_prisma_artigo_generico_out_dir#1` | `_prisma_artigo_generico_out_dir` | `_prisma_artigo_generico_out_dir_impl_001` | 1513–1521 |
| `_prisma_artigo_generico_run_export#1` | `_prisma_artigo_generico_run_export` | `_prisma_artigo_generico_run_export_impl_001` | 1523–1542 |
| `_prisma_artigo_generico_run_freeze#1` | `_prisma_artigo_generico_run_freeze` | `_prisma_artigo_generico_run_freeze_impl_001` | 1544–1567 |

## Estágios extraídos do primeiro main

| Estágio | AST | Origem | Runtime | Produz |
|---|---:|---:|---|---|
| `run_prisma_stage_001` | 2–2 | 717–724 | `_prisma_curadoria_dispatch` | nenhum |
| `run_prisma_stage_002` | 78–79 | 975–976 | `cfg`, `external_search_enabled`, `is_external_prisma_run`, `output_paths`, `research_output_paths` | `is_external_prisma_run`, `out_dir`, `prefix` |
| `run_prisma_stage_003` | 90–90 | 997–997 | `stage` | nenhum |
| `run_prisma_stage_004` | 95–95 | 1006–1068 | `PIPELINE_VERSION`, `Path`, `artifacts`, `cache_dir`, `cfg`, `client`, `is_external_prisma_run`, `make_client`, `make_run_report`, `model`, `org_path`, `out_dir`, `outputs`, `pdf_path`, `precheck`, `prefix`, `print_outputs`, `prisma_outputs`, `prompt_lock`, `prompt_lock_md`, `prompt_lock_path`, `render_external_prisma_outputs`, `report`, `report_json_path`, `run_external_prisma_search`, `search_cfg`, `stage`, `warnings`, `work_dir`, `write_json`, `write_outputs_manifest`, `write_prompt_lock`, `write_prompt_lock_markdown` | `artifacts`, `client`, `model`, `org_path`, `outputs`, `pdf_path`, `prisma_outputs`, `prompt_lock`, `prompt_lock_md`, `prompt_lock_path`, `report`, `report_json_path`, `search_cfg` |
| `run_prisma_stage_005` | 107–107 | 1097–1097 | nenhum | `prisma_outputs` |
| `run_prisma_stage_006` | 145–145 | 1342–1352 | `_json_or_none`, `bib_path`, `document`, `document_json_path`, `docx_path`, `org_path`, `out_dir`, `paper_abstract_bundle`, `paper_abstract_path`, `pdf_path`, `prisma_outputs`, `translated_outputs` | `outputs` |
| `run_prisma_stage_007` | 151–151 | 1362–1362 | `stage` | nenhum |
| `run_prisma_stage_008` | 162–162 | 1391–1391 | `stage` | nenhum |

## Segundo main

- Implementação: `run_prisma_generic_entrypoint`.
- Origem: linhas 1571–1593.
- O wrapper histórico permanece com a mesma assinatura.

## Integridade

- Orquestrador antes: `da4c6c9b817d6607873e0412b5829729e36c3d70a1745b4b7d39ea4e31d31367`.
- Orquestrador depois: `431882d57a5a6ed334985b51a04db782ceda8de5cff1aa0e3bf856c4aa2c5b3a`.
- Parser AP-003B: `f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8`.
- Despacho AP-003C: `42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3`.
- Orquestração documental AP-003D: `3f2a3c95e08ccc3c19e3019a225c36fcf532cf4468f75b13c56b7c43bbc88a8e`.
- Helpers extraídos: **30**.
- Estágios do primeiro `main()`: **8**.
- Dois `main()` preservados.
- Alias `_original_main_before_prisma_artigo_generico_wrapper` preservado.
- Os três `xfail` históricos não foram alterados.

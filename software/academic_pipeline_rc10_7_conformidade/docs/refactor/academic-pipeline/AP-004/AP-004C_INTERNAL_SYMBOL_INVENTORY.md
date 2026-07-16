# AP-004C — inventário de símbolos internos (v1.3)

> Levantamento somente preparatório. Nenhum símbolo produtivo foi renomeado.

## Estado Git e base canônica

- Branch: `ap-refactor/03-orchestrator-decomposition`.
- HEAD local/remoto: `aa9829f09a5c1b9e69c634637c311b03f360b07e`.
- Commit AP-004B: `aa9829f09a5c1b9e69c634637c311b03f360b07e`.
- Inventário AP-004A: revisão `4.2`.
- Aplicação AP-004B: `module-file-application-v1.4`.

## Resumo

- Candidatos e controles: **73**.
- Arquivos definidores: **5**.
- Referências Python: **232**.
- Resumos de referências textuais: **491**.
- Arquivos no manifesto: **12**.
- Colisões de destino: **0**.
- Onda 1 pronta: **7**.
- Onda 2 vinculada a contratos: **13**.
- Adiados: **49**.
- Protegidos: **4**.
- Código produtivo alterado: **não**.

## Matriz de decisão

| Símbolo atual | Sugestão | Categoria | Arquivo:linha | Classificação AP-004A | Disposição AP-004C | Onda | Externas/contratos |
|---|---|---|---|---|---|---|---:|
| `_refs_v6_disabled_impl` | `_refs_disabled_impl` | função | `academic_pipeline/document_orchestration.py:173` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 1 |
| `_refs_v6_apply_runtime_policy_impl` | `_refs_apply_runtime_policy_impl` | função | `academic_pipeline/document_orchestration.py:185` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 1 |
| `_refs_v6_clear_document_bibliography_impl` | `_refs_clear_document_bibliography_impl` | função | `academic_pipeline/document_orchestration.py:225` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 1 |
| `_refs_v6_strip_org_impl` | `_refs_strip_org_impl` | função | `academic_pipeline/document_orchestration.py:238` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 1 |
| `_ap003d_impl_output_paths` | `_impl_output_paths` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:304` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl_apply_cli_path_overrides` | `_impl_apply_cli_path_overrides` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:339` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl_load_existing_document_json` | `_impl_load_existing_document_json` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:344` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl_resolve_bib_for_existing_document` | `_impl_resolve_bib_for_existing_document` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:350` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl__resolve_latex_paths_for_recompile` | `_impl_resolve_latex_paths_for_recompile` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:368` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl_run_recompile` | `_impl_run_recompile` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:373` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl_render_additional_language_versions` | `_impl_render_additional_language_versions` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:396` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003f_pipeline_core` | `_run_pipeline` | função | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:498` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 15 |
| `_ap003e_stage_001` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:504` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_001` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:515` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_002` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:526` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_003` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:537` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_004` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:548` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_005` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:559` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_006` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:570` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_007` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:581` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_008` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:592` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_009` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:603` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_010` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:614` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_011` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:625` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_001` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:636` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_012` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:673` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_013` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:683` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_014` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:694` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_015` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:705` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_016` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:716` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_017` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:727` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_018` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:738` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003c_dispatch_019` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:752` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003e_stage_002` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:764` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_002` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:781` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003e_stage_003` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:800` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003e_stage_004` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:818` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_003` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:855` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003e_stage_005` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:882` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_004` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1053` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_005` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1064` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_006` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1078` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_007` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1099` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_008` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1121` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003e_stage_006` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1138` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003e_stage_007` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1159` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_009` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1169` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_010` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1182` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003e_stage_008` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1197` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_011` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1207` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_ap003d_stage_012` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1223` | renomeação de alto risco | `deferred_structural_symbol` | AP-004C/AP-004D (revisão manual) | 0 |
| `_refs_v6_disabled` | `_refs_disabled` | função | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1252` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 5 |
| `_ap003d_impl__refs_v6_disabled` | `_impl_refs_disabled` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1253` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_refs_v6_apply_runtime_policy` | `_refs_apply_runtime_policy` | função | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1257` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 4 |
| `_ap003d_impl__refs_v6_apply_runtime_policy` | `_impl_refs_apply_runtime_policy` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1258` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl_load_config` | `_impl_load_config` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1265` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003d_impl_build_bibliography` | `_impl_build_bibliography` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1273` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_refs_v6_clear_document_bibliography` | `_refs_clear_document_bibliography` | função | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1277` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 1 |
| `_ap003d_impl__refs_v6_clear_document_bibliography` | `_impl_refs_clear_document_bibliography` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1278` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_refs_v6_strip_org` | `—` | função | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1282` | nome que deve permanecer | `protected_xfail_out_of_scope` | fora da AP-004 | 5 |
| `_ap003d_impl__refs_v6_strip_org` | `—` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1283` | nome que deve permanecer | `protected_xfail_out_of_scope` | fora da AP-004 | 0 |
| `_ap003d_impl_render_org_latex` | `_impl_render_org_latex` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1291` | renomeação segura | `ready_contract_bound_ast_rename` | AP-004C — onda 2 | 0 |
| `_ap003e_entrypoint` | `_entrypoint` | alias | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1327` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 3 |
| `_generate_interactive_before_wizard_documentos_locais_v4` | `_generate_interactive_before_wizard_documentos_locais` | alias | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4830` | renomeação segura | `ready_local_ast_rename` | AP-004C — onda 1 | 0 |
| `_generate_interactive_with_wizard_documentos_locais_v4` | `_generate_interactive_with_wizard_documentos_locais` | função | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4833` | renomeação segura | `ready_local_ast_rename` | AP-004C — onda 1 | 0 |
| `_WIZ_V5_REFERENCE_POLICY` | `_WIZ_REFERENCE_POLICY` | constante | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4855` | renomeação de alto risco | `manual_semantic_name_required` | AP-004C/AP-004D | 0 |
| `_v5_is_local_document` | `_is_local_document` | função | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4858` | renomeação segura | `ready_local_ast_rename` | AP-004C — onda 1 | 0 |
| `_v5_reference_default` | `_reference_default` | função | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4867` | renomeação segura | `ready_local_ast_rename` | AP-004C — onda 1 | 0 |
| `_v5_normalise_prompt` | `_normalise_prompt` | função | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4874` | renomeação segura | `ready_local_ast_rename` | AP-004C — onda 1 | 0 |
| `_v5_configure_reference_policy` | `_configure_reference_policy` | função | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4884` | renomeação segura | `ready_local_ast_rename` | AP-004C — onda 1 | 0 |
| `_v5_ensure_reference_policy` | `_ensure_reference_policy` | função | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4978` | renomeação segura | `ready_local_ast_rename` | AP-004C — onda 1 | 0 |
| `_normalize` | `—` | função | `app_bundle/scripts/pipeline/article_workflow/state.py:108` | nome que deve permanecer | `protected_xfail_out_of_scope` | fora da AP-004 | 2 |
| `extract_org_abstracts` | `—` | função | `app_bundle/scripts/pipeline/render_docx_canonico.py:656` | nome que deve permanecer | `protected_xfail_out_of_scope` | fora da AP-004 | 3 |

## Aliases seguros herdados da AP-004A

- `_ap003d_impl_output_paths` → `_impl_output_paths`.
- `_ap003d_impl_apply_cli_path_overrides` → `_impl_apply_cli_path_overrides`.
- `_ap003d_impl_load_existing_document_json` → `_impl_load_existing_document_json`.
- `_ap003d_impl_resolve_bib_for_existing_document` → `_impl_resolve_bib_for_existing_document`.
- `_ap003d_impl__resolve_latex_paths_for_recompile` → `_impl_resolve_latex_paths_for_recompile`.
- `_ap003d_impl_run_recompile` → `_impl_run_recompile`.
- `_ap003d_impl_render_additional_language_versions` → `_impl_render_additional_language_versions`.
- `_ap003d_impl__refs_v6_disabled` → `_impl_refs_disabled`.
- `_ap003d_impl__refs_v6_apply_runtime_policy` → `_impl_refs_apply_runtime_policy`.
- `_ap003d_impl_load_config` → `_impl_load_config`.
- `_ap003d_impl_build_bibliography` → `_impl_build_bibliography`.
- `_ap003d_impl__refs_v6_clear_document_bibliography` → `_impl_refs_clear_document_bibliography`.
- `_ap003d_impl_render_org_latex` → `_impl_render_org_latex`.

## Controles xfail protegidos

- `_refs_v6_strip_org` em `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`.
- `extract_org_abstracts` em `app_bundle/scripts/pipeline/render_docx_canonico.py`.
- `WorkflowState._normalize` em `app_bundle/scripts/pipeline/article_workflow/state.py`.
- `_ap003d_impl__refs_v6_strip_org` em `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`.

## Distribuição por disposição

| Disposição | Quantidade |
|---|---:|
| `ready_local_ast_rename` | 7 |
| `ready_contract_bound_ast_rename` | 13 |
| `contract_update_required` | 0 |
| `compatibility_required` | 0 |
| `deferred_structural_symbol` | 40 |
| `manual_semantic_name_required` | 9 |
| `blocked_destination_collision` | 0 |
| `protected_xfail_out_of_scope` | 4 |

## Consumidores efetivos dos candidatos prontos

| Símbolo | Categoria | Arquivo:linha | Tipo | Escopo |
|---|---|---|---|---|
| `_ap003d_impl__refs_v6_disabled` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1254` | `name_load` | `_refs_v6_disabled` |
| `_v5_ensure_reference_policy` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4985` | `name_load` | `<module>` |
| `_ap003d_impl_resolve_bib_for_existing_document` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:351` | `name_load` | `resolve_bib_for_existing_document` |
| `_v5_normalise_prompt` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4925` | `name_load` | `collect_outputs_and_options.policy_bool` |
| `_v5_normalise_prompt` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4937` | `name_load` | `collect_outputs_and_options.policy_choice` |
| `_ap003d_impl_load_existing_document_json` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:345` | `name_load` | `load_existing_document_json` |
| `_v5_is_local_document` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4912` | `name_load` | `collect_outputs_and_options` |
| `_v5_is_local_document` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4964` | `name_load` | `render_toml` |
| `_v5_configure_reference_policy` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4915` | `name_load` | `collect_outputs_and_options` |
| `_ap003d_impl_run_recompile` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:374` | `name_load` | `run_recompile` |
| `_ap003d_impl_load_config` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1266` | `name_load` | `load_config` |
| `_generate_interactive_with_wizard_documentos_locais_v4` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4847` | `name_load` | `<module>` |
| `_ap003d_impl_apply_cli_path_overrides` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:340` | `name_load` | `apply_cli_path_overrides` |
| `_ap003d_impl__refs_v6_clear_document_bibliography` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1279` | `name_load` | `_refs_v6_clear_document_bibliography` |
| `_ap003d_impl_build_bibliography` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1274` | `name_load` | `build_bibliography` |
| `_generate_interactive_before_wizard_documentos_locais_v4` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4838` | `name_load` | `_generate_interactive_with_wizard_documentos_locais_v4` |
| `_ap003d_impl_render_additional_language_versions` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:397` | `name_load` | `render_additional_language_versions` |
| `_ap003d_impl__resolve_latex_paths_for_recompile` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:369` | `name_load` | `_resolve_latex_paths_for_recompile` |
| `_ap003d_impl_render_org_latex` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1292` | `name_load` | `render_org_latex` |
| `_v5_reference_default` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4892` | `name_load` | `_v5_configure_reference_policy` |
| `_ap003d_impl__refs_v6_apply_runtime_policy` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1259` | `name_load` | `_refs_v6_apply_runtime_policy` |
| `_ap003d_impl_output_paths` | `same_module_static` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:305` | `name_load` | `output_paths` |

## Colisões

Nenhuma colisão de destino foi detectada.

## Manifesto

O JSON registra hashes de **12** arquivos relevantes, além dos contratos AST de todas as definições selecionadas.

## Validação

- `py_compile`: `passed`.
- `git diff --check`: `passed`.
- Suíte específica: `15 passed`.
- Suíte consolidada: `short test summary info`.

## Decisão de fase

O aplicador produtivo da AP-004C permanece bloqueado até aprovação expressa deste inventário.

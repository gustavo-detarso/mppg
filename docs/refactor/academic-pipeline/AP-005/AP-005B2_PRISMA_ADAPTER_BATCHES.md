# AP-005B2 — Lotes de adapters PRISMA

> Plano preparatório. Nenhum código produtivo é alterado nesta etapa.

## Baseline

- Commit: `6ef568b250390e12dc2e86b86a8c530188604a28`
- Fingerprint: `c5c6ab8734707cdf792cef3aa3b81ecb67b4b9aa17015bd1e2b83dcdf7122664`
- Hash PRISMA: `f250487a7787c967a0bad0ac38d5dbe210ff63981078d3c65e1d77655ff5f072`
- Hash RC10: `b7d2e0c8039e0a35ef1ffde343fa315dd15670728fe099fb1dd2c5c7b3fe517d`

## Partição

| Lote | Adapters | Escopo |
|---|---:|---|
| AP-005B2.1 | 6 | Núcleo e utilitários PRISMA |
| AP-005B2.2 | 10 | Configuração e argumentos da curadoria |
| AP-005B2.3 | 9 | Execução da curadoria |
| AP-005B2.4 | 6 | Artigo genérico e entrypoint |

## Regras obrigatórias

1. Um lote é atômico: não pode permanecer parcialmente aplicado.
2. Cada adapter mantém exatamente a assinatura do wrapper correspondente.
3. O adapter assume a chamada direta a `_invoke_with_runtime`.
4. O wrapper histórico permanece e passa a delegar ao adapter.
5. Adapter e wrapper permanecem em `_PROTECTED_RUNTIME_NAMES`.
6. O adapter é acrescentado a `__all__`.
7. O `rc10` migra o nome importado, mas preserva o alias local.
8. Bodies, wrappers e aliases locais não podem ser removidos.

## Matriz nominal

| Lote | Wrapper | Adapter | Body | Consumidor RC10 | Alias local |
|---|---|---|---|---|---|
| AP-005B2.1 | `stage_impl_001` | `stage_with_runtime` | `_ap003e_body_stage_1` | `stage` | `_ap003e_impl_stage_1` |
| AP-005B2.1 | `_json_or_none_impl_001` | `json_or_none_with_runtime` | `_ap003e_body__json_or_none_1` | `_json_or_none` | `_ap003e_impl__json_or_none_1` |
| AP-005B2.1 | `make_client_impl_001` | `make_client_with_runtime` | `_ap003e_body_make_client_1` | `make_client` | `_ap003e_impl_make_client_1` |
| AP-005B2.1 | `_section_impl_001` | `section_with_runtime` | `_ap003e_body__section_1` | `_section` | `_ap003e_impl__section_1` |
| AP-005B2.1 | `research_output_paths_impl_001` | `research_output_paths_with_runtime` | `_ap003e_body_research_output_paths_1` | `research_output_paths` | `_ap003e_impl_research_output_paths_1` |
| AP-005B2.1 | `render_external_prisma_outputs_impl_001` | `render_external_prisma_outputs_with_runtime` | `_ap003e_body_render_external_prisma_outputs_1` | `render_external_prisma_outputs` | `_ap003e_impl_render_external_prisma_outputs_1` |
| AP-005B2.2 | `_prisma_curadoria_default_config_impl_001` | `prisma_curadoria_default_config_with_runtime` | `_ap003e_body__prisma_curadoria_default_config_1` | `_prisma_curadoria_default_config` | `_ap003e_impl__prisma_curadoria_default_config_1` |
| AP-005B2.2 | `_prisma_curadoria_default_out_dir_impl_001` | `prisma_curadoria_default_out_dir_with_runtime` | `_ap003e_body__prisma_curadoria_default_out_dir_1` | `_prisma_curadoria_default_out_dir` | `_ap003e_impl__prisma_curadoria_default_out_dir_1` |
| AP-005B2.2 | `_prisma_curadoria_default_prompt_impl_001` | `prisma_curadoria_default_prompt_with_runtime` | `_ap003e_body__prisma_curadoria_default_prompt_1` | `_prisma_curadoria_default_prompt` | `_ap003e_impl__prisma_curadoria_default_prompt_1` |
| AP-005B2.2 | `_prisma_curadoria_script_path_impl_001` | `prisma_curadoria_script_path_with_runtime` | `_ap003e_body__prisma_curadoria_script_path_1` | `_prisma_curadoria_script_path` | `_ap003e_impl__prisma_curadoria_script_path_1` |
| AP-005B2.2 | `_prisma_curadoria_arg_impl_001` | `prisma_curadoria_arg_with_runtime` | `_ap003e_body__prisma_curadoria_arg_1` | `_prisma_curadoria_arg` | `_ap003e_impl__prisma_curadoria_arg_1` |
| AP-005B2.2 | `_prisma_curadoria_config_from_args_impl_001` | `prisma_curadoria_config_from_args_with_runtime` | `_ap003e_body__prisma_curadoria_config_from_args_1` | `_prisma_curadoria_config_from_args` | `_ap003e_impl__prisma_curadoria_config_from_args_1` |
| AP-005B2.2 | `_prisma_curadoria_out_from_args_impl_001` | `prisma_curadoria_out_from_args_with_runtime` | `_ap003e_body__prisma_curadoria_out_from_args_1` | `_prisma_curadoria_out_from_args` | `_ap003e_impl__prisma_curadoria_out_from_args_1` |
| AP-005B2.2 | `_prisma_curadoria_prompt_from_args_impl_001` | `prisma_curadoria_prompt_from_args_with_runtime` | `_ap003e_body__prisma_curadoria_prompt_from_args_1` | `_prisma_curadoria_prompt_from_args` | `_ap003e_impl__prisma_curadoria_prompt_from_args_1` |
| AP-005B2.2 | `_prisma_curadoria_input_from_args_impl_001` | `prisma_curadoria_input_from_args_with_runtime` | `_ap003e_body__prisma_curadoria_input_from_args_1` | `_prisma_curadoria_input_from_args` | `_ap003e_impl__prisma_curadoria_input_from_args_1` |
| AP-005B2.2 | `_prisma_curadoria_run_command_impl_001` | `prisma_curadoria_run_command_with_runtime` | `_ap003e_body__prisma_curadoria_run_command_1` | `_prisma_curadoria_run_command` | `_ap003e_impl__prisma_curadoria_run_command_1` |
| AP-005B2.3 | `_prisma_curadoria_build_cmd_impl_001` | `prisma_curadoria_build_cmd_with_runtime` | `_ap003e_body__prisma_curadoria_build_cmd_1` | `_prisma_curadoria_build_cmd` | `_ap003e_impl__prisma_curadoria_build_cmd_1` |
| AP-005B2.3 | `_prisma_curadoria_run_ia_impl_001` | `prisma_curadoria_run_ia_with_runtime` | `_ap003e_body__prisma_curadoria_run_ia_1` | `_prisma_curadoria_run_ia` | `_ap003e_impl__prisma_curadoria_run_ia_1` |
| AP-005B2.3 | `_prisma_curadoria_reexportar_xlsx_impl_001` | `prisma_curadoria_reexportar_xlsx_with_runtime` | `_ap003e_body__prisma_curadoria_reexportar_xlsx_1` | `_prisma_curadoria_reexportar_xlsx` | `_ap003e_impl__prisma_curadoria_reexportar_xlsx_1` |
| AP-005B2.3 | `_prisma_curadoria_pipeline_supports_flag_impl_001` | `prisma_curadoria_pipeline_supports_flag_with_runtime` | `_ap003e_body__prisma_curadoria_pipeline_supports_flag_1` | `_prisma_curadoria_pipeline_supports_flag` | `_ap003e_impl__prisma_curadoria_pipeline_supports_flag_1` |
| AP-005B2.3 | `_prisma_curadoria_importar_no_pipeline_impl_001` | `prisma_curadoria_importar_no_pipeline_with_runtime` | `_ap003e_body__prisma_curadoria_importar_no_pipeline_1` | `_prisma_curadoria_importar_no_pipeline` | `_ap003e_impl__prisma_curadoria_importar_no_pipeline_1` |
| AP-005B2.3 | `_prisma_curadoria_fluxo_completo_impl_001` | `prisma_curadoria_fluxo_completo_with_runtime` | `_ap003e_body__prisma_curadoria_fluxo_completo_1` | `_prisma_curadoria_fluxo_completo` | `_ap003e_impl__prisma_curadoria_fluxo_completo_1` |
| AP-005B2.3 | `_prisma_curadoria_mostrar_caminhos_impl_001` | `prisma_curadoria_mostrar_caminhos_with_runtime` | `_ap003e_body__prisma_curadoria_mostrar_caminhos_1` | `_prisma_curadoria_mostrar_caminhos` | `_ap003e_impl__prisma_curadoria_mostrar_caminhos_1` |
| AP-005B2.3 | `_prisma_curadoria_menu_impl_001` | `prisma_curadoria_menu_with_runtime` | `_ap003e_body__prisma_curadoria_menu_1` | `_prisma_curadoria_menu` | `_ap003e_impl__prisma_curadoria_menu_1` |
| AP-005B2.3 | `_prisma_curadoria_dispatch_impl_001` | `prisma_curadoria_dispatch_with_runtime` | `_ap003e_body__prisma_curadoria_dispatch_1` | `_prisma_curadoria_dispatch` | `_ap003e_impl__prisma_curadoria_dispatch_1` |
| AP-005B2.4 | `_prisma_artigo_generico_get_arg_impl_001` | `prisma_artigo_generico_get_arg_with_runtime` | `_ap003e_body__prisma_artigo_generico_get_arg_1` | `_prisma_artigo_generico_get_arg` | `_ap003e_impl__prisma_artigo_generico_get_arg_1` |
| AP-005B2.4 | `_prisma_artigo_generico_strip_impl_001` | `prisma_artigo_generico_strip_with_runtime` | `_ap003e_body__prisma_artigo_generico_strip_1` | `_prisma_artigo_generico_strip` | `_ap003e_impl__prisma_artigo_generico_strip_1` |
| AP-005B2.4 | `_prisma_artigo_generico_out_dir_impl_001` | `prisma_artigo_generico_out_dir_with_runtime` | `_ap003e_body__prisma_artigo_generico_out_dir_1` | `_prisma_artigo_generico_out_dir` | `_ap003e_impl__prisma_artigo_generico_out_dir_1` |
| AP-005B2.4 | `_prisma_artigo_generico_run_export_impl_001` | `prisma_artigo_generico_run_export_with_runtime` | `_ap003e_body__prisma_artigo_generico_run_export_1` | `_prisma_artigo_generico_run_export` | `_ap003e_impl__prisma_artigo_generico_run_export_1` |
| AP-005B2.4 | `_prisma_artigo_generico_run_freeze_impl_001` | `prisma_artigo_generico_run_freeze_with_runtime` | `_ap003e_body__prisma_artigo_generico_run_freeze_1` | `_prisma_artigo_generico_run_freeze` | `_ap003e_impl__prisma_artigo_generico_run_freeze_1` |
| AP-005B2.4 | `run_prisma_generic_entrypoint` | `run_prisma_generic_with_runtime` | `_ap003e_body_main_2` | `main` | `_ap003e_entrypoint` |

## Contratos históricos a atualizar

- `software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap003e_prisma_generic_contract.py`
- `software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap003g_stabilization_contract.py`

## Bloqueios atuais

```text
alteração produtiva = bloqueada
aplicador produtivo = bloqueado
rollout parcial de lote = bloqueado
remoção de wrappers = bloqueada
remoção de bodies = bloqueada
staging = bloqueado
commit = bloqueado
push = bloqueado
```

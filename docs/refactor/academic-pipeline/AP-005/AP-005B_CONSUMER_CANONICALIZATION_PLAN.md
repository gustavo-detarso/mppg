# AP-005B — Plano de canonicalização de consumidores

> Plano preparatório e reproduzível. Nenhum código produtivo é alterado nesta etapa.

## Baseline

- Commit-base: `6ef568b250390e12dc2e86b86a8c530188604a28`
- Fingerprint do contrato: `e659a91460dd5058ba6e49942454c26650eb4455e42f1d6e2ce450125f6284c8`

## Conclusão de escopo

- Superfícies herdadas da onda de migração: **38**
- Superfícies executáveis na AP-005B: **31**
- Contratos reclassificados para preservação: **3**
- Aliases adiados para a AP-005C: **4**
- Arquivos consumidores distintos: **4**
- Evidências AST internas: **76**

## Decisão arquitetural

A tentativa controlada da AP-005B1 foi rejeitada pela suíte canônica e revertida. A fachada `academic_pipeline.cli.main` e os wrappers documentais extraídos constituem contratos vigentes, não consumidores a serem eliminados.

Os quatro aliases `_original` do gerador TOML capturam bindings anteriores às redefinições. Eles não são imports comuns e não podem ser substituídos pelos nomes correntes sem risco de recursão ou alteração da ordem dos patches.

Os 31 wrappers PRISMA delegam simultaneamente a `_invoke_with_runtime` e a uma função-corpo `_ap003e_body_*`. O helper isolado não constitui destino canônico suficiente. A AP-005B2 deverá introduzir adapters nomeados antes de migrar os consumidores do `academic_pipeline_rc10.py`.

## Lotes

| Lote | Superfícies | Situação |
|---|---:|---|
| PRESERVAÇÃO | 3 | preservados após rejeição controlada da aplicação AP-005B1 |
| AP-005B2 | 31 | aguardando desenho nominal dos adapters e testes de equivalência |
| AP-005C | 4 | adiado; substituição direta é proibida |

## Contagens estruturais

- Contratos preservados: **3**
- AP-005B2 dependentes de adapters: **31**
- AP-005C adiadas: **4**
- Evidências internas de baixa confiança: **0**
- Consumidores dinâmicos no escopo: **0**
- Ciclos no escopo: **0**
- Candidatos à remoção: **0**

## Matriz nominal

| ID | Superfície | Cluster | Lote | Destino ou disposição |
|---|---|---|---|---|
| `AP004E-054764be4586` | `_original_ensure_reference_policy` | toml_assignment_aliases | AP-005C | `_WizInputController._ensure_reference_policy` |
| `AP004E-0a600a9d1a44` | `_prisma_curadoria_arg_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_arg_with_runtime` |
| `AP004E-0d3fc308271c` | `_json_or_none_impl_001` | prisma_runtime_adapters | AP-005B2 | `json_or_none_with_runtime` |
| `AP004E-160e1b08feaf` | `_prisma_curadoria_reexportar_xlsx_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_reexportar_xlsx_with_runtime` |
| `AP004E-1847360516ee` | `_prisma_curadoria_default_config_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_default_config_with_runtime` |
| `AP004E-187e66036dd3` | `_prisma_curadoria_importar_no_pipeline_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_importar_no_pipeline_with_runtime` |
| `AP004E-1b290f50b5e3` | `_prisma_curadoria_default_prompt_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_default_prompt_with_runtime` |
| `AP004E-1ecfd96dfa69` | `_prisma_curadoria_script_path_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_script_path_with_runtime` |
| `AP004E-301f38a187b2` | `run_prisma_generic_entrypoint` | prisma_runtime_adapters | AP-005B2 | `run_prisma_generic_with_runtime` |
| `AP004E-317f17a716c7` | `_prisma_artigo_generico_strip_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_artigo_generico_strip_with_runtime` |
| `AP004E-34041cff6510` | `_prisma_curadoria_out_from_args_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_out_from_args_with_runtime` |
| `AP004E-36edca88a8f6` | `stage_impl_001` | prisma_runtime_adapters | AP-005B2 | `stage_with_runtime` |
| `AP004E-389bb46f68e6` | `_prisma_artigo_generico_get_arg_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_artigo_generico_get_arg_with_runtime` |
| `AP004E-3cddb3d56457` | `_prisma_artigo_generico_run_freeze_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_artigo_generico_run_freeze_with_runtime` |
| `AP004E-40f450199df1` | `load_config_impl` | document_orchestration | PRESERVAÇÃO | `academic_pipeline.document_orchestration.load_config_impl` |
| `AP004E-5fa6e68ff3fc` | `_wiz_disable_references_original` | toml_assignment_aliases | AP-005C | `_wiz_disable_references` |
| `AP004E-6151cb3f2843` | `_prisma_curadoria_mostrar_caminhos_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_mostrar_caminhos_with_runtime` |
| `AP004E-648d0683722c` | `_prisma_curadoria_pipeline_supports_flag_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_pipeline_supports_flag_with_runtime` |
| `AP004E-656eec3d1374` | `_prisma_curadoria_fluxo_completo_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_fluxo_completo_with_runtime` |
| `AP004E-760b7614a4e3` | `_prisma_curadoria_run_command_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_run_command_with_runtime` |
| `AP004E-7a1f70069b96` | `render_external_prisma_outputs_impl_001` | prisma_runtime_adapters | AP-005B2 | `render_external_prisma_outputs_with_runtime` |
| `AP004E-9094555eaafc` | `_prisma_curadoria_run_ia_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_run_ia_with_runtime` |
| `AP004E-90e1169791e3` | `_prisma_curadoria_input_from_args_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_input_from_args_with_runtime` |
| `AP004E-936e788786e4` | `_render_toml_original` | toml_assignment_aliases | AP-005C | `render_toml` |
| `AP004E-a1f507c8ca1e` | `load_existing_document_json_impl` | document_orchestration | PRESERVAÇÃO | `academic_pipeline.document_orchestration.load_existing_document_json_impl` |
| `AP004E-b0edfe05c4e4` | `_prisma_curadoria_build_cmd_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_build_cmd_with_runtime` |
| `AP004E-b81340a16e30` | `_prisma_artigo_generico_run_export_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_artigo_generico_run_export_with_runtime` |
| `AP004E-bba253a35116` | `_prisma_artigo_generico_out_dir_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_artigo_generico_out_dir_with_runtime` |
| `AP004E-c3f6df07093a` | `_collect_outputs_and_options_original` | toml_assignment_aliases | AP-005C | `collect_outputs_and_options` |
| `AP004E-c9c406150ca6` | `_prisma_curadoria_menu_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_menu_with_runtime` |
| `AP004E-e0be0597d7fe` | `_section_impl_001` | prisma_runtime_adapters | AP-005B2 | `section_with_runtime` |
| `AP004E-e72e9bb23f1e` | `main` | cli_entrypoints | PRESERVAÇÃO | `academic_pipeline.cli.main` |
| `AP004E-ebc42e1d4c6c` | `_prisma_curadoria_prompt_from_args_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_prompt_from_args_with_runtime` |
| `AP004E-edc8203917be` | `_prisma_curadoria_default_out_dir_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_default_out_dir_with_runtime` |
| `AP004E-f04bcf304fe1` | `_prisma_curadoria_dispatch_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_dispatch_with_runtime` |
| `AP004E-f8594e08fa3d` | `make_client_impl_001` | prisma_runtime_adapters | AP-005B2 | `make_client_with_runtime` |
| `AP004E-fe333ce8096a` | `_prisma_curadoria_config_from_args_impl_001` | prisma_runtime_adapters | AP-005B2 | `prisma_curadoria_config_from_args_with_runtime` |
| `AP004E-ff473436589c` | `research_output_paths_impl_001` | prisma_runtime_adapters | AP-005B2 | `research_output_paths_with_runtime` |

## Gates seguintes

1. Manter os três contratos reclassificados sem alteração produtiva.
2. Auditar nominalmente os 31 adapters propostos para a AP-005B2.
3. Criar testes de equivalência entre adapters canônicos e wrappers legados.
4. Aplicar a AP-005B2 em lotes pequenos com rollback transacional.
5. Manter os quatro aliases do gerador TOML fora da AP-005B.

## Bloqueios

```text
alteração produtiva = bloqueada
aplicador produtivo = bloqueado
staging = bloqueado
commit = bloqueado
push = bloqueado
remoção = bloqueada
```

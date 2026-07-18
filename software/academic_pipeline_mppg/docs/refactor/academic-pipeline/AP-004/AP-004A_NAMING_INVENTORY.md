# AP-004A — inventário e convenção canônica (v4.2)

> Levantamento somente preparatório. Nenhum arquivo produtivo foi modificado.

## Estado Git confirmado

- Branch: `ap-refactor/03-orchestrator-decomposition`.
- HEAD local: `59ec50368de7302a9f25fe45809649e4baf2c144`.
- Referência local `origin/ap-refactor/03-orchestrator-decomposition`: `59ec50368de7302a9f25fe45809649e4baf2c144`.
- HEAD publicado: `59ec50368de7302a9f25fe45809649e4baf2c144`.
- Verificação remota: `passed`.
- Estado inicial aceito: `ap004a-artifacts-only`.

## Encerramento AP-003G confirmado

- Commit: `59ec50368de7302a9f25fe45809649e4baf2c144`.
- Assunto: `test(academic-pipeline): encerrar estabilização da AP-003G`.
- Data: `2026-07-15T19:06:21-03:00`.
- Commit ancestral do HEAD local e do HEAD publicado.
- Alterações produtivas no commit: nenhuma.

## Escopo técnico

- Arquivos rastreados: **328**.
- Python analisados por AST: **110**.
- Textos analisados: **260**.
- Testes e documentação: evidência, não candidatos por palavra isolada.
- Saídas operacionais, scripts históricos de manutenção e assets: protegidos.
- Métodos `__dunder__` e imports consumidores não duplicam decisões acionáveis.
- Código produtivo alterado: **não**.

## Totais v4.2

- Ocorrências brutas: **2067**.
- Candidatos acionáveis: **154**.
- Nomes operacionais protegidos: **105**.
- Registros históricos/testes: **101**.
- Ocorrências contextuais não acionáveis: **356**.
- Colisões de destino: **1**.
- Revisões manuais: **118**.
- Renomeação segura: **20**.
- Renomeação com compatibilidade: **4**.
- Renomeação de alto risco: **118**.
- Nome que deve permanecer: **12**.

## Candidatos acionáveis

| ID | Categoria | Nome atual | Sugestão | Arquivo:linha | Classificação | Fase |
|---|---|---|---|---|---|---|
| `787d65689024b728` | função | `_generate_interactive_with_wizard_documentos_locais_v4` | `_generate_interactive_with_wizard_documentos_locais` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4833` | renomeação segura | AP-004C/AP-004D |
| `44fe3d8dad6b1bf2` | função | `_v5_is_local_document` | `_is_local_document` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4858` | renomeação segura | AP-004C/AP-004D |
| `deae148a0fdbb03e` | função | `_v5_reference_default` | `_reference_default` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4867` | renomeação segura | AP-004C/AP-004D |
| `23fa7235a4211ec1` | função | `_v5_normalise_prompt` | `_normalise_prompt` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4874` | renomeação segura | AP-004C/AP-004D |
| `4d72c85eb6a79da7` | função | `_v5_configure_reference_policy` | `_configure_reference_policy` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4884` | renomeação segura | AP-004C/AP-004D |
| `0075c507d6fb990b` | função | `_v5_ensure_reference_policy` | `_ensure_reference_policy` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4978` | renomeação segura | AP-004C/AP-004D |
| `fbdd696256d2975d` | alias | `_ap003d_impl_output_paths` | `_impl_output_paths` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:304` | renomeação segura | AP-004C/AP-004D |
| `7ac5ba52736d3c34` | alias | `_ap003d_impl_apply_cli_path_overrides` | `_impl_apply_cli_path_overrides` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:339` | renomeação segura | AP-004C/AP-004D |
| `3b3ba3bd8bca538a` | alias | `_ap003d_impl_load_existing_document_json` | `_impl_load_existing_document_json` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:344` | renomeação segura | AP-004C/AP-004D |
| `0559b106e3df76fd` | alias | `_ap003d_impl_resolve_bib_for_existing_document` | `_impl_resolve_bib_for_existing_document` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:350` | renomeação segura | AP-004C/AP-004D |
| `c83168acf6d6cf56` | alias | `_ap003d_impl__resolve_latex_paths_for_recompile` | `_impl_resolve_latex_paths_for_recompile` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:368` | renomeação segura | AP-004C/AP-004D |
| `55935f0a2cd7579c` | alias | `_ap003d_impl_run_recompile` | `_impl_run_recompile` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:373` | renomeação segura | AP-004C/AP-004D |
| `bfdca14c08a3c949` | alias | `_ap003d_impl_render_additional_language_versions` | `_impl_render_additional_language_versions` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:396` | renomeação segura | AP-004C/AP-004D |
| `0061094161b58e22` | alias | `_ap003d_impl__refs_v6_disabled` | `_impl_refs_disabled` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1253` | renomeação segura | AP-004C/AP-004D |
| `f3ea31fda50e835f` | alias | `_ap003d_impl__refs_v6_apply_runtime_policy` | `_impl_refs_apply_runtime_policy` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1258` | renomeação segura | AP-004C/AP-004D |
| `60bd0b92d74b062f` | alias | `_ap003d_impl_load_config` | `_impl_load_config` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1265` | renomeação segura | AP-004C/AP-004D |
| `b090ac1faa24ee63` | alias | `_ap003d_impl_build_bibliography` | `_impl_build_bibliography` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1273` | renomeação segura | AP-004C/AP-004D |
| `a840c953eab45df1` | alias | `_ap003d_impl__refs_v6_clear_document_bibliography` | `_impl_refs_clear_document_bibliography` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1278` | renomeação segura | AP-004C/AP-004D |
| `ddceb34a9e1bc1ac` | alias | `_ap003d_impl_render_org_latex` | `_impl_render_org_latex` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1291` | renomeação segura | AP-004C/AP-004D |
| `bb2ff1489df45496` | alias | `_generate_interactive_before_wizard_documentos_locais_v4` | `_generate_interactive_before_wizard_documentos_locais` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4830` | renomeação segura | AP-004C/AP-004D |
| `1506ffdf6c57e609` | arquivo/módulo | `academic_pipeline_rc10.py` | `app_bundle/scripts/pipeline/pipeline_orchestrator.py` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | renomeação com compatibilidade | AP-004B/AP-004E |
| `daec8f8818d1c10e` | arquivo/módulo | `academic_pipeline_toml_generator_v0_3_1.py` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py` | renomeação com compatibilidade | AP-004B/AP-004E |
| `d9e93689f63e667e` | arquivo/módulo | `configurar_pretriagem_ia_prisma_v16.py` | `configurar_pretriagem_ia_prisma.py` | `configurar_pretriagem_ia_prisma_v16.py` | renomeação com compatibilidade | AP-004B/AP-004E |
| `9e6c0771276663d8` | arquivo/módulo | `gerar_log_diagnostico_artigo_v1_18.py` | `gerar_log_diagnostico_artigo.py` | `gerar_log_diagnostico_artigo_v1_18.py` | renomeação com compatibilidade | AP-004B/AP-004E |
| `76e7a298c383ade6` | arquivo/módulo | `executar_artigo_longo_fulltext_v1_13.py` | suspensa: `executar_artigo_longo_fulltext.py` | `executar_artigo_longo_fulltext_v1_13.py` | renomeação de alto risco | revisão manual antes da AP-004B/AP-004C |
| `4e3ed95d84e12846` | arquivo/módulo | `executar_artigo_longo_fulltext_v1_14.py` | suspensa: `executar_artigo_longo_fulltext.py` | `executar_artigo_longo_fulltext_v1_14.py` | renomeação de alto risco | revisão manual antes da AP-004B/AP-004C |
| `c2b69b709021f518` | função | `_refs_v6_disabled_impl` | `_refs_disabled_impl` | `academic_pipeline/document_orchestration.py:173` | renomeação de alto risco | AP-004B/AP-004C/AP-004D |
| `c8326cb72b5c4410` | função | `_refs_v6_apply_runtime_policy_impl` | `_refs_apply_runtime_policy_impl` | `academic_pipeline/document_orchestration.py:185` | renomeação de alto risco | AP-004B/AP-004C/AP-004D |
| `6b57f34b87dab9c0` | função | `_refs_v6_clear_document_bibliography_impl` | `_refs_clear_document_bibliography_impl` | `academic_pipeline/document_orchestration.py:225` | renomeação de alto risco | AP-004B/AP-004C/AP-004D |
| `9b6170b1abf1ebd9` | função | `_refs_v6_strip_org_impl` | `_refs_strip_org_impl` | `academic_pipeline/document_orchestration.py:238` | renomeação de alto risco | AP-004B/AP-004C/AP-004D |
| `d264a131a30585a2` | função | `_ap003e_body_stage_1` | — | `academic_pipeline/prisma_generic_orchestration.py:33` | renomeação de alto risco | revisão manual |
| `7b95a15e49c7d7bc` | função | `_ap003e_body__json_or_none_1` | — | `academic_pipeline/prisma_generic_orchestration.py:40` | renomeação de alto risco | revisão manual |
| `f0155c5742dc5a3a` | função | `_ap003e_body_make_client_1` | — | `academic_pipeline/prisma_generic_orchestration.py:51` | renomeação de alto risco | revisão manual |
| `f7e22780f039f7fd` | função | `_ap003e_body__section_1` | — | `academic_pipeline/prisma_generic_orchestration.py:61` | renomeação de alto risco | revisão manual |
| `4b19745e3939bbe8` | função | `_ap003e_body_research_output_paths_1` | — | `academic_pipeline/prisma_generic_orchestration.py:68` | renomeação de alto risco | revisão manual |
| `967f3bd0ccb6f0fd` | função | `_ap003e_body_render_external_prisma_outputs_1` | — | `academic_pipeline/prisma_generic_orchestration.py:90` | renomeação de alto risco | revisão manual |
| `3f38e8e7f7e507e8` | função | `_ap003e_body__prisma_curadoria_default_config_1` | — | `academic_pipeline/prisma_generic_orchestration.py:126` | renomeação de alto risco | revisão manual |
| `37aae9a00c2fb725` | função | `_ap003e_body__prisma_curadoria_default_out_dir_1` | — | `academic_pipeline/prisma_generic_orchestration.py:132` | renomeação de alto risco | revisão manual |
| `276931a110e02b3c` | função | `_ap003e_body__prisma_curadoria_default_prompt_1` | — | `academic_pipeline/prisma_generic_orchestration.py:138` | renomeação de alto risco | revisão manual |
| `621f86f409b2517e` | função | `_ap003e_body__prisma_curadoria_script_path_1` | — | `academic_pipeline/prisma_generic_orchestration.py:144` | renomeação de alto risco | revisão manual |
| `6844dc105ecfe888` | função | `_ap003e_body__prisma_curadoria_arg_1` | — | `academic_pipeline/prisma_generic_orchestration.py:150` | renomeação de alto risco | revisão manual |
| `c5820909b64b986e` | função | `_ap003e_body__prisma_curadoria_config_from_args_1` | — | `academic_pipeline/prisma_generic_orchestration.py:156` | renomeação de alto risco | revisão manual |
| `392d2ea5658fe5f1` | função | `_ap003e_body__prisma_curadoria_out_from_args_1` | — | `academic_pipeline/prisma_generic_orchestration.py:162` | renomeação de alto risco | revisão manual |
| `a030284ef88d6718` | função | `_ap003e_body__prisma_curadoria_prompt_from_args_1` | — | `academic_pipeline/prisma_generic_orchestration.py:168` | renomeação de alto risco | revisão manual |
| `a82d4ce792ad3d5c` | função | `_ap003e_body__prisma_curadoria_input_from_args_1` | — | `academic_pipeline/prisma_generic_orchestration.py:174` | renomeação de alto risco | revisão manual |
| `278335c7a7152b60` | função | `_ap003e_body__prisma_curadoria_run_command_1` | — | `academic_pipeline/prisma_generic_orchestration.py:186` | renomeação de alto risco | revisão manual |
| `a6fa01e24a280033` | função | `_ap003e_body__prisma_curadoria_build_cmd_1` | — | `academic_pipeline/prisma_generic_orchestration.py:202` | renomeação de alto risco | revisão manual |
| `a2c712d00352bb77` | função | `_ap003e_body__prisma_curadoria_run_ia_1` | — | `academic_pipeline/prisma_generic_orchestration.py:235` | renomeação de alto risco | revisão manual |
| `1938aaf9c6215fd3` | função | `_ap003e_body__prisma_curadoria_reexportar_xlsx_1` | — | `academic_pipeline/prisma_generic_orchestration.py:242` | renomeação de alto risco | revisão manual |
| `7d8c0422b71eab16` | função | `_ap003e_body__prisma_curadoria_pipeline_supports_flag_1` | — | `academic_pipeline/prisma_generic_orchestration.py:249` | renomeação de alto risco | revisão manual |
| `a81b87b3ada8b135` | função | `_ap003e_body__prisma_curadoria_importar_no_pipeline_1` | — | `academic_pipeline/prisma_generic_orchestration.py:261` | renomeação de alto risco | revisão manual |
| `00b43fa7e88d0963` | função | `_ap003e_body__prisma_curadoria_fluxo_completo_1` | — | `academic_pipeline/prisma_generic_orchestration.py:282` | renomeação de alto risco | revisão manual |
| `af4e9101b288ee2a` | função | `_ap003e_body__prisma_curadoria_mostrar_caminhos_1` | — | `academic_pipeline/prisma_generic_orchestration.py:291` | renomeação de alto risco | revisão manual |
| `f6f483a58f82c30a` | função | `_ap003e_body__prisma_curadoria_menu_1` | — | `academic_pipeline/prisma_generic_orchestration.py:314` | renomeação de alto risco | revisão manual |
| `91e0179721304c61` | função | `_ap003e_body__prisma_curadoria_dispatch_1` | — | `academic_pipeline/prisma_generic_orchestration.py:352` | renomeação de alto risco | revisão manual |
| `b9d6ea4eef6c6e92` | função | `_ap003e_body__prisma_artigo_generico_get_arg_1` | — | `academic_pipeline/prisma_generic_orchestration.py:369` | renomeação de alto risco | revisão manual |
| `2b507eacedd6106f` | função | `_ap003e_body__prisma_artigo_generico_strip_1` | — | `academic_pipeline/prisma_generic_orchestration.py:380` | renomeação de alto risco | revisão manual |
| `6f113ccc47f6bdf1` | função | `_ap003e_body__prisma_artigo_generico_out_dir_1` | — | `academic_pipeline/prisma_generic_orchestration.py:403` | renomeação de alto risco | revisão manual |
| `5836b39f09639d02` | função | `_ap003e_body__prisma_artigo_generico_run_export_1` | — | `academic_pipeline/prisma_generic_orchestration.py:418` | renomeação de alto risco | revisão manual |
| `0f6b2bb67619c717` | função | `_ap003e_body__prisma_artigo_generico_run_freeze_1` | — | `academic_pipeline/prisma_generic_orchestration.py:447` | renomeação de alto risco | revisão manual |
| `cfe71af0fd5880a4` | função | `_ap003e_body_main_2` | — | `academic_pipeline/prisma_generic_orchestration.py:685` | renomeação de alto risco | revisão manual |
| `15a1ef6a5b9b5e93` | função | `_ap003f_pipeline_core` | `_run_pipeline` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:498` | renomeação de alto risco | AP-004C/AP-004D |
| `708475d026033381` | função | `_refs_v6_disabled` | `_refs_disabled` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1252` | renomeação de alto risco | AP-004B/AP-004C/AP-004D |
| `e99bc9165fb000b3` | função | `_refs_v6_apply_runtime_policy` | `_refs_apply_runtime_policy` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1257` | renomeação de alto risco | AP-004B/AP-004C/AP-004D |
| `bfbad1a7ac56dc39` | função | `_refs_v6_clear_document_bibliography` | `_refs_clear_document_bibliography` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1277` | renomeação de alto risco | AP-004C/AP-004D |
| `7c4aa0c07f0c9102` | constante | `_WIZ_V5_REFERENCE_POLICY` | `_WIZ_REFERENCE_POLICY` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4855` | renomeação de alto risco | AP-004C/AP-004D |
| `013a06087dc6b031` | alias | `_ap003e_impl_stage_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:274` | renomeação de alto risco | revisão manual |
| `5c429e281670943e` | alias | `_ap003e_impl__json_or_none_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:279` | renomeação de alto risco | revisão manual |
| `1a015db2238ef230` | alias | `_ap003e_impl_make_client_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:293` | renomeação de alto risco | revisão manual |
| `0c91529eef19ad7d` | alias | `_ap003e_impl__section_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:298` | renomeação de alto risco | revisão manual |
| `412115995788dc24` | alias | `_ap003e_impl_research_output_paths_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:316` | renomeação de alto risco | revisão manual |
| `498f4d146acf9a91` | alias | `_ap003e_impl_render_external_prisma_outputs_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:385` | renomeação de alto risco | revisão manual |
| `625cc56fd67cb306` | alias | `_ap003e_impl__prisma_curadoria_default_config_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:403` | renomeação de alto risco | revisão manual |
| `6883deaafbb5e287` | alias | `_ap003e_impl__prisma_curadoria_default_out_dir_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:408` | renomeação de alto risco | revisão manual |
| `88a5014a701b9d28` | alias | `_ap003e_impl__prisma_curadoria_default_prompt_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:413` | renomeação de alto risco | revisão manual |
| `5e47b767b23b3c9f` | alias | `_ap003e_impl__prisma_curadoria_script_path_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:418` | renomeação de alto risco | revisão manual |
| `174f06ac099669b5` | alias | `_ap003e_impl__prisma_curadoria_arg_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:423` | renomeação de alto risco | revisão manual |
| `530ce8fb66f14ebd` | alias | `_ap003e_impl__prisma_curadoria_config_from_args_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:428` | renomeação de alto risco | revisão manual |
| `8145a41cbcce564f` | alias | `_ap003e_impl__prisma_curadoria_out_from_args_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:433` | renomeação de alto risco | revisão manual |
| `eb2e6dbbcfba5c75` | alias | `_ap003e_impl__prisma_curadoria_prompt_from_args_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:438` | renomeação de alto risco | revisão manual |
| `a975853539a17f28` | alias | `_ap003e_impl__prisma_curadoria_input_from_args_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:443` | renomeação de alto risco | revisão manual |
| `33e0167ce3eba4de` | alias | `_ap003e_impl__prisma_curadoria_run_command_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:448` | renomeação de alto risco | revisão manual |
| `1cc5304b40103d97` | alias | `_ap003e_impl__prisma_curadoria_build_cmd_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:453` | renomeação de alto risco | revisão manual |
| `d899be9109e2c607` | alias | `_ap003e_impl__prisma_curadoria_run_ia_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:458` | renomeação de alto risco | revisão manual |
| `0ea0a5d91563157f` | alias | `_ap003e_impl__prisma_curadoria_reexportar_xlsx_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:463` | renomeação de alto risco | revisão manual |
| `9b13509193005e95` | alias | `_ap003e_impl__prisma_curadoria_pipeline_supports_flag_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:468` | renomeação de alto risco | revisão manual |
| `72493085f806cc94` | alias | `_ap003e_impl__prisma_curadoria_importar_no_pipeline_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:473` | renomeação de alto risco | revisão manual |
| `469d04fd62a59189` | alias | `_ap003e_impl__prisma_curadoria_fluxo_completo_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:478` | renomeação de alto risco | revisão manual |
| `5eb38061e849051f` | alias | `_ap003e_impl__prisma_curadoria_mostrar_caminhos_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:483` | renomeação de alto risco | revisão manual |
| `f4c65ca9fbb5a361` | alias | `_ap003e_impl__prisma_curadoria_menu_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:488` | renomeação de alto risco | revisão manual |
| `ce2b166d606744d0` | alias | `_ap003e_impl__prisma_curadoria_dispatch_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:493` | renomeação de alto risco | revisão manual |
| `850d12b4e7133fb0` | alias | `_ap003e_stage_001` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:504` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `f2b3b2d1e9db7074` | alias | `_ap003c_dispatch_001` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:515` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `269806bee5f2da71` | alias | `_ap003c_dispatch_002` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:526` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `052ad1b825fed830` | alias | `_ap003c_dispatch_003` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:537` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `2d7518709a880d3a` | alias | `_ap003c_dispatch_004` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:548` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `6ce557e9a6031547` | alias | `_ap003c_dispatch_005` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:559` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `be6297d3eff6ce6a` | alias | `_ap003c_dispatch_006` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:570` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `9842e9dd41829eab` | alias | `_ap003c_dispatch_007` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:581` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `b45e5baf215c682f` | alias | `_ap003c_dispatch_008` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:592` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `4f830f8239df8c22` | alias | `_ap003c_dispatch_009` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:603` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `c62c7e84a68da1cc` | alias | `_ap003c_dispatch_010` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:614` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `93bbf2a72c86b797` | alias | `_ap003c_dispatch_011` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:625` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `55141a658f75373f` | alias | `_ap003d_stage_001` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:636` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `41e00036d6f6a48b` | alias | `_ap003c_dispatch_012` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:673` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `6a5b9f15fbfc7985` | alias | `_ap003c_dispatch_013` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:683` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `57b669852db4d148` | alias | `_ap003c_dispatch_014` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:694` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `366f01fba386bbc6` | alias | `_ap003c_dispatch_015` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:705` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `fee109c3a9892ab1` | alias | `_ap003c_dispatch_016` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:716` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `d361726e39ccd798` | alias | `_ap003c_dispatch_017` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:727` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `b51aabfb6d8b984e` | alias | `_ap003c_dispatch_018` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:738` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `c1a18d7f073a78e6` | alias | `_ap003c_dispatch_019` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:752` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `b3ae0ed72abbf3df` | alias | `_ap003e_stage_002` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:764` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `5bced5d553137f91` | alias | `_ap003d_stage_002` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:781` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `5d31f6c4067acda7` | alias | `_ap003e_stage_003` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:800` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `566f9831e64711ab` | alias | `_ap003e_stage_004` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:818` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `939c396827a788e2` | alias | `_ap003d_stage_003` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:855` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `63d250fe8484f5c5` | alias | `_ap003e_stage_005` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:882` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `93262ccf320f53fb` | alias | `_ap003d_stage_004` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1053` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `c50435eb9130e58a` | alias | `_ap003d_stage_005` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1064` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `c2767a95f976faaf` | alias | `_ap003d_stage_006` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1078` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `4cc6ff6b6b1e95b6` | alias | `_ap003d_stage_007` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1099` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `8de6cb0b05f434e1` | alias | `_ap003d_stage_008` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1121` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `cfc56baf104eb306` | alias | `_ap003e_stage_006` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1138` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `9f0f4dd6dde5571a` | alias | `_ap003e_stage_007` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1159` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `c783b84990958c9f` | alias | `_ap003d_stage_009` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1169` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `d72d4206e94d9fdc` | alias | `_ap003d_stage_010` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1182` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `8278c8e1cf0353dd` | alias | `_ap003e_stage_008` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1197` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `9f2512d651469f62` | alias | `_ap003d_stage_011` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1207` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `b8a6553ee09a8975` | alias | `_ap003d_stage_012` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1223` | renomeação de alto risco | AP-004C/AP-004D (revisão manual) |
| `e440e18701f7e590` | alias | `_refs_v6_original_load_config` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1263` | renomeação de alto risco | revisão manual |
| `1627418a54ad7de4` | alias | `_refs_v6_original_build_bibliography` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1271` | renomeação de alto risco | revisão manual |
| `7e960c4d72226e20` | alias | `_refs_v6_original_render_org_latex` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1289` | renomeação de alto risco | revisão manual |
| `4d610c1e8d66eb6c` | alias | `_ap003e_impl__prisma_artigo_generico_get_arg_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1306` | renomeação de alto risco | revisão manual |
| `875dc608dd5709be` | alias | `_ap003e_impl__prisma_artigo_generico_strip_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1310` | renomeação de alto risco | revisão manual |
| `31449f9e2eefdb78` | alias | `_ap003e_impl__prisma_artigo_generico_out_dir_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1314` | renomeação de alto risco | revisão manual |
| `a6b1ce8eda3475d7` | alias | `_ap003e_impl__prisma_artigo_generico_run_export_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1318` | renomeação de alto risco | revisão manual |
| `6a4e619299a9ef95` | alias | `_ap003e_impl__prisma_artigo_generico_run_freeze_1` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1322` | renomeação de alto risco | revisão manual |
| `065e09e45ed4274b` | alias | `_ap003e_entrypoint` | `_entrypoint` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1327` | renomeação de alto risco | AP-004C/AP-004D |
| `9f481ff147329df9` | alias | `_v5_collect_outputs_and_options_original` | — | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4907` | renomeação de alto risco | revisão manual |
| `aa9aea38dfb1967b` | alias | `_v5_render_toml_original` | — | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4958` | renomeação de alto risco | revisão manual |
| `55ca57cb9fd7cc3b` | alias | `_wiz_disable_references_pre_v5_2` | — | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4993` | renomeação de alto risco | revisão manual |
| `3c3f757dcff94918` | arquivo/módulo | `academic_pipeline_rc10_7_conformidade` | — | `.` | nome que deve permanecer | AP-006 |
| `4095f92c34b94e24` | arquivo/módulo | `legacy.py` | — | `academic_pipeline/legacy.py` | nome que deve permanecer | AP-004E (revisão de compatibilidade) |
| `6b2d99dee96a3164` | função | `ensure_legacy_path` | — | `academic_pipeline/legacy.py:46` | nome que deve permanecer | AP-004E (revisão de compatibilidade) |
| `8c05d9bf73fa5d98` | função | `load_legacy_module` | — | `academic_pipeline/legacy.py:71` | nome que deve permanecer | AP-004E (revisão de compatibilidade) |
| `b6bc887de54fc486` | função | `run_legacy` | — | `academic_pipeline/legacy.py:97` | nome que deve permanecer | AP-004E (revisão de compatibilidade) |
| `55c13dceda83d146` | função | `_refs_v6_strip_org` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1282` | nome que deve permanecer | fora da AP-004 |
| `7a1194d2acc5dcfd` | função | `WorkflowState._normalize` | — | `app_bundle/scripts/pipeline/article_workflow/state.py:108` | nome que deve permanecer | fora da AP-004 |
| `901cd9c8b8ba6f8d` | função | `extract_org_abstracts` | — | `app_bundle/scripts/pipeline/render_docx_canonico.py:656` | nome que deve permanecer | fora da AP-004 |
| `1861287abc88d4f3` | classe | `LegacyRuntimeError` | — | `academic_pipeline/legacy.py:24` | nome que deve permanecer | AP-004E (revisão de compatibilidade) |
| `fa8845bddbfc2c72` | constante | `LEGACY_PIPELINE_DIR` | — | `academic_pipeline/legacy.py:19` | nome que deve permanecer | AP-004E (revisão de compatibilidade) |
| `06d0e184674abcb5` | constante | `LEGACY_MODULE_NAME` | — | `academic_pipeline/legacy.py:20` | nome que deve permanecer | AP-004E (revisão de compatibilidade) |
| `091a4acaa0f7f503` | alias | `_ap003d_impl__refs_v6_strip_org` | — | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1283` | nome que deve permanecer | fora da AP-004 |

## Superfícies relacionadas

- `academic_pipeline_rc10.py`:
  - python-main-guard: `app_bundle.scripts.pipeline.academic_pipeline_rc10` → `['SystemExit', 'main']` em `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`
- `academic_pipeline_toml_generator_v0_3_1.py`:
  - python-main-guard: `app_bundle.scripts.pipeline.academic_pipeline_toml_generator_v0_3_1` → `['SystemExit', 'main']` em `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py`
- `configurar_pretriagem_ia_prisma_v16.py`:
  - python-main-guard: `configurar_pretriagem_ia_prisma_v16` → `['SystemExit', 'main', 'print', 'SystemExit']` em `configurar_pretriagem_ia_prisma_v16.py`
- `gerar_log_diagnostico_artigo_v1_18.py`:
  - python-main-guard: `gerar_log_diagnostico_artigo_v1_18` → `['main']` em `gerar_log_diagnostico_artigo_v1_18.py`

## Colisões de destino

- `executar_artigo_longo_fulltext.py` — sugestão suspensa para 2 origens.
  - `executar_artigo_longo_fulltext_v1_13.py` em `executar_artigo_longo_fulltext_v1_13.py`
  - `executar_artigo_longo_fulltext_v1_14.py` em `executar_artigo_longo_fulltext_v1_14.py`

## Caminhos operacionais protegidos

- `aplicar_docx_canonico_v10.py` — script histórico de aplicação, atualização ou migração na raiz.
- `aplicar_docx_canonico_v11.py` — script histórico de aplicação, atualização ou migração na raiz.
- `aplicar_docx_canonico_v12.py` — script histórico de aplicação, atualização ou migração na raiz.
- `aplicar_docx_canonico_v13.py` — script histórico de aplicação, atualização ou migração na raiz.
- `aplicar_docx_capa_disciplina_v14.py` — script histórico de aplicação, atualização ou migração na raiz.
- `aplicar_docx_only_gerador_unificado_v9.py` — script histórico de aplicação, atualização ou migração na raiz.
- `app_bundle/institutions/fgv/assets/README.md` — asset operacional fora do escopo da AP-004.
- `app_bundle/institutions/fgv/assets/fgv.png` — asset operacional fora do escopo da AP-004.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.bbl` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.bib` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.check_config_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.compliance_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.compliance_report.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.document.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.org` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.outputs.txt` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.pdf` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.prompt_lock.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.prompt_lock.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.quality_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.quality_report.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.rc10_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.run_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.tex` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.tex~` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein_bibliografia_diagnostico.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein_export_pdf.el` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.bbl` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.bib` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.check_config_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.compliance_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.compliance_report.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.document.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.org` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.outputs.txt` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.pdf` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.prompt_lock.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.prompt_lock.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.quality_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.quality_report.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.rc10_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.run_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.tex` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub_bibliografia_diagnostico.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub_export_pdf.el` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.busca_prisma_log.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.candidatos_brutos.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.candidatos_deduplicados.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.check_config_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.outputs.txt` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.pre_triagem_ia.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.prisma_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.protocolo_busca_prisma.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.rc10_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.bbl` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.org` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.pdf` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.tex` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar_export_pdf.el` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.run_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.triagem_titulo_resumo.csv` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.triagem_titulo_resumo.xlsx` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/referencias_incluidas_seminario_atestmed_pmf.xlsx` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.busca_prisma_log.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.candidatos_brutos.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.candidatos_deduplicados.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.check_config_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_log.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_referencias.xlsx` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_resumo.txt` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.diagrama_prisma.png` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.diagrama_prisma_contagens.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.matriz_estudos_incluidos.csv` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.matriz_estudos_incluidos.xlsx` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.outputs.txt` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.pre_triagem_ia.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prisma_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prisma_report_final.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.protocolo_busca_prisma.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.rc10_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas.bib` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas_seminario.csv` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.bbl` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.md` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.org` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.pdf` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.tex` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.tex~` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final_export_pdf.el` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.bbl` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.org` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.pdf` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.tex` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.tex~` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar_export_pdf.el` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.run_report.json` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.triagem_humana.csv` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.triagem_titulo_resumo.csv` — artefato operacional gerado ou execução histórica.
- `app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.triagem_titulo_resumo.xlsx` — artefato operacional gerado ou execução histórica.
- `atualizar_academic_pipeline_bundle.py` — script histórico de aplicação, atualização ou migração na raiz.
- `install_rc10.sh` — instalador ou script operacional fora do escopo da AP-004.

## Entry points identificados

| Tipo | Nome | Destino | Arquivo |
|---|---|---|---|
| package-main-module | `python -m academic_pipeline` | `academic_pipeline.__main__` | `academic_pipeline/__main__.py` |
| project-script | `academic-pipeline` | `academic_pipeline.cli:main` | `pyproject.toml` |
| python-main-guard | `academic_pipeline.__main__` | `['SystemExit', 'main']` | `academic_pipeline/__main__.py` |
| python-main-guard | `aplicar_docx_canonico_v10` | `['SystemExit', 'main']` | `aplicar_docx_canonico_v10.py` |
| python-main-guard | `aplicar_docx_canonico_v11` | `['SystemExit', 'main']` | `aplicar_docx_canonico_v11.py` |
| python-main-guard | `aplicar_docx_canonico_v12` | `['SystemExit', 'main']` | `aplicar_docx_canonico_v12.py` |
| python-main-guard | `aplicar_docx_canonico_v13` | `['SystemExit', 'main']` | `aplicar_docx_canonico_v13.py` |
| python-main-guard | `aplicar_docx_capa_disciplina_v14` | `['SystemExit', 'main']` | `aplicar_docx_capa_disciplina_v14.py` |
| python-main-guard | `aplicar_docx_only_gerador_unificado_v9` | `['SystemExit', 'main']` | `aplicar_docx_only_gerador_unificado_v9.py` |
| python-main-guard | `app_bundle.scripts.pipeline.academic_pipeline_gui` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/academic_pipeline_gui.py` |
| python-main-guard | `app_bundle.scripts.pipeline.academic_pipeline_rc10` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` |
| python-main-guard | `app_bundle.scripts.pipeline.academic_pipeline_toml_generator_interativo` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py` |
| python-main-guard | `app_bundle.scripts.pipeline.academic_pipeline_toml_generator_v0_3_1` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py` |
| python-main-guard | `app_bundle.scripts.pipeline.academic_pipeline_tui` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/academic_pipeline_tui.py` |
| python-main-guard | `app_bundle.scripts.pipeline.artigo_prisma_workflow` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/artigo_prisma_workflow.py` |
| python-main-guard | `app_bundle.scripts.pipeline.clean_bundle` | `['argparse.ArgumentParser', 'parser.add_argument', 'parser.add_argument', 'parser.add_argument', 'parser.add_argument', 'parser.add_argument', 'parser.parse_args', 'clean_institutional_tree', 'print', 'res.get', 'CleanAction', 'render_clean_report', 'print', 'Path(args.base_dir).expanduser().resolve', 'Path(args.base_dir).expanduser', 'Path']` | `app_bundle/scripts/pipeline/clean_bundle.py` |
| python-main-guard | `app_bundle.scripts.pipeline.gerar_artigo_final_unificado` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/gerar_artigo_final_unificado.py` |
| python-main-guard | `app_bundle.scripts.pipeline.gerar_artigo_longo_fulltext_secional` | `['main']` | `app_bundle/scripts/pipeline/gerar_artigo_longo_fulltext_secional.py` |
| python-main-guard | `app_bundle.scripts.pipeline.preparar_artigo_longo_fulltext` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/preparar_artigo_longo_fulltext.py` |
| python-main-guard | `app_bundle.scripts.pipeline.prisma_baixar_fulltext_artigos` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/prisma_baixar_fulltext_artigos.py` |
| python-main-guard | `app_bundle.scripts.pipeline.prisma_congelar_artigo` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/prisma_congelar_artigo.py` |
| python-main-guard | `app_bundle.scripts.pipeline.prisma_curadoria_ia_referencias` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/prisma_curadoria_ia_referencias.py` |
| python-main-guard | `app_bundle.scripts.pipeline.prisma_diagrama_fluxo` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/prisma_diagrama_fluxo.py` |
| python-main-guard | `app_bundle.scripts.pipeline.prisma_exportar_bib` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/prisma_exportar_bib.py` |
| python-main-guard | `app_bundle.scripts.pipeline.prisma_fulltext_garantido` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/prisma_fulltext_garantido.py` |
| python-main-guard | `app_bundle.scripts.pipeline.render_docx_canonico` | `['SystemExit', 'main']` | `app_bundle/scripts/pipeline/render_docx_canonico.py` |
| python-main-guard | `app_bundle.scripts.pipeline.validar_artigo_longo_fulltext` | `['main']` | `app_bundle/scripts/pipeline/validar_artigo_longo_fulltext.py` |
| python-main-guard | `app_bundle.tests.test_rc10_smoke` | `['test_doctor_returns_dict', 'test_check_config_detects_duplicate_program_course', 'test_render_org_without_empty_citation', 'print', 'Path', 'tempfile.mkdtemp', '_rc10_4_imports', 'print', 'print']` | `app_bundle/tests/test_rc10_smoke.py` |
| python-main-guard | `atualizar_academic_pipeline_bundle` | `['SystemExit', 'main']` | `atualizar_academic_pipeline_bundle.py` |
| python-main-guard | `configurar_pretriagem_ia_prisma_v16` | `['SystemExit', 'main', 'print', 'SystemExit']` | `configurar_pretriagem_ia_prisma_v16.py` |
| python-main-guard | `diagnosticar_fontes_prisma` | `['SystemExit', 'main']` | `diagnosticar_fontes_prisma.py` |
| python-main-guard | `gerar_artigo_final_unificado` | `['SystemExit', 'main']` | `gerar_artigo_final_unificado.py` |
| python-main-guard | `gerar_docx_canonico` | `['SystemExit', 'main']` | `gerar_docx_canonico.py` |
| python-main-guard | `gerar_log_diagnostico_artigo_v1_18` | `['main']` | `gerar_log_diagnostico_artigo_v1_18.py` |
| python-main-guard | `tools.refactor.ap003a_inventory_orchestrator` | `['SystemExit', 'main']` | `tools/refactor/ap003a_inventory_orchestrator.py` |

## Nomes protegidos

- Diretório físico: `academic_pipeline_rc10_7_conformidade` — permanece até AP-006.
- `_refs_v6_strip_org` — xfail histórico congelado.
- `extract_org_abstracts` — xfail histórico congelado.
- `WorkflowState._normalize` — xfail histórico congelado.
- `legacy` permanece quando identifica compatibilidade real.
- `academic-pipeline` e `python -m academic_pipeline` permanecem contratos públicos.

## Validação

- `py_compile`: `passed`.
- `git diff --check`: `passed`.
- Suíte específica: `10 passed`.
- Suíte consolidada: `418 passed, 3 xfailed`.

## Decisão de fase

A AP-004B permanece bloqueada. A matriz v4.2 deve ser revisada e aprovada antes de qualquer renomeação ou commit.

# AP-007D.0 — Inventário operacional corrigido

## Resultado executivo

- HEAD auditado: `ab066e68947ac5f33f1c12c9a7db5086d0f93790`.
- Opções de parser inventariadas: **152**.
- Comandos realmente ainda no fallback legado: **64**.
- Elegíveis pelo filtro conservador da primeira onda: **7**.
- Recomendados provisoriamente para caracterização na AP-007D.1: **`--check-institution-compliance`, `--list-profiles`, `--make-doi-manifest`**.

## Correção aplicada

A auditoria inicial validou corretamente o contrato nativo, mas a associação posterior entre opção do parser e rota pública voltou a usar `legacy_fallback` como padrão. Isso reinseriu `--explain-profile` na lista legada, embora a própria auditoria o tivesse confirmado em `native_first_wave`. O inventário foi reclassificado para tornar o contrato nativo uma fonte vinculante em todas as etapas.

Também foi endurecida a seleção da primeira onda. Comandos com escrita em disco, semântica mutante, subprocessos, rede, interatividade, credenciais, estado global, volatilidade, ausência de testes ou ausência de estágio histórico explícito não podem mais ser recomendados automaticamente.

## Contrato de rotas nativas preservado

- `--help` → `native_first_wave`
- `--list-toml-profiles` → `native_first_wave`
- `--list-institutions` → `native_first_wave`
- `--list-layouts` → `native_first_wave`
- `--explain-profile` → `native_first_wave`
- `--doctor` → `native_doctor`
- `--check-config` → `native_check_config`

## Ranking dos comandos ainda legados

| # | Flag | Destino | Pontos | Elegível | Exclusões conservadoras | Testes |
|---:|---|---|---:|:---:|---|---:|
| 1 | `--apply` | `apply` | 30 | não | historical_dispatch_not_located, mutating_command_semantics | 9 |
| 2 | `--init-project` | `init_project` | 30 | não | mutating_command_semantics | 3 |
| 3 | `--no-clear` | `no_clear` | 30 | não | historical_dispatch_not_located | 3 |
| 4 | `--write` | `write` | 30 | não | historical_dispatch_not_located, mutating_command_semantics | 4 |
| 5 | `--check-institution-compliance` | `check_institution_compliance` | 29 | sim | — | 2 |
| 6 | `--list-profiles` | `list_profiles` | 29 | sim | — | 1 |
| 7 | `--make-doi-manifest` | `make_doi_manifest` | 29 | sim | — | 1 |
| 8 | `--overwrite-project` | `overwrite_project` | 29 | não | historical_dispatch_not_located | 1 |
| 9 | `--write-prompt-lock` | `write_prompt_lock` | 29 | não | mutating_command_semantics | 1 |
| 10 | `--compile` | `compile` | 28 | não | mutating_command_semantics | 10 |
| 11 | `--dry-run` | `dry_run` | 28 | não | historical_dispatch_not_located | 1 |
| 12 | `--gui` | `gui` | 28 | sim | — | 6 |
| 13 | `--recompile` | `recompile` | 28 | sim | — | 4 |
| 14 | `--tui` | `tui` | 28 | sim | — | 8 |
| 15 | `--check` | `check` | 27 | não | historical_dispatch_not_located, mutating_command_semantics, subprocess_execution | 31 |
| 16 | `--docx-only` | `docx_only` | 27 | não | no_existing_tests | 0 |
| 17 | `--emit-docx` | `emit_docx` | 27 | não | no_existing_tests | 0 |
| 18 | `--filtrar-incluidas` | `filtrar_incluidas` | 27 | não | historical_dispatch_not_located, no_existing_tests | 0 |
| 19 | `--init-toml` | `init_toml` | 27 | não | mutating_command_semantics | 1 |
| 20 | `--no-clean` | `no_clean` | 27 | não | historical_dispatch_not_located, mutating_command_semantics, no_existing_tests | 0 |
| 21 | `--no-output-subdir` | `no_output_subdir` | 27 | não | handler_not_located, historical_dispatch_not_located | 1 |
| 22 | `--prisma-cfg` | `prisma_cfg` | 27 | não | complete_or_external_pipeline, historical_dispatch_not_located, mutating_command_semantics, no_existing_tests | 0 |
| 23 | `--prisma-curadoria-fluxo-completo` | `prisma_curadoria_fluxo_completo` | 27 | não | complete_or_external_pipeline, global_process_state, historical_dispatch_not_located | 1 |
| 24 | `--prisma-curadoria-importar` | `prisma_curadoria_importar` | 27 | não | complete_or_external_pipeline, global_process_state, historical_dispatch_not_located, mutating_command_semantics | 1 |
| 25 | `--prisma-curadoria-input` | `prisma_curadoria_input` | 27 | não | complete_or_external_pipeline, historical_dispatch_not_located, mutating_command_semantics | 1 |
| 26 | `--prisma-curadoria-max-incluir` | `prisma_curadoria_max_incluir` | 27 | não | complete_or_external_pipeline, historical_dispatch_not_located | 1 |
| 27 | `--prisma-curadoria-menu` | `prisma_curadoria_menu` | 27 | não | complete_or_external_pipeline, handler_not_located, historical_dispatch_not_located | 1 |
| 28 | `--prisma-curadoria-prompt` | `prisma_curadoria_prompt` | 27 | não | complete_or_external_pipeline, historical_dispatch_not_located | 1 |
| 29 | `--prisma-curadoria-reexportar-xlsx` | `prisma_curadoria_reexportar_xlsx` | 27 | não | complete_or_external_pipeline, historical_dispatch_not_located, mutating_command_semantics | 1 |
| 30 | `--prisma-importar-triagem` | `prisma_importar_triagem` | 27 | não | complete_or_external_pipeline | 1 |
| 31 | `--quality-report` | `quality_report` | 27 | não | no_existing_tests | 0 |
| 32 | `--reexportar-xlsx` | `reexportar_xlsx` | 27 | sim | — | 1 |
| 33 | `--show-prompts` | `show_prompts` | 27 | não | filesystem_writes | 1 |
| 34 | `--forcar-regeneracao-mapa-mental` | `forcar_regeneracao_mapa_mental` | 26 | não | complete_or_external_pipeline, filesystem_writes | 1 |
| 35 | `--reusar-mapa-mental` | `reusar_mapa_mental` | 26 | não | complete_or_external_pipeline, filesystem_writes | 1 |
| 36 | `--skip-tests` | `skip_tests` | 26 | não | historical_dispatch_not_located, no_existing_tests | 0 |
| 37 | `--somente-mapa-mental` | `somente_mapa_mental` | 26 | não | complete_or_external_pipeline, filesystem_writes | 1 |
| 38 | `--somente-renderizar` | `somente_renderizar` | 26 | não | complete_or_external_pipeline, filesystem_writes | 1 |
| 39 | `--ativar-selecao-final` | `ativar_selecao_final` | 25 | não | no_existing_tests | 0 |
| 40 | `--exigir-prioritarios` | `exigir_prioritarios` | 25 | não | no_existing_tests | 0 |
| 41 | `--fail-if-insufficient` | `fail_if_insufficient` | 25 | não | no_existing_tests | 0 |
| 42 | `--gerar-artigo-final` | `gerar_artigo_final` | 25 | não | mutating_command_semantics, no_existing_tests, subprocess_execution | 0 |
| 43 | `--gerar-toml-artigo` | `gerar_toml_artigo` | 25 | não | mutating_command_semantics, no_existing_tests | 0 |
| 44 | `--json` | `as_json` | 25 | não | filesystem_writes, historical_dispatch_not_located, no_existing_tests | 0 |
| 45 | `--keep-backups` | `keep_backups` | 25 | não | handler_not_located, historical_dispatch_not_located, no_existing_tests | 0 |
| 46 | `--no-fetch` | `no_fetch` | 25 | não | historical_dispatch_not_located, no_existing_tests | 0 |
| 47 | `--pipeline-script` | `pipeline_script` | 25 | não | complete_or_external_pipeline, no_existing_tests | 0 |
| 48 | `--prisma-curadoria-ia` | `prisma_curadoria_ia` | 25 | não | complete_or_external_pipeline, historical_dispatch_not_located, no_existing_tests | 0 |
| 49 | `--prisma-curadoria-limiar-minimo` | `prisma_curadoria_limiar_minimo` | 25 | não | complete_or_external_pipeline, historical_dispatch_not_located, no_existing_tests | 0 |
| 50 | `--prisma-curadoria-out-dir` | `prisma_curadoria_out_dir` | 25 | não | complete_or_external_pipeline, historical_dispatch_not_located, mutating_command_semantics, no_existing_tests | 0 |
| 51 | `--prisma-curadoria-sem-ia` | `prisma_curadoria_sem_ia` | 25 | não | complete_or_external_pipeline, handler_not_located, historical_dispatch_not_located, no_existing_tests | 0 |
| 52 | `--prisma-curadoria-top-n-candidatos` | `prisma_curadoria_top_n_candidatos` | 25 | não | complete_or_external_pipeline, historical_dispatch_not_located, no_existing_tests | 0 |
| 53 | `--remove-output` | `remove_output` | 25 | não | handler_not_located, historical_dispatch_not_located, mutating_command_semantics, no_existing_tests | 0 |
| 54 | `--remove-projects` | `remove_projects` | 25 | não | handler_not_located, historical_dispatch_not_located, mutating_command_semantics, no_existing_tests | 0 |
| 55 | `--skip-generate` | `skip_generate` | 25 | não | mutating_command_semantics, no_existing_tests | 0 |
| 56 | `--skip-prepare` | `skip_prepare` | 25 | não | no_existing_tests | 0 |
| 57 | `--tui-theme` | `tui_theme` | 25 | não | historical_dispatch_not_located, no_existing_tests, subprocess_execution | 0 |
| 58 | `--usar-ia` | `usar_ia` | 25 | não | mutating_command_semantics, no_existing_tests | 0 |
| 59 | `--verify-post` | `verify_post` | 24 | não | historical_dispatch_not_located, no_existing_tests, subprocess_execution | 0 |
| 60 | `--artigo-prefix` | `artigo_prefix` | 17 | não | complete_or_external_pipeline, destructive_filesystem_effects, filesystem_writes, historical_dispatch_not_located, mutating_command_semantics, no_existing_tests, score_below_conservative_threshold, subprocess_execution | 0 |
| 61 | `--fallback-proxy-oa` | `fallback_proxy_oa` | 16 | não | network_access, no_existing_tests, score_below_conservative_threshold | 0 |
| 62 | `--prisma-out-dir` | `prisma_out_dir` | 15 | não | complete_or_external_pipeline, credentials, destructive_filesystem_effects, filesystem_writes, historical_dispatch_not_located, no_existing_tests, score_below_conservative_threshold | 0 |
| 63 | `--artigo-dir` | `artigo_dir` | 14 | não | complete_or_external_pipeline, destructive_filesystem_effects, filesystem_writes, historical_dispatch_not_located, mutating_command_semantics, no_existing_tests, score_below_conservative_threshold, subprocess_execution, volatile_behavior | 0 |
| 64 | `--prisma-config` | `prisma_config` | 8 | não | complete_or_external_pipeline, credentials, destructive_filesystem_effects, filesystem_writes, historical_dispatch_not_located, mutating_command_semantics, network_access, no_existing_tests, score_below_conservative_threshold, subprocess_execution, volatile_behavior | 0 |

## Primeira onda provisória

A recomendação corrigida aceita apenas comandos ainda legados, com estágio histórico e testes localizados, pontuação mínima de 24/30 e ausência de escrita, destruição, rede, UI, subprocessos, credenciais, volatilidade, mutação de cwd ou estado global. A AP-007D.1 ainda deverá caracterizar cada candidato antes de qualquer adaptador ou alteração de rota.

- `--check-institution-compliance` — 29/30; handlers: dispatch_stage_015.branch@239; testes localizados: 2.
- `--list-profiles` — 29/30; handlers: main.branch@4250; testes localizados: 1.
- `--make-doi-manifest` — 29/30; handlers: dispatch_stage_010.branch@160; testes localizados: 1.

## Riscos remanescentes

- A análise permanece estática e não prova a ausência de efeitos transitivos em funções chamadas.
- A presença de testes localizados por token não prova cobertura comportamental suficiente.
- A recomendação é apenas uma fila de caracterização; nenhuma rota pública foi alterada.
- Comandos mutantes como --apply e --init-project foram excluídos da recomendação automática.

## Gate

`[GATE] AP-007D.0: INVENTÁRIO OPERACIONAL RECLASSIFICADO E VALIDADO; FALSO CANDIDATO NATIVO REMOVIDO, RECOMENDAÇÃO CONSERVADORA MATERIALIZADA, SEM MODIFICAÇÃO PRODUTIVA, STAGING, COMMIT, TAG OU PUSH.`

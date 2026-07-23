# AP-007C.5 — Integração pública do `--check-config`

## Resultado

`--check-config` passa a usar o adaptador nativo materializado na
AP-007C.4. O `--doctor` permanece nativo.

## Precedência

A primeira onda mantém prioridade. O doctor mantém prioridade sobre
check-config porque corresponde a `dispatch_stage_016`, anterior ao
`dispatch_stage_017`.

Quando um destino de estágio anterior está ativo sem doctor, o runtime
mantém a execução no fallback legado.

Destinos bloqueadores anteriores:
check_institution_compliance, doctor, explain_profile, forcar_regeneracao_mapa_mental, gui, init_project, init_toml, inspect_bib, list_institutions, list_layouts, list_toml_profiles, make_doi_manifest, output, reusar_mapa_mental, show_prompts, somente_mapa_mental, somente_renderizar, tui, write_prompt_lock.

Probe de precedência:
`--check-institution-compliance --check-config`.

## Contratos

- sem `--config`: retorno de processo `1` e mensagem obrigatória;
- relatório válido: retorno `0`;
- relatório com problemas: retorno `2`;
- `--doctor --check-config`: rota doctor;
- comandos de estágios anteriores: fallback;
- primeira onda: precedência nativa preservada;
- sem mutação de `sys.argv`, `sys.path` ou cwd.

Nenhum staging, commit, tag ou push está autorizado.

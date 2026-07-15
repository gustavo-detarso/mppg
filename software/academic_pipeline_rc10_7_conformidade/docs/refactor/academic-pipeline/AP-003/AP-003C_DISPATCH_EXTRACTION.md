# AP-003C — extração do despacho de comandos

## Escopo

Foram extraídos somente blocos de despacho terminais comprovados por AST. Cada delegação permaneceu na posição estrutural do bloco original, preservando preparação, ordem e precedência.

## Critérios conservadores

- instrução `if` de nível superior;
- ausência de `else` ou `elif`;
- término garantido em `return` ou `raise`;
- rejeição de escopos aninhados, `yield`, `await`, controle de laços, atribuições aumentadas e expressões de atribuição;
- resolução tardia de nomes externos por mapa de runtime;
- dois `main()` e alias histórico preservados;
- parser da AP-003B preservado byte a byte.

## Estágios extraídos

| Estágio | Origem | Condição | Atributos | Runtime |
|---|---:|---|---|---|
| `dispatch_stage_001` | 914–920 | `args.gui` | `args.gui` | `run_gui` |
| `dispatch_stage_002` | 922–928 | `args.tui` | `args.no_clear`, `args.tui` | `run_tui` |
| `dispatch_stage_003` | 930–937 | `args.list_toml_profiles` | `args.list_toml_profiles` | `print_profiles` |
| `dispatch_stage_004` | 939–946 | `args.init_toml` | `args.init_toml`, `args.no_clear`, `args.toml_profile` | `generate_interactive` |
| `dispatch_stage_005` | 948–950 | `args.list_institutions` | `args.list_institutions` | `describe_institution_profiles` |
| `dispatch_stage_006` | 952–967 | `args.list_layouts` | `args.config`, `args.list_layouts` | `Path`, `available_layouts`, `load_config`, `resolve_layout_spec` |
| `dispatch_stage_007` | 969–971 | `args.explain_profile` | `args.explain_profile` | `explain_profile` |
| `dispatch_stage_008` | 973–978 | `args.show_prompts` | `args.config`, `args.show_prompts` | `Path`, `json`, `load_config`, `prompt_report_for_cfg` |
| `dispatch_stage_009` | 980–990 | `args.init_project` | `args.base_dir`, `args.init_project`, `args.institution`, `args.overwrite_project`, `args.project_type` | `Path`, `init_project` |
| `dispatch_stage_010` | 992–1009 | `args.make_doi_manifest` | `args.input_dir`, `args.input_zip`, `args.make_doi_manifest`, `args.output` | `Path`, `make_doi_manifest` |
| `dispatch_stage_011` | 1011–1017 | `args.inspect_bib` | `args.inspect_bib` | `Path`, `inspect_bib`, `render_bib_inspection_markdown` |
| `dispatch_stage_012` | 1045–1046 | `args.somente_renderizar and args.somente_mapa_mental` | `args.somente_mapa_mental`, `args.somente_renderizar` | nenhum |
| `dispatch_stage_013` | 1047–1048 | `args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental` | `args.forcar_regeneracao_mapa_mental`, `args.reusar_mapa_mental` | nenhum |
| `dispatch_stage_014` | 1050–1060 | `args.write_prompt_lock` | `args.write_prompt_lock` | `cfg`, `external_search_enabled`, `output_paths`, `research_output_paths`, `write_prompt_lock`, `write_prompt_lock_markdown` |
| `dispatch_stage_015` | 1062–1074 | `args.check_institution_compliance` | `args.bib`, `args.check_institution_compliance`, `args.docx`, `args.org`, `args.pdf` | `Path`, `cfg`, `output_paths`, `render_compliance_markdown`, `run_institution_compliance`, `write_compliance_reports` |
| `dispatch_stage_016` | 1076–1082 | `args.doctor` | `args.doctor` | `cfg`, `external_search_enabled`, `output_paths`, `print_doctor_report`, `research_output_paths`, `run_doctor`, `write_json` |
| `dispatch_stage_017` | 1084–1091 | `args.check_config` | `args.check_config` | `cfg`, `check_config`, `external_search_enabled`, `output_paths`, `print_check_config_report`, `research_output_paths`, `write_json` |
| `dispatch_stage_018` | 1093–1094 | `args.recompile` | `args.recompile` | `cfg`, `run_recompile` |
| `dispatch_stage_019` | 1099–1139 | `args.prisma_importar_triagem` | `args.prisma_importar_triagem` | `PIPELINE_VERSION`, `Path`, `cfg`, `external_search_enabled`, `import_manual_prisma_triage`, `make_run_report`, `print_outputs`, `render_external_prisma_outputs`, `research_output_paths`, `stage`, `write_json`, `write_outputs_manifest` |

## Candidatos mantidos

| Origem | Condição | Motivo |
|---:|---|---|
| 1019–1039 | `args.quality_report` | contém nó não extraível: NamedExpr |
| 1041–1041 | `` | tipo Assign, não If |
| 1141–1141 | `` | tipo Assign, não If |
| 1142–1142 | `` | tipo Assign, não If |
| 1160–1160 | `` | tipo Assign, não If |
| 1162–1224 | `is_external_prisma_run` | carrega nome antes da primeira atribuição local: model |
| 1226–1279 | `args.somente_mapa_mental` | carrega nome antes da primeira atribuição local: model |
| 1286–1439 | `args.somente_renderizar` | possui else/elif |
| 1473–1499 | `args.somente_renderizar` | possui else/elif |
| 1544–1559 | `` | tipo Assign, não If |

## Integridade

- Orquestrador antes: `51af32106184df8fd5810222a8ccdb5cc0818aa3e167ff8bd2e1c96199ef1a0f`.
- Orquestrador depois: `4261568e60308764ef1f56ab1e13d6ccfd886d76dce965ec6d3e8fd66cdee51d`.
- Parser AP-003B: `f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8`.
- Estágios extraídos: **19**.
- Candidatos mantidos: **10**.

A orquestração documental e os fluxos PRISMA/artigo genérico permanecem no arquivo histórico para as AP-003D e AP-003E.

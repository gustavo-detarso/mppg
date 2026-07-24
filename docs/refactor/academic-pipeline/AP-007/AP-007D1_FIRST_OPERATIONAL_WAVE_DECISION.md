# AP-007D.1 — Decisão da primeira onda operacional

## Resultado

- Status: `selected`
- Comandos selecionados: `--list-profiles`
- Limite deliberado da primeira onda: 1 comando.
- Integração pública: não realizada.
- Modificação produtiva: nenhuma.

## Método

A decisão combina o inventário AST corrigido da AP-007D.0 com resolução transitiva dirigida de chamadas locais e importadas, inspeção de efeitos colaterais, localização de testes, precedência histórica e análise do contrato do parser. A seleção é fail-closed: qualquer escrita, rede, interface, subprocesso, credencial, mutação de estado do processo, superfície transitiva excessiva ou ausência de evidência bloqueia a promoção automática.

## Classificação dos candidatos

| Comando | Pontuação | Risco | Elegível | Funções transitivas | Testes | Exclusões |
|---|---:|---|---|---:|---:|---|
| `--list-profiles` | 32/33 | low | sim | 2 | 43 | nenhuma |
| `--check-institution-compliance` | 31/33 | low | sim | 4 | 23 | nenhuma |
| `--make-doi-manifest` | 26/33 | moderate | não | 1 | 3 | mutating_or_generating_command_semantics |

## Onda selecionada

### `--list-profiles`

- Handler(s) resolvido(s): software/academic_pipeline_mppg/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:main.branch@4250:4250
- Dependências diretas: nenhuma
- Testes localizados: software/academic_pipeline_mppg/app_bundle/tests/test_entrypoints_orchestration_characterization.py, software/academic_pipeline_mppg/app_bundle/tests/test_official_package_entrypoint.py, software/academic_pipeline_mppg/app_bundle/tests/test_package_imports_document_core.py, software/academic_pipeline_mppg/app_bundle/tests/test_package_imports_entrypoints.py, software/academic_pipeline_mppg/app_bundle/tests/test_package_imports_prisma_core.py, software/academic_pipeline_mppg/app_bundle/tests/test_package_imports_rendering.py, software/academic_pipeline_mppg/app_bundle/tests/test_package_imports_support_services.py, software/academic_pipeline_mppg/app_bundle/tests/test_packaging_metadata.py, software/academic_pipeline_mppg/app_bundle/tests/test_paper_abstracts_layouts_characterization.py, software/academic_pipeline_mppg/app_bundle/tests/test_rc10_smoke.py, software/academic_pipeline_mppg/tests/characterization/test_ap003a_orchestrator_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap003b_parser_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap003c_dispatch_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap003d_document_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap003e_prisma_generic_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap003f_main_unification_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap003g_stabilization_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap004a_naming_inventory_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap004b_module_file_application_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap004b_module_file_inventory_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap004c_internal_symbol_application_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap004f_closure_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap005b_consumer_canonicalization_plan_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap005b_consumer_contract_reclassification.py, software/academic_pipeline_mppg/tests/characterization/test_ap005c1_toml_capture_alias_application_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap005c2_stabilization_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap005c3_closure_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap005d_facade_inventory_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap005e1_installation_entrypoint_inventory_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap005e2_isolated_build_installation_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap006c_physical_materialization.py, software/academic_pipeline_mppg/tests/characterization/test_ap006d1_runtime_consumer_migration.py, software/academic_pipeline_mppg/tests/characterization/test_ap006d2_contract_validator_migration.py, software/academic_pipeline_mppg/tests/characterization/test_ap006d3_operational_source_migration.py, software/academic_pipeline_mppg/tests/characterization/test_ap006d4d_source_csv_provenance_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap006e3_consumer_stabilization.py, software/academic_pipeline_mppg/tests/characterization/test_ap006e5_closure_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap006f4_comparative_source_distribution_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap006f5_closure_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap007b_native_runtime_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap007c3_doctor_public_integration_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap007c5_check_config_public_integration_contract.py, software/academic_pipeline_mppg/tests/characterization/test_ap007c6_closure_contract.py
- Códigos de retorno observados estaticamente: [0]
- Chamadas não resolvidas: textwrap.fill
- Próxima subfase: construir adaptador nativo isolado, mantendo a rota pública no fallback legado.

## Contratos obrigatórios para a AP-007D.2

1. Executar adaptador e histórico com `argv` explícito em diretório temporário.
2. Comparar código de retorno, stdout, stderr e arquivos gerados.
3. Cobrir caminho válido, erro de uso e combinações de precedência com `--help`, `--doctor` e `--check-config`.
4. Confirmar preservação de `sys.argv`, `sys.path` e diretório corrente.
5. Não alterar a rota pública até a equivalência direta ser aprovada.

## Escopo

Somente os quatro artefatos AP-007D.1 foram produzidos. Runtime, CLI, parser, dispatcher, monólito e módulos produtivos permanecem inalterados.

## Gate

`[GATE] AP-007D.1: DECISÃO DA PRIMEIRA ONDA MATERIALIZADA; comandos=--list-profiles; NENHUMA INTEGRAÇÃO PÚBLICA.`

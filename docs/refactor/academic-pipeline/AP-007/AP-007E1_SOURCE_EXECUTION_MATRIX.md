# AP-007E.1 — Matriz de execução pela fonte

## Resultado

- **Status:** `matrix_approved_with_classified_historical_debt`
- **Commit preservado:** `766956710435f1c338d2e0332d24e55106b981b7`
- **Fonte:** `software/academic_pipeline_mppg`
- **Console target declarado:** `academic-pipeline = academic_pipeline.cli:main`
- **Casos:** `6`
- **Superfícies:** `6`
- **Execuções:** `36`
- **Falhas:** `0`
- **Avisos:** `6`
- **PYTHONPATH:** removido em todos os subprocessos
- **Build/instalação:** não executados

## Equivalência das superfícies canônicas pela fonte

| Caso | Equivalente | Retorno documentado | Sem timeout |
| --- | --- | --- | --- |
| help | True | True | True |
| list_institutions | True | True | True |
| list_profiles | True | True | True |
| check_config_missing_config | True | True | True |
| institution_compliance_missing_config | True | True | True |
| doi_manifest_missing_input | True | True | True |

As superfícies canônicas comparadas foram `direct_import_call_source_root`, `python_minus_m_source_root` e `official_target_source_root`. A comparação usa return code, stdout e stderr após normalização nominalmente documentada.

## Probes de descoberta

| Probe | Retorno | module.__file__ |
| --- | --- | --- |
| source_root_import | 0 | /home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline-ap005/software/academic_pipeline_mppg/academic_pipeline/__init__.py |
| worktree_root_import | 1 | — |
| neutral_cwd_import | 1 | — |

Os probes em worktree root e CWD neutro caracterizam a dependência de descoberta pela fonte. Sucesso fora do source root seria tratado como contaminação por instalação ambiente ou bridge de caminho.

## Preservação do processo e import guard

| Superfície | Caso | argv | sys.path | cwd | origem fonte | tentativas monólito |
| --- | --- | --- | --- | --- | --- | --- |
| direct_import_call_source_root | help | True | True | True | True | 0 |
| direct_import_call_source_root | list_institutions | True | True | True | True | 0 |
| direct_import_call_source_root | list_profiles | True | True | True | True | 0 |
| direct_import_call_source_root | check_config_missing_config | True | True | True | True | 0 |
| direct_import_call_source_root | institution_compliance_missing_config | True | True | True | True | 0 |
| direct_import_call_source_root | doi_manifest_missing_input | True | True | True | True | 0 |
| official_target_source_root | help | True | True | True | True | 0 |
| official_target_source_root | list_institutions | True | True | True | True | 0 |
| official_target_source_root | list_profiles | True | True | True | True | 0 |
| official_target_source_root | check_config_missing_config | True | True | True | True | 0 |
| official_target_source_root | institution_compliance_missing_config | True | True | True | True | 0 |
| official_target_source_root | doi_manifest_missing_input | True | True | True | True | 0 |

O import guard bloqueou nominalmente qualquer tentativa de carregar `academic_pipeline_rc10`, registrando também tentativas relacionadas a `project_tools`, `bibliography_manager`, `dotenv` e `pydantic`.

## Testes históricos destinados à AP-007E

| Teste | Node ID | Status | Classificação | Bloqueante | Retorno |
| --- | --- | --- | --- | --- | --- |
| test_legacy_script_help_remains_supported | app_bundle/tests/test_official_package_entrypoint.py::test_legacy_script_help_remains_supported | classified_failure | legacy_direct_source_bridge_absent | False | 1 |
| test_official_and_legacy_help_expose_same_options | app_bundle/tests/test_official_package_entrypoint.py::test_official_and_legacy_help_expose_same_options | classified_failure | legacy_direct_source_bridge_absent | False | 1 |
| test_official_and_legacy_list_institutions_match | app_bundle/tests/test_official_package_entrypoint.py::test_official_and_legacy_list_institutions_match | classified_failure | legacy_direct_source_bridge_absent | False | 1 |
| test_legacy_entrypoint_still_matches_console_target | app_bundle/tests/test_packaging_metadata.py::test_legacy_entrypoint_still_matches_console_target | classified_failure | legacy_direct_source_bridge_absent | False | 1 |

Cada teste foi executado individualmente por node ID exato. Falhas com a assinatura nominal `ModuleNotFoundError: No module named 'academic_pipeline'` no entrypoint legado direto são registradas como dívida de bridge da fonte destinada à AP-007F; qualquer assinatura diferente permanece bloqueante.

## Superfícies não canônicas

| Superfície | Retornos | Classificação |
| --- | --- | --- |
| python_minus_m_worktree_root | [1] | cwd_or_direct_file_dependency_characterized |
| python_minus_m_neutral_cwd | [1] | cwd_or_direct_file_dependency_characterized |
| absolute_main_neutral_cwd | [1] | cwd_or_direct_file_dependency_characterized |

## Normalizações autorizadas

- replace absolute worktree path with <REPO>
- replace source root with <SOURCE_ROOT>
- replace run sandbox with <RUN_ROOT>
- replace canonical Python path with <PYTHON>
- normalize argparse usage program token to <PROG>
- normalize argparse error program prefix to <PROG>
- normalize CRLF to LF and remove trailing horizontal whitespace
- normalize Python traceback frames to the terminal exception line only

## Falhas

Nenhuma.

## Avisos

- `CWD_DEPENDENCY_CHARACTERIZED`
- `CWD_DEPENDENCY_CHARACTERIZED`
- `HISTORICAL_LEGACY_SOURCE_BRIDGE_DEBT`
- `HISTORICAL_LEGACY_SOURCE_BRIDGE_DEBT`
- `HISTORICAL_LEGACY_SOURCE_BRIDGE_DEBT`
- `HISTORICAL_LEGACY_SOURCE_BRIDGE_DEBT`

## Gate

```text
[GATE] AP-007E.1: MATRIZ DE EXECUÇÃO PELA FONTE APROVADA; DEPENDÊNCIAS DE CWD/PYTHONPATH E DÍVIDAS HISTÓRICAS CLASSIFICADAS NOMINALMENTE; NENHUM BUILD, INSTALAÇÃO, MODIFICAÇÃO PRODUTIVA, STAGING, COMMIT, TAG OU PUSH.
```

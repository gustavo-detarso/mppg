# AP-007E.4 — Equivalência entre fonte, wheel e sdist

Status: **equivalence_approved_with_classified_findings**. Baseline: `766956710435f1c338d2e0332d24e55106b981b7`.

## Desenho da comparação

Três ambientes virtuais descartáveis usaram o mesmo estado mínimo de dependências. A fonte foi executada a partir do snapshot filtrado; wheel e sdist foram instalados sem dependências e sem índice. A matriz contém cinco superfícies e seis casos, totalizando 30 execuções e 24 comparações contra a fonte.

| Superfície | Origem | Casos |
|---|---|---|
| source_python_m | filtered_source_snapshot | 6 |
| wheel_python_m | installed_direct_wheel | 6 |
| wheel_console | installed_direct_wheel | 6 |
| sdist_python_m | installed_sdist_derived_wheel | 6 |
| sdist_console | installed_sdist_derived_wheel | 6 |

## Resultado

| Indicador | Valor |
|---|---|
| Execuções | 30 |
| Comparações | 24 |
| Comparações equivalentes | 24 |
| Divergências | 0 |
| Códigos esperados | True |
| Timeouts ausentes | True |

## Paridade de conteúdo

| Contrato | Resultado |
|---|---|
| Hashes dos módulos `academic_pipeline/*.py` | True |
| Hashes dos oito recursos críticos | True |
| Vazamentos da instalação para a fonte | 0 |
| Ambiente canônico preservado | True |

## Casos

| Caso | Argumentos | RC esperado |
|---|---|---|
| help | --help | 0 |
| list_institutions | --list-institutions | 0 |
| list_profiles | --list-profiles | 0 |
| check_config_without_config | --check-config | 1 |
| check_institution_compliance_without_config | --check-institution-compliance | 1 |
| make_doi_manifest_without_input | --make-doi-manifest | 1 |

## Achados classificados

| Código | Bloqueante | Fase-alvo |
|---|---|---|
| HISTORICAL_LEGACY_SOURCE_BRIDGE_DEBT_CARRIED_FORWARD | False | AP-007F |
| TRACKED_RESIDUAL_SOURCE_MEMBERS_EXCLUDED_BEFORE_EXTRACTION | False | AP-007F |
| PACKAGE_DATA_CANDIDATES_NOT_PRESENT_IN_WHEEL | False | AP-007F |
| DECLARED_RUNTIME_DEPENDENCIES_ABSENT_BY_SYMMETRIC_MINIMAL_ENV_POLICY | False | — |
| RECONSTRUCTED_SDIST_RAW_HASH_DIFFERS_NORMALIZED_EQUAL | False | — |

## Decisão

A execução pela fonte, pelo wheel e pelo wheel derivado do sdist é equivalente para os seis casos selecionados. A normalização substitui apenas caminhos voláteis e, quando há traceback, conserva a linha terminal da exceção. As dívidas históricas, resíduos rastreados e candidatos de package data permanecem encaminhados para a AP-007F.

# AP-007E.3 — Instalação isolada em ambientes descartáveis

Status: **installation_approved_with_classified_findings**. Baseline: `766956710435f1c338d2e0332d24e55106b981b7`.

## Proveniência dos artefatos

O wheel e o sdist foram reconstruídos em sandbox a partir de `git archive` do commit publicado, com o mesmo filtro pré-extração e o mesmo backend PEP 517 da AP-007E.2. Seus manifestos normalizados foram comparados aos dois builds da AP-007E.2 antes da instalação.

| Origem | Instalação | Ambiente | Dependências adicionais |
|---|---|---|---|
| wheel | wheel direto | venv descartável | nenhuma |
| sdist | wheel derivado do sdist pelo backend declarado | venv descartável independente | nenhuma |

## Isolamento

| Contrato | Resultado |
|---|---|
| PYTHONPATH removido | True |
| Vazamentos para árvore de fonte | 0 |
| Módulos dentro dos venvs | True |
| Ambiente canônico preservado | True |
| Rede permitida | False |

## Recursos críticos

| Origem | Recurso | Presente | SHA-256 |
|---|---|---|---|
| wheel | institutions/fgv/institution_profile.toml | True | 04e75cebcebf90dfb28fed83fa8d6b7b4297b8f803aee49c2bdda7b91601d800 |
| wheel | institutions/fgv/assets/fgv.png | True | b3412d7db84838428d2267778fea15f1e575ced37ed5b2c77ec13fbc84359856 |
| wheel | institutions/fgv/docx/reference_fgv.docx | True | 8cca5b96bb49f589950e0a48a4c60303f7d231f856812067d8ab1c74b2ae010d |
| wheel | institutions/fgv/latex/fgv-paper.sty | True | 298501fd18248e4d5d0c7925b6bd9afedd6dc6f0fe282e6460b6f9be22de85b5 |
| wheel | institutions/fgv/templates/template_atividade.org | True | 2210824f824b62d4a21a57c5db338fab32018a4d4ec48b105b888ace6aed24b6 |
| wheel | misc/academic-writing.el | True | fb3f922a09d5b3ad29d3d2a22d5a422bb14f3b92ae07a956a80935ee4591d299 |
| wheel | templates/template_paper.org | True | ebe2081d166b19947cd6d17be14616b3a75800cd61a7670089024626deb3ef70 |
| wheel | templates/csl/associacao-brasileira-de-normas-tecnicas.csl | True | 4f3f2eadb94c314476608e070acb889253f7007fbbe3e2c251ffc90e2163eaae |
| sdist | institutions/fgv/institution_profile.toml | True | 04e75cebcebf90dfb28fed83fa8d6b7b4297b8f803aee49c2bdda7b91601d800 |
| sdist | institutions/fgv/assets/fgv.png | True | b3412d7db84838428d2267778fea15f1e575ced37ed5b2c77ec13fbc84359856 |
| sdist | institutions/fgv/docx/reference_fgv.docx | True | 8cca5b96bb49f589950e0a48a4c60303f7d231f856812067d8ab1c74b2ae010d |
| sdist | institutions/fgv/latex/fgv-paper.sty | True | 298501fd18248e4d5d0c7925b6bd9afedd6dc6f0fe282e6460b6f9be22de85b5 |
| sdist | institutions/fgv/templates/template_atividade.org | True | 2210824f824b62d4a21a57c5db338fab32018a4d4ec48b105b888ace6aed24b6 |
| sdist | misc/academic-writing.el | True | fb3f922a09d5b3ad29d3d2a22d5a422bb14f3b92ae07a956a80935ee4591d299 |
| sdist | templates/template_paper.org | True | ebe2081d166b19947cd6d17be14616b3a75800cd61a7670089024626deb3ef70 |
| sdist | templates/csl/associacao-brasileira-de-normas-tecnicas.csl | True | 4f3f2eadb94c314476608e070acb889253f7007fbbe3e2c251ffc90e2163eaae |

## Comandos instalados

| Origem | Caso | RC | Classificação |
|---|---|---|---|
| wheel | python_m_help | 0 | passed |
| wheel | python_m_list_institutions | 0 | passed |
| wheel | python_m_list_profiles | 0 | passed |
| wheel | console_help | 0 | passed |
| wheel | console_list_institutions | 0 | passed |
| wheel | console_list_profiles | 0 | passed |
| sdist | python_m_help | 0 | passed |
| sdist | python_m_list_institutions | 0 | passed |
| sdist | python_m_list_profiles | 0 | passed |
| sdist | console_help | 0 | passed |
| sdist | console_list_institutions | 0 | passed |
| sdist | console_list_profiles | 0 | passed |

## Dependências

Dependências declaradas: **6**. A instalação foi executada com `--no-deps` e `--no-index`; lacunas declaradas são registradas, enquanto dependência não declarada permanece bloqueante.

## Achados classificados

| Código | Bloqueante | Ambiente | Detalhe |
|---|---|---|---|
| TRACKED_RESIDUAL_SOURCE_MEMBERS_EXCLUDED_BEFORE_EXTRACTION | False |  | AP-007F |
| PACKAGE_DATA_CANDIDATES_NOT_PRESENT_IN_WHEEL | False |  | AP-007E.4 |
| DECLARED_RUNTIME_DEPENDENCIES_NOT_INSTALLED_BY_POLICY | False | wheel-origin | artifact installed with --no-deps and --no-index by phase policy |
| DECLARED_RUNTIME_DEPENDENCIES_NOT_INSTALLED_BY_POLICY | False | sdist-origin | artifact installed with --no-deps and --no-index by phase policy |
| RECONSTRUCTED_SDIST_RAW_HASH_DIFFERS_NORMALIZED_EQUAL | False |  |  |

## Gate

`[GATE] AP-007E.3: WHEEL E SDIST INSTALADOS EM AMBIENTES DESCARTÁVEIS, IMPORTS, RECURSOS, CONSOLE SCRIPT E PYTHON -M AUDITADOS SEM PYTHONPATH OU REUSO DA FONTE; AMBIENTE CANÔNICO PRESERVADO, SEM DEPENDÊNCIAS ADICIONAIS, STAGING, COMMIT, TAG OU PUSH.`

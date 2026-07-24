# AP-007F.0 — Auditoria final do legado residual

## Estado e escopo

| Campo | Valor |
|---|---|
| Status | residual_legacy_audit_complete |
| HEAD AP-007E | ba43b7d606378501d6faafa62ad8c8a6697665e5 |
| Tree AP-007E | 078326090dd64572fb12a026e8d92968bf106d0f |
| Verificação remota | https_readonly_fallback |
| Remoto ao vivo verificado | YES |
| Runtime SHA-256 | b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c |
| Master protegida | 6adc5e7c6ce510a49eba13266eabfa227fbeae31 |
| Python canônico | /home/gustavodetarso/.local/share/virtualenvs/academic_pipeline_rc10_7_conformidade-D8fBLuIA/bin/python |
| Arquivos rastreados examinados | 1474 |
| Arquivos Python dirigidos examinados por AST | 102 |
| Dívidas classificadas | 70 |
| Casos direct-source reproduzidos | 4 |
| Alteração produtiva | não |
| Instalação | não |
| Git de escrita | não |

Esta subfase realizou auditoria dirigida e materializou somente os quatro artefatos AP-007F.0 autorizados. Nenhum módulo produtivo foi alterado; nenhuma dependência foi instalada; nenhuma operação de staging, commit, tag ou push foi executada.

## Arquitetura pública observada

- Distribuição declarada: `academic-pipeline-mppg`.
- Console publicado: `academic-pipeline`.
- Alvo do console: `academic_pipeline.cli:main`.
- Módulo público: `python -m academic_pipeline`.
- Definição de `run_legacy`: `software/academic_pipeline_mppg/academic_pipeline/legacy.py:97`.
- Classificação provisória de `legacy.py`: `adaptador/carregador`.

## Consumidores de `run_legacy`

### Chamadas produtivas

- Nenhum item encontrado.

### Comandos/branches em `RuntimeRoute.LEGACY_FALLBACK`

- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:329` — ``
- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:339` — ``
- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:349` — ``
- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:350` — ``
- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:359` — ``
- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:360` — ``
- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:369` — ``
- `software/academic_pipeline_mppg/academic_pipeline/runtime.py:378` — ``

### Imports, monkeypatches e doubles

- Imports AST: **1**.
- Chamadas AST totais: **3**.
- Referências a fallback: **26**.
- Monkeypatches/doubles relacionados: **14**.

## Execução direta histórica

Foram coletados e executados individualmente os quatro node IDs classificados como `legacy_direct_source_bridge_absent`. Todos foram realmente coletados, nenhum retornou código 4 e todos falharam com a assinatura exata `ModuleNotFoundError: No module named 'academic_pipeline'`, sem `PYTHONPATH`, `.pth`, instalação ou recriação da ponte.

- `?` — `app_bundle/tests/test_official_package_entrypoint.py::test_legacy_script_help_remains_supported`
- `?` — `app_bundle/tests/test_official_package_entrypoint.py::test_official_and_legacy_help_expose_same_options`
- `?` — `app_bundle/tests/test_official_package_entrypoint.py::test_official_and_legacy_list_institutions_match`
- `?` — `app_bundle/tests/test_packaging_metadata.py::test_legacy_entrypoint_still_matches_console_target`

## Matriz mínima de superfícies

| Campo | Valor |
|---|---|
| source_import_from_canonical_root | rc=0 cwd=/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline-ap005/software/academic_pipeline_mppg |
| source_python_m_help_from_canonical_root | rc=0 cwd=/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline-ap005/software/academic_pipeline_mppg |
| neutral_import_without_pythonpath | rc=1 cwd=/home/gustavodetarso/Downloads/mppg-logs/ap007f0_auditoria_final_legado_residual_20260724_151024_1950458/neutral-cwd |
| neutral_python_m_help_without_pythonpath | rc=1 cwd=/home/gustavodetarso/Downloads/mppg-logs/ap007f0_auditoria_final_legado_residual_20260724_151024_1950458/neutral-cwd |
| console_help_from_neutral_cwd | rc=None cwd=/home/gustavodetarso/Downloads/mppg-logs/ap007f0_auditoria_final_legado_residual_20260724_151024_1950458/neutral-cwd |

Os probes neutros são observacionais nesta subfase: a AP-007F.0 não instala o projeto nem dependências no ambiente canônico. A equivalência wheel/sdist permanece gate obrigatório da AP-007F.2.

## Catálogo das 70 dívidas

Fonte selecionada: `docs/refactor/academic-pipeline/AP-007/ap007e5_closure_manifest.json` em `integrated_regression_census.classifications`.

### Contagem por classe

| Campo | Valor |
|---|---|
| direct_source_wrapper_debt | 4 |
| frozen_snapshot_or_hash | 29 |
| historical_documentation_or_tool | 9 |
| obsolete_inventory_boundary | 3 |
| phase_local_precommit_validator | 5 |
| published_compatibility_debt | 20 |

### Contagem por decisão provisória

| Campo | Valor |
|---|---|
| manual_review_required | 4 |
| preserve_historical_only | 43 |
| preserve_published_compatibility | 20 |
| supersede_test_contract_only | 3 |

A classificação automática é conservadora. Itens `manual_review_required` não autorizam alteração; exigem decisão nominal na AP-007F.1.

## Respostas obrigatórias

1. **`run_legacy` ainda é chamado no fluxo público real?** `True`. Evidências no inventário JSON, campos `run_legacy.active_runtime_calls` e `runtime_fallback.commands`.
2. **Quais argumentos chegam a ele?** Assinatura estática: `argv: Sequence[str] | None=None, *, program_name: str=OFFICIAL_PROGRAM_NAME`; chamadas e contextos no inventário.
3. **Quais comandos ainda dependem dele?** `[]`; branches ambíguos estão registrados separadamente.
4. **Quais comandos já são nativos, mas têm contratos esperando fallback?** Itens `superseded_route_expectation` no catálogo.
5. **`legacy.py` contém lógica ou apenas adaptador?** `adaptador/carregador, provisoriamente`.
6. **Execução direta é superfície pública?** `não foi encontrada publicação por metadata; os quatro casos são históricos/documentados até prova contrária`.
7. **É possível corrigir sem ponte e sem `PYTHONPATH`?** Em princípio sim, mas somente mediante decisão por wrapper e sem mutação nesta subfase.
8. **Mecanismo legítimo?** Bootstrap mínimo explícito somente se o wrapper for público, mudança de entrypoint ou aposentadoria formal.
9. **Quais dívidas superseder?** As classificadas com `supersede_test_contract_only`.
10. **Quais permanecer imutáveis?** As classificadas com `preserve_historical_only`.
11. **`run_legacy` pode ser removido integralmente?** `False`. run_legacy não pode ser removido integralmente nesta auditoria. Consumidores produtivos=[]; comandos literais em fallback=[].
12. **Menor residual defensável?** Somente os call sites produtivos, branches de fallback e compatibilidades publicadas listados no JSON.
13. **Escopo exato recomendado para AP-007F.1?** Candidatos da matriz com `active_runtime_debt` ou `direct_source_wrapper_debt`; excluir documentação histórica, snapshots, hashes, validadores fase-locais e contratos da ponte removida.
14. **Riscos e rollback?** Preservar códigos de saída, stdout/stderr, arquivos, estado do processo, CWD/sys.path, equivalência source/wheel; aplicar patches transacionais e restaurar somente caminhos manifestados.
15. **Gate da AP-007F.1?** Caracterização nominal de cada fallback, equivalência source/módulo/console/subprocesso, CWD neutro sem `PYTHONPATH`, wheel/sdist isolados, suíte produtiva e censo histórico sem falhas novas ou `xpass`.

## Recomendação para AP-007F.1

run_legacy não pode ser removido integralmente nesta auditoria. Consumidores produtivos=[]; comandos literais em fallback=[].

Nenhum aplicador é produzido nesta subfase. A AP-007F.1 somente deve ser preparada depois da análise integral do log desta execução e da revisão nominal dos itens `manual_review_required`.

## Gate

```text
[GATE] AP-007F.0: LEGADO RESIDUAL, CONSUMIDORES DE RUN_LEGACY, COMANDOS EM FALLBACK, WRAPPERS PUBLICADOS E 70 DÍVIDAS HISTÓRICAS/FASE-LOCAIS INVENTARIADOS E CLASSIFICADOS; ESCOPO DIRIGIDO PARA AP-007F.1 DEFINIDO, SEM ALTERAÇÃO PRODUTIVA, INSTALAÇÃO, STAGING, COMMIT, TAG OU PUSH.
```

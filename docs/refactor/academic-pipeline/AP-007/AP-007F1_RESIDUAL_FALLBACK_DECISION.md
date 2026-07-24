# AP-007F.1 — Decisão sobre o fallback residual

## Resultado

| Campo | Valor |
|---|---|
| Status | residual_fallback_preserved_no_productive_edit |
| HEAD | ba43b7d606378501d6faafa62ad8c8a6697665e5 |
| Runtime SHA-256 | b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c |
| Condições ancestrais registradas pela AP-007F.0 | 8 |
| Retornos reais `LEGACY_FALLBACK` | 6 |
| Injeções `legacy_runner=run_legacy` em `cli.main` | 1 |
| Chamadas indiretas a `legacy_runner` em `runtime.run` | 1 |
| Alteração produtiva | não |
| Instalação | não |
| Git de escrita | não |

A AP-007F.0 registrou oito condições ancestrais que continham retornos
descendentes. A inspeção AST desta subfase encontrou **seis retornos reais**
a `RuntimeRoute.LEGACY_FALLBACK`.

## Decisão arquitetural

`run_legacy` **não deve ser removido**. A ausência de call sites diretos era um
efeito da injeção de dependência: `academic_pipeline.cli:main` fornece
`run_legacy` como `legacy_runner`, e `academic_pipeline.runtime:run` o invoca
quando a rota selecionada é `LEGACY_FALLBACK`.

O adaptador `legacy.py` permanece necessário para comandos não migrados,
precedência histórica e combinações não exatas ou mistas.

## Retornos reais de fallback

| Linha | Retorno |
|---:|---|
| 332 | `return RuntimeRoute.LEGACY_FALLBACK` |
| 342 | `return RuntimeRoute.LEGACY_FALLBACK` |
| 351 | `return RuntimeRoute.LEGACY_FALLBACK` |
| 361 | `return RuntimeRoute.LEGACY_FALLBACK` |
| 379 | `return RuntimeRoute.LEGACY_FALLBACK` |
| 382 | `return RuntimeRoute.LEGACY_FALLBACK` |

Esses retornos são compatibilidade ativa mínima, não dívida removível.

## Matriz dinâmica de roteamento

| Caso | Argumentos | Rota |
|---|---|---|
| `help` | `--help` | `native_first_wave` |
| `institution_exact` | `--check-institution-compliance` | `native_institution_compliance` |
| `institution_with_config` | `--config x.toml --check-institution-compliance` | `native_institution_compliance` |
| `institution_mixed_doctor` | `--doctor --check-institution-compliance` | `legacy_fallback` |
| `institution_mixed_check` | `--check-config --check-institution-compliance` | `legacy_fallback` |
| `institution_mixed_profiles` | `--list-profiles --check-institution-compliance` | `legacy_fallback` |
| `doi_exact` | `--make-doi-manifest --input-dir in --output out.csv` | `native_doi_manifest` |
| `doi_mixed_doctor` | `--doctor --make-doi-manifest` | `legacy_fallback` |
| `doi_mixed_check` | `--check-config --make-doi-manifest` | `legacy_fallback` |
| `doi_mixed_profiles` | `--list-profiles --make-doi-manifest` | `legacy_fallback` |
| `doctor_exact` | `--doctor` | `native_doctor` |
| `doctor_preceding_guard` | `--check-institution-compliance --doctor` | `legacy_fallback` |
| `check_exact` | `--check-config` | `native_check_config` |
| `check_preceding_guard` | `--check-institution-compliance --check-config` | `legacy_fallback` |
| `profiles_exact` | `--list-profiles` | `native_list_profiles` |
| `profiles_unrelated_guard` | `--list-profiles --config=x.toml` | `legacy_fallback` |
| `empty_default` | `` | `legacy_fallback` |
| `unknown_default` | `--ap007f-unmigrated-command` | `legacy_fallback` |

A matriz também foi executada por `runtime.run` com doubles seguros. Em cada
rota nativa, exatamente um handler nativo foi chamado; em cada fallback,
somente o `legacy_runner` foi chamado.

## Execução direta do script histórico

- `app_bundle/tests/test_official_package_entrypoint.py::test_legacy_script_help_remains_supported` — superseder apenas o contrato de execução direta.
- `app_bundle/tests/test_official_package_entrypoint.py::test_official_and_legacy_help_expose_same_options` — superseder apenas o contrato de execução direta.
- `app_bundle/tests/test_official_package_entrypoint.py::test_official_and_legacy_list_institutions_match` — superseder apenas o contrato de execução direta.
- `app_bundle/tests/test_packaging_metadata.py::test_legacy_entrypoint_still_matches_console_target` — superseder apenas o contrato de execução direta.

Decisão: `supersede_test_contract_only`.

O arquivo histórico `academic_pipeline_rc10.py` permanece porque é carregado
internamente por `legacy.py`. Não serão recriados ponte física, `PYTHONPATH`,
`.pth` ou instalação ad hoc.

## Escopo resultante

```text
run_legacy preservado como adaptador residual deliberado
seis retornos reais de fallback preservados
execução direta histórica formalmente aposentada
quatro testes antigos supersedidos por contrato novo, sem edição retroativa
```

Nenhum módulo produtivo necessita alteração na AP-007F.1. A próxima subfase é
a AP-007F.2, destinada à validação final de fonte, módulo, console,
subprocessos, CWD neutro, wheel, sdist, recursos e censo histórico.

## Gate

```text
[GATE] AP-007F.1: RUN_LEGACY E FALLBACK RESIDUAL CONFIRMADOS COMO COMPATIBILIDADE ATIVA MÍNIMA; RETORNOS REAIS DE FALLBACK, PRECEDÊNCIA, DISPATCH E QUATRO CONTRATOS DIRECT-SOURCE FORMALMENTE CLASSIFICADOS, SEM ALTERAÇÃO PRODUTIVA, INSTALAÇÃO OU ESCRITA GIT.
```

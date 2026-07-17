# AP-005C.2 — Validação de estabilização

## Baseline

- Commit: `9372de8f621c9012a28d4c4a9a64e252a398bdf3`
- Branch: `ap-refactor/04-consumer-canonicalization`
- Fingerprint: `9cfc858992cdb30343d02d6526eb36ae6e8f2cc82fecf762f6849673022528f1`

## Resultado estrutural

- Capturas canônicas: 4/4
- Aliases históricos preservados: 4/4
- Consumidores internos canônicos: 6/6
- Consumidores legados restantes: 0
- Novos exports públicos: 0

## Diff produtivo

- Arquivos: 1
- Inserções: 14
- Remoções: 10
- Módulo: `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py`

## Gates de estabilização

- Testes legados relacionados: 106 passed
- Testes AP-005C: 24 passed
- Regressão focada: 53 passed
- Suíte canônica: 532 passed, 3 xfailed

## Manifesto candidato

Total de arquivos candidatos: 12

1. `docs/refactor/academic-pipeline/AP-005/AP-005C2_STABILIZATION_VALIDATION.md`
2. `docs/refactor/academic-pipeline/AP-005/AP-005C_TOML_CAPTURE_ALIAS_STRATEGY.md`
3. `docs/refactor/academic-pipeline/AP-005/ap005c2_stabilization_manifest.json`
4. `docs/refactor/academic-pipeline/AP-005/ap005c_toml_capture_alias_inventory.json`
5. `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py`
6. `software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c1_toml_capture_alias_application_contract.py`
7. `software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c2_stabilization_contract.py`
8. `software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c_toml_capture_alias_inventory_contract.py`
9. `software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c_toml_capture_alias_semantics_characterization.py`
10. `tools/refactor/ap005c1_apply_toml_capture_aliases.py`
11. `tools/refactor/ap005c2_validate_stabilization.py`
12. `tools/refactor/ap005c_inventory_toml_capture_aliases.py`

## Decisão

A AP-005C.1 está estabilizada. Não foram identificadas regressões, consumidores legados ou alterações colaterais fora do módulo previsto.

A consolidação permanece bloqueada até a AP-005C.3 realizar a auditoria final do manifesto e receber autorização explícita para commit e publicação.

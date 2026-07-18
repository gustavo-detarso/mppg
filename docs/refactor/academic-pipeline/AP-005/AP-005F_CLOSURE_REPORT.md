# AP-005F — Relatório de encerramento da AP-005

## 1. Decisão

A AP-005 está tecnicamente estabilizada e pronta para consolidação
explícita. A decisão registrada é
`ready_for_explicit_commit_and_publication_approval`.

A AP-005F não altera código produtivo. Seu escopo é congelar as
evidências de encerramento, registrar os contratos finais e preparar a
baseline da AP-006.

## 2. Baseline

- Branch: `ap-refactor/04-consumer-canonicalization`
- HEAD publicado: `e5e0d85178d8498c303ad2e8ccc9102f2c8222c8`
- Upstream: `origin/ap-refactor/04-consumer-canonicalization`
- Origin: `git@github.com:gustavo-detarso/mppg.git`
- Fingerprint do manifesto: `9dfc5c8de044b2bc22e57c02efa297a37659341268440f038500f4c7a2cd8c98`

## 3. Resultados consolidados

- Suíte canônica: **573 passed e 3 xfailed**.
- Recursos não Python no wheel: **38**.
- Módulos passivos instalados: **65**.
- Falhas de importação passiva: **0**.
- `pip check`: aprovado.
- Console `academic-pipeline`: aprovado.
- Módulo `python -P -m academic_pipeline`: aprovado.
- Perfil institucional FGV: aprovado.
- Avisos residuais classificados de build: **0**.
- Documentos AP-005 anteriores ao encerramento: **21**.
- Contratos AP-005 anteriores ao encerramento: **14**.

## 4. Trajetória registrada

- `6ef568b250390e12dc2e86b86a8c530188604a28` — refactor(academic-pipeline): inventariar consumidores da AP-005A
- `9372de8f621c9012a28d4c4a9a64e252a398bdf3` — refactor(ap005): canonicalize PRISMA consumers
- `b8cb7ba3a3175ac79799b78a5d0678224076ef80` — refactor(ap005): canonicalize TOML capture aliases
- `78f3be0fce0dd8f79e55729a7111a9359c9edb8d` — fix(ap005): support post-commit validation of AP-005C
- `ba28822c826c37022581bf88c6a1b488e2c618de` — docs(ap005): formalize AP-005D facade preservation
- `162df76eea94b3a5889ca217a907690f4d62c649` — fix(academic-pipeline): congelar universo histórico da AP-005D
- `0d553c975ad7948762f74aa4fcff3903578712de` — chore(academic-pipeline): materializar inventário da AP-005E.1
- `b16d1389486f220f829235e87adf88a191cefa87` — test(academic-pipeline): caracterizar instalação isolada AP-005E.2
- `71b0c490463edfeb24d6c733ce0a6c698b970510` — fix(academic-pipeline): corrigir instalação distribuída AP-005E.3
- `e5e0d85178d8498c303ad2e8ccc9102f2c8222c8` — chore(academic-pipeline): estabilizar metadados de distribuição AP-005E.4

## 5. Síntese por subfase

- **AP-005A:** inventário de consumidores e estratégia de migração.
- **AP-005B:** canonicalização dos consumidores PRISMA.
- **AP-005C:** migração dos aliases de captura TOML e estabilização.
- **AP-005D:** consolidação das fachadas e preservação explícita da API.
- **AP-005E:** metadados, build, instalação isolada, recursos e entrypoints.
- **AP-005F:** auditoria integrada e encerramento documental/contratual.

## 6. Defeitos legados mantidos como xfail

- `app_bundle/tests/test_article_workflow_characterization.py::test_refresh_from_files_should_keep_downstream_stages_blocked_after_first_failure` — legacy_defect_catalogued
- `app_bundle/tests/test_canonical_docx_characterization.py::test_extract_resumos_should_separate_inline_keywords_from_heading_abstract` — legacy_defect_catalogued
- `app_bundle/tests/test_rc10_configuration_characterization.py::test_reference_strip_should_remove_parenthetical_citations` — legacy_defect_catalogued

Esses três defeitos permanecem fora do escopo da AP-005 e não impedem
o encerramento porque estão catalogados e preservados como `xfail`.

## 7. Evidência distributiva

- Wheel SHA-256 observado:
  `2d97c03fa36475d3813497219867d51d58adbd89023d3fafbe82d1623af283c1`
- sdist SHA-256 observado:
  `85035062e279334c748cd38c85df4af887de3b2b4e25d594a935721c2052d1db`
- `Description-Content-Type`: `text/markdown`
- Entrypoint:
  `academic-pipeline = academic_pipeline.cli:main`

Os hashes registram a execução de auditoria de encerramento de
18/07/2026. Não constituem requisito de reprodutibilidade byte a byte,
pois os formatos de distribuição podem incorporar metadados temporais.

## 8. Artefatos de encerramento

- `docs/refactor/academic-pipeline/AP-005/AP-005F_CLOSURE_REPORT.md`
- `docs/refactor/academic-pipeline/AP-005/ap005f_closure_manifest.json`
- `software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005f_closure_contract.py`
- `tools/refactor/ap005f_validate_closure.py`

## 9. Limites e transição

A AP-005 não renomeia fisicamente
`software/academic_pipeline_rc10_7_conformidade`. Essa eventual mudança
permanece reservada para a AP-006.

A baseline para a próxima fase é o estado publicado em
`e5e0d85178d8498c303ad2e8ccc9102f2c8222c8`, acrescido somente dos quatro artefatos
de encerramento desta AP-005F após aprovação explícita de commit e
publicação.

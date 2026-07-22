# AP-006F.3 — Materialização mínima

- Baseline: `4db60736cfb4d2be53af32babdcdbfed84c3e6b4`
- Ponte `software/academic_pipeline_rc10_7_conformidade`: removida.
- Raiz canônica `software/academic_pipeline_mppg`: preservada.
- Fallback `academic_pipeline.legacy:run_legacy`: preservado.
- Evidência: árvore-fonte e wheel sem ponte aprovados na AP-006F.2; a retirada de `run_legacy` interrompeu o console e `python -m`.
- Gate AP-006F.4: `PASS`

## Integridade

- Master preservado: `True`
- 22 caminhos paralelos preservados: `True`
- Inventário `.pth` preservado: `True`
- Staging vazio: `True`

A materialização não cria commit, tag ou publicação.

## Reparo de compatibilidade contratual

A matriz AP-006F.1 permanece histórica. Seu validador agora distingue o pré-commit F1, o pré-commit F3 e commits descendentes, sem aceitar diffs ou artefatos estranhos ao contrato.

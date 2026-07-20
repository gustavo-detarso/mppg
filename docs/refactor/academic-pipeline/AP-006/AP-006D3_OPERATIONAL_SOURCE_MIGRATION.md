# AP-006D.3 — Migração dos fontes operacionais

## Baseline

- Commit: `0367a1988ee494200e912538e638703cb73dc74e`
- Tree OID: `1ba11ffaba1d2bae00fd4327e0db395c82e004b9`
- Worktree externo: `/home/gustavodetarso/Documentos/mppg`
- Branch externa: `master`
- HEAD externo auditado: `ef9acfd739274139637fae934c0c5dca4416728e`

## Materialização

Foram alterados somente os fontes versionados nesta branch:

- `software/academic_pipeline_mppg/app_bundle/docs/README_rc10.md`: 3 linhas;
- `software/academic_pipeline_mppg/app_bundle/docs/SETUP_PIPENV.md`: 1 linha operacional;
- `disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo/artigo_final_atestmed_abnt.toml`: 7 linhas.

O fonte `software/academic_pipeline_mppg/atualizar_academic_pipeline_bundle.py` já estava canônico e foi preservado.

## Preservações

A referência não operacional de `SETUP_PIPENV.md` continua legada.
A ponte de compatibilidade permanece até a AP-006F.

## Regeneração externa

Os três consumidores externos não foram alterados. A cópia dos fontes e o
commit separado no worktree `master` ficam adiados para a AP-006D.4, conforme
o bloco `external_regeneration` do contrato JSON.

## Escopo

- 3 fontes modificados;
- 4 artefatos adicionados;
- 11 linhas migradas;
- nenhum staging, commit ou push.

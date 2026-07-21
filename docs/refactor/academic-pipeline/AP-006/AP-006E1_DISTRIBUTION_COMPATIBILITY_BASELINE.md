# AP-006E.1 — Baseline de distribuição e compatibilidade

## Estado

- Commit-fonte: `aed79d72f6c26fabcdda00f25d058b32fdc3fd75`
- Tree-fonte: `f4004337607e21e1ba89330928b4473ffd739dcd`
- Fingerprint: `36be68ba900b7afe83f952290b5978816f62f3ef7bbdbce6c44faeea27068c9f`
- Natureza: contrato declarativo e teste de caracterização.
- Código produtivo alterado nesta subfase: **zero**.

## Contrato público preservado

- Distribuição: `academic-pipeline-mppg` versão `0.1.0`.
- Python: `>=3.11`.
- Console: `academic-pipeline = academic_pipeline.cli:main`.
- Módulos públicos: `academic_pipeline` e `app_bundle`.
- `python -m academic_pipeline` converge para o mesmo `main`.
- O fallback `academic_pipeline.legacy:run_legacy` permanece preservado.

## Ponte de compatibilidade

- `software/academic_pipeline_rc10_7_conformidade -> academic_pipeline_mppg`.
- Retenção obrigatória durante toda a AP-006E.
- Remoção, substituição ou retenção definitiva pertence à AP-006F.

## Partição das referências físicas

- Linhas classificadas: **39415**.
- Ocorrências dos dois nomes: **40724**.
- Registros ativos para revisão: **166**.
- Caminhos candidatos com nome legado: **50**.

- `canonical_runtime_or_config`: 30
- `canonical_test_or_validator`: 121
- `explicit_backup_archive_component`: 156
- `external_document_source`: 8
- `external_operational_source`: 7
- `historical_evidence`: 37877
- `other_tracked_reference`: 1
- `pathological_recursive_backup_evidence`: 273
- `scan_deferred_component`: 942

As classes são disjuntas, cobrem integralmente as linhas encontradas e
separam backups explícitos, caminhos adiados, evidência patológica e
documentação histórica de consumidores operacionais.

## Decisão ambiental

A ausência da distribuição no virtualenv persistente e a falha de importação
em diretório neutro sem instalação são observações ambientais. Elas não
contradizem o contrato da árvore-fonte. A validação distributiva deverá ser
feita em ambiente temporário isolado, sem instalação persistente e sem `.pth`
residual.

## Próximo gate

Executar build, instalação, console, `python -m`, subprocessos e imports em
clone/venv temporários. Somente evidência dessa validação poderá justificar
alterações produtivas na AP-006E.2 ou AP-006E.3.

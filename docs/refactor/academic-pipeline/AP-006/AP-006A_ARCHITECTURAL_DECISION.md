# AP-006A — Decisão arquitetural de escopo

## Status

**Aceita para planejamento.** Esta decisão não autoriza alteração produtiva,
staging, commit ou publicação.

## Decisão

Adotar a estratégia `encapsulate_then_migrate_with_compatibility`.

A renomeação direta de
`software/academic_pipeline_rc10_7_conformidade` fica rejeitada sem ponte de
compatibilidade.

A AP-006 fica dividida em:

1. AP-006A — auditoria e decisão de escopo;
2. AP-006B — arquitetura-alvo e contrato de compatibilidade;
3. AP-006C — materialização física controlada;
4. AP-006D — migração de consumidores e integrações;
5. AP-006E — estabilização distributiva e compatibilidade;
6. AP-006F — validação integrada e encerramento.

## Restrições para a AP-006B

A AP-006B deverá escolher o destino físico, excluir
`software/academic_pipeline`, preservar a distribuição
`academic-pipeline-mppg`, o entrypoint `academic-pipeline`, os imports
`academic_pipeline` e a compatibilidade necessária de `app_bundle`.

Também deverá definir um resolvedor canônico de recursos, a ponte temporária, a
matriz de consumidores por onda e os critérios de retirada da compatibilidade.

Documentos históricos não serão reescritos. Snapshots e saídas geradas serão
preservados ou regenerados. A divisão física imediata entre pacote e
`app_bundle` permanece fora do escopo até nova evidência.

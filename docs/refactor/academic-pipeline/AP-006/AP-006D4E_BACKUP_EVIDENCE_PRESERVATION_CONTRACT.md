# AP-006D.4E — Contrato de preservação de backups e evidências históricas

## Decisão

Os 177 arquivos rastreados classificados como backups, snapshots, árvores legadas ou evidências históricas devem permanecer nos caminhos atuais e com os mesmos blobs Git. A ausência de referências produtivas não autoriza exclusão, movimentação, renomeação ou alteração de conteúdo.

## Evidência

- Candidatos preservados: **177**
- Referências produtivas: **0**
- Auditoria executada exclusivamente por objetos Git
- Caminhos recursivos patológicos não foram percorridos pelo filesystem
- Fingerprint contratual: `0932843e67044970aec49962b2dc52332a38fcf6d57ae933fff40f7feeec3bbd`

## Classificações

- `duplicate_historical_snapshot`: **14**
- `editor_or_tool_backup_evidence`: **12**
- `explicit_backup_tree_evidence`: **39**
- `legacy_project_evidence`: **5**
- `named_historical_evidence`: **1**
- `patch_backup_historical_snapshot`: **9**
- `pathological_recursive_backup_evidence`: **90**
- `unique_historical_evidence`: **7**

## Política

É proibido remover, mover, renomear ou modificar os 177 candidatos sem uma fase futura explicitamente autorizada, com inventário por objetos Git, prova de ausência de dependência e plano de preservação substitutivo. A ponte de compatibilidade permanece preservada.

O manifesto verificável está em `docs/refactor/academic-pipeline/AP-006/ap006d4e_backup_evidence_preservation_contract.json`.

# AP-006D.1 — Migração dos consumidores residuais de runtime

## Objetivo

Migrar referências operacionais residuais do nome físico legado
`academic_pipeline_rc10_7_conformidade` para a raiz canônica
`academic_pipeline_mppg`, preservando a ponte de compatibilidade definida na
AP-006B e materializada na AP-006C.

## Baseline

- Commit: `7eddf9cf16f57b6433a325080fc992beab7a3184`
- Tree OID: `394bcdd413f9c62de99eecb769ebd47651559e4e`
- Ponte temporária:
  `software/academic_pipeline_rc10_7_conformidade -> academic_pipeline_mppg`

## Decisão semântica

A auditoria geral encontrou 12 referências em sete arquivos inicialmente
classificados como runtime residual. A revisão semântica determinou:

- 10 referências operacionais devem ser migradas;
- seis arquivos recebem alterações;
- duas referências em
  `app_bundle/clean_institutional_tree_report.json` são evidência histórica e
  permanecem inalteradas.

O relatório preservado registra o caminho vigente e uma ação executada no
momento em que foi produzido. Reescrever esses valores alteraria a proveniência
do artefato e não reduziria dependência operacional da ponte.

## Arquivos migrados

1. `aplicar_docx_canonico_v11.py`
2. `aplicar_docx_canonico_v12.py`
3. `aplicar_docx_canonico_v13.py`
4. `aplicar_docx_capa_disciplina_v14.py`
5. `app_bundle/.academic_pipeline_tui_state.json`
6. `atualizar_academic_pipeline_bundle.py`

## Restrições

- A ponte não é removida na AP-006D.1.
- Nenhuma evidência histórica é reescrita.
- Não há substituição em massa.
- Consumidores externos permanecem para ondas posteriores da AP-006D.
- A remoção da ponte somente poderá ser decidida na AP-006F.

# AP-006A — Auditoria de impacto da organização física

## Baseline e proveniência

- Commit: `a9d0fa1e100af966329d48629ec234e32da6ded7`
- Branch: `ap-refactor/04-consumer-canonicalization`
- Projeto: `software/academic_pipeline_rc10_7_conformidade`
- Log semântico revisado: `ap006a_classificacao_semantica_dependencias_20260718_111944.log`
- Fingerprint semântico: `fe2e4262338815154f42b8eaca7d33c8d3d87bb9e1302c9bcc08fc2b5326a179`

## Resultado consolidado

| Indicador | Valor |
|---|---:|
| Arquivos rastreados | 1283 |
| Linhas únicas com referência | 39141 |
| Arquivos com referência exata | 289 |
| Arquivos com acoplamento material | 153 |
| Consumidores externos ativos ou reutilizáveis | 38 |
| Arquivos produtivos internos | 96 |
| Testes e validadores contratuais | 19 |
| Arquivos da superfície distributiva | 91 |
| Camadas arquiteturais confirmadas | 6 |

A maior parte das 39.141 linhas pertence a evidência histórica congelada. A
classificação semântica separou histórico, snapshots, saídas regeneráveis,
consumidores ativos, fontes reutilizáveis, código produtivo e contratos.

## Conclusão

Foram confirmadas seis camadas: estrutura produtiva, consumidores externos,
testes e validadores, distribuição e entrypoints, fontes e artefatos gerados, e
evidência histórica. A AP-006 deve possuir **seis subfases**.

O destino `software/academic_pipeline` está ocupado por 63 entradas rastreadas e
fica excluído de renomeação direta. `software/academic_pipeline_mppg` e
`software/academic-pipeline` permanecem como candidatos para avaliação na AP-006B.

A renomeação direta do diretório atual é rejeitada. A estratégia segura é
encapsular a resolução de caminhos, preservar compatibilidade temporária e migrar
consumidores por ondas.

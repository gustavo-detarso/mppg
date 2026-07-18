# AP-006B — Arquitetura-alvo

## Status

**Decisão aceita para materialização na AP-006C.** Esta fase não executa a
mudança física.

## Baseline

- Commit: `9745f3727403c7ed7637faa12141cd16310f18c1`
- Caminho atual: `software/academic_pipeline_rc10_7_conformidade`
- Fingerprint da auditoria: `248523bda24d985e1b1d7a0aa7de3c4cafade6bd640860c93da79d79b08ffbc7`

## Destino selecionado

O destino físico canônico será:

`software/academic_pipeline_mppg`

A escolha obteve 84 pontos, não possui colisões, elimina os marcadores de versão
e conformidade e corresponde à forma normalizada da distribuição
`academic-pipeline-mppg`.

`software/academic-pipeline` permanece como alternativa rejeitada, com 67
pontos. `software/academic_pipeline` permanece excluído por colisão com 63
entradas rastreadas.

## Topologia transitória

A AP-006C deverá:

1. mover a árvore rastreada para o destino canônico;
2. criar um symlink relativo rastreado do caminho antigo para o novo;
3. introduzir um resolvedor canônico para recursos do repositório;
4. preservar distribuição, console e imports públicos;
5. proibir uma árvore duplicada como segunda fonte de verdade.

O symlink é uma ponte transitória. Sua retirada somente poderá ser decidida na
AP-006F.

## Limites da decisão

A AP-006B não autoriza rename, symlink físico, alteração de imports ou migração
de consumidores. Essas operações pertencem às AP-006C e AP-006D.

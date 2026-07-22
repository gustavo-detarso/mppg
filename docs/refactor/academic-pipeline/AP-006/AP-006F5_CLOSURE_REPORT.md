# AP-006F.5 — Encerramento formal da AP-006F

## Situação final

A AP-006F está encerrada tecnicamente em estado pré-commit. A ponte histórica `software/academic_pipeline_rc10_7_conformidade` foi removida, a raiz física canônica `software/academic_pipeline_mppg` permanece como única implementação e o adaptador ativo `academic_pipeline.legacy:run_legacy` foi preservado.

## Resultados consolidados

A AP-006F.1 formalizou a matriz de decisão; a AP-006F.2 comprovou em ensaio descartável que a ponte poderia ser removida e que `run_legacy` ainda era necessário; a AP-006F.3 materializou a remoção da ponte; e a AP-006F.4 comprovou paridade entre fonte e wheel após reparar o dispatch público.

O reparo de `academic_pipeline/command_dispatch.py` possui SHA-256 `9255c4b924fd61b7120b8c5e02684d338f6788de42ae7c352b049a488a308afe` e restringe-se a oito alvos de importação distributiva e quatro chamadas diretas das funções importadas.

## Validações

- suíte focada: 95 aprovados e 1 `xfailed`;
- testes atuais não históricos: 603 aprovados e 3 `xfailed`;
- contratos históricos no `HEAD` limpo com Python canônico: 33 aprovados;
- consolidação lógica: 636 aprovados e 3 `xfailed`;
- contratos formais F1, F3, F4 e F5: 8 aprovados;
- comparação funcional fonte/wheel: aprovada;
- wheel: 110 membros, com diferenças autorizadas limitadas ao dispatch e metadados derivados.

## Escopo candidato ao commit

O conjunto candidato contém 18 caminhos: a exclusão da ponte, a modificação do dispatch e 16 artefatos formais das fases F1, F3, F4 e F5. O staging permanece vazio.

## Integridade

O master operacional preserva exatamente 22 caminhos paralelos, com snapshot `f12b966a7c3e33ea3d1274219529cdb4db58454769fe8448bf1b6c820f318449`. Não houve alteração persistente de `.pth`, staging, commit, amend, tag ou push.

## Evidência de fechamento

A materialização formal da AP-006F.4 está registrada em `ap006f4_materializacao_validacao_comparativa_formal_v1_20260721_191932.log`, SHA-256 `828ddc8e1d4fe7f7a776a6228b5a3da4fd4697a06a1e85470785632ff18a42e1`.

## Decisão

A AP-006F está pronta para um commit isolado e publicação, mas essas operações continuam condicionadas à autorização explícita. Este encerramento não concede autorização automática para staging, commit ou push.

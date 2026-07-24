# AP-007D.4 — Adaptador nativo isolado de `--check-institution-compliance`

## Decisão

A caracterização da segunda onda selecionou o comando com risco baixo,
pontuação 31/33, um handler estrutural e quatro funções transitivas, sem
efeitos bloqueadores.

Esta etapa materializa somente o adaptador. A rota pública continua em
`legacy_fallback` até uma integração posterior e explicitamente validada.

## Estratégia

O adaptador reutiliza o parser e o estágio canônicos. Ele carrega a
configuração por dependências explícitas, aplica os overrides da CLI e chama
`dispatch_stage_015` com um contexto tipado, imutável e mínimo.

O contrato histórico preservado inclui:

- `--config` obrigatório;
- resolução de ORG, BIB, DOCX e PDF;
- geração dos relatórios de conformidade;
- impressão do relatório Markdown e dos caminhos gerados;
- retorno `0` quando `report.ok` é verdadeiro e `2` quando é falso.

## Restrições

- sem `globals()` ou `locals()`;
- sem importação do monólito histórico;
- sem mutação de `sys.argv`, `sys.path` ou cwd;
- sem alteração de runtime, parser, CLI ou dispatcher;
- sem staging, commit, tag ou push.

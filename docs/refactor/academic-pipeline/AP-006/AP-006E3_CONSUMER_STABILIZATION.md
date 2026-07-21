# AP-006E.3 — Estabilização dos consumidores operacionais

## Decisão retificada

A auditoria contextual encontrou cinco scripts executáveis com referências ao
nome físico anterior. Todos são ferramentas históricas das fases AP-004A–C,
não módulos do wheel, entrypoints ou consumidores ativos. Elas permanecem
byte a byte inalteradas para preservar a reprodutibilidade.

A primeira materialização tentou atualizar `SETUP_PIPENV.md`. A validação
integrada demonstrou que essa referência já estava classificada e congelada
pela AP-006D.3 como documentação não operacional de compatibilidade. A
alteração foi revertida integralmente.

## Escopo final

A AP-006E.3 não modifica código produtivo nem arquivo rastreado existente.
Seu resultado material consiste apenas neste relatório, no contrato JSON, no
validador e no teste de caracterização.

Os validadores históricos sensíveis ao estado da worktree são executados em
clone limpo do commit canônico. Os contratos AP-006E são executados na
worktree candidata, que contém os novos artefatos ainda não rastreados.

A decisão definitiva sobre ponte e fallbacks permanece reservada à AP-006F.

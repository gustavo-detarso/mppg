# AP-007D.5 — Caracterização controlada de `--make-doi-manifest`

## Resultado

A lógica DOI mínima foi extraída do módulo canônico por fechamento AST no mesmo
arquivo e executada com sucesso para entradas por diretório e ZIP. A rota
pública permanece no `legacy_fallback`, mas não é executável pelo Python
canônico porque o bootstrap legado falha antes de alcançar o handler DOI. A
primeira dependência ausente observada de forma consistente foi
`dotenv`.

## Prova dinâmica

- lógica DOI mínima: aprovada para diretório e ZIP;
- equivalência entre as duas formas de entrada: aprovada por CSV normalizado;
- timeout explícito: 30 segundos por execução;
- rede: bloqueada por `sitecustomize`;
- fontes, worktree e runtime: preservados;
- arquivos inesperados: nenhum;
- rota pública legada: falha controlada e reproduzível por acoplamento do
  bootstrap legado a uma dependência ausente não utilizada pela closure DOI.

## Decisão

O comando foi selecionado para um adaptador nativo isolado de desacoplamento.
O adaptador deverá reproduzir a closure DOI verificada sem depender do
bootstrap legado nem de `dotenv`, preservar sandbox, snapshots,
hashes, timeout e rollback. A rota pública não muda nesta subfase.

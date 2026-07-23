# AP-007C.3 — Integração pública do `--doctor`

`--doctor` passa a usar o adaptador nativo da AP-007C.2. `--check-config`
permanece no fallback legado.

A seleção preserva a ordem histórica: a primeira onda continua prioritária e,
quando um destino que aciona estágio anterior a `dispatch_stage_016` está
ativo, a combinação permanece no fallback. `--doctor --check-config` continua
executando doctor primeiro, como no encadeamento histórico.

Os códigos diagnósticos `0` e `2`, a escrita do relatório JSON e o estado do
processo são preservados. Nenhum staging, commit, tag ou push está autorizado.

# AP-007C.2 — Adaptador nativo isolado de `--doctor`

## Decisão

A AP-007C.1 confirmou `dispatch_stage_016` como a próxima superfície
de baixo risco e determinou que `--doctor` e `--check-config` devem ser
migrados em ondas separadas.

Esta subfase materializa somente o adaptador de `--doctor`. O runtime e
o entrypoint públicos permanecem byte a byte inalterados. A integração
pública pertence à AP-007C.3.

## Contrato

- sem configuração, executa o diagnóstico e não grava relatório;
- com configuração, aplica os overrides de caminhos da CLI;
- seleciona saída documental ou de pesquisa conforme o perfil;
- grava `<prefix>.doctor_report.json`;
- retorna `0` quando `report.ok` é verdadeiro e `2` quando é falso;
- usa dependências explícitas, sem `globals()`, `locals()` ou ponte
  para o monólito histórico.

## Limites

`--check-config` permanece no fallback legado e constitui a onda
seguinte. Não há autorização de staging, commit, tag ou push.

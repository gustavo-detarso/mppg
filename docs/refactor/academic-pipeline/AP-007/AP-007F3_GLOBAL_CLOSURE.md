# AP-007F.3 — Encerramento global da AP-007

## Decisão

A AP-007 está **formalmente encerrada** e pronta para a decisão de commit isolado.
Nenhuma operação de staging, commit, tag ou push integra esta subfase.

## Resultado consolidado

- Baseline preservado: `ba43b7d606378501d6faafa62ad8c8a6697665e5`.
- Runtime preservado: `b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c`.
- Casos de superfície: **6**.
- Superfícies: **5**.
- Execuções: **30**.
- Comparações: **24**.
- Divergências: **0**.
- Suíte produtiva: **759 passed**, **72 deselected**, **3 xfailed**.
- Falhas, erros e xpasses novos: **0**.
- Dívidas históricas: **70**.
- Contratos fase-locais: **2**.
- Contratos direct-source supersedidos: **4**.
- Retornos residuais `LEGACY_FALLBACK` preservados: **6**.
- Evidências AP-007F.0/F.1/F.2 preservadas: **12**.
- Evidências AP-007F.3 materializadas: **4**.
- Escopo final para decisão de commit: **16 caminhos**.

## Compatibilidade residual

`run_legacy`, o adaptador `legacy.py` e os seis retornos reais de fallback
permanecem como compatibilidade ativa mínima. Os quatro testes históricos de
execução direta foram classificados como contratos de teste supersedidos, sem
recriação de ponte, `PYTHONPATH` ou arquivo `.pth`.

## Distribuição e isolamento

Fonte, wheel direto, console instalado, wheel derivado de sdist e seu console
foram equivalentes nas 24 comparações. Módulos e recursos críticos mantiveram
paridade, e o ambiente canônico não foi alterado.

## Estado Git

O worktree contém somente as 16 evidências autorizadas e permanece com staging
vazio. A master protegida e seus 22 caminhos paralelos permanecem intactos.

## Próxima decisão

O commit e a publicação devem ocorrer somente após autorização explícita. A
mensagem proposta é:

```text
refactor(academic-pipeline): close AP-007 compatibility boundary
```

## Gate

```text
[GATE] AP-007F.3: AP-007 ENCERRADA FORMALMENTE E PRONTA PARA DECISÃO DE COMMIT ISOLADO; ESCOPO EXATO DE 16 EVIDÊNCIAS, BASELINE E MASTER PRESERVADOS, SEM ALTERAÇÃO PRODUTIVA, INSTALAÇÃO OU ESCRITA GIT. COMMIT, TAG E PUSH CONTINUAM BLOQUEADOS ATÉ AUTORIZAÇÃO EXPLÍCITA.
```

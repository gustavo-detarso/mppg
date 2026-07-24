# AP-007F.2 — Validação final integrada

## Resultado

- Status: **final_integrated_validation_complete**.
- HEAD: `ba43b7d606378501d6faafa62ad8c8a6697665e5`.
- Runtime SHA-256: `b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c`.
- Retornos reais de fallback preservados: **6**.
- Casos de roteamento/dispatch AP-007F.1: **18**.
- Casos de superfície: **6**.
- Superfícies: **5**.
- Execuções: **30**.
- Comparações: **24**.
- Divergências: **0**.
- Dívidas históricas deselecionadas nominalmente: **70**.
- Contratos fase-locais deselecionados nominalmente: **2**.
- Total de argumentos `--deselect`: **72**.
- Testes direct-source reproduzidos: **4**.
- Suíte produtiva atual: **759 passed**, **3 xfailed**, **0 failed**, **0 errors**, **0 xpassed**.
- Alteração produtiva: **não**.
- Staging, commit, tag ou push: **não**.
- Manifesto JSON: **pré-validado, serializável e relido por roundtrip**.

## Distribuição

Foi criado um snapshot candidato por `git archive`, com exclusão dirigida apenas
de `software/academic_pipeline_mppg/backups/`. Essa árvore contém cópias históricas recursivas e não integra
a distribuição executável. Foram omitidos **130**
caminhos rastreados, registrados por hash NUL SHA-256
`ebb5b6154fe96b724bbdc593148a8f4be5f3b50e17fc642b26c9ca9bac3b86e0`.

Os contratos AP-007F.0, F.1 e F.2 foram sobrepostos explicitamente. O backend
PEP 517 `setuptools.build_meta` produziu um sdist, um wheel direto e um wheel derivado
do sdist.

Os dois wheels foram instalados em ambientes virtuais descartáveis com
`--no-index --no-deps --no-cache-dir --no-compile`. Nenhuma dependência foi
instalada e o ambiente canônico permaneceu inalterado.

## Equivalência

As superfícies validadas foram:

1. fonte por `python -m academic_pipeline`;
2. wheel por `python -m academic_pipeline`;
3. wheel pelo console `academic-pipeline`;
4. wheel derivado de sdist por `python -m academic_pipeline`;
5. wheel derivado de sdist pelo console `academic-pipeline`.

Os seis casos reproduziram ajuda, instituições, perfis, ausência de configuração
para `check-config`, ausência de configuração para conformidade institucional e
ausência de origem para manifesto DOI. Todas as 24 comparações foram equivalentes.

## Legado residual

`run_legacy`, `legacy.py` e os seis retornos `LEGACY_FALLBACK` permanecem como
compatibilidade ativa mínima. Os quatro contratos de execução direta foram
reproduzidos com a assinatura exata de ponte ausente e permanecem formalmente
supersedidos, sem recriação de ponte, `PYTHONPATH`, `.pth` ou instalação ad hoc.

## Regressão

O catálogo publicado de 70 dívidas foi aplicado por node ID exato. Dois
contratos fase-locais foram deselecionados separadamente:

1. o validador AP-007E.0, congelado para o HEAD, a árvore e o escopo daquela fase;
2. o escopo AP-007F.0 de exatamente quatro caminhos, substituído pelos contratos
   acumulados F.1 e F.2.

A suíte produtiva corrente passou sem falha, erro ou xpass novo. Nenhum teste
histórico foi editado, removido ou reclassificado como dívida produtiva.

## Próxima subfase

A AP-007F.3 deve materializar o encerramento formal global da AP-007 e preparar
o escopo exato para decisão de commit isolado, exigindo autorização explícita.

## Gate

```text
[GATE] AP-007F.2: VALIDAÇÃO FINAL INTEGRADA APROVADA; FONTE, WHEEL E WHEEL DERIVADO DE SDIST EQUIVALENTES EM 30 EXECUÇÕES E 24 COMPARAÇÕES, 70 DÍVIDAS E 2 CONTRATOS FASE-LOCAIS DESELECIONADOS NOMINALMENTE, QUATRO CONTRATOS DIRECT-SOURCE REPRODUZIDOS, SUÍTE PRODUTIVA ATUAL APROVADA E ESCOPO EXATO DE 12 CAMINHOS PRESERVADO, SEM ALTERAÇÃO PRODUTIVA OU ESCRITA GIT.
```

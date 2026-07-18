# AP-005D — Estratégia de facades e reexports

## Baseline

A classificação foi realizada sobre a baseline:

    78f3be0fce0dd8f79e55729a7111a9359c9edb8d

O inventário canônico possui fingerprint:

    93727ee23a696b46f3fa8b0e08f8316903b958d574c4ad446b15936ddbede75f

A evidência externa da AP-005D.1 foi preservada pelos identificadores:

    SHA-256: 45d0f51577e6d8cdf4f840b40546dd86878812e93c980f372d805efd0c301788
    fingerprint: 9a5ee044b249cd600e4af8c7d934073b979c899d6dd2ae6dbcaba7f86438b7ab

## Fronteira temporal do universo

O universo histórico de 145 arquivos Python é obtido da árvore Git do commit baseline:

    78f3be0fce0dd8f79e55729a7111a9359c9edb8d

O inventariador não usa o índice do `HEAD` descendente para definir novos caminhos.
Assim, testes e ferramentas adicionados pela AP-005E ou por fases posteriores não
alteram retrospectivamente o corpus da AP-005D. Os conteúdos dos caminhos que já
existiam na baseline continuam sendo lidos no checkout atual, preservando a
detecção de alterações ou remoções posteriores nas superfícies históricas.

## Decisão

A AP-005D não requer alteração produtiva.

A única facade verdadeira é:

    app_bundle.scripts.pipeline.article_workflow

Ela agrega e publica deliberadamente `STAGES`, `StageRecord`,
`WorkflowState`, `ArticleWorkflow` e `StageValidation`. Essa superfície
deve permanecer inalterada.

A raiz pública `academic_pipeline` também deve ser preservada. Seu
`__all__` contém somente `main`. `Sequence` e `annotations` são falsos
positivos do inventário inicial e não representam reexports públicos.

Os módulos abaixo possuem `__all__` declarativo, mas implementam
localmente seus símbolos e, portanto, não são facades:

    academic_pipeline.cli_parser
    academic_pipeline.command_dispatch
    academic_pipeline.document_orchestration
    academic_pipeline.prisma_generic_orchestration

Seus consumidores internos permanecem válidos. Nenhuma importação deve
ser redirecionada nesta fase.

## Limites

A AP-005D não deve remover reexports, alterar entrypoints, modificar
metadados de empacotamento, renomear pacotes nem migrar caminhos
`academic_pipeline.*`.

Qualquer reorganização ampla desses caminhos pertence a uma fase
posterior e exige contratos próprios.

As contagens brutas de arestas de importação não são gates da decisão,
pois variam conforme o tratamento de aliases, imports relativos,
chamadas dinâmicas e arquivos residuais. Os gates contratuais são as
seis superfícies AST, seus `__all__`, a classificação e a decisão de
preservação.

## Artefatos

    tools/refactor/ap005d_inventory_facades.py
    docs/refactor/academic-pipeline/AP-005/ap005d_facade_inventory.json
    software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005d_facade_inventory_contract.py

# AP-004C — estratégia de símbolos internos

> Documento preparatório. Nenhum símbolo produtivo foi renomeado.

## Objetivo técnico

Normalizar identificadores internos herdados das fases AP-003 e de marcadores de
versão sem alterar comportamento, estrutura do orquestrador, contratos públicos,
conteúdo gerado ou os três defeitos legados congelados.

## Base canônica

- Branch: `ap-refactor/03-orchestrator-decomposition`.
- HEAD local/remoto: `aa9829f09a5c1b9e69c634637c311b03f360b07e`.
- Fechamento AP-004B: `refactor(academic-pipeline): consolidar módulos e arquivos da AP-004B`.
- Baseline: `448 passed, 3 xfailed`.

## Ondas propostas

### Onda 1 — símbolos privados locais

Abrange somente `ready_local_ast_rename`. A transformação futura deverá usar AST,
validar colisões, preservar assinaturas e provar que nenhum consumidor externo ou
dinâmico foi alterado. Candidatos atuais: **7**.

### Onda 2 — símbolos vinculados a contratos

Abrange `ready_contract_bound_ast_rename` e `contract_update_required`. Inclui
aliases privados do orquestrador e outros símbolos cuja renomeação exige atualizar
contratos caracterizadores e rebaselinar hashes, sem mudar a estrutura consolidada
na AP-003. Candidatos atuais: **13**.

### Símbolos adiados

Símbolos opacos de stage/dispatch, núcleo `_ap003f_pipeline_core`, nomes sem
sugestão semântica, colisões e superfícies com compatibilidade ficam fora do
primeiro aplicador. Total adiado: **49**.

### Proteções absolutas

`_refs_v6_strip_org`, `_ap003d_impl__refs_v6_strip_org`,
`extract_org_abstracts` e `WorkflowState._normalize` permanecem fora do escopo.
Nenhum xfail será corrigido, renomeado ou convertido em teste aprovado.

## Regras do futuro aplicador

- branch, HEAD e remoto exatos;
- árvore limitada aos cinco artefatos preparatórios e aos dois contratos AP-004B mantidos;
- hashes de todo o `source_manifest`;
- renomeação por AST, nunca substituição global;
- atualização somente de referências vinculadas ao mesmo `candidate_id`;
- bloqueio diante de string dinâmica, consumidor externo inesperado ou colisão;
- preservação dos módulos e wrappers tratados na AP-004B;
- backup externo, escrita atômica e rollback integral;
- `py_compile`, `git diff --check`, suíte específica e suíte consolidada;
- nenhum commit sem aprovação expressa.

## Estado

A criação do aplicador produtivo permanece bloqueada até aprovação deste
inventário e definição das ondas que realmente serão executadas.

# AP-004D — Estratégia de consolidação dos marcadores de versão

> **Gate:** aplicador produtivo bloqueado até aprovação expressa do inventário `2059d15dceb68a105e6b03b4fa15e900730ab398e1dc1eb03dd13143578571b1`.

## Objetivo

Consolidar somente marcadores internos de versão sem contrato público, sem função de compatibilidade, sem vínculo com artefatos históricos congelados e sem alteração comportamental.

## Baseline

- Branch: `ap-refactor/03-orchestrator-decomposition`
- Commit AP-004C: `81293d79e86da8b4d0407b483fc3dedaf27768cb`
- Remoto sincronizado: `81293d79e86da8b4d0407b483fc3dedaf27768cb`
- Divergência: `0 0`
- Registros inventariados: **14742**
- Candidatos AST: **20**
- Candidatos textuais: **0**
- Colisões: **0**

## Ondas propostas

### Onda 0 — preservação explícita

Não alterar símbolos AP-003, os quatro símbolos protegidos, os três defeitos congelados, os dois arquivos fulltext, caminhos físicos fora do escopo, superfícies públicas, wrappers, snapshots, fixtures, manifestos e metadados de compatibilidade.

### Onda 1 — símbolos privados por AST

Considerar exclusivamente registros classificados como `marcador_privado_renomeavel_ast`, sem colisão e com definição/consumidores identificados. A transformação deverá resolver todas as referências da onda antes da primeira escrita; substituição textual é proibida.

### Onda 2 — comentários internos removíveis

Considerar apenas registros classificados como `marcador_interno_removivel`. Cada alteração deve preservar o significado do comentário e não pode alcançar strings, documentos gerados ou registros históricos.

### Onda 3 — revisão manual

Adiar strings operacionais, identificadores sem definição local, ocorrências documentais, caminhos versionados e qualquer item ambíguo. Esses itens exigem contrato explícito ou prova adicional de ausência de consumidores.

## Contrato do futuro aplicador

O eventual aplicador deverá ser idempotente, pré-validar todas as ondas de uma vez, usar backup externo, escrita atômica e rollback integral, informar caminhos alterados, executar `py_compile`, `git diff --check`, testes específicos e suíte consolidada, e nunca criar commit ou publicar automaticamente.

## Critério de autorização

A criação do aplicador produtivo só poderá começar após aprovação expressa dos candidatos listados no inventário. Aprovação genérica da fase não substitui a aprovação da lista de candidatos e dos destinos propostos.

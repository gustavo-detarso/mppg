# AP-007D.4 — Caracterização da segunda onda operacional

## Comando selecionado

`--check-institution-compliance`

## Resultado

- status: `selected_for_isolated_adapter`;
- risco: `low`;
- pontuação: `31/33`;
- handlers AST confirmados: `1`;
- funções transitivas: `4`;
- testes localizados: `23`;
- falsos positivos reclassificados: `re.sub, resolve, unicodedata.combining`;
- dependências de projeto não resolvidas: `nenhuma`;
- efeitos bloqueadores: `nenhum`.

## Método

A evidência transitiva dirigida da AP-007D.1 foi reutilizada e revalidada contra o estado atual: registro do parser, caminhos rastreados, rejeição de backups, handlers por arquivo/linha/qualname e ausência de efeitos bloqueadores. Chamadas por atributo foram reclassificadas somente quando o AST comprovou tratar-se de módulo da biblioteca padrão ou do método não mutante `resolve()` sobre caminho.

## Decisão

O comando está apto à materialização de adaptador nativo isolado. A rota pública permanece em `legacy_fallback` até equivalência direta e contratos próprios.

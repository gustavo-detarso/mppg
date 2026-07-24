# AP-007D.5 — Adaptador nativo isolado de `--make-doi-manifest`

## Decisão

A caracterização controlada comprovou a lógica DOI mínima para entradas por
diretório e ZIP. A rota pública histórica permaneceu indisponível no Python
canônico porque o bootstrap legado exige `dotenv` antes de alcançar o handler.

Este artefato materializa somente o adaptador isolado. A rota pública continua
em `legacy_fallback` até uma integração posterior e explicitamente validada.

## Estratégia

O módulo é formado pelos segmentos AST canônicos de `project_tools.py`
necessários a `make_doi_manifest`, incluindo os dois iteradores, constantes e
imports padrão. Ele não importa `project_tools`, `bibliography_manager`, o
monólito histórico, `dotenv` ou `pydantic`.

O entrypoint aceita exatamente uma origem (`--input-dir` ou `--input-zip`),
exige `--output`, preserva sobrescrita controlada e retorna `0` após gerar o
CSV. Diretório e ZIP são verificados por equivalência de conteúdo normalizado.

## Restrições

- sem rede, subprocessos ou avaliação dinâmica;
- sem mutação de `sys.argv`, `sys.path` ou cwd;
- sem alteração de runtime, parser, CLI ou dispatcher;
- sem staging, commit, tag ou push.

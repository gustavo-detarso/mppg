# AP-007D.5 — Integração pública de `--make-doi-manifest`

O comando passou a usar a rota `native_doi_manifest` para solicitações compostas somente por `--make-doi-manifest`, uma origem (`--input-dir` ou `--input-zip`) e `--output`.

A integração usa o adaptador produzido pela closure AST mínima de `project_tools.py`, sem importar o bootstrap legado nem depender de `dotenv`. Erros de uso retornam `1`; gerações válidas retornam `0`.

Runtime anterior: `b54d7b47b7eca7c02af5d4e0f004e9243b3e9ec386c736d0edf06d17bbc07061`. Runtime integrado: `b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c`.

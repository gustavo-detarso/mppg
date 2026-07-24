# AP-007D.4 — Integração pública de `--check-institution-compliance`

O comando passou a usar a rota `native_institution_compliance` quando invocado isoladamente com `--config` e os overrides de caminho explicitamente permitidos. Combinações com outros comandos operacionais permanecem no fallback legado.

Códigos preservados: `0` para relatório válido, `2` para inconformidade sem falha técnica e `1` para erro de uso, inclusive ausência de `--config`.

Runtime anterior: `98f84244f3e447c108f627b9af55ab4782ed20347952b73c63f19e44a2b5371d`. Runtime integrado: `b54d7b47b7eca7c02af5d4e0f004e9243b3e9ec386c736d0edf06d17bbc07061`.
Origem da integração formalizada: `reconciled_preexisting_exact_write`.

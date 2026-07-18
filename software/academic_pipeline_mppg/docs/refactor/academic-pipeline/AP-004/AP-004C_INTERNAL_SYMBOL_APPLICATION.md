# AP-004C — aplicação produtiva de símbolos internos (v1.4)

> Aplicação em duas ondas atômicas. Nenhum commit foi criado.

## Base canônica

- Branch: `ap-refactor/03-orchestrator-decomposition`.
- HEAD local/remoto: `aa9829f09a5c1b9e69c634637c311b03f360b07e`.
- Inventário aprovado: `internal-symbol-inventory-v1.3-read-only`.
- Baseline: `463 passed, 3 xfailed`.

## Onda 1 — símbolos locais

Arquivo: `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py`.

- `_generate_interactive_before_wizard_documentos_locais_v4` → `_generate_interactive_before_wizard_documentos_locais` (2 tokens).
- `_generate_interactive_with_wizard_documentos_locais_v4` → `_generate_interactive_with_wizard_documentos_locais` (2 tokens).
- `_v5_is_local_document` → `_is_local_document` (3 tokens).
- `_v5_reference_default` → `_reference_default` (2 tokens).
- `_v5_normalise_prompt` → `_normalise_prompt` (3 tokens).
- `_v5_configure_reference_policy` → `_configure_reference_policy` (2 tokens).
- `_v5_ensure_reference_policy` → `_ensure_reference_policy` (2 tokens).

## Onda 2 — aliases vinculados a contratos

Arquivo: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`.

- `_ap003d_impl_output_paths` → `_impl_output_paths` (2 tokens).
- `_ap003d_impl_apply_cli_path_overrides` → `_impl_apply_cli_path_overrides` (2 tokens).
- `_ap003d_impl_load_existing_document_json` → `_impl_load_existing_document_json` (2 tokens).
- `_ap003d_impl_resolve_bib_for_existing_document` → `_impl_resolve_bib_for_existing_document` (2 tokens).
- `_ap003d_impl__resolve_latex_paths_for_recompile` → `_impl_resolve_latex_paths_for_recompile` (2 tokens).
- `_ap003d_impl_run_recompile` → `_impl_run_recompile` (2 tokens).
- `_ap003d_impl_render_additional_language_versions` → `_impl_render_additional_language_versions` (2 tokens).
- `_ap003d_impl__refs_v6_disabled` → `_impl_refs_disabled` (2 tokens).
- `_ap003d_impl__refs_v6_apply_runtime_policy` → `_impl_refs_apply_runtime_policy` (2 tokens).
- `_ap003d_impl_load_config` → `_impl_load_config` (2 tokens).
- `_ap003d_impl_build_bibliography` → `_impl_build_bibliography` (2 tokens).
- `_ap003d_impl__refs_v6_clear_document_bibliography` → `_impl_refs_clear_document_bibliography` (2 tokens).
- `_ap003d_impl_render_org_latex` → `_impl_render_org_latex` (2 tokens).

A AST normalizada das duas ondas permanece idêntica ao baseline; somente os identificadores aprovados foram alterados.

## Contratos atualizados

- `tests/characterization/test_ap003d_document_contract.py`
- `tests/characterization/test_ap003f_main_unification_contract.py`
- `tests/characterization/test_ap003g_stabilization_contract.py`
- `tests/characterization/test_ap004b_module_file_application_contract.py`
- `tests/characterization/test_ap004c_internal_symbol_inventory_contract.py`

## Proteções

- `_refs_v6_strip_org` preservado.
- `_ap003d_impl__refs_v6_strip_org` preservado.
- `WorkflowState._normalize` preservado.
- `extract_org_abstracts` preservado.
- Símbolos adiados: **49**, sem alteração.
- Nenhum módulo, arquivo ou diretório foi renomeado.

## Validação

- `py_compile`: `passed`.
- `git diff --check`: `passed`.
- Contrato da aplicação: `19 passed`.
- Suíte específica: `161 passed, 1 xfailed`.
- Suíte consolidada: `482 passed, 3 xfailed`.

## Estado

A consolidação permanece bloqueada até revisão do diff e aprovação expressa.

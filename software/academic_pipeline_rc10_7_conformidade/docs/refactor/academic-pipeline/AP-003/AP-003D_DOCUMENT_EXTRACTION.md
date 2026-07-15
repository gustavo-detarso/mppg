# AP-003D — extração da orquestração documental

## Estratégia

A extração foi dirigida pelo inventário AST da AP-003D. Helpers documentais foram mantidos no módulo histórico como wrappers finos. As faixas documentais do primeiro `main()` foram substituídas por estágios que devolvem explicitamente as variáveis produzidas.

## Helpers extraídos

| Helper histórico | Implementação | Origem | Runtime externo |
|---|---|---:|---|
| `load_config` | `load_config_impl` | 1648–1649 | `_refs_v6_apply_runtime_policy`, `_refs_v6_original_load_config` |
| `output_paths` | `output_paths_impl` | 308–317 | `Path`, `_section`, `resolve_path` |
| `apply_cli_path_overrides` | `apply_cli_path_overrides_impl` | 353–386 | nenhum |
| `load_existing_document_json` | `load_existing_document_json_impl` | 389–390 | `AcademicDocument` |
| `resolve_bib_for_existing_document` | `resolve_bib_for_existing_document_impl` | 393–420 | `Path`, `c`, `m` |
| `_resolve_latex_paths_for_recompile` | `_resolve_latex_paths_for_recompile_impl` | 436–442 | `Path`, `resolve_path` |
| `run_recompile` | `run_recompile_impl` | 445–471 | `Path`, `_resolve_latex_paths_for_recompile`, `clean_aux_files`, `make_run_report`, `print_outputs`, `run_compile_sequence`, `stage`, `write_json`, `write_outputs_manifest` |
| `render_additional_language_versions` | `render_additional_language_versions_impl` | 535–648 | `Any`, `Path`, `build_quality_report`, `inject_paper_abstracts_into_docx`, `inject_paper_abstracts_into_org`, `item`, `raise_if_errors`, `render_docx`, `render_org_latex`, `requested_translation_languages`, `resolve_path`, `run_compile_sequence`, `shutil`, `stage`, `translate_document_model`, `translation_batch_size`, `validate_docx_file`, `validate_org_text`, `write_json`, `write_quality_report` |
| `_refs_v6_disabled` | `_refs_v6_disabled_impl` | 1587–1602 | nenhum |
| `_refs_v6_apply_runtime_policy` | `_refs_v6_apply_runtime_policy_impl` | 1605–1643 | `_refs_v6_disabled` |
| `build_bibliography` | `build_bibliography_impl` | 1655–1666 | `Path`, `_refs_v6_disabled`, `_refs_v6_original_build_bibliography` |
| `_refs_v6_clear_document_bibliography` | `_refs_v6_clear_document_bibliography_impl` | 1669–1680 | nenhum |
| `_refs_v6_strip_org` | `_refs_v6_strip_org_impl` | 1683–1711 | nenhum |
| `render_org_latex` | `render_org_latex_impl` | 1717–1743 | `Path`, `_refs_v6_clear_document_bibliography`, `_refs_v6_disabled`, `_refs_v6_original_render_org_latex`, `_refs_v6_strip_org` |

## Estágios documentais extraídos

| Estágio | AST | Origem | Variáveis devolvidas | Runtime externo |
|---|---:|---:|---|---|
| `run_document_stage_001` | 36–36 | 1035–1055 | `bib_entry_key`, `bib_keys`, `bib_path`, `document`, `document_json`, `e`, `k`, `org`, `out`, `report`, `split_bib_entries` | `Path`, `build_quality_report`, `k`, `load_existing_document_json`, `write_quality_report` |
| `run_document_stage_002` | 68–68 | 1155–1155 | `doc_cfg` | `cfg` |
| `run_document_stage_003` | 80–80 | 1236–1289 | `client`, `document`, `mm_diag`, `model`, `outputs`, `removed_mindmap_files`, `report`, `w` | `PIPELINE_VERSION`, `Path`, `attach_existing_mindmap_if_available`, `cfg`, `delete_existing_mindmap_outputs`, `document_json_path`, `generate_and_attach_mindmap`, `json`, `load_existing_document_json`, `make_client`, `make_run_report`, `mm_diag`, `model`, `out_dir`, `prefix`, `print_outputs`, `should_generate_mindmap`, `stage`, `warnings`, `write_json`, `write_outputs_manifest` |
| `run_document_stage_004` | 86–86 | 1451–1451 | nenhuma | `stage` |
| `run_document_stage_005` | 88–91 | 1453–1458 | `org_text` | `bib_keys`, `bib_path`, `cfg`, `document`, `inject_paper_abstracts_into_org`, `main_document_abstract_languages`, `org_path`, `paper_abstract_bundle`, `prefix`, `raise_if_errors`, `render_org_latex`, `stage`, `validate_org_text` |
| `run_document_stage_006` | 93–93 | 1461–1466 | `academic_writing`, `latex_extra`, `pdf_engine`, `pdf_path` | `config_dir`, `doc_cfg`, `latex_cfg`, `org_path`, `resolve_path`, `run_compile_sequence`, `stage` |
| `run_document_stage_007` | 96–96 | 1470–1480 | `docx_cfg`, `docx_path`, `docx_validation`, `ref`, `w` | `bib_path`, `cfg`, `config_dir`, `doc_cfg`, `document`, `inject_paper_abstracts_into_docx`, `main_document_abstract_languages`, `out_dir`, `paper_abstract_bundle`, `prefix`, `render_docx`, `resolve_path`, `stage`, `validate_docx_file`, `w`, `warnings` |
| `run_document_stage_008` | 98–98 | 1483–1509 | `exc`, `translated_outputs`, `translation_warnings` | `TranslationError`, `bib_keys`, `bib_path`, `cfg`, `client`, `config_dir`, `doc_cfg`, `document`, `latex_cfg`, `model`, `out_dir`, `paper_abstract_bundle`, `prefix`, `render_additional_language_versions`, `requested_translation_languages`, `warnings` |
| `run_document_stage_009` | 106–106 | 1532–1538 | `compliance_report` | `bib_path`, `cfg`, `docx_path`, `org_path`, `pdf_path`, `run_institution_compliance` |
| `run_document_stage_010` | 108–108 | 1540–1540 | nenhuma | `compliance_md`, `outputs` |
| `run_document_stage_011` | 112–112 | 1547–1547 | `quality` | `bib_keys`, `build_quality_report`, `document`, `org_path` |
| `run_document_stage_012` | 116–117 | 1552–1569 | `report` | `Path`, `cache_dir`, `cfg`, `docx_validation`, `make_run_report`, `model`, `out_dir`, `outputs`, `precheck`, `prefix`, `quality_path`, `warnings`, `work_dir` |

## Integridade

- Orquestrador antes: `4261568e60308764ef1f56ab1e13d6ccfd886d76dce965ec6d3e8fd66cdee51d`.
- Orquestrador depois: `da4c6c9b817d6607873e0412b5829729e36c3d70a1745b4b7d39ea4e31d31367`.
- Parser AP-003B: `f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8`.
- Despacho AP-003C: `42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3`.
- Helpers extraídos: **14**.
- Estágios extraídos: **12**.
- Dois `main()` preservados.
- Alias `_original_main_before_prisma_artigo_generico_wrapper` preservado.
- Blocos de sobreposição com PRISMA não foram movidos.

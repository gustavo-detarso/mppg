# AP-003F — preparação da unificação do main

> Inventário AST somente leitura. Nenhum módulo produtivo foi alterado.

## Baseline

- Branch: `ap-refactor/03-orchestrator-decomposition`
- HEAD: `f493ab1d09f38467bf47de05760963500abd554d`
- Upstream: `origin/ap-refactor/03-orchestrator-decomposition`
- HEAD remoto: `f493ab1d09f38467bf47de05760963500abd554d`
- Orquestrador: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`
- SHA-256: `431882d57a5a6ed334985b51a04db782ceda8de5cff1aa0e3bf856c4aa2c5b3a`
- Definições `main()`: **2**
- Alias histórico: **1**
- Guardas `__main__`: **1**

## Ordem estrutural

| Elemento | Índice AST | Linhas |
|---|---:|---:|
| Primeiro `main()` | 71 | 498–1243 |
| Alias histórico | 87 | 1325–1325 |
| Segundo `main()` | 88 | 1327–1329 |
| Guarda direta | 89 | 1333–1334 |

## Primeiro main canônico

- Instruções de nível superior: **213**
- Chamadas principais:
  - `', '.join`
  - `'\n- '.join`
  - `(cfg.get('documentos_locais', {}) if isinstance(cfg.get('documentos_locais'), dict) else {}).get`
  - `FileNotFoundError`
  - `Path`
  - `Path(args.document_json).expanduser`
  - `Path(args.document_json).expanduser().resolve`
  - `Path(str(cfg.get('__config_dir__'))).resolve`
  - `RuntimeError`
  - `_ap003c_dispatch_001`
  - `_ap003c_dispatch_002`
  - `_ap003c_dispatch_003`
  - `_ap003c_dispatch_004`
  - `_ap003c_dispatch_005`
  - `_ap003c_dispatch_006`
  - `_ap003c_dispatch_007`
  - `_ap003c_dispatch_008`
  - `_ap003c_dispatch_009`
  - `_ap003c_dispatch_010`
  - `_ap003c_dispatch_011`
  - `_ap003c_dispatch_012`
  - `_ap003c_dispatch_013`
  - `_ap003c_dispatch_014`
  - `_ap003c_dispatch_015`
  - `_ap003c_dispatch_016`
  - `_ap003c_dispatch_017`
  - `_ap003c_dispatch_018`
  - `_ap003c_dispatch_019`
  - `_ap003d_stage_001`
  - `_ap003d_stage_002`
  - `_ap003d_stage_003`
  - `_ap003d_stage_004`
  - `_ap003d_stage_005`
  - `_ap003d_stage_006`
  - `_ap003d_stage_007`
  - `_ap003d_stage_008`
  - `_ap003d_stage_009`
  - `_ap003d_stage_010`
  - `_ap003d_stage_011`
  - `_ap003d_stage_012`
  - `_ap003e_stage_001`
  - `_ap003e_stage_002`
  - `_ap003e_stage_003`
  - `_ap003e_stage_004`
  - `_ap003e_stage_005`
  - `_ap003e_stage_006`
  - `_ap003e_stage_007`
  - `_ap003e_stage_008`
  - `_load_optional_config`
  - `_openai_model_from_cfg`
  - `abstract_sidecar_path`
  - `apply_cli_path_overrides`
  - `attach_existing_mindmap_if_available`
  - `bool`
  - `build_bibliography`
  - `build_document_model`
  - `cfg.get`
  - `check_config`
  - `collect_orientation_docs`
  - `compliance_report.get`
  - `copy_documents_to_fulltext_cache`
  - `delete_existing_mindmap_outputs`
  - `discover_local_documents`
  - `document.model_dump`
  - `document_json_path.exists`
  - `e.get`
  - `generate_and_attach_mindmap`
  - `generate_paper_abstract_bundle`
  - `globals`
  - `isinstance`
  - `json.dumps`
  - `load_existing_document_json`
  - `locals`
  - `make_client`
  - `make_run_report`
  - `paper_abstract_path.exists`
  - `paper_abstracts_enabled`
  - `parse_cli_args`
  - `pipeline_cfg.get`
  - `precheck.get`
  - `print`
  - `print_outputs`
  - `prisma_enabled`
  - `quality.get`
  - `raise_if_errors`
  - `read_paper_abstract_bundle`
  - `resolve_bib_for_existing_document`
  - `resumo_cfg_for_stage.get`
  - `run_prisma_report_outputs`
  - `sanitize_document_model_raw_bibkeys`
  - `sanitize_document_model_technical_leaks`
  - `should_generate_mindmap`
  - `stage`
  - `str`
  - `validate_document_model`
  - `w.get`
  - `warnings.append`
  - `warnings.extend`
  - `work_cache_paths`
  - `write_compliance_reports`
  - `write_json`
  - `write_outputs_manifest`
  - `write_paper_abstract_bundle`
  - `write_prompt_lock`
  - `write_prompt_lock_markdown`
  - `write_quality_report`

## Segundo main histórico

- Instruções: **2**
- Candidato a wrapper fino: **sim**
- Delegação AP-003E:
  - `_ap003e_entrypoint`
- Chamadas ao alias histórico: **nenhuma**.

## Superfícies de entrada

- Guarda direta chama `main()`: **sim**
- Guarda usa `SystemExit`/`sys.exit`: **sim**
- `academic_pipeline.__main__`: `academic_pipeline/__main__.py`
- Chamadas no módulo de pacote: `SystemExit`, `main`

## Região candidata à remoção

- Linhas: **1325–1329**
- Índices AST: **87–88**
- Conteúdo:
  - alias histórico;
  - segundo `main()` wrapper da AP-003E.

```python
   1323:     return _ap003e_impl__prisma_artigo_generico_run_freeze_1({**globals(), **locals()}, argv, silent)
   1324:
>> 1325: _original_main_before_prisma_artigo_generico_wrapper = main
>> 1326:
>> 1327: def main(*args, **kwargs):
>> 1328:     from academic_pipeline.prisma_generic_orchestration import run_prisma_generic_entrypoint as _ap003e_entrypoint
>> 1329:     return _ap003e_entrypoint({**globals(), **locals()}, *args, **kwargs)
   1330: # <<< PATCH_PRISMA_ARTIGO_GENERICO_WRAPPER_V1_5 <<<
   1331:
```

## Condições obrigatórias do aplicador

- remover somente o alias histórico e o segundo `main()`;
- preservar integralmente o primeiro `main()`;
- preservar a guarda direta e fazê-la continuar chamando `main()`;
- preservar `academic_pipeline.__main__` byte a byte;
- preservar parser, despacho, documento e PRISMA byte a byte;
- resultar em exatamente um `main()` de nível superior;
- eliminar completamente `_original_main_before_prisma_artigo_generico_wrapper` do orquestrador;
- manter script direto, `python -m academic_pipeline` e `academic-pipeline` equivalentes;
- preservar os três `xfail` conhecidos;
- executar a suíte consolidada em `app_bundle/tests tests`.

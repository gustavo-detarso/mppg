# AP-003C — preparação da extração do despacho

> Relatório somente leitura gerado por AST. Nenhum módulo produtivo foi alterado.

## Baseline

- Branch: `ap-refactor/03-orchestrator-decomposition`
- HEAD: `90dc81b60720c25d2514ca6a3dec29dcba91efe8`
- Upstream: `origin/ap-refactor/03-orchestrator-decomposition`
- HEAD remoto: `90dc81b60720c25d2514ca6a3dec29dcba91efe8`
- Orquestrador: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`
- SHA-256: `51af32106184df8fd5810222a8ccdb5cc0818aa3e167ff8bd2e1c96199ef1a0f`
- Primeiro `main()`: linhas 899–1568
- Parse de argumentos: linhas 902–902
- Definições `main()` preservadas: **2**

## Resumo do despacho observado

- Candidatos baseados em `args`: **29**
- Candidatos terminais (`return`/`raise`): **23**
- Candidatos não terminais: **6**
- Atributos distintos de `args`: **37**

### Atributos observados

```text
base_dir
bib
check_config
check_institution_compliance
config
doctor
document_json
docx
explain_profile
forcar_regeneracao_mapa_mental
gui
init_project
init_toml
input_dir
input_zip
inspect_bib
institution
list_institutions
list_layouts
list_toml_profiles
make_doi_manifest
no_clear
org
output
overwrite_project
pdf
prisma_importar_triagem
project_type
quality_report
recompile
reusar_mapa_mental
show_prompts
somente_mapa_mental
somente_renderizar
toml_profile
tui
write_prompt_lock
```

## Tabela de candidatos

| # | Linhas | Tipo | Condição | Flags/atributos | Terminal | Chamadas |
|---:|---:|---|---|---|:---:|---|
| 1 | 914–920 | `if` | `args.gui` | `gui` | sim | `run_gui` |
| 2 | 922–928 | `if` | `args.tui` | `no_clear`, `tui` | sim | `bool`, `run_tui` |
| 3 | 930–937 | `if` | `args.list_toml_profiles` | `list_toml_profiles` | sim | `print_profiles` |
| 4 | 939–946 | `if` | `args.init_toml` | `init_toml`, `no_clear`, `toml_profile` | sim | `bool`, `generate_interactive` |
| 5 | 948–950 | `if` | `args.list_institutions` | `list_institutions` | sim | `describe_institution_profiles`, `print` |
| 6 | 952–967 | `if` | `args.list_layouts` | `config`, `list_layouts` | sim | `Path`, `Path(args.config).expanduser`, `Path(args.config).expanduser().resolve`, `RuntimeError`, `available_layouts`, `layouts.items`, `load_config`, `print`, `resolve_layout_spec`, `spec.get`, `str`, `str(spec.get('description') or spec.get('descricao') or '').strip`, `str(spec.get('genero_academico') or '').strip` |
| 7 | 969–971 | `if` | `args.explain_profile` | `explain_profile` | sim | `explain_profile`, `print` |
| 8 | 973–978 | `if` | `args.show_prompts` | `config`, `show_prompts` | sim | `Path`, `Path(args.config).expanduser`, `Path(args.config).expanduser().resolve`, `RuntimeError`, `json.dumps`, `load_config`, `print`, `prompt_report_for_cfg` |
| 9 | 980–990 | `if` | `args.init_project` | `base_dir`, `init_project`, `institution`, `overwrite_project`, `project_type` | sim | `Path`, `Path(args.base_dir).expanduser`, `Path(args.base_dir).expanduser().resolve`, `bool`, `init_project`, `print` |
| 10 | 992–1009 | `if` | `args.make_doi_manifest` | `input_dir`, `input_zip`, `make_doi_manifest`, `output` | sim | `Path`, `Path(args.input_dir).expanduser`, `Path(args.input_dir).expanduser().resolve`, `Path(args.input_zip).expanduser`, `Path(args.input_zip).expanduser().resolve`, `Path(args.output).expanduser`, `Path(args.output).expanduser().resolve`, `RuntimeError`, `make_doi_manifest`, `print` |
| 11 | 1011–1017 | `if` | `args.inspect_bib` | `inspect_bib` | sim | `Path`, `Path(args.inspect_bib).expanduser`, `Path(args.inspect_bib).expanduser().resolve`, `bib.with_name`, `inspect_bib`, `print`, `render_bib_inspection_markdown`, `report.get`, `str` |
| 12 | 1019–1039 | `if` | `args.quality_report` | `bib`, `document_json`, `org`, `quality_report` | sim | `Path`, `Path(args.bib).expanduser`, `Path(args.bib).expanduser().resolve`, `Path(args.document_json).expanduser`, `Path(args.document_json).expanduser().resolve`, `Path(args.org).expanduser`, `Path(args.org).expanduser().resolve`, `RuntimeError`, `bib_entry_key`, `bib_path.exists`, `bib_path.read_text`, `build_quality_report`, `document_json.with_suffix`, `list`, `load_existing_document_json`, `print`, `report.get`, `split_bib_entries`, `write_quality_report` |
| 13 | 1041–1041 | `assignment` | `` | `config` | não | `_load_optional_config` |
| 14 | 1045–1046 | `if` | `args.somente_renderizar and args.somente_mapa_mental` | `somente_mapa_mental`, `somente_renderizar` | sim | `RuntimeError` |
| 15 | 1047–1048 | `if` | `args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental` | `forcar_regeneracao_mapa_mental`, `reusar_mapa_mental` | sim | `RuntimeError` |
| 16 | 1050–1060 | `if` | `args.write_prompt_lock` | `write_prompt_lock` | sim | `RuntimeError`, `external_search_enabled`, `output_paths`, `print`, `research_output_paths`, `write_prompt_lock`, `write_prompt_lock_markdown` |
| 17 | 1062–1074 | `if` | `args.check_institution_compliance` | `bib`, `check_institution_compliance`, `docx`, `org`, `pdf` | sim | `Path`, `Path(args.bib).expanduser`, `Path(args.bib).expanduser().resolve`, `Path(args.docx).expanduser`, `Path(args.docx).expanduser().resolve`, `Path(args.org).expanduser`, `Path(args.org).expanduser().resolve`, `Path(args.pdf).expanduser`, `Path(args.pdf).expanduser().resolve`, `RuntimeError`, `output_paths`, `print`, `render_compliance_markdown`, `report.get`, `run_institution_compliance`, `write_compliance_reports` |
| 18 | 1076–1082 | `if` | `args.doctor` | `doctor` | sim | `external_search_enabled`, `output_paths`, `print_doctor_report`, `report.get`, `research_output_paths`, `run_doctor`, `write_json` |
| 19 | 1084–1091 | `if` | `args.check_config` | `check_config` | sim | `RuntimeError`, `check_config`, `external_search_enabled`, `output_paths`, `print_check_config_report`, `report.get`, `research_output_paths`, `write_json` |
| 20 | 1093–1094 | `if` | `args.recompile` | `recompile` | sim | `run_recompile` |
| 21 | 1099–1139 | `if` | `args.prisma_importar_triagem` | `prisma_importar_triagem` | sim | `Path`, `RuntimeError`, `artifacts.get`, `cfg.get`, `external_search_enabled`, `import_manual_prisma_triage`, `isinstance`, `make_run_report`, `print_outputs`, `prisma_outputs.setdefault`, `render_external_prisma_outputs`, `research_output_paths`, `stage`, `str`, `write_json`, `write_outputs_manifest` |
| 22 | 1141–1141 | `assignment` | `` | `somente_renderizar` | não | `bool` |
| 23 | 1142–1142 | `assignment` | `` | `somente_renderizar` | não | `external_search_enabled` |
| 24 | 1160–1160 | `assignment` | `` | `document_json` | não | `Path`, `Path(args.document_json).expanduser`, `Path(args.document_json).expanduser().resolve` |
| 25 | 1162–1224 | `if` | `is_external_prisma_run` | `somente_mapa_mental` | sim | `Path`, `RuntimeError`, `artifacts.get`, `bool`, `cfg.get`, `isinstance`, `make_client`, `make_run_report`, `print_outputs`, `prisma_outputs.setdefault`, `render_external_prisma_outputs`, `run_external_prisma_search`, `search_cfg.get`, `stage`, `str`, `write_json`, `write_outputs_manifest`, `write_prompt_lock`, `write_prompt_lock_markdown` |
| 26 | 1226–1279 | `if` | `args.somente_mapa_mental` | `forcar_regeneracao_mapa_mental`, `reusar_mapa_mental`, `somente_mapa_mental` | sim | `(mm_diag or {}).get`, `FileNotFoundError`, `Path`, `RuntimeError`, `attach_existing_mindmap_if_available`, `bool`, `cfg.get`, `delete_existing_mindmap_outputs`, `dict`, `document.model_dump`, `document_json_path.exists`, `generate_and_attach_mindmap`, `json.dumps`, `load_existing_document_json`, `make_client`, `make_run_report`, `print`, `print_outputs`, `should_generate_mindmap`, `stage`, `str`, `warnings.append`, `write_json`, `write_outputs_manifest` |
| 27 | 1286–1439 | `if` | `args.somente_renderizar` | `forcar_regeneracao_mapa_mental`, `reusar_mapa_mental`, `somente_renderizar` | sim | `', '.join`, `(cfg.get('documentos_locais', {}) if isinstance(cfg.get('documentos_locais'), dict) else {}).get`, `FileNotFoundError`, `Path`, `RuntimeError`, `attach_existing_mindmap_if_available`, `bool`, `build_bibliography`, `build_document_model`, `cfg.get`, `collect_orientation_docs`, `copy_documents_to_fulltext_cache`, `delete_existing_mindmap_outputs`, `discover_local_documents`, `document.model_dump`, `document_json_path.exists`, `generate_and_attach_mindmap`, `generate_paper_abstract_bundle`, `isinstance`, `json.dumps`, `load_existing_document_json`, `make_client`, `make_run_report`, `paper_abstract_path.exists`, `paper_abstracts_enabled`, `pipeline_cfg.get`, `print_outputs`, `prisma_enabled`, `raise_if_errors`, `read_paper_abstract_bundle`, `resolve_bib_for_existing_document`, `resumo_cfg_for_stage.get`, `run_prisma_report_outputs`, `sanitize_document_model_raw_bibkeys`, `sanitize_document_model_technical_leaks`, `should_generate_mindmap`, `stage`, `str`, `validate_document_model`, `warnings.append`, `write_json`, `write_outputs_manifest`, `write_paper_abstract_bundle`, `write_prompt_lock`, `write_prompt_lock_markdown` |
| 28 | 1473–1499 | `if` | `args.somente_renderizar` | `somente_renderizar` | não | `render_additional_language_versions`, `requested_translation_languages`, `warnings.append`, `warnings.extend` |
| 29 | 1544–1559 | `assignment` | `` | `somente_renderizar` | não | `Path`, `cfg.get`, `make_run_report`, `str` |

## Blocos-fonte candidatos

### Candidato 1: linhas 914–920

- Tipo: `if`
- Condição: `args.gui`
- Atributos: `args.gui`
- Fluxo terminal: sim
- Retornos/lançamentos: `return run_gui()`

```python
    912:         return _prisma_curadoria_dispatch(args)
    913:     # <<< PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_DISPATCH <<<
>>  914:     if args.gui:
>>  915:         # Compatibilidade temporária entre pacote e script direto.
>>  916:         if __package__:
>>  917:             from .academic_pipeline_gui import run_gui
>>  918:         else:
>>  919:             from academic_pipeline_gui import run_gui
>>  920:         return run_gui()
    921:
    922:     if args.tui:
```

### Candidato 2: linhas 922–928

- Tipo: `if`
- Condição: `args.tui`
- Atributos: `args.no_clear`, `args.tui`
- Fluxo terminal: sim
- Retornos/lançamentos: `return run_tui(no_clear=bool(args.no_clear))`

```python
    920:         return run_gui()
    921:
>>  922:     if args.tui:
>>  923:         # Compatibilidade temporária entre pacote e script direto.
>>  924:         if __package__:
>>  925:             from .academic_pipeline_tui import run_tui
>>  926:         else:
>>  927:             from academic_pipeline_tui import run_tui
>>  928:         return run_tui(no_clear=bool(args.no_clear))
    929:
    930:     if args.list_toml_profiles:
```

### Candidato 3: linhas 930–937

- Tipo: `if`
- Condição: `args.list_toml_profiles`
- Atributos: `args.list_toml_profiles`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`

```python
    928:         return run_tui(no_clear=bool(args.no_clear))
    929:
>>  930:     if args.list_toml_profiles:
>>  931:         # Compatibilidade temporária entre pacote e script direto.
>>  932:         if __package__:
>>  933:             from .academic_pipeline_toml_generator_interativo import print_profiles
>>  934:         else:
>>  935:             from academic_pipeline_toml_generator_interativo import print_profiles
>>  936:         print_profiles()
>>  937:         return 0
    938:
    939:     if args.init_toml:
```

### Candidato 4: linhas 939–946

- Tipo: `if`
- Condição: `args.init_toml`
- Atributos: `args.init_toml`, `args.no_clear`, `args.toml_profile`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`

```python
    937:         return 0
    938:
>>  939:     if args.init_toml:
>>  940:         # Compatibilidade temporária entre pacote e script direto.
>>  941:         if __package__:
>>  942:             from .academic_pipeline_toml_generator_interativo import generate_interactive
>>  943:         else:
>>  944:             from academic_pipeline_toml_generator_interativo import generate_interactive
>>  945:         generate_interactive(non_interactive_profile=args.toml_profile or None, no_clear=bool(args.no_clear))
>>  946:         return 0
    947:
    948:     if args.list_institutions:
```

### Candidato 5: linhas 948–950

- Tipo: `if`
- Condição: `args.list_institutions`
- Atributos: `args.list_institutions`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`

```python
    946:         return 0
    947:
>>  948:     if args.list_institutions:
>>  949:         print(describe_institution_profiles())
>>  950:         return 0
    951:
    952:     if args.list_layouts:
```

### Candidato 6: linhas 952–967

- Tipo: `if`
- Condição: `args.list_layouts`
- Atributos: `args.config`, `args.list_layouts`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`, `raise raise RuntimeError("--list-layouts exige --config caminho.toml")`

```python
    950:         return 0
    951:
>>  952:     if args.list_layouts:
>>  953:         if not args.config:
>>  954:             raise RuntimeError("--list-layouts exige --config caminho.toml")
>>  955:         cfg_layouts = load_config(Path(args.config).expanduser().resolve())
>>  956:         layouts = available_layouts(cfg_layouts)
>>  957:         if not layouts:
>>  958:             print("Nenhum layout declarado no perfil institucional.")
>>  959:         else:
>>  960:             print("Layouts disponíveis:")
>>  961:             for layout_id, spec in layouts.items():
>>  962:                 desc = str(spec.get("description") or spec.get("descricao") or "").strip()
>>  963:                 genero = str(spec.get("genero_academico") or "").strip()
>>  964:                 print(f"- {layout_id}" + (f" ({genero})" if genero else "") + (f": {desc}" if desc else ""))
>>  965:             resolved = resolve_layout_spec(cfg_layouts)
>>  966:             print(f"Layout resolvido para este TOML: {resolved.id}")
>>  967:         return 0
    968:
    969:     if args.explain_profile:
```

### Candidato 7: linhas 969–971

- Tipo: `if`
- Condição: `args.explain_profile`
- Atributos: `args.explain_profile`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`

```python
    967:         return 0
    968:
>>  969:     if args.explain_profile:
>>  970:         print(explain_profile(args.explain_profile))
>>  971:         return 0
    972:
    973:     if args.show_prompts:
```

### Candidato 8: linhas 973–978

- Tipo: `if`
- Condição: `args.show_prompts`
- Atributos: `args.config`, `args.show_prompts`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`, `raise raise RuntimeError("--show-prompts exige --config caminho.toml")`

```python
    971:         return 0
    972:
>>  973:     if args.show_prompts:
>>  974:         if not args.config:
>>  975:             raise RuntimeError("--show-prompts exige --config caminho.toml")
>>  976:         cfg_preview = load_config(Path(args.config).expanduser().resolve())
>>  977:         print(json.dumps(prompt_report_for_cfg(cfg_preview), ensure_ascii=False, indent=2))
>>  978:         return 0
    979:
    980:     if args.init_project:
```

### Candidato 9: linhas 980–990

- Tipo: `if`
- Condição: `args.init_project`
- Atributos: `args.base_dir`, `args.init_project`, `args.institution`, `args.overwrite_project`, `args.project_type`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`

```python
    978:         return 0
    979:
>>  980:     if args.init_project:
>>  981:         base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else None
>>  982:         result = init_project(args.init_project, project_type=args.project_type, base_dir=base_dir, overwrite=bool(args.overwrite_project), institution=args.institution)
>>  983:         print("Projeto criado:")
>>  984:         print(f"- Diretório: {result.project_dir}")
>>  985:         print(f"- TOML: {result.config_path}")
>>  986:         print(f"- DOI manifest: {result.doi_manifest_path}")
>>  987:         print(f"- Documentos ZIP: {result.documentos_zip_path}")
>>  988:         print(f"- Orientações ZIP: {result.orientacoes_zip_path}")
>>  989:         print(f"- README: {result.readme_path}")
>>  990:         return 0
    991:
    992:     if args.make_doi_manifest:
```

### Candidato 10: linhas 992–1009

- Tipo: `if`
- Condição: `args.make_doi_manifest`
- Atributos: `args.input_dir`, `args.input_zip`, `args.make_doi_manifest`, `args.output`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`, `raise raise RuntimeError("Use --make-doi-manifest com --input-zip ou --input-dir.")`

```python
    990:         return 0
    991:
>>  992:     if args.make_doi_manifest:
>>  993:         input_zip = Path(args.input_zip).expanduser().resolve() if args.input_zip else None
>>  994:         input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
>>  995:         if args.output:
>>  996:             output = Path(args.output).expanduser().resolve()
>>  997:         else:
>>  998:             if input_zip:
>>  999:                 output = input_zip.parent / "doi_manifest.csv"
>> 1000:             elif input_dir:
>> 1001:                 output = input_dir / "doi_manifest.csv"
>> 1002:             else:
>> 1003:                 raise RuntimeError("Use --make-doi-manifest com --input-zip ou --input-dir.")
>> 1004:         result = make_doi_manifest(input_zip, input_dir, output, overwrite=True)
>> 1005:         print("DOI manifest gerado:")
>> 1006:         print(f"- Fonte: {result['source']}")
>> 1007:         print(f"- Saída: {result['output']}")
>> 1008:         print(f"- Arquivos listados: {result['total_files']}")
>> 1009:         return 0
   1010:
   1011:     if args.inspect_bib:
```

### Candidato 11: linhas 1011–1017

- Tipo: `if`
- Condição: `args.inspect_bib`
- Atributos: `args.inspect_bib`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0 if report.get("ok") else 1`

```python
   1009:         return 0
   1010:
>> 1011:     if args.inspect_bib:
>> 1012:         bib = Path(args.inspect_bib).expanduser().resolve()
>> 1013:         prefix = bib.with_name(bib.name + "_inspection")
>> 1014:         report = inspect_bib(bib, output_prefix=prefix)
>> 1015:         print(render_bib_inspection_markdown(report))
>> 1016:         print(f"Relatórios: {str(prefix)}.md e {str(prefix)}.json")
>> 1017:         return 0 if report.get("ok") else 1
   1018:
   1019:     if args.quality_report:
```

### Candidato 12: linhas 1019–1039

- Tipo: `if`
- Condição: `args.quality_report`
- Atributos: `args.bib`, `args.document_json`, `args.org`, `args.quality_report`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0 if report.get("ok") else 1`, `raise raise RuntimeError("--quality-report exige --document-json caminho/document.json")`

```python
   1017:         return 0 if report.get("ok") else 1
   1018:
>> 1019:     if args.quality_report:
>> 1020:         if not args.document_json:
>> 1021:             raise RuntimeError("--quality-report exige --document-json caminho/document.json")
>> 1022:         document_json = Path(args.document_json).expanduser().resolve()
>> 1023:         document = load_existing_document_json(document_json)
>> 1024:         org = Path(args.org).expanduser().resolve() if args.org else None
>> 1025:         bib_keys: list[str] = []
>> 1026:         if args.bib:
>> 1027:             # Compatibilidade temporária entre pacote e script direto.
>> 1028:             if __package__:
>> 1029:                 from .bibliography_manager import split_bib_entries, bib_entry_key
>> 1030:             else:
>> 1031:                 from bibliography_manager import split_bib_entries, bib_entry_key
>> 1032:             bib_path = Path(args.bib).expanduser().resolve()
>> 1033:             if bib_path.exists():
>> 1034:                 bib_keys = [k for e in split_bib_entries(bib_path.read_text(encoding='utf-8', errors='ignore')) if (k := bib_entry_key(e))]
>> 1035:         report = build_quality_report(document, org_path=org, bib_keys=bib_keys or list(document.bibliography.entries_used or []))
>> 1036:         out = document_json.with_suffix(".quality_report.md")
>> 1037:         write_quality_report(report, out)
>> 1038:         print(f"Relatório de qualidade: {out}")
>> 1039:         return 0 if report.get("ok") else 1
   1040:
   1041:     cfg = _load_optional_config(args.config) if args.config else None
```

### Candidato 13: linhas 1041–1041

- Tipo: `assignment`
- Condição: ``
- Atributos: `args.config`
- Fluxo terminal: não
- Retornos/lançamentos: nenhum

```python
   1039:         return 0 if report.get("ok") else 1
   1040:
>> 1041:     cfg = _load_optional_config(args.config) if args.config else None
   1042:     if cfg:
   1043:         cfg = apply_cli_path_overrides(cfg, args)
```

### Candidato 14: linhas 1045–1046

- Tipo: `if`
- Condição: `args.somente_renderizar and args.somente_mapa_mental`
- Atributos: `args.somente_mapa_mental`, `args.somente_renderizar`
- Fluxo terminal: sim
- Retornos/lançamentos: `raise raise RuntimeError("Use apenas um entre --somente-renderizar e --somente-mapa-mental.")`

```python
   1043:         cfg = apply_cli_path_overrides(cfg, args)
   1044:
>> 1045:     if args.somente_renderizar and args.somente_mapa_mental:
>> 1046:         raise RuntimeError("Use apenas um entre --somente-renderizar e --somente-mapa-mental.")
   1047:     if args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental:
   1048:         raise RuntimeError("Use apenas um entre --reusar-mapa-mental e --forcar-regeneracao-mapa-mental.")
```

### Candidato 15: linhas 1047–1048

- Tipo: `if`
- Condição: `args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental`
- Atributos: `args.forcar_regeneracao_mapa_mental`, `args.reusar_mapa_mental`
- Fluxo terminal: sim
- Retornos/lançamentos: `raise raise RuntimeError("Use apenas um entre --reusar-mapa-mental e --forcar-regeneracao-mapa-mental.")`

```python
   1045:     if args.somente_renderizar and args.somente_mapa_mental:
   1046:         raise RuntimeError("Use apenas um entre --somente-renderizar e --somente-mapa-mental.")
>> 1047:     if args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental:
>> 1048:         raise RuntimeError("Use apenas um entre --reusar-mapa-mental e --forcar-regeneracao-mapa-mental.")
   1049:
   1050:     if args.write_prompt_lock:
```

### Candidato 16: linhas 1050–1060

- Tipo: `if`
- Condição: `args.write_prompt_lock`
- Atributos: `args.write_prompt_lock`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`, `raise raise RuntimeError("--write-prompt-lock exige --config caminho.toml")`

```python
   1048:         raise RuntimeError("Use apenas um entre --reusar-mapa-mental e --forcar-regeneracao-mapa-mental.")
   1049:
>> 1050:     if args.write_prompt_lock:
>> 1051:         if not cfg:
>> 1052:             raise RuntimeError("--write-prompt-lock exige --config caminho.toml")
>> 1053:         out_dir, prefix = research_output_paths(cfg) if external_search_enabled(cfg) else output_paths(cfg)
>> 1054:         lock_path = out_dir / f"{prefix}.prompt_lock.json"
>> 1055:         lock_md = out_dir / f"{prefix}.prompt_lock.md"
>> 1056:         lock = write_prompt_lock(cfg, lock_path)
>> 1057:         write_prompt_lock_markdown(lock, lock_md)
>> 1058:         print(f"Prompt lock gerado: {lock_path}")
>> 1059:         print(f"Prompt lock markdown: {lock_md}")
>> 1060:         return 0
   1061:
   1062:     if args.check_institution_compliance:
```

### Candidato 17: linhas 1062–1074

- Tipo: `if`
- Condição: `args.check_institution_compliance`
- Atributos: `args.bib`, `args.check_institution_compliance`, `args.docx`, `args.org`, `args.pdf`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0 if report.get("ok") else 2`, `raise raise RuntimeError("--check-institution-compliance exige --config caminho.toml")`

```python
   1060:         return 0
   1061:
>> 1062:     if args.check_institution_compliance:
>> 1063:         if not cfg:
>> 1064:             raise RuntimeError("--check-institution-compliance exige --config caminho.toml")
>> 1065:         out_dir, prefix = output_paths(cfg)
>> 1066:         org = Path(args.org).expanduser().resolve() if args.org else out_dir / f"{prefix}.org"
>> 1067:         bib = Path(args.bib).expanduser().resolve() if args.bib else out_dir / f"{prefix}.bib"
>> 1068:         docx = Path(args.docx).expanduser().resolve() if args.docx else out_dir / f"{prefix}.docx"
>> 1069:         pdf = Path(args.pdf).expanduser().resolve() if args.pdf else out_dir / f"{prefix}.pdf"
>> 1070:         report = run_institution_compliance(cfg, org_path=org, bib_path=bib, docx_path=docx, pdf_path=pdf)
>> 1071:         md_path, json_path = write_compliance_reports(report, out_dir / prefix)
>> 1072:         print(render_compliance_markdown(report))
>> 1073:         print(f"Relatórios: {md_path} e {json_path}")
>> 1074:         return 0 if report.get("ok") else 2
   1075:
   1076:     if args.doctor:
```

### Candidato 18: linhas 1076–1082

- Tipo: `if`
- Condição: `args.doctor`
- Atributos: `args.doctor`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0 if report.get("ok") else 2`

```python
   1074:         return 0 if report.get("ok") else 2
   1075:
>> 1076:     if args.doctor:
>> 1077:         report = run_doctor(cfg)
>> 1078:         print_doctor_report(report)
>> 1079:         if cfg:
>> 1080:             out_dir, prefix = research_output_paths(cfg) if external_search_enabled(cfg) else output_paths(cfg)
>> 1081:             write_json(out_dir / f"{prefix}.doctor_report.json", report)
>> 1082:         return 0 if report.get("ok") else 2
   1083:
   1084:     if args.check_config:
```

### Candidato 19: linhas 1084–1091

- Tipo: `if`
- Condição: `args.check_config`
- Atributos: `args.check_config`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0 if report.get("ok") else 2`, `raise raise RuntimeError("--check-config exige --config caminho.toml")`

```python
   1082:         return 0 if report.get("ok") else 2
   1083:
>> 1084:     if args.check_config:
>> 1085:         if not cfg:
>> 1086:             raise RuntimeError("--check-config exige --config caminho.toml")
>> 1087:         report = check_config(cfg)
>> 1088:         print_check_config_report(report)
>> 1089:         out_dir, prefix = research_output_paths(cfg) if external_search_enabled(cfg) else output_paths(cfg)
>> 1090:         write_json(out_dir / f"{prefix}.check_config_report.json", report)
>> 1091:         return 0 if report.get("ok") else 2
   1092:
   1093:     if args.recompile:
```

### Candidato 20: linhas 1093–1094

- Tipo: `if`
- Condição: `args.recompile`
- Atributos: `args.recompile`
- Fluxo terminal: sim
- Retornos/lançamentos: `return run_recompile(args, cfg)`

```python
   1091:         return 0 if report.get("ok") else 2
   1092:
>> 1093:     if args.recompile:
>> 1094:         return run_recompile(args, cfg)
   1095:
   1096:     if not cfg:
```

### Candidato 21: linhas 1099–1139

- Tipo: `if`
- Condição: `args.prisma_importar_triagem`
- Atributos: `args.prisma_importar_triagem`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`, `raise raise RuntimeError("--prisma-importar-triagem exige um TOML do perfil relatorio_prisma_busca_orientada_fgv.")`

```python
   1097:         raise RuntimeError("Informe --config, ou use --doctor sem config.")
   1098:
>> 1099:     if args.prisma_importar_triagem:
>> 1100:         if not external_search_enabled(cfg):
>> 1101:             raise RuntimeError("--prisma-importar-triagem exige um TOML do perfil relatorio_prisma_busca_orientada_fgv.")
>> 1102:         out_dir, prefix = research_output_paths(cfg)
>> 1103:         stage("Importando planilha de triagem PRISMA preenchida")
>> 1104:         prisma_outputs = import_manual_prisma_triage(cfg, out_dir, prefix, Path(args.prisma_importar_triagem))
>> 1105:         org_path, pdf_path = render_external_prisma_outputs(
>> 1106:             cfg,
>> 1107:             out_dir,
>> 1108:             prefix,
>> 1109:             prisma_outputs,
>> 1110:             phase="final",
>> 1111:         )
>> 1112:         artifacts = prisma_outputs.setdefault("artefatos", {}) if isinstance(prisma_outputs, dict) else {}
>> 1113:         if org_path:
>> 1114:             artifacts["relatorio_org"] = str(org_path)
>> 1115:         if pdf_path:
>> 1116:             artifacts["relatorio_pdf"] = str(pdf_path)
>> 1117:         report_json_path = artifacts.get("prisma_report_json") if isinstance(artifacts, dict) else ""
>> 1118:         if report_json_path:
>> 1119:             write_json(Path(str(report_json_path)), prisma_outputs)
>> 1120:         outputs = {
>> 1121:             "output_dir": str(out_dir),
>> 1122:             "org": str(org_path) if org_path else None,
>> 1123:             "pdf": str(pdf_path) if pdf_path else None,
>> 1124:             "relatorio_pesquisa": prisma_outputs,
>> 1125:         }
>> 1126:         report = make_run_report(
>> 1127:             cfg=cfg,
>> 1128:             config_path=Path(str(cfg.get("__config_path__"))),
>> 1129:             out_dir=out_dir,
>> 1130:             prefix=prefix,
>> 1131:             model=None,
>> 1132:             outputs=outputs,
>> 1133:             warnings=[],
>> 1134:             extra={"mode": "prisma_importar_triagem"},
>> 1135:         )
>> 1136:         write_json(out_dir / f"{prefix}.run_report.json", report)
>> 1137:         write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
>> 1138:         print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — triagem PRISMA consolidada")
>> 1139:         return 0
   1140:
   1141:     cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
```

### Candidato 22: linhas 1141–1141

- Tipo: `assignment`
- Condição: ``
- Atributos: `args.somente_renderizar`
- Fluxo terminal: não
- Retornos/lançamentos: nenhum

```python
   1139:         return 0
   1140:
>> 1141:     cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
   1142:     is_external_prisma_run = external_search_enabled(cfg) and not args.somente_renderizar
   1143:     out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
```

### Candidato 23: linhas 1142–1142

- Tipo: `assignment`
- Condição: ``
- Atributos: `args.somente_renderizar`
- Fluxo terminal: não
- Retornos/lançamentos: nenhum

```python
   1140:
   1141:     cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
>> 1142:     is_external_prisma_run = external_search_enabled(cfg) and not args.somente_renderizar
   1143:     out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
   1144:     work_dir, cache_dir = work_cache_paths(cfg, prefix)
```

### Candidato 24: linhas 1160–1160

- Tipo: `assignment`
- Condição: ``
- Atributos: `args.document_json`
- Fluxo terminal: não
- Retornos/lançamentos: nenhum

```python
   1158:         raise RuntimeError("Configuração inválida:\n- " + "\n- ".join(precheck["errors"]))
   1159:
>> 1160:     document_json_path = Path(args.document_json).expanduser().resolve() if args.document_json else out_dir / f"{prefix}.document.json"
   1161:
   1162:     if is_external_prisma_run:
```

### Candidato 25: linhas 1162–1224

- Tipo: `if`
- Condição: `is_external_prisma_run`
- Atributos: `args.somente_mapa_mental`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`, `raise raise RuntimeError("O perfil de busca PRISMA não produz document.json; use a geração normal ou --prisma-importar-triagem.")`

```python
   1160:     document_json_path = Path(args.document_json).expanduser().resolve() if args.document_json else out_dir / f"{prefix}.document.json"
   1161:
>> 1162:     if is_external_prisma_run:
>> 1163:         if args.somente_mapa_mental:
>> 1164:             raise RuntimeError("O perfil de busca PRISMA não produz document.json; use a geração normal ou --prisma-importar-triagem.")
>> 1165:         search_cfg = cfg.get("busca_prisma", {}) if isinstance(cfg.get("busca_prisma"), dict) else {}
>> 1166:         if bool(search_cfg.get("pre_triagem_ia", False)):
>> 1167:             stage("Inicializando cliente OpenAI para pré-triagem assistida")
>> 1168:             client, model = make_client(model)
>> 1169:         stage("Executando busca bibliográfica externa e preparando triagem humana")
>> 1170:         prisma_outputs = run_external_prisma_search(
>> 1171:             cfg,
>> 1172:             out_dir,
>> 1173:             prefix,
>> 1174:             progress=stage,
>> 1175:             client=client,
>> 1176:             model=model,
>> 1177:         )
>> 1178:         org_path, pdf_path = render_external_prisma_outputs(
>> 1179:             cfg,
>> 1180:             out_dir,
>> 1181:             prefix,
>> 1182:             prisma_outputs,
>> 1183:             phase="preliminar",
>> 1184:         )
>> 1185:         artifacts = prisma_outputs.setdefault("artefatos", {}) if isinstance(prisma_outputs, dict) else {}
>> 1186:         if org_path:
>> 1187:             artifacts["relatorio_org"] = str(org_path)
>> 1188:         if pdf_path:
>> 1189:             artifacts["relatorio_pdf"] = str(pdf_path)
>> 1190:         report_json_path = artifacts.get("prisma_report_json") if isinstance(artifacts, dict) else ""
>> 1191:         if report_json_path:
>> 1192:             write_json(Path(str(report_json_path)), prisma_outputs)
>> 1193:         prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
>> 1194:         prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
>> 1195:         stage("Registrando prompt_lock")
>> 1196:         prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
>> 1197:         write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
>> 1198:         outputs = {
>> 1199:             "output_dir": str(out_dir),
>> 1200:             "work_dir": str(work_dir),
>> 1201:             "cache_dir": str(cache_dir),
>> 1202:             "document_json": None,
>> 1203:             "org": str(org_path) if org_path else None,
>> 1204:             "bib": None,
>> 1205:             "pdf": str(pdf_path) if pdf_path else None,
>> 1206:             "docx": None,
>> 1207:             "relatorio_pesquisa": prisma_outputs,
>> 1208:             "prompt_lock": str(prompt_lock_path),
>> 1209:         }
>> 1210:         report = make_run_report(
>> 1211:             cfg=cfg,
>> 1212:             config_path=Path(str(cfg.get("__config_path__"))),
>> 1213:             out_dir=out_dir,
>> 1214:             prefix=prefix,
>> 1215:             model=None,
>> 1216:             outputs=outputs,
>> 1217:             warnings=warnings,
>> 1218:             extra={"mode": "prisma_busca_externa", "precheck": precheck},
>> 1219:         )
>> 1220:         write_json(out_dir / f"{prefix}.run_report.json", report)
>> 1221:         write_json(out_dir / f"{prefix}.rc10_report.json", outputs)
>> 1222:         write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
>> 1223:         print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — busca PRISMA concluída; aguarda triagem humana")
>> 1224:         return 0
   1225:
   1226:     if args.somente_mapa_mental:
```

### Candidato 26: linhas 1226–1279

- Tipo: `if`
- Condição: `args.somente_mapa_mental`
- Atributos: `args.forcar_regeneracao_mapa_mental`, `args.reusar_mapa_mental`, `args.somente_mapa_mental`
- Fluxo terminal: sim
- Retornos/lançamentos: `return 0`, `raise raise FileNotFoundError(f"document.json não encontrado para --somente-mapa-mental: {document_json_path}")`, `raise raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --somente-mapa-mental.")`

```python
   1224:         return 0
   1225:
>> 1226:     if args.somente_mapa_mental:
>> 1227:         if not document_json_path.exists():
>> 1228:             raise FileNotFoundError(f"document.json não encontrado para --somente-mapa-mental: {document_json_path}")
>> 1229:         if not should_generate_mindmap(cfg):
>> 1230:             raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --somente-mapa-mental.")
>> 1231:         stage("Carregando document.json existente")
>> 1232:         document = load_existing_document_json(document_json_path)
>> 1233:         removed_mindmap_files: list[str] = []
>> 1234:         if args.forcar_regeneracao_mapa_mental:
>> 1235:             stage("Removendo mapa mental existente")
>> 1236:             removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
>> 1237:         mm_diag = None
>> 1238:         if args.reusar_mapa_mental:
>> 1239:             stage("Tentando reutilizar mapa mental existente")
>> 1240:             mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
>> 1241:             if not mm_diag:
>> 1242:                 warnings.append("Mapa mental existente não encontrado; gerando novo mapa mental.")
>> 1243:         if not mm_diag:
>> 1244:             stage("Inicializando cliente OpenAI")
>> 1245:             client, model = make_client(model)
>> 1246:             stage("Gerando/renderizando apenas o mapa mental")
>> 1247:             mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
>> 1248:         if removed_mindmap_files:
>> 1249:             mm_diag = dict(mm_diag or {})
>> 1250:             mm_diag["removed_before_regeneration"] = removed_mindmap_files
>> 1251:         document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
>> 1252:         stage("Salvando document.json atualizado")
>> 1253:         write_json(document_json_path, document.model_dump())
>> 1254:         outputs = {
>> 1255:             "output_dir": str(out_dir),
>> 1256:             "document_json": str(document_json_path),
>> 1257:             "mindmap_puml": (mm_diag or {}).get("puml_path") if mm_diag else None,
>> 1258:             "mindmap_image": (mm_diag or {}).get("image_path") if mm_diag else None,
>> 1259:             "mindmap_reused": bool((mm_diag or {}).get("reused")),
>> 1260:             "mindmap_removed": removed_mindmap_files,
>> 1261:         }
>> 1262:         report = make_run_report(
>> 1263:             cfg=cfg,
>> 1264:             config_path=Path(str(cfg.get("__config_path__"))),
>> 1265:             out_dir=out_dir,
>> 1266:             prefix=prefix,
>> 1267:             model=model,
>> 1268:             outputs=outputs,
>> 1269:             warnings=warnings,
>> 1270:             extra={"mode": "somente_mapa_mental"},
>> 1271:         )
>> 1272:         write_json(out_dir / f"{prefix}.run_report.json", report)
>> 1273:         write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
>> 1274:         print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — mapa mental renderizado")
>> 1275:         if warnings:
>> 1276:             print("Avisos:")
>> 1277:             for w in warnings:
>> 1278:                 print(f"- {w}")
>> 1279:         return 0
   1280:
   1281:     prisma_outputs = None
```

### Candidato 27: linhas 1286–1439

- Tipo: `if`
- Condição: `args.somente_renderizar`
- Atributos: `args.forcar_regeneracao_mapa_mental`, `args.reusar_mapa_mental`, `args.somente_renderizar`
- Fluxo terminal: sim
- Retornos/lançamentos: `raise raise FileNotFoundError(f"document.json não encontrado para --somente-renderizar: {document_json_path}")`, `return 0`, `raise raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --forcar-regeneracao-mapa-mental.")`, `raise raise RuntimeError("Falha ao gerar o resumo acadêmico do paper: " + str(exc)) from exc`

```python
   1284:     paper_abstract_path = abstract_sidecar_path(out_dir, prefix)
   1285:
>> 1286:     if args.somente_renderizar:
>> 1287:         if not document_json_path.exists():
>> 1288:             raise FileNotFoundError(f"document.json não encontrado para --somente-renderizar: {document_json_path}")
>> 1289:         stage("Carregando document.json existente")
>> 1290:         document = load_existing_document_json(document_json_path)
>> 1291:         stage("Resolvendo bibliografia para renderização")
>> 1292:         bib_path, bib_keys = resolve_bib_for_existing_document(document, document_json_path, out_dir, prefix)
>> 1293:         stage("Saneando document_model existente")
>> 1294:         document, leak_repairs = sanitize_document_model_technical_leaks(document)
>> 1295:         if leak_repairs:
>> 1296:             warnings.append("Menções técnicas removidas/reescritas no document_model existente: " + ", ".join(leak_repairs[:20]))
>> 1297:         document, raw_key_repairs = sanitize_document_model_raw_bibkeys(document, bib_keys)
>> 1298:         if raw_key_repairs:
>> 1299:             warnings.append("Chaves BibTeX cruas convertidas em citações LaTeX no document_model existente: " + ", ".join(raw_key_repairs[:20]))
>> 1300:         if paper_abstracts_enabled(cfg):
>> 1301:             if paper_abstract_path.exists():
>> 1302:                 paper_abstract_bundle = read_paper_abstract_bundle(paper_abstract_path)
>> 1303:             else:
>> 1304:                 warnings.append(
>> 1305:                     "RESUMO: arquivo de resumos não encontrado no modo --somente-renderizar; "
>> 1306:                     "o ORG/DOCX será recompilado sem inserir resumo. Execute uma geração completa para criá-lo."
>> 1307:                 )
>> 1308:         if args.forcar_regeneracao_mapa_mental:
>> 1309:             if not should_generate_mindmap(cfg):
>> 1310:                 raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --forcar-regeneracao-mapa-mental.")
>> 1311:             stage("Removendo mapa mental existente")
>> 1312:             removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
>> 1313:             if removed_mindmap_files:
>> 1314:                 warnings.append("Arquivos de mapa mental removidos antes da regeneração: " + ", ".join(removed_mindmap_files[:10]))
>> 1315:             stage("Inicializando cliente OpenAI para regenerar mapa mental")
>> 1316:             client, model = make_client(model)
>> 1317:             stage("Regenerando mapa mental antes da renderização")
>> 1318:             mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
>> 1319:             document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
>> 1320:             stage("Salvando document.json atualizado com novo mapa mental")
>> 1321:             write_json(document_json_path, document.model_dump())
>> 1322:         elif args.reusar_mapa_mental:
>> 1323:             stage("Tentando reutilizar mapa mental existente")
>> 1324:             mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
>> 1325:             if mm_diag:
>> 1326:                 document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
>> 1327:                 write_json(document_json_path, document.model_dump())
>> 1328:             else:
>> 1329:                 warnings.append("--reusar-mapa-mental informado, mas nenhum mapa existente foi encontrado; renderização seguirá com figuras já registradas no document.json.")
>> 1330:     else:
>> 1331:         pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline"), dict) else {}
>> 1332:         executar_documento = bool(pipeline_cfg.get("executar_documento", True))
>> 1333:         stage("Inicializando cliente OpenAI")
>> 1334:         client, model = make_client(model)
>> 1335:         stage("Descobrindo e extraindo documentos locais")
>> 1336:         docs, source_info = discover_local_documents(cfg, work_dir)
>> 1337:         clean_cache = bool((cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}).get("limpar_cache_anterior", True))
>> 1338:         stage("Copiando documentos para fulltext_cache")
>> 1339:         copy_documents_to_fulltext_cache(docs, cache_dir, clean=clean_cache)
>> 1340:         stage("Carregando orientações do projeto")
>> 1341:         orientations = collect_orientation_docs(cfg, work_dir)
>> 1342:         stage("Gerando e validando bibliografia")
>> 1343:         bib_result = build_bibliography(cfg, docs, out_dir, prefix, client, model)
>> 1344:         bib_path = bib_result.bib_path
>> 1345:         bib_keys = bib_result.keys
>> 1346:         stage("Verificando geração de relatório PRISMA")
>> 1347:         prisma_outputs = run_prisma_report_outputs(cfg, docs, orientations, bib_result, out_dir, prefix) if prisma_enabled(cfg) else None
>> 1348:
>> 1349:         if not executar_documento:
>> 1350:             outputs = {
>> 1351:                 "output_dir": str(out_dir),
>> 1352:         "work_dir": str(work_dir),
>> 1353:         "cache_dir": str(cache_dir),
>> 1354:                 "work_dir": str(work_dir),
>> 1355:                 "cache_dir": str(cache_dir),
>> 1356:                 "document_json": None,
>> 1357:                 "org": None,
>> 1358:                 "bib": str(bib_path),
>> 1359:                 "pdf": None,
>> 1360:                 "docx": None,
>> 1361:                 "relatorio_pesquisa": prisma_outputs,
>> 1362:             }
>> 1363:             prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
>> 1364:             prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
>> 1365:             stage("Registrando prompt_lock")
>> 1366:             prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
>> 1367:             write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
>> 1368:             outputs["prompt_lock"] = str(prompt_lock_path)
>> 1369:             report = make_run_report(
>> 1370:                 cfg=cfg,
>> 1371:                 config_path=Path(str(cfg.get("__config_path__"))),
>> 1372:                 out_dir=out_dir,
>> 1373:                 prefix=prefix,
>> 1374:                 model=model,
>> 1375:                 outputs=outputs,
>> 1376:                 warnings=warnings,
>> 1377:                 extra={"mode": "research_only", "source_info": source_info, "work_dir": str(work_dir), "cache_dir": str(cache_dir)},
>> 1378:             )
>> 1379:             write_json(out_dir / f"{prefix}.run_report.json", report)
>> 1380:             write_json(out_dir / f"{prefix}.rc10_report.json", outputs)  # compatibilidade
>> 1381:             write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
>> 1382:             print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} concluído sem documento acadêmico")
>> 1383:             return 0
>> 1384:
>> 1385:         resumo_cfg_for_stage = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
>> 1386:         if resumo_cfg_for_stage.get("ativo") and bool(resumo_cfg_for_stage.get("geracao_em_etapas", True)):
>> 1387:             stage("Gerando document.json canônico com IA em etapas")
>> 1388:         else:
>> 1389:             stage("Gerando document.json canônico com IA")
>> 1390:         document = build_document_model(
>> 1391:             client,
>> 1392:             model,
>> 1393:             cfg,
>> 1394:             docs,
>> 1395:             orientations,
>> 1396:             bib_keys,
>> 1397:             bib_path,
>> 1398:             progress=stage,
>> 1399:             checkpoint_dir=out_dir,
>> 1400:             prefix=prefix,
>> 1401:         )
>> 1402:         if not document.bibliography.entries_used:
>> 1403:             document.bibliography.entries_used = bib_keys
>> 1404:         if not document.bibliography.bib_path:
>> 1405:             document.bibliography.bib_path = bib_path.name
>> 1406:         stage("Saneando linguagem técnica do document_model")
>> 1407:         document, leak_repairs = sanitize_document_model_technical_leaks(document)
>> 1408:         if leak_repairs:
>> 1409:             warnings.append("Menções técnicas removidas/reescritas no document_model: " + ", ".join(leak_repairs[:20]))
>> 1410:         document, raw_key_repairs = sanitize_document_model_raw_bibkeys(document, bib_keys)
>> 1411:         if raw_key_repairs:
>> 1412:             warnings.append("Chaves BibTeX cruas convertidas em citações LaTeX no document_model: " + ", ".join(raw_key_repairs[:20]))
>> 1413:         stage("Validando document_model")
>> 1414:         raise_if_errors(validate_document_model(document, bib_keys), "Validação do document_model falhou")
>> 1415:         if paper_abstracts_enabled(cfg):
>> 1416:             stage("Gerando resumo e palavras-chave do paper")
>> 1417:             try:
>> 1418:                 paper_abstract_bundle = generate_paper_abstract_bundle(client, model, document, cfg)
>> 1419:                 write_paper_abstract_bundle(paper_abstract_path, paper_abstract_bundle)
>> 1420:             except PaperAbstractError as exc:
>> 1421:                 raise RuntimeError("Falha ao gerar o resumo acadêmico do paper: " + str(exc)) from exc
>> 1422:         stage("Gerando/anexando mapa mental, se configurado")
>> 1423:         mm_diag = None
>> 1424:         if args.forcar_regeneracao_mapa_mental:
>> 1425:             removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
>> 1426:             if removed_mindmap_files:
>> 1427:                 warnings.append("Arquivos de mapa mental removidos antes da regeneração: " + ", ".join(removed_mindmap_files[:10]))
>> 1428:         if args.reusar_mapa_mental:
>> 1429:             mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
>> 1430:             if not mm_diag:
>> 1431:                 warnings.append("Mapa mental existente não encontrado; gerando novo mapa mental.")
>> 1432:         if not mm_diag:
>> 1433:             mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
>> 1434:         document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
>> 1435:         document.diagnostics.source_info_json = json.dumps(source_info or {}, ensure_ascii=False)
>> 1436:         if prisma_outputs:
>> 1437:             document.diagnostics.relatorio_pesquisa_json = json.dumps(prisma_outputs, ensure_ascii=False)
>> 1438:         stage("Salvando document.json")
>> 1439:         write_json(document_json_path, document.model_dump())
   1440:
   1441:     stage("Renderizando ORG/LaTeX")
```

### Candidato 28: linhas 1473–1499

- Tipo: `if`
- Condição: `args.somente_renderizar`
- Atributos: `args.somente_renderizar`
- Fluxo terminal: não
- Retornos/lançamentos: nenhum

```python
   1471:
   1472:     translated_outputs: dict[str, Any] = {}
>> 1473:     if args.somente_renderizar:
>> 1474:         if requested_translation_languages(cfg):
>> 1475:             warnings.append(
>> 1476:                 "Versões adicionais por IA não foram atualizadas no modo --somente-renderizar. "
>> 1477:                 "Execute a geração completa para traduzir o document.json canônico."
>> 1478:             )
>> 1479:     elif requested_translation_languages(cfg):
>> 1480:         try:
>> 1481:             translated_outputs, translation_warnings = render_additional_language_versions(
>> 1482:                 client=client,
>> 1483:                 model=model,
>> 1484:                 cfg=cfg,
>> 1485:                 document=document,
>> 1486:                 bib_path=bib_path,
>> 1487:                 bib_keys=bib_keys,
>> 1488:                 out_dir=out_dir,
>> 1489:                 prefix=prefix,
>> 1490:                 doc_cfg=doc_cfg,
>> 1491:                 latex_cfg=latex_cfg,
>> 1492:                 config_dir=config_dir,
>> 1493:                 abstract_bundle=paper_abstract_bundle or None,
>> 1494:             )
>> 1495:             warnings.extend(translation_warnings)
>> 1496:         except TranslationError as exc:
>> 1497:             # Traduções são saídas opcionais: uma falha nelas não invalida o
>> 1498:             # paper principal que já foi gerado e validado.
>> 1499:             warnings.append(f"TRADUÇÃO: {exc}")
   1500:
   1501:     outputs = {
```

### Candidato 29: linhas 1544–1559

- Tipo: `assignment`
- Condição: ``
- Atributos: `args.somente_renderizar`
- Fluxo terminal: não
- Retornos/lançamentos: nenhum

```python
   1542:     outputs["quality_report"] = str(quality_path)
   1543:
>> 1544:     report = make_run_report(
>> 1545:         cfg=cfg,
>> 1546:         config_path=Path(str(cfg.get("__config_path__"))),
>> 1547:         out_dir=out_dir,
>> 1548:         prefix=prefix,
>> 1549:         model=model,
>> 1550:         outputs=outputs,
>> 1551:         warnings=warnings,
>> 1552:         extra={
>> 1553:             "mode": "somente_renderizar" if args.somente_renderizar else "full",
>> 1554:             "work_dir": str(work_dir),
>> 1555:             "cache_dir": str(cache_dir),
>> 1556:             "precheck": precheck,
>> 1557:             "docx_validation": docx_validation,
>> 1558:         },
>> 1559:     )
   1560:     write_json(out_dir / f"{prefix}.run_report.json", report)
   1561:     write_json(out_dir / f"{prefix}.rc10_report.json", outputs)  # compatibilidade com scripts antigos
```

## Regras para a extração AP-003C

- Não mover ainda a orquestração documental.
- Não mover ainda PRISMA ou artigo genérico.
- Não alterar o parser extraído na AP-003B.
- Preservar os dois `main()` e o alias histórico.
- Separar comandos terminais de ajustes não terminais.
- Manter ordem e precedência do despacho atual.
- Testar chamada direta, módulo e comando instalável.
- Executar a suíte consolidada somente em `app_bundle/tests tests`.

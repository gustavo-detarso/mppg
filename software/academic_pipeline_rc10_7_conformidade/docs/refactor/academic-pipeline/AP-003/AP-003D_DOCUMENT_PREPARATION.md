# AP-003D — preparação da orquestração documental

> Inventário AST somente leitura. Nenhum módulo produtivo foi alterado.

## Baseline

- Branch: `ap-refactor/03-orchestrator-decomposition`
- HEAD: `148e067ebbfd1cbdc7e72e2a4d5189893c06f8b7`
- Upstream: `origin/ap-refactor/03-orchestrator-decomposition`
- HEAD remoto: `148e067ebbfd1cbdc7e72e2a4d5189893c06f8b7`
- Orquestrador: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`
- SHA-256: `4261568e60308764ef1f56ab1e13d6ccfd886d76dce965ec6d3e8fd66cdee51d`
- Primeiro `main()`: linhas 899–1578
- Instruções de nível superior no primeiro `main()`: **124**
- Definições `main()` preservadas: **2**
- Alias histórico: `_original_main_before_prisma_artigo_generico_wrapper`

## Resumo

- Helpers de nível superior: **48**
- Helpers documentais candidatos: **14**
- Helpers de sobreposição com PRISMA: **27**
- Instruções documentais candidatas no primeiro `main()`: **16**
- Instruções de sobreposição: **7**
- Faixas contíguas candidatas: **12**

## Helpers documentais candidatos

| Função | Linhas | Chamada pelo primeiro main | Score | Motivo |
|---|---:|:---:|---:|---|
| `load_config` | 1648–1649 | não | 6 | helpers documentais explícitos: _refs_v6_apply_runtime_policy; termos documentais: refs |
| `output_paths` | 308–317 | sim | 5 | helper documental explícito: output_paths; termos documentais: document, documento, output |
| `apply_cli_path_overrides` | 353–386 | sim | 3 | termos documentais: document, documento, output |
| `load_existing_document_json` | 389–390 | sim | 5 | helper documental explícito: load_existing_document_json; termos documentais: document |
| `resolve_bib_for_existing_document` | 393–420 | sim | 5 | helper documental explícito: resolve_bib_for_existing_document; termos documentais: bib, bibliography, document, output, render |
| `_resolve_latex_paths_for_recompile` | 436–442 | não | 5 | helper documental explícito: _resolve_latex_paths_for_recompile; termos documentais: latex, org, pdf, recompile |
| `run_recompile` | 445–471 | não | 10 | helper documental explícito: run_recompile; helpers documentais explícitos: _resolve_latex_paths_for_recompile; termos documentais: latex, org, output, pdf, recompile |
| `render_additional_language_versions` | 535–648 | sim | 20 | helper documental explícito: render_additional_language_versions; helpers documentais explícitos: render_org_latex; termos documentais: abstract, bib, bibliografia, bibliography, document, docx, idioma, language, latex, org, output, pdf, quality, reference, render |
| `_refs_v6_disabled` | 1587–1602 | não | 7 | helper documental explícito: _refs_v6_disabled; termos documentais: bib, bibliografia, bibliography, document, documento, latex, refs |
| `_refs_v6_apply_runtime_policy` | 1605–1643 | não | 13 | helper documental explícito: _refs_v6_apply_runtime_policy; helpers documentais explícitos: _refs_v6_disabled; termos documentais: bib, bibliografia, bibliography, document, documento, latex, pdf, refs |
| `build_bibliography` | 1655–1666 | sim | 8 | helper documental explícito: build_bibliography; helpers documentais explícitos: _refs_v6_disabled; termos documentais: bib, bibliography, refs |
| `_refs_v6_clear_document_bibliography` | 1669–1680 | não | 5 | helper documental explícito: _refs_v6_clear_document_bibliography; termos documentais: bib, bibliography, document, refs |
| `_refs_v6_strip_org` | 1683–1711 | não | 8 | helper documental explícito: _refs_v6_strip_org; termos documentais: bib, bibliografia, bibliography, document, latex, org, refs, render |
| `render_org_latex` | 1717–1743 | sim | 22 | helper documental explícito: render_org_latex; helpers documentais explícitos: _refs_v6_clear_document_bibliography, _refs_v6_disabled, _refs_v6_strip_org; termos documentais: bib, bibliography, document, latex, org, refs, render |

## Helpers de sobreposição mantidos fora da AP-003D

| Função | Linhas | Motivo |
|---|---:|---|
| `research_output_paths` | 320–337 | helper de sobreposição: research_output_paths; termos documentais: bib, document, documento, output; excluído por PRISMA/artigo genérico: prisma |
| `render_external_prisma_outputs` | 475–532 | helper de sobreposição: render_external_prisma_outputs; termos documentais: document, latex, org, output, pdf, render; excluído por PRISMA/artigo genérico: prisma, triagem |
| `_prisma_curadoria_default_config` | 653–654 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_default_out_dir` | 657–658 | termos documentais: output; excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_default_prompt` | 661–662 | termos documentais: document, documento; excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_script_path` | 665–666 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_arg` | 669–670 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_config_from_args` | 673–678 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_out_from_args` | 681–682 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_prompt_from_args` | 685–686 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_input_from_args` | 689–696 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_run_command` | 699–710 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_build_cmd` | 713–759 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_run_ia` | 762–764 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_reexportar_xlsx` | 767–769 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_pipeline_supports_flag` | 772–786 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_importar_no_pipeline` | 789–809 | excluído por PRISMA/artigo genérico: curadoria, prisma, triagem |
| `_prisma_curadoria_fluxo_completo` | 812–819 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `_prisma_curadoria_mostrar_caminhos` | 822–841 | termos documentais: output; excluído por PRISMA/artigo genérico: curadoria, prisma, triagem |
| `_prisma_curadoria_menu` | 844–880 | excluído por PRISMA/artigo genérico: curadoria, prisma, triagem |
| `_prisma_curadoria_dispatch` | 883–895 | excluído por PRISMA/artigo genérico: curadoria, prisma |
| `main` | 1837–1859 | termos documentais: bib; excluído por PRISMA/artigo genérico: artigo_generico, curadoria, prisma |
| `_prisma_artigo_generico_get_arg` | 1756–1762 | excluído por PRISMA/artigo genérico: artigo_generico, prisma |
| `_prisma_artigo_generico_strip` | 1764–1777 | termos documentais: bib, output; excluído por PRISMA/artigo genérico: artigo_generico, prisma |
| `_prisma_artigo_generico_out_dir` | 1779–1787 | termos documentais: output; excluído por PRISMA/artigo genérico: artigo_generico, curadoria, prisma |
| `_prisma_artigo_generico_run_export` | 1789–1808 | termos documentais: bib, latex, output; excluído por PRISMA/artigo genérico: artigo_generico, prisma |
| `_prisma_artigo_generico_run_freeze` | 1810–1833 | termos documentais: output; excluído por PRISMA/artigo genérico: artigo_generico, prisma |

## Faixas documentais candidatas no primeiro main

| Faixa AST | Linhas | Instruções |
|---|---:|---:|
| 36–36 | 1035–1055 | 1 |
| 68–68 | 1155–1155 | 1 |
| 80–80 | 1236–1289 | 1 |
| 86–86 | 1451–1451 | 1 |
| 88–91 | 1453–1458 | 4 |
| 93–93 | 1461–1466 | 1 |
| 96–96 | 1470–1480 | 1 |
| 98–98 | 1483–1509 | 1 |
| 106–106 | 1532–1538 | 1 |
| 108–108 | 1540–1540 | 1 |
| 112–112 | 1547–1547 | 1 |
| 116–117 | 1552–1569 | 2 |

## Instruções documentais candidatas

| AST | Linhas | Tipo | Condição | Score | Carrega | Grava |
|---:|---:|---|---|---:|---|---|
| 36 | 1035–1055 | `if` | `args.quality_report` | 10 | `Path`, `args`, `bib_entry_key`, `bib_keys`, `bib_path`, `build_quality_report`, `document`, `document_json`, `e`, `k`, `load_existing_document_json`, `org`, `out`, `report`, `split_bib_entries`, `write_quality_report` | `bib_keys`, `bib_path`, `document`, `document_json`, `e`, `k`, `org`, `out`, `report` |
| 68 | 1155–1155 | `assignment` | `` | 2 | `cfg` | `doc_cfg` |
| 80 | 1236–1289 | `if` | `args.somente_mapa_mental` | 10 | `PIPELINE_VERSION`, `Path`, `args`, `attach_existing_mindmap_if_available`, `cfg`, `client`, `delete_existing_mindmap_outputs`, `document`, `document_json_path`, `generate_and_attach_mindmap`, `json`, `load_existing_document_json`, `make_client`, `make_run_report`, `mm_diag`, `model`, `out_dir`, `outputs`, `prefix`, `print_outputs`, `removed_mindmap_files`, `report`, `should_generate_mindmap`, `stage`, `w`, `warnings`, `write_json`, `write_outputs_manifest` | `client`, `document`, `mm_diag`, `model`, `outputs`, `removed_mindmap_files`, `report`, `w` |
| 86 | 1451–1451 | `expression` | `` | 3 | `stage` |  |
| 88 | 1453–1453 | `assignment` | `` | 10 | `bib_keys`, `bib_path`, `cfg`, `document`, `org_path`, `prefix`, `render_org_latex` | `org_text` |
| 89 | 1454–1456 | `if` | `paper_abstract_bundle` | 4 | `cfg`, `inject_paper_abstracts_into_org`, `main_document_abstract_languages`, `org_path`, `paper_abstract_bundle`, `stage` | `org_text` |
| 90 | 1457–1457 | `expression` | `` | 2 | `stage` |  |
| 91 | 1458–1458 | `expression` | `` | 3 | `bib_keys`, `org_text`, `raise_if_errors`, `validate_org_text` |  |
| 93 | 1461–1466 | `if` | `bool(doc_cfg.get("exportar_pdf", True))` | 3 | `academic_writing`, `config_dir`, `doc_cfg`, `latex_cfg`, `latex_extra`, `org_path`, `pdf_engine`, `resolve_path`, `run_compile_sequence`, `stage` | `academic_writing`, `latex_extra`, `pdf_engine`, `pdf_path` |
| 96 | 1470–1480 | `if` | `bool(doc_cfg.get("exportar_docx", True))` | 8 | `bib_path`, `cfg`, `config_dir`, `doc_cfg`, `document`, `docx_cfg`, `docx_path`, `docx_validation`, `inject_paper_abstracts_into_docx`, `main_document_abstract_languages`, `out_dir`, `paper_abstract_bundle`, `prefix`, `ref`, `render_docx`, `resolve_path`, `stage`, `validate_docx_file`, `w`, `warnings` | `docx_cfg`, `docx_path`, `docx_validation`, `ref`, `w` |
| 98 | 1483–1509 | `if` | `args.somente_renderizar` | 12 | `TranslationError`, `args`, `bib_keys`, `bib_path`, `cfg`, `client`, `config_dir`, `doc_cfg`, `document`, `exc`, `latex_cfg`, `model`, `out_dir`, `paper_abstract_bundle`, `prefix`, `render_additional_language_versions`, `requested_translation_languages`, `translation_warnings`, `warnings` | `translated_outputs`, `translation_warnings` |
| 106 | 1532–1538 | `assignment` | `` | 5 | `bib_path`, `cfg`, `docx_path`, `org_path`, `pdf_path`, `run_institution_compliance` | `compliance_report` |
| 108 | 1540–1540 | `assignment` | `` | 2 | `compliance_md`, `outputs` |  |
| 112 | 1547–1547 | `assignment` | `` | 4 | `bib_keys`, `build_quality_report`, `document`, `org_path` | `quality` |
| 116 | 1552–1552 | `assignment` | `` | 2 | `outputs`, `quality_path` |  |
| 117 | 1554–1569 | `assignment` | `` | 3 | `Path`, `args`, `cache_dir`, `cfg`, `docx_validation`, `make_run_report`, `model`, `out_dir`, `outputs`, `precheck`, `prefix`, `warnings`, `work_dir` | `report` |

## Trechos candidatos

### Candidato 1: AST 36, linhas 1035–1055

- Motivos: helpers documentais explícitos: load_existing_document_json; termos documentais: bib, bibliography, document, org, quality
- Chamadas: `Path`, `Path(args.bib).expanduser`, `Path(args.bib).expanduser().resolve`, `Path(args.document_json).expanduser`, `Path(args.document_json).expanduser().resolve`, `Path(args.org).expanduser`, `Path(args.org).expanduser().resolve`, `RuntimeError`, `bib_entry_key`, `bib_path.exists`, `bib_path.read_text`, `build_quality_report`, `document_json.with_suffix`, `list`, `load_existing_document_json`, `print`, `report.get`, `split_bib_entries`, `write_quality_report`
- Fluxo terminal: sim

```python
   1033:         return _ap003c_result_011.value
   1034:
>> 1035:     if args.quality_report:
>> 1036:         if not args.document_json:
>> 1037:             raise RuntimeError("--quality-report exige --document-json caminho/document.json")
>> 1038:         document_json = Path(args.document_json).expanduser().resolve()
>> 1039:         document = load_existing_document_json(document_json)
>> 1040:         org = Path(args.org).expanduser().resolve() if args.org else None
>> 1041:         bib_keys: list[str] = []
>> 1042:         if args.bib:
>> 1043:             # Compatibilidade temporária entre pacote e script direto.
>> 1044:             if __package__:
>> 1045:                 from .bibliography_manager import split_bib_entries, bib_entry_key
>> 1046:             else:
>> 1047:                 from bibliography_manager import split_bib_entries, bib_entry_key
>> 1048:             bib_path = Path(args.bib).expanduser().resolve()
>> 1049:             if bib_path.exists():
>> 1050:                 bib_keys = [k for e in split_bib_entries(bib_path.read_text(encoding='utf-8', errors='ignore')) if (k := bib_entry_key(e))]
>> 1051:         report = build_quality_report(document, org_path=org, bib_keys=bib_keys or list(document.bibliography.entries_used or []))
>> 1052:         out = document_json.with_suffix(".quality_report.md")
>> 1053:         write_quality_report(report, out)
>> 1054:         print(f"Relatório de qualidade: {out}")
>> 1055:         return 0 if report.get("ok") else 1
   1056:
   1057:     cfg = _load_optional_config(args.config) if args.config else None
```

### Candidato 2: AST 68, linhas 1155–1155

- Motivos: termos documentais: document, documento
- Chamadas: `cfg.get`, `isinstance`
- Fluxo terminal: não

```python
   1153:     out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
   1154:     work_dir, cache_dir = work_cache_paths(cfg, prefix)
>> 1155:     doc_cfg = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
   1156:     latex_cfg = cfg.get("latex", {}) if isinstance(cfg.get("latex"), dict) else {}
   1157:     config_dir = Path(str(cfg.get("__config_dir__"))).resolve()
```

### Candidato 3: AST 80, linhas 1236–1289

- Motivos: helpers documentais explícitos: load_existing_document_json; termos documentais: document, mapa, mental, output, render
- Chamadas: `(mm_diag or {}).get`, `FileNotFoundError`, `Path`, `RuntimeError`, `attach_existing_mindmap_if_available`, `bool`, `cfg.get`, `delete_existing_mindmap_outputs`, `dict`, `document.model_dump`, `document_json_path.exists`, `generate_and_attach_mindmap`, `json.dumps`, `load_existing_document_json`, `make_client`, `make_run_report`, `print`, `print_outputs`, `should_generate_mindmap`, `stage`, `str`, `warnings.append`, `write_json`, `write_outputs_manifest`
- Fluxo terminal: sim

```python
   1234:         return 0
   1235:
>> 1236:     if args.somente_mapa_mental:
>> 1237:         if not document_json_path.exists():
>> 1238:             raise FileNotFoundError(f"document.json não encontrado para --somente-mapa-mental: {document_json_path}")
>> 1239:         if not should_generate_mindmap(cfg):
>> 1240:             raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --somente-mapa-mental.")
>> 1241:         stage("Carregando document.json existente")
>> 1242:         document = load_existing_document_json(document_json_path)
>> 1243:         removed_mindmap_files: list[str] = []
>> 1244:         if args.forcar_regeneracao_mapa_mental:
>> 1245:             stage("Removendo mapa mental existente")
>> 1246:             removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
>> 1247:         mm_diag = None
>> 1248:         if args.reusar_mapa_mental:
>> 1249:             stage("Tentando reutilizar mapa mental existente")
>> 1250:             mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
>> 1251:             if not mm_diag:
>> 1252:                 warnings.append("Mapa mental existente não encontrado; gerando novo mapa mental.")
>> 1253:         if not mm_diag:
>> 1254:             stage("Inicializando cliente OpenAI")
>> 1255:             client, model = make_client(model)
>> 1256:             stage("Gerando/renderizando apenas o mapa mental")
>> 1257:             mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
>> 1258:         if removed_mindmap_files:
>> 1259:             mm_diag = dict(mm_diag or {})
>> 1260:             mm_diag["removed_before_regeneration"] = removed_mindmap_files
>> 1261:         document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
>> 1262:         stage("Salvando document.json atualizado")
>> 1263:         write_json(document_json_path, document.model_dump())
>> 1264:         outputs = {
>> 1265:             "output_dir": str(out_dir),
>> 1266:             "document_json": str(document_json_path),
>> 1267:             "mindmap_puml": (mm_diag or {}).get("puml_path") if mm_diag else None,
>> 1268:             "mindmap_image": (mm_diag or {}).get("image_path") if mm_diag else None,
>> 1269:             "mindmap_reused": bool((mm_diag or {}).get("reused")),
>> 1270:             "mindmap_removed": removed_mindmap_files,
>> 1271:         }
>> 1272:         report = make_run_report(
>> 1273:             cfg=cfg,
>> 1274:             config_path=Path(str(cfg.get("__config_path__"))),
>> 1275:             out_dir=out_dir,
>> 1276:             prefix=prefix,
>> 1277:             model=model,
>> 1278:             outputs=outputs,
>> 1279:             warnings=warnings,
>> 1280:             extra={"mode": "somente_mapa_mental"},
>> 1281:         )
>> 1282:         write_json(out_dir / f"{prefix}.run_report.json", report)
>> 1283:         write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
>> 1284:         print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — mapa mental renderizado")
>> 1285:         if warnings:
>> 1286:             print("Avisos:")
>> 1287:             for w in warnings:
>> 1288:                 print(f"- {w}")
>> 1289:         return 0
   1290:
   1291:     prisma_outputs = None
```

### Candidato 4: AST 86, linhas 1451–1451

- Motivos: termos documentais: latex, org, render
- Chamadas: `stage`
- Fluxo terminal: não

```python
   1449:         write_json(document_json_path, document.model_dump())
   1450:
>> 1451:     stage("Renderizando ORG/LaTeX")
   1452:     org_path = out_dir / f"{prefix}.org"
   1453:     org_text = render_org_latex(document, org_path, bib_path.name if 'bib_path' in locals() else f"{prefix}.bib", cfg=cfg, bib_keys=bib_keys if 'bib_keys' in locals() else None)
```

### Candidato 5: AST 88, linhas 1453–1453

- Motivos: helpers documentais explícitos: render_org_latex; termos documentais: bib, document, latex, org, render
- Chamadas: `locals`, `render_org_latex`
- Fluxo terminal: não

```python
   1451:     stage("Renderizando ORG/LaTeX")
   1452:     org_path = out_dir / f"{prefix}.org"
>> 1453:     org_text = render_org_latex(document, org_path, bib_path.name if 'bib_path' in locals() else f"{prefix}.bib", cfg=cfg, bib_keys=bib_keys if 'bib_keys' in locals() else None)
   1454:     if paper_abstract_bundle:
   1455:         stage("Inserindo resumo e palavras-chave no ORG")
```

### Candidato 6: AST 89, linhas 1454–1456

- Motivos: termos documentais: abstract, document, language, org
- Chamadas: `inject_paper_abstracts_into_org`, `main_document_abstract_languages`, `stage`
- Fluxo terminal: não

```python
   1452:     org_path = out_dir / f"{prefix}.org"
   1453:     org_text = render_org_latex(document, org_path, bib_path.name if 'bib_path' in locals() else f"{prefix}.bib", cfg=cfg, bib_keys=bib_keys if 'bib_keys' in locals() else None)
>> 1454:     if paper_abstract_bundle:
>> 1455:         stage("Inserindo resumo e palavras-chave no ORG")
>> 1456:         org_text = inject_paper_abstracts_into_org(org_path, paper_abstract_bundle, main_document_abstract_languages(cfg))
   1457:     stage("Validando ORG renderizado")
   1458:     raise_if_errors(validate_org_text(org_text, bib_keys), "Validação do ORG renderizado falhou")
```

### Candidato 7: AST 90, linhas 1457–1457

- Motivos: termos documentais: org, render
- Chamadas: `stage`
- Fluxo terminal: não

```python
   1455:         stage("Inserindo resumo e palavras-chave no ORG")
   1456:         org_text = inject_paper_abstracts_into_org(org_path, paper_abstract_bundle, main_document_abstract_languages(cfg))
>> 1457:     stage("Validando ORG renderizado")
   1458:     raise_if_errors(validate_org_text(org_text, bib_keys), "Validação do ORG renderizado falhou")
   1459:
```

### Candidato 8: AST 91, linhas 1458–1458

- Motivos: termos documentais: bib, org, render
- Chamadas: `raise_if_errors`, `validate_org_text`
- Fluxo terminal: não

```python
   1456:         org_text = inject_paper_abstracts_into_org(org_path, paper_abstract_bundle, main_document_abstract_languages(cfg))
   1457:     stage("Validando ORG renderizado")
>> 1458:     raise_if_errors(validate_org_text(org_text, bib_keys), "Validação do ORG renderizado falhou")
   1459:
   1460:     pdf_path = None
```

### Candidato 9: AST 93, linhas 1461–1466

- Motivos: termos documentais: latex, org, pdf
- Chamadas: `bool`, `doc_cfg.get`, `latex_cfg.get`, `resolve_path`, `run_compile_sequence`, `stage`, `str`
- Fluxo terminal: não

```python
   1459:
   1460:     pdf_path = None
>> 1461:     if bool(doc_cfg.get("exportar_pdf", True)):
>> 1462:         academic_writing = resolve_path(latex_cfg.get("org_latex_class_init"), config_dir)
>> 1463:         latex_extra = resolve_path(latex_cfg.get("latex_extra_path"), config_dir)
>> 1464:         pdf_engine = str(latex_cfg.get("pdf_engine") or "lualatex")
>> 1465:         stage("Compilando PDF via Emacs/LaTeX")
>> 1466:         pdf_path = run_compile_sequence(org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
   1467:
   1468:     docx_path = None
```

### Candidato 10: AST 96, linhas 1470–1480

- Motivos: termos documentais: abstract, bib, bibliography, document, docx, language, reference, render
- Chamadas: `bool`, `cfg.get`, `doc_cfg.get`, `docx_cfg.get`, `docx_validation.get`, `inject_paper_abstracts_into_docx`, `isinstance`, `main_document_abstract_languages`, `render_docx`, `resolve_path`, `stage`, `validate_docx_file`, `warnings.extend`
- Fluxo terminal: não

```python
   1468:     docx_path = None
   1469:     docx_validation: dict[str, Any] | None = None
>> 1470:     if bool(doc_cfg.get("exportar_docx", True)):
>> 1471:         docx_cfg = cfg.get("docx", {}) if isinstance(cfg.get("docx"), dict) else {}
>> 1472:         ref = resolve_path(docx_cfg.get("reference_docx") or doc_cfg.get("docx_reference"), config_dir)
>> 1473:         stage("Renderizando DOCX")
>> 1474:         docx_path = render_docx(document, out_dir / f"{prefix}.docx", bib_path=bib_path, reference_docx=ref, cfg=cfg)
>> 1475:         if paper_abstract_bundle:
>> 1476:             stage("Inserindo resumo e palavras-chave no DOCX")
>> 1477:             inject_paper_abstracts_into_docx(docx_path, paper_abstract_bundle, main_document_abstract_languages(cfg))
>> 1478:         docx_validation = validate_docx_file(docx_path, expected_title=document.metadata.titulo, require_references=bool(document.bibliography.entries_used))
>> 1479:         if docx_validation and docx_validation.get("warnings"):
>> 1480:             warnings.extend([f"DOCX: {w}" for w in docx_validation.get("warnings", [])])
   1481:
   1482:     translated_outputs: dict[str, Any] = {}
```

### Candidato 11: AST 98, linhas 1483–1509

- Motivos: helpers documentais explícitos: render_additional_language_versions; termos documentais: abstract, bib, document, language, latex, output, render
- Chamadas: `render_additional_language_versions`, `requested_translation_languages`, `warnings.append`, `warnings.extend`
- Fluxo terminal: não

```python
   1481:
   1482:     translated_outputs: dict[str, Any] = {}
>> 1483:     if args.somente_renderizar:
>> 1484:         if requested_translation_languages(cfg):
>> 1485:             warnings.append(
>> 1486:                 "Versões adicionais por IA não foram atualizadas no modo --somente-renderizar. "
>> 1487:                 "Execute a geração completa para traduzir o document.json canônico."
>> 1488:             )
>> 1489:     elif requested_translation_languages(cfg):
>> 1490:         try:
>> 1491:             translated_outputs, translation_warnings = render_additional_language_versions(
>> 1492:                 client=client,
>> 1493:                 model=model,
>> 1494:                 cfg=cfg,
>> 1495:                 document=document,
>> 1496:                 bib_path=bib_path,
>> 1497:                 bib_keys=bib_keys,
>> 1498:                 out_dir=out_dir,
>> 1499:                 prefix=prefix,
>> 1500:                 doc_cfg=doc_cfg,
>> 1501:                 latex_cfg=latex_cfg,
>> 1502:                 config_dir=config_dir,
>> 1503:                 abstract_bundle=paper_abstract_bundle or None,
>> 1504:             )
>> 1505:             warnings.extend(translation_warnings)
>> 1506:         except TranslationError as exc:
>> 1507:             # Traduções são saídas opcionais: uma falha nelas não invalida o
>> 1508:             # paper principal que já foi gerado e validado.
>> 1509:             warnings.append(f"TRADUÇÃO: {exc}")
   1510:
   1511:     outputs = {
```

### Candidato 12: AST 106, linhas 1532–1538

- Motivos: termos documentais: bib, compliance, docx, org, pdf
- Chamadas: `run_institution_compliance`
- Fluxo terminal: não

```python
   1530:     # Conformidade institucional: valida artefatos contra o perfil escolhido.
   1531:     stage("Executando conformidade institucional")
>> 1532:     compliance_report = run_institution_compliance(
>> 1533:         cfg,
>> 1534:         org_path=org_path,
>> 1535:         bib_path=bib_path,
>> 1536:         docx_path=docx_path,
>> 1537:         pdf_path=pdf_path,
>> 1538:     )
   1539:     compliance_md, compliance_json = write_compliance_reports(compliance_report, out_dir / prefix)
   1540:     outputs["compliance_report"] = str(compliance_md)
```

### Candidato 13: AST 108, linhas 1540–1540

- Motivos: termos documentais: compliance, output
- Chamadas: `str`
- Fluxo terminal: não

```python
   1538:     )
   1539:     compliance_md, compliance_json = write_compliance_reports(compliance_report, out_dir / prefix)
>> 1540:     outputs["compliance_report"] = str(compliance_md)
   1541:     if compliance_report.get("warnings"):
   1542:         warnings.extend([f"CONFORMIDADE: {w.get('message')}" for w in compliance_report.get("warnings", [])])
```

### Candidato 14: AST 112, linhas 1547–1547

- Motivos: termos documentais: bib, document, org, quality
- Chamadas: `build_quality_report`
- Fluxo terminal: não

```python
   1545:
   1546:     stage("Gerando relatório de qualidade")
>> 1547:     quality = build_quality_report(document, org_path=org_path, bib_keys=bib_keys)
   1548:     quality_path = out_dir / f"{prefix}.quality_report.md"
   1549:     write_quality_report(quality, quality_path)
```

### Candidato 15: AST 116, linhas 1552–1552

- Motivos: termos documentais: output, quality
- Chamadas: `str`
- Fluxo terminal: não

```python
   1550:     if quality.get("warnings"):
   1551:         warnings.extend([f"QUALIDADE: {w}" for w in quality.get("warnings", [])])
>> 1552:     outputs["quality_report"] = str(quality_path)
   1553:
   1554:     report = make_run_report(
```

### Candidato 16: AST 117, linhas 1554–1569

- Motivos: termos documentais: docx, output, render
- Chamadas: `Path`, `cfg.get`, `make_run_report`, `str`
- Fluxo terminal: não

```python
   1552:     outputs["quality_report"] = str(quality_path)
   1553:
>> 1554:     report = make_run_report(
>> 1555:         cfg=cfg,
>> 1556:         config_path=Path(str(cfg.get("__config_path__"))),
>> 1557:         out_dir=out_dir,
>> 1558:         prefix=prefix,
>> 1559:         model=model,
>> 1560:         outputs=outputs,
>> 1561:         warnings=warnings,
>> 1562:         extra={
>> 1563:             "mode": "somente_renderizar" if args.somente_renderizar else "full",
>> 1564:             "work_dir": str(work_dir),
>> 1565:             "cache_dir": str(cache_dir),
>> 1566:             "precheck": precheck,
>> 1567:             "docx_validation": docx_validation,
>> 1568:         },
>> 1569:     )
   1570:     write_json(out_dir / f"{prefix}.run_report.json", report)
   1571:     write_json(out_dir / f"{prefix}.rc10_report.json", outputs)  # compatibilidade com scripts antigos
```

## Restrições para o aplicador AP-003D

- não alterar `academic_pipeline/cli_parser.py`;
- não alterar `academic_pipeline/command_dispatch.py`;
- não mover helpers ou blocos classificados como PRISMA;
- preservar as posições relativas dos despachos da AP-003C;
- preservar os dois `main()` até a AP-003F;
- preservar o alias histórico;
- manter compatibilidade direta, por pacote e pelo comando instalável;
- validar o fluxo documental com testes de caracterização;
- executar a suíte consolidada em `app_bundle/tests tests`.

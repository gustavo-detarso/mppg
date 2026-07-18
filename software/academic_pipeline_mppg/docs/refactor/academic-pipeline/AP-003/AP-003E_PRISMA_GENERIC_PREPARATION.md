# AP-003E — preparação PRISMA e artigo genérico

> Inventário AST somente leitura. Nenhum módulo produtivo foi alterado.

## Baseline

- Branch: `ap-refactor/03-orchestrator-decomposition`
- HEAD: `698d37d54e164ea3c4163b747a6659bc1082f635`
- Upstream: `origin/ap-refactor/03-orchestrator-decomposition`
- HEAD remoto: `698d37d54e164ea3c4163b747a6659bc1082f635`
- Orquestrador: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`
- SHA-256: `da4c6c9b817d6607873e0412b5829729e36c3d70a1745b4b7d39ea4e31d31367`
- Primeiro `main()`: linhas 711–1428
- Segundo `main()`: linhas 1571–1593
- Alias histórico: linhas 1569–1569

## Resumo

- Helpers selecionados: **30**
- Helpers com sobreposição de fases anteriores: **1**
- Instruções PRISMA no primeiro `main()`: **9**
- Instruções no segundo `main()`: **13**
- Faixas no primeiro `main()`: **8**
- Faixas no segundo `main()`: **1**

## Helpers selecionados

| Ocorrência | Linhas | Fontes de seleção | Chamadores | Dependências |
|---|---:|---|---|---|
| `stage#1` | 272–274 | chamada de bloco PRISMA no primeiro main | main#1, render_external_prisma_outputs#1 | nenhuma |
| `_json_or_none#1` | 277–283 | chamada de bloco PRISMA no primeiro main | main#1 | nenhuma |
| `make_client#1` | 295–300 | chamada de bloco PRISMA no primeiro main | main#1 | nenhuma |
| `_section#1` | 303–305 | dependência transitiva de helper selecionado | research_output_paths#1, work_cache_paths#1 | nenhuma |
| `research_output_paths#1` | 314–331 | evidência lexical, chamada de bloco PRISMA no primeiro main | main#1 | _section#1 |
| `render_external_prisma_outputs#1` | 392–449 | evidência lexical, chamada de bloco PRISMA no primeiro main | main#1 | stage#1 |
| `_prisma_curadoria_default_config#1` | 465–466 | evidência lexical | _prisma_curadoria_config_from_args#1 | nenhuma |
| `_prisma_curadoria_default_out_dir#1` | 469–470 | evidência lexical | _prisma_curadoria_out_from_args#1 | nenhuma |
| `_prisma_curadoria_default_prompt#1` | 473–474 | evidência lexical | _prisma_curadoria_prompt_from_args#1 | nenhuma |
| `_prisma_curadoria_script_path#1` | 477–478 | evidência lexical | _prisma_curadoria_build_cmd#1, _prisma_curadoria_mostrar_caminhos#1 | nenhuma |
| `_prisma_curadoria_arg#1` | 481–482 | evidência lexical | _prisma_curadoria_build_cmd#1, _prisma_curadoria_config_from_args#1, _prisma_curadoria_dispatch#1, _prisma_curadoria_fluxo_completo#1, _prisma_curadoria_input_from_args#1, _prisma_curadoria_out_from_args#1, _prisma_curadoria_prompt_from_args#1 | nenhuma |
| `_prisma_curadoria_config_from_args#1` | 485–490 | evidência lexical | _prisma_curadoria_build_cmd#1, _prisma_curadoria_importar_no_pipeline#1, _prisma_curadoria_mostrar_caminhos#1 | _prisma_curadoria_arg#1, _prisma_curadoria_default_config#1 |
| `_prisma_curadoria_out_from_args#1` | 493–494 | evidência lexical | _prisma_curadoria_build_cmd#1, _prisma_curadoria_importar_no_pipeline#1, _prisma_curadoria_input_from_args#1, _prisma_curadoria_mostrar_caminhos#1 | _prisma_curadoria_arg#1, _prisma_curadoria_default_out_dir#1 |
| `_prisma_curadoria_prompt_from_args#1` | 497–498 | evidência lexical | _prisma_curadoria_build_cmd#1, _prisma_curadoria_mostrar_caminhos#1 | _prisma_curadoria_arg#1, _prisma_curadoria_default_prompt#1 |
| `_prisma_curadoria_input_from_args#1` | 501–508 | evidência lexical | _prisma_curadoria_build_cmd#1 | _prisma_curadoria_arg#1, _prisma_curadoria_out_from_args#1 |
| `_prisma_curadoria_run_command#1` | 511–522 | evidência lexical | _prisma_curadoria_importar_no_pipeline#1, _prisma_curadoria_reexportar_xlsx#1, _prisma_curadoria_run_ia#1 | nenhuma |
| `_prisma_curadoria_build_cmd#1` | 525–571 | evidência lexical | _prisma_curadoria_reexportar_xlsx#1, _prisma_curadoria_run_ia#1 | _prisma_curadoria_arg#1, _prisma_curadoria_config_from_args#1, _prisma_curadoria_input_from_args#1, _prisma_curadoria_out_from_args#1, _prisma_curadoria_prompt_from_args#1, _prisma_curadoria_script_path#1 |
| `_prisma_curadoria_run_ia#1` | 574–576 | evidência lexical | _prisma_curadoria_dispatch#1, _prisma_curadoria_fluxo_completo#1, _prisma_curadoria_menu#1 | _prisma_curadoria_build_cmd#1, _prisma_curadoria_run_command#1 |
| `_prisma_curadoria_reexportar_xlsx#1` | 579–581 | evidência lexical | _prisma_curadoria_dispatch#1, _prisma_curadoria_menu#1 | _prisma_curadoria_build_cmd#1, _prisma_curadoria_run_command#1 |
| `_prisma_curadoria_pipeline_supports_flag#1` | 584–598 | evidência lexical | _prisma_curadoria_importar_no_pipeline#1 | nenhuma |
| `_prisma_curadoria_importar_no_pipeline#1` | 601–621 | evidência lexical | _prisma_curadoria_dispatch#1, _prisma_curadoria_fluxo_completo#1, _prisma_curadoria_menu#1 | _prisma_curadoria_config_from_args#1, _prisma_curadoria_out_from_args#1, _prisma_curadoria_pipeline_supports_flag#1, _prisma_curadoria_run_command#1 |
| `_prisma_curadoria_fluxo_completo#1` | 624–631 | evidência lexical | _prisma_curadoria_dispatch#1, _prisma_curadoria_menu#1 | _prisma_curadoria_arg#1, _prisma_curadoria_importar_no_pipeline#1, _prisma_curadoria_run_ia#1 |
| `_prisma_curadoria_mostrar_caminhos#1` | 634–653 | evidência lexical | _prisma_curadoria_menu#1 | _prisma_curadoria_config_from_args#1, _prisma_curadoria_out_from_args#1, _prisma_curadoria_prompt_from_args#1, _prisma_curadoria_script_path#1 |
| `_prisma_curadoria_menu#1` | 656–692 | evidência lexical | _prisma_curadoria_dispatch#1 | _prisma_curadoria_fluxo_completo#1, _prisma_curadoria_importar_no_pipeline#1, _prisma_curadoria_mostrar_caminhos#1, _prisma_curadoria_reexportar_xlsx#1, _prisma_curadoria_run_ia#1 |
| `_prisma_curadoria_dispatch#1` | 695–707 | evidência lexical, chamada de bloco PRISMA no primeiro main | main#1 | _prisma_curadoria_arg#1, _prisma_curadoria_fluxo_completo#1, _prisma_curadoria_importar_no_pipeline#1, _prisma_curadoria_menu#1, _prisma_curadoria_reexportar_xlsx#1, _prisma_curadoria_run_ia#1 |
| `_prisma_artigo_generico_get_arg#1` | 1490–1496 | evidência lexical | _prisma_artigo_generico_out_dir#1, _prisma_artigo_generico_run_export#1, _prisma_artigo_generico_run_freeze#1 | nenhuma |
| `_prisma_artigo_generico_strip#1` | 1498–1511 | evidência lexical, chamada direta do segundo main | main#2 | nenhuma |
| `_prisma_artigo_generico_out_dir#1` | 1513–1521 | evidência lexical | _prisma_artigo_generico_run_export#1, _prisma_artigo_generico_run_freeze#1 | _prisma_artigo_generico_get_arg#1 |
| `_prisma_artigo_generico_run_export#1` | 1523–1542 | evidência lexical, chamada direta do segundo main | main#2 | _prisma_artigo_generico_get_arg#1, _prisma_artigo_generico_out_dir#1 |
| `_prisma_artigo_generico_run_freeze#1` | 1544–1567 | evidência lexical, chamada direta do segundo main | main#2 | _prisma_artigo_generico_get_arg#1, _prisma_artigo_generico_out_dir#1 |

## Sobreposições preservadas fora da AP-003E

| Ocorrência | Linhas | Motivo |
|---|---:|---|
| `output_paths#1` | 308–311 | sobreposição documental AP-003D: _ap003d_, document_orchestration |

## Primeiro main — blocos PRISMA/artigo genérico

| AST | Linhas | Tipo | Condição | Helpers chamados | Carrega | Grava |
|---:|---:|---|---|---|---|---|
| 2 | 717–724 | `if` | `getattr(args, "prisma_curadoria_menu", False)
        or getattr(args, "prisma_curadoria_ia", False)
        or getattr(args, "prisma_curadoria_reexportar_xlsx", False)
        or getattr(args, "prisma_curadoria_importar", False)
        or getattr(args, "prisma_curadoria_fluxo_completo", False)` | `_prisma_curadoria_dispatch#1` | `_prisma_curadoria_dispatch`, `args` | nenhum |
| 78 | 975–975 | `assignment` | `` | nenhum | `args`, `cfg`, `external_search_enabled` | `is_external_prisma_run` |
| 79 | 976–976 | `assignment` | `` | `output_paths#1`, `research_output_paths#1` | `cfg`, `is_external_prisma_run`, `output_paths`, `research_output_paths` | `out_dir`, `prefix` |
| 90 | 997–997 | `expression` | `` | `stage#1` | `stage` | nenhum |
| 95 | 1006–1068 | `if` | `is_external_prisma_run` | `make_client#1`, `render_external_prisma_outputs#1`, `stage#1` | `PIPELINE_VERSION`, `Path`, `args`, `artifacts`, `cache_dir`, `cfg`, `client`, `is_external_prisma_run`, `make_client`, `make_run_report`, `model`, `org_path`, `out_dir`, `outputs`, `pdf_path`, `precheck`, `prefix`, `print_outputs`, `prisma_outputs`, `prompt_lock`, `prompt_lock_md`, `prompt_lock_path`, `render_external_prisma_outputs`, `report`, `report_json_path`, `run_external_prisma_search`, `search_cfg`, `stage`, `warnings`, `work_dir`, `write_json`, `write_outputs_manifest`, `write_prompt_lock`, `write_prompt_lock_markdown` | `artifacts`, `client`, `model`, `org_path`, `outputs`, `pdf_path`, `prisma_outputs`, `prompt_lock`, `prompt_lock_md`, `prompt_lock_path`, `report`, `report_json_path`, `search_cfg` |
| 107 | 1097–1097 | `assignment` | `` | nenhum | nenhum | `prisma_outputs` |
| 145 | 1342–1352 | `assignment` | `` | `_json_or_none#1` | `_json_or_none`, `bib_path`, `document`, `document_json_path`, `docx_path`, `org_path`, `out_dir`, `paper_abstract_bundle`, `paper_abstract_path`, `pdf_path`, `prisma_outputs`, `translated_outputs` | `outputs` |
| 151 | 1362–1362 | `expression` | `` | `stage#1` | `stage` | nenhum |
| 162 | 1391–1391 | `expression` | `` | `stage#1` | `stage` | nenhum |

### Trechos

#### Bloco 1: AST 2, linhas 717–724

- Seleção: evidência lexical PRISMA/artigo genérico; chamada de helper PRISMA selecionado
- Chamadas: `_prisma_curadoria_dispatch`, `getattr`
- Fluxo terminal: sim

```python
    715:
    716:     # >>> PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_DISPATCH >>>
>>  717:     if (
>>  718:         getattr(args, "prisma_curadoria_menu", False)
>>  719:         or getattr(args, "prisma_curadoria_ia", False)
>>  720:         or getattr(args, "prisma_curadoria_reexportar_xlsx", False)
>>  721:         or getattr(args, "prisma_curadoria_importar", False)
>>  722:         or getattr(args, "prisma_curadoria_fluxo_completo", False)
>>  723:     ):
>>  724:         return _prisma_curadoria_dispatch(args)
    725:     # <<< PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_DISPATCH <<<
    726:     from academic_pipeline.command_dispatch import (
```

#### Bloco 2: AST 78, linhas 975–975

- Seleção: evidência lexical PRISMA/artigo genérico
- Chamadas: `external_search_enabled`
- Fluxo terminal: não

```python
    973:
    974:     cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
>>  975:     is_external_prisma_run = external_search_enabled(cfg) and not args.somente_renderizar
    976:     out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
    977:     work_dir, cache_dir = work_cache_paths(cfg, prefix)
```

#### Bloco 3: AST 79, linhas 976–976

- Seleção: evidência lexical PRISMA/artigo genérico; chamada de helper PRISMA selecionado
- Chamadas: `output_paths`, `research_output_paths`
- Fluxo terminal: não

```python
    974:     cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
    975:     is_external_prisma_run = external_search_enabled(cfg) and not args.somente_renderizar
>>  976:     out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
    977:     work_dir, cache_dir = work_cache_paths(cfg, prefix)
    978:     from academic_pipeline.document_orchestration import (
```

#### Bloco 4: AST 90, linhas 997–997

- Seleção: chamada de helper PRISMA selecionado
- Chamadas: `stage`
- Fluxo terminal: não

```python
    995:
    996:     # Validação preventiva leve; não bloqueia warnings.
>>  997:     stage("Validando configuração preventiva")
    998:     precheck = check_config(cfg)
    999:     if precheck.get("warnings"):
```

#### Bloco 5: AST 95, linhas 1006–1068

- Seleção: evidência lexical PRISMA/artigo genérico; chamada de helper PRISMA selecionado
- Chamadas: `Path`, `RuntimeError`, `artifacts.get`, `bool`, `cfg.get`, `isinstance`, `make_client`, `make_run_report`, `print_outputs`, `prisma_outputs.setdefault`, `render_external_prisma_outputs`, `run_external_prisma_search`, `search_cfg.get`, `stage`, `str`, `write_json`, `write_outputs_manifest`, `write_prompt_lock`, `write_prompt_lock_markdown`
- Fluxo terminal: sim

```python
   1004:     document_json_path = Path(args.document_json).expanduser().resolve() if args.document_json else out_dir / f"{prefix}.document.json"
   1005:
>> 1006:     if is_external_prisma_run:
>> 1007:         if args.somente_mapa_mental:
>> 1008:             raise RuntimeError("O perfil de busca PRISMA não produz document.json; use a geração normal ou --prisma-importar-triagem.")
>> 1009:         search_cfg = cfg.get("busca_prisma", {}) if isinstance(cfg.get("busca_prisma"), dict) else {}
>> 1010:         if bool(search_cfg.get("pre_triagem_ia", False)):
>> 1011:             stage("Inicializando cliente OpenAI para pré-triagem assistida")
>> 1012:             client, model = make_client(model)
>> 1013:         stage("Executando busca bibliográfica externa e preparando triagem humana")
>> 1014:         prisma_outputs = run_external_prisma_search(
>> 1015:             cfg,
>> 1016:             out_dir,
>> 1017:             prefix,
>> 1018:             progress=stage,
>> 1019:             client=client,
>> 1020:             model=model,
>> 1021:         )
>> 1022:         org_path, pdf_path = render_external_prisma_outputs(
>> 1023:             cfg,
>> 1024:             out_dir,
>> 1025:             prefix,
>> 1026:             prisma_outputs,
>> 1027:             phase="preliminar",
>> 1028:         )
>> 1029:         artifacts = prisma_outputs.setdefault("artefatos", {}) if isinstance(prisma_outputs, dict) else {}
>> 1030:         if org_path:
>> 1031:             artifacts["relatorio_org"] = str(org_path)
>> 1032:         if pdf_path:
>> 1033:             artifacts["relatorio_pdf"] = str(pdf_path)
>> 1034:         report_json_path = artifacts.get("prisma_report_json") if isinstance(artifacts, dict) else ""
>> 1035:         if report_json_path:
>> 1036:             write_json(Path(str(report_json_path)), prisma_outputs)
>> 1037:         prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
>> 1038:         prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
>> 1039:         stage("Registrando prompt_lock")
>> 1040:         prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
>> 1041:         write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
>> 1042:         outputs = {
>> 1043:             "output_dir": str(out_dir),
>> 1044:             "work_dir": str(work_dir),
>> 1045:             "cache_dir": str(cache_dir),
>> 1046:             "document_json": None,
>> 1047:             "org": str(org_path) if org_path else None,
>> 1048:             "bib": None,
>> 1049:             "pdf": str(pdf_path) if pdf_path else None,
>> 1050:             "docx": None,
>> 1051:             "relatorio_pesquisa": prisma_outputs,
>> 1052:             "prompt_lock": str(prompt_lock_path),
>> 1053:         }
>> 1054:         report = make_run_report(
>> 1055:             cfg=cfg,
>> 1056:             config_path=Path(str(cfg.get("__config_path__"))),
>> 1057:             out_dir=out_dir,
>> 1058:             prefix=prefix,
>> 1059:             model=None,
>> 1060:             outputs=outputs,
>> 1061:             warnings=warnings,
>> 1062:             extra={"mode": "prisma_busca_externa", "precheck": precheck},
>> 1063:         )
>> 1064:         write_json(out_dir / f"{prefix}.run_report.json", report)
>> 1065:         write_json(out_dir / f"{prefix}.rc10_report.json", outputs)
>> 1066:         write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
>> 1067:         print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — busca PRISMA concluída; aguarda triagem humana")
>> 1068:         return 0
   1069:
   1070:     from academic_pipeline.document_orchestration import (
```

#### Bloco 6: AST 107, linhas 1097–1097

- Seleção: evidência lexical PRISMA/artigo genérico
- Chamadas: nenhuma
- Fluxo terminal: não

```python
   1095:         w = _ap003d_result_003.values['w']
   1096:
>> 1097:     prisma_outputs = None
   1098:     source_info: dict[str, Any] | None = None
   1099:     paper_abstract_bundle: dict[str, Any] = {}
```

#### Bloco 7: AST 145, linhas 1342–1352

- Seleção: evidência lexical PRISMA/artigo genérico; chamada de helper PRISMA selecionado
- Chamadas: `_json_or_none`, `getattr`, `str`
- Fluxo terminal: não

```python
   1340:         translation_warnings = _ap003d_result_008.values['translation_warnings']
   1341:
>> 1342:     outputs = {
>> 1343:         "output_dir": str(out_dir),
>> 1344:         "document_json": str(document_json_path),
>> 1345:         "org": str(org_path),
>> 1346:         "bib": str(bib_path),
>> 1347:         "pdf": str(pdf_path) if pdf_path else None,
>> 1348:         "docx": str(docx_path) if docx_path else None,
>> 1349:         "resumos_paper": str(paper_abstract_path) if paper_abstract_bundle else None,
>> 1350:         "idiomas_adicionais": translated_outputs,
>> 1351:         "relatorio_pesquisa": _json_or_none(getattr(document.diagnostics, "relatorio_pesquisa_json", "")) if getattr(document, "diagnostics", None) else prisma_outputs,
>> 1352:     }
   1353:
   1354:     # Prompt lock: rastreabilidade exata dos prompts/diretivas usados.
```

#### Bloco 8: AST 151, linhas 1362–1362

- Seleção: chamada de helper PRISMA selecionado
- Chamadas: `stage`
- Fluxo terminal: não

```python
   1360:
   1361:     # Conformidade institucional: valida artefatos contra o perfil escolhido.
>> 1362:     stage("Executando conformidade institucional")
   1363:     from academic_pipeline.document_orchestration import (
   1364:         run_document_stage_009 as _ap003d_stage_009,
```

#### Bloco 9: AST 162, linhas 1391–1391

- Seleção: chamada de helper PRISMA selecionado
- Chamadas: `stage`
- Fluxo terminal: não

```python
   1389:         warnings.extend([f"CONFORMIDADE CRÍTICA: {e.get('message')}" for e in compliance_report.get("errors", [])])
   1390:
>> 1391:     stage("Gerando relatório de qualidade")
   1392:     from academic_pipeline.document_orchestration import (
   1393:         run_document_stage_011 as _ap003d_stage_011,
```


## Segundo main — wrapper histórico

| AST | Linhas | Tipo | Condição | Helpers chamados | Carrega | Grava |
|---:|---:|---|---|---|---|---|
| 0 | 1572–1572 | `Import` | `` | nenhum | nenhum | nenhum |
| 1 | 1573–1573 | `assignment` | `` | nenhum | `sys` | `original_argv` |
| 2 | 1574–1574 | `assignment` | `` | nenhum | `original_argv` | `has_import` |
| 3 | 1575–1575 | `assignment` | `` | nenhum | `original_argv` | `wants_export` |
| 4 | 1576–1576 | `assignment` | `` | nenhum | `original_argv` | `wants_freeze` |
| 5 | 1577–1577 | `assignment` | `` | nenhum | `original_argv` | `wants_toml` |
| 6 | 1578–1578 | `assignment` | `` | nenhum | `original_argv` | `wants_final` |
| 7 | 1579–1580 | `if` | `not has_import and wants_export and not wants_freeze and not wants_toml and not wants_final` | `_prisma_artigo_generico_run_export#1` | `_prisma_artigo_generico_run_export`, `has_import`, `original_argv`, `wants_export`, `wants_final`, `wants_freeze`, `wants_toml` | nenhum |
| 8 | 1581–1583 | `if` | `not has_import and (wants_freeze or wants_toml or wants_final)` | `_prisma_artigo_generico_run_export#1`, `_prisma_artigo_generico_run_freeze#1` | `_prisma_artigo_generico_run_export`, `_prisma_artigo_generico_run_freeze`, `has_import`, `original_argv`, `wants_final`, `wants_freeze`, `wants_toml` | nenhum |
| 9 | 1584–1590 | `if` | `has_import and (wants_export or wants_freeze or wants_toml or wants_final)` | `_prisma_artigo_generico_strip#1` | `_original_main_before_prisma_artigo_generico_wrapper`, `_prisma_artigo_generico_strip`, `args`, `has_import`, `kwargs`, `old_argv`, `original_argv`, `sys`, `wants_export`, `wants_final`, `wants_freeze`, `wants_toml` | `old_argv`, `rc` |
| 10 | 1591–1591 | `if` | `has_import` | `_prisma_artigo_generico_run_export#1` | `_prisma_artigo_generico_run_export`, `has_import`, `original_argv` | nenhum |
| 11 | 1592–1592 | `if` | `wants_freeze or wants_toml or wants_final` | `_prisma_artigo_generico_run_freeze#1` | `_prisma_artigo_generico_run_freeze`, `original_argv`, `wants_final`, `wants_freeze`, `wants_toml` | nenhum |
| 12 | 1593–1593 | `return` | `` | nenhum | `rc` | nenhum |

### Trechos

#### Bloco 1: AST 0, linhas 1572–1572

- Seleção: instrução do segundo main histórico
- Chamadas: nenhuma
- Fluxo terminal: não

```python
   1570:
   1571: def main(*args, **kwargs):
>> 1572:     import sys
   1573:     original_argv=list(sys.argv[1:])
   1574:     has_import="--prisma-curadoria-importar" in original_argv or "--prisma-curadoria-fluxo-completo" in original_argv
```

#### Bloco 2: AST 1, linhas 1573–1573

- Seleção: instrução do segundo main histórico
- Chamadas: `list`
- Fluxo terminal: não

```python
   1571: def main(*args, **kwargs):
   1572:     import sys
>> 1573:     original_argv=list(sys.argv[1:])
   1574:     has_import="--prisma-curadoria-importar" in original_argv or "--prisma-curadoria-fluxo-completo" in original_argv
   1575:     wants_export="--prisma-exportar-bib" in original_argv
```

#### Bloco 3: AST 2, linhas 1574–1574

- Seleção: instrução do segundo main histórico
- Chamadas: nenhuma
- Fluxo terminal: não

```python
   1572:     import sys
   1573:     original_argv=list(sys.argv[1:])
>> 1574:     has_import="--prisma-curadoria-importar" in original_argv or "--prisma-curadoria-fluxo-completo" in original_argv
   1575:     wants_export="--prisma-exportar-bib" in original_argv
   1576:     wants_freeze="--prisma-congelar-artigo" in original_argv
```

#### Bloco 4: AST 3, linhas 1575–1575

- Seleção: instrução do segundo main histórico
- Chamadas: nenhuma
- Fluxo terminal: não

```python
   1573:     original_argv=list(sys.argv[1:])
   1574:     has_import="--prisma-curadoria-importar" in original_argv or "--prisma-curadoria-fluxo-completo" in original_argv
>> 1575:     wants_export="--prisma-exportar-bib" in original_argv
   1576:     wants_freeze="--prisma-congelar-artigo" in original_argv
   1577:     wants_toml="--prisma-gerar-toml-artigo" in original_argv
```

#### Bloco 5: AST 4, linhas 1576–1576

- Seleção: instrução do segundo main histórico
- Chamadas: nenhuma
- Fluxo terminal: não

```python
   1574:     has_import="--prisma-curadoria-importar" in original_argv or "--prisma-curadoria-fluxo-completo" in original_argv
   1575:     wants_export="--prisma-exportar-bib" in original_argv
>> 1576:     wants_freeze="--prisma-congelar-artigo" in original_argv
   1577:     wants_toml="--prisma-gerar-toml-artigo" in original_argv
   1578:     wants_final="--prisma-gerar-artigo-final" in original_argv
```

#### Bloco 6: AST 5, linhas 1577–1577

- Seleção: instrução do segundo main histórico
- Chamadas: nenhuma
- Fluxo terminal: não

```python
   1575:     wants_export="--prisma-exportar-bib" in original_argv
   1576:     wants_freeze="--prisma-congelar-artigo" in original_argv
>> 1577:     wants_toml="--prisma-gerar-toml-artigo" in original_argv
   1578:     wants_final="--prisma-gerar-artigo-final" in original_argv
   1579:     if not has_import and wants_export and not wants_freeze and not wants_toml and not wants_final:
```

#### Bloco 7: AST 6, linhas 1578–1578

- Seleção: instrução do segundo main histórico
- Chamadas: nenhuma
- Fluxo terminal: não

```python
   1576:     wants_freeze="--prisma-congelar-artigo" in original_argv
   1577:     wants_toml="--prisma-gerar-toml-artigo" in original_argv
>> 1578:     wants_final="--prisma-gerar-artigo-final" in original_argv
   1579:     if not has_import and wants_export and not wants_freeze and not wants_toml and not wants_final:
   1580:         return _prisma_artigo_generico_run_export(original_argv, silent=False)
```

#### Bloco 8: AST 7, linhas 1579–1580

- Seleção: instrução do segundo main histórico
- Chamadas: `_prisma_artigo_generico_run_export`
- Fluxo terminal: sim

```python
   1577:     wants_toml="--prisma-gerar-toml-artigo" in original_argv
   1578:     wants_final="--prisma-gerar-artigo-final" in original_argv
>> 1579:     if not has_import and wants_export and not wants_freeze and not wants_toml and not wants_final:
>> 1580:         return _prisma_artigo_generico_run_export(original_argv, silent=False)
   1581:     if not has_import and (wants_freeze or wants_toml or wants_final):
   1582:         _prisma_artigo_generico_run_export(original_argv, silent=True)
```

#### Bloco 9: AST 8, linhas 1581–1583

- Seleção: instrução do segundo main histórico
- Chamadas: `_prisma_artigo_generico_run_export`, `_prisma_artigo_generico_run_freeze`
- Fluxo terminal: sim

```python
   1579:     if not has_import and wants_export and not wants_freeze and not wants_toml and not wants_final:
   1580:         return _prisma_artigo_generico_run_export(original_argv, silent=False)
>> 1581:     if not has_import and (wants_freeze or wants_toml or wants_final):
>> 1582:         _prisma_artigo_generico_run_export(original_argv, silent=True)
>> 1583:         return _prisma_artigo_generico_run_freeze(original_argv, silent=False)
   1584:     if has_import and (wants_export or wants_freeze or wants_toml or wants_final):
   1585:         old_argv=sys.argv[:]
```

#### Bloco 10: AST 9, linhas 1584–1590

- Seleção: instrução do segundo main histórico
- Chamadas: `_original_main_before_prisma_artigo_generico_wrapper`, `_prisma_artigo_generico_strip`
- Fluxo terminal: não

```python
   1582:         _prisma_artigo_generico_run_export(original_argv, silent=True)
   1583:         return _prisma_artigo_generico_run_freeze(original_argv, silent=False)
>> 1584:     if has_import and (wants_export or wants_freeze or wants_toml or wants_final):
>> 1585:         old_argv=sys.argv[:]
>> 1586:         sys.argv=[sys.argv[0]]+_prisma_artigo_generico_strip(original_argv)
>> 1587:         try: rc=_original_main_before_prisma_artigo_generico_wrapper(*args, **kwargs)
>> 1588:         finally: sys.argv=old_argv
>> 1589:     else:
>> 1590:         rc=_original_main_before_prisma_artigo_generico_wrapper(*args, **kwargs)
   1591:     if has_import: _prisma_artigo_generico_run_export(original_argv, silent=True)
   1592:     if wants_freeze or wants_toml or wants_final: _prisma_artigo_generico_run_freeze(original_argv, silent=False)
```

#### Bloco 11: AST 10, linhas 1591–1591

- Seleção: instrução do segundo main histórico
- Chamadas: `_prisma_artigo_generico_run_export`
- Fluxo terminal: não

```python
   1589:     else:
   1590:         rc=_original_main_before_prisma_artigo_generico_wrapper(*args, **kwargs)
>> 1591:     if has_import: _prisma_artigo_generico_run_export(original_argv, silent=True)
   1592:     if wants_freeze or wants_toml or wants_final: _prisma_artigo_generico_run_freeze(original_argv, silent=False)
   1593:     return rc
```

#### Bloco 12: AST 11, linhas 1592–1592

- Seleção: instrução do segundo main histórico
- Chamadas: `_prisma_artigo_generico_run_freeze`
- Fluxo terminal: não

```python
   1590:         rc=_original_main_before_prisma_artigo_generico_wrapper(*args, **kwargs)
   1591:     if has_import: _prisma_artigo_generico_run_export(original_argv, silent=True)
>> 1592:     if wants_freeze or wants_toml or wants_final: _prisma_artigo_generico_run_freeze(original_argv, silent=False)
   1593:     return rc
   1594: # <<< PATCH_PRISMA_ARTIGO_GENERICO_WRAPPER_V1_5 <<<
```

#### Bloco 13: AST 12, linhas 1593–1593

- Seleção: instrução do segundo main histórico
- Chamadas: nenhuma
- Fluxo terminal: sim

```python
   1591:     if has_import: _prisma_artigo_generico_run_export(original_argv, silent=True)
   1592:     if wants_freeze or wants_toml or wants_final: _prisma_artigo_generico_run_freeze(original_argv, silent=False)
>> 1593:     return rc
   1594: # <<< PATCH_PRISMA_ARTIGO_GENERICO_WRAPPER_V1_5 <<<
   1595:
```

## Restrições para o aplicador AP-003E

- não alterar `academic_pipeline/cli_parser.py`;
- não alterar `academic_pipeline/command_dispatch.py`;
- não alterar `academic_pipeline/document_orchestration.py`;
- preservar as delegações AP-003B, AP-003C e AP-003D;
- preservar os dois `main()` até a AP-003F;
- preservar o alias histórico e sua posição relativa;
- mover somente helpers pertencentes ao fechamento selecionado pelo grafo de chamadas;
- não incorporar helpers documentais por sobreposição lexical;
- preservar os três `xfail` conhecidos;
- executar a suíte consolidada somente em `app_bundle/tests tests`.

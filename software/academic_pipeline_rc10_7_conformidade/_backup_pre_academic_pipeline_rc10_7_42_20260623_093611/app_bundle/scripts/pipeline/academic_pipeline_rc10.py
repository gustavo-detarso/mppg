#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""academic_pipeline rc10.7.20 — seção [paths] e override de saída por CLI.

A IA gera AcademicDocument/document.json. Renderizadores determinísticos geram
ORG/PDF/DOCX e relatório PRISMA. Esta versão acrescenta:
- --doctor para diagnóstico de ambiente;
- --check-config para validação preventiva do TOML;
- --recompile para recompilar ORG sem chamar IA;
- run_report.json e outputs.txt para rastreabilidade;
- validação básica do DOCX;
- modo --somente-renderizar sem exigir OPENAI_API_KEY;
- --init-project para criar estrutura de projeto;
- --make-doi-manifest para gerar CSV a partir de ZIP/pasta;
- --inspect-bib para diagnóstico bibliográfico;
- quality_report.md/json após a geração;
- perfis institucionais em app_bundle/institutions/<perfil>;
- prompt bank reutilizável em app_bundle/prompts e app_bundle/institutions/<perfil>/prompts;
- conformidade institucional auditável;
- prompt_lock.json/md para rastreabilidade de diretivas;
- schema estrito compatível com OpenAI Structured Outputs;
- impressão de etapas em tempo real com [ETAPA];
- saneamento de menções técnicas no conteúdo visível antes da validação.
- perfil resumo_artigos_local_fgv e perguntas condicionais por tipo documental.
- perfil de resumo de artigos com profundidade analítica configurável, matriz por texto e comparação transversal robusta.
- seção [paths] como fonte única para saídas/work/cache e overrides por linha de comando.
- --tui para central operacional visual de atividades acadêmicas em terminal.
- --gui para interface gráfica opcional de atividades acadêmicas.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from bibliography_manager import build_bibliography
from corpus_manager import collect_orientation_docs, copy_documents_to_fulltext_cache, discover_local_documents
from document_builder import build_document_model
from document_model import AcademicDocument
from document_validator import raise_if_errors, validate_document_model, validate_org_text, sanitize_document_model_technical_leaks, sanitize_document_model_raw_bibkeys
from diagnostics import (
    PIPELINE_VERSION,
    check_config,
    clean_aux_files,
    make_run_report,
    print_check_config_report,
    print_doctor_report,
    print_outputs,
    run_doctor,
    validate_docx_file,
    write_outputs_manifest,
)
from latex_compile import run_compile_sequence
from mindmap_manager import generate_and_attach_mindmap, should_generate_mindmap, attach_existing_mindmap_if_available, delete_existing_mindmap_outputs
from render_docx import render_docx
from render_org_latex import render_org_latex
from prisma_pipeline import run_prisma_report_outputs, prisma_enabled
from utils import write_json, resolve_path
from project_tools import init_project, make_doi_manifest, inspect_bib, render_bib_inspection_markdown
from quality_report import build_quality_report, write_quality_report
from institution_profiles import apply_institution_profile, describe_institution_profiles
from institution_layouts import available_layouts, resolve_layout_spec
from prompt_manager import prompt_report_for_cfg, load_prompt_bundle
from institution_explainer import explain_profile
from institution_compliance import run_institution_compliance, render_compliance_markdown, write_compliance_reports
from prompt_lock import write_prompt_lock, write_prompt_lock_markdown, build_prompt_lock

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")


def stage(message: str) -> None:
    """Mostra etapas de execução em tempo real."""
    print(f"[ETAPA] {message}", flush=True)


def _json_or_none(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["__config_path__"] = str(path.resolve())
    cfg["__config_dir__"] = str(path.resolve().parent)
    cfg = apply_institution_profile(cfg)
    return cfg


def make_client(model_override: str | None = None) -> tuple[Any, str]:
    from openai import OpenAI
    load_dotenv(override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY não encontrado no ambiente/.env.")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), (model_override or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = cfg.get(name, {})
    return sec if isinstance(sec, dict) else {}


def output_paths(cfg: dict[str, Any]) -> tuple[Path, str]:
    """Resolve a saída final do documento pela seção [paths]."""
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    paths = _section(cfg, "paths")
    projeto = _section(cfg, "projeto")
    prefix = str(paths.get("document_prefix") or projeto.get("nome") or "documento").strip() or "documento"
    out_base = resolve_path(paths.get("document_output_dir") or "../../output/documento", config_dir) or (config_dir / "output/documento")
    out_dir = out_base / prefix if bool(paths.get("create_document_subdir", True)) else out_base
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, prefix


def work_cache_paths(cfg: dict[str, Any], prefix: str) -> tuple[Path, Path]:
    """Resolve diretórios operacionais pela seção [paths]."""
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    paths = _section(cfg, "paths")
    work_base = resolve_path(paths.get("work_dir") or "../../output/work", config_dir) or (config_dir / "output/work")
    cache_base = resolve_path(paths.get("cache_dir") or "../../output/cache", config_dir) or (config_dir / "output/cache")
    work_dir = work_base / prefix if bool(paths.get("create_work_subdir", True)) else work_base
    cache_dir = cache_base / prefix if bool(paths.get("create_cache_subdir", True)) else cache_base
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return work_dir, cache_dir


def apply_cli_path_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Aplica overrides de caminhos informados na linha de comando.

    A prioridade fica: CLI > TOML. Os caminhos continuam sendo resolvidos
    posteriormente em relação ao diretório do TOML, salvo quando absolutos.
    """
    paths = cfg.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        cfg["paths"] = paths
    if getattr(args, "output_dir", ""):
        paths["document_output_dir"] = args.output_dir
    if getattr(args, "work_dir", ""):
        paths["work_dir"] = args.work_dir
    if getattr(args, "cache_dir", ""):
        paths["cache_dir"] = args.cache_dir
    if getattr(args, "research_output_dir", ""):
        paths["research_output_dir"] = args.research_output_dir
    if getattr(args, "output_prefix", ""):
        paths["document_prefix"] = args.output_prefix
    if getattr(args, "no_output_subdir", False):
        paths["create_document_subdir"] = False

    doc = cfg.setdefault("documento", {})
    if not isinstance(doc, dict):
        doc = {}
        cfg["documento"] = doc
    if getattr(args, "layout", ""):
        doc["layout"] = args.layout
    if getattr(args, "tipo_conteudo", ""):
        doc["tipo_conteudo"] = args.tipo_conteudo
    if getattr(args, "genero_academico", ""):
        doc["genero_academico"] = args.genero_academico
    return cfg


def load_existing_document_json(path: Path) -> AcademicDocument:
    return AcademicDocument.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_bib_for_existing_document(document: AcademicDocument, document_json_path: Path, out_dir: Path, prefix: str) -> tuple[Path, list[str]]:
    """Resolve o .bib em modo --somente-renderizar sem exigir que ele já esteja no output_dir."""
    raw = str(document.bibliography.bib_path or f"{prefix}.bib").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([document_json_path.parent / p, out_dir / p])
    candidates.extend([
        document_json_path.with_suffix(".bib"),
        document_json_path.with_name(prefix + ".bib"),
        out_dir / f"{prefix}.bib",
    ])
    found = next((c.resolve() for c in candidates if c.exists()), out_dir / f"{prefix}.bib")
    target = out_dir / found.name
    if found.exists() and found.resolve() != target.resolve():
        import shutil
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, target)
        found = target
    keys = list(document.bibliography.entries_used or [])
    if found.exists() and not keys:
        import re
        text = found.read_text(encoding="utf-8", errors="ignore")
        keys = [m.group(1).strip() for m in re.finditer(r"@[^{}]+\{\s*([^,]+)\s*,", text)]
    return found, keys


def _openai_model_from_cfg(cfg: dict[str, Any]) -> str:
    return str((cfg.get("openai", {}) if isinstance(cfg.get("openai"), dict) else {}).get("model") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)


def _load_optional_config(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config não encontrada: {p}")
    return load_config(p)


def _resolve_latex_paths_for_recompile(args: argparse.Namespace, cfg: dict[str, Any] | None) -> tuple[Path | None, Path | None, str]:
    base = Path(str((cfg or {}).get("__config_dir__") or Path.cwd())).resolve()
    latex_cfg = (cfg or {}).get("latex", {}) if isinstance((cfg or {}).get("latex", {}), dict) else {}
    academic_writing = resolve_path(args.academic_writing or latex_cfg.get("org_latex_class_init"), base)
    latex_extra = resolve_path(args.latex_extra_path or latex_cfg.get("latex_extra_path"), base)
    pdf_engine = str(args.pdf_engine or latex_cfg.get("pdf_engine") or "lualatex")
    return academic_writing, latex_extra, pdf_engine


def run_recompile(args: argparse.Namespace, cfg: dict[str, Any] | None) -> int:
    if not args.org:
        raise RuntimeError("Use --recompile --org caminho/arquivo.org")
    org_path = Path(args.org).expanduser().resolve()
    if not org_path.exists():
        raise FileNotFoundError(f"ORG não encontrado: {org_path}")
    academic_writing, latex_extra, pdf_engine = _resolve_latex_paths_for_recompile(args, cfg)
    removed = [] if args.no_clean else clean_aux_files(org_path)
    pdf = run_compile_sequence(org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
    out_dir = org_path.parent
    prefix = org_path.stem
    outputs = {"org": str(org_path), "pdf": str(pdf), "removed_aux": removed}
    write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
    stage("Gerando run_report e manifestos")
    report = make_run_report(
        cfg=cfg or {"__config_dir__": str(Path.cwd())},
        config_path=Path(str((cfg or {}).get("__config_path__"))) if cfg and cfg.get("__config_path__") else None,
        out_dir=out_dir,
        prefix=prefix,
        model=None,
        outputs=outputs,
        warnings=[],
        extra={"mode": "recompile"},
    )
    write_json(out_dir / f"{prefix}.run_report.json", report)
    print_outputs(outputs, title="Recompilação concluída")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"academic_pipeline {PIPELINE_VERSION} — document_model canônico")
    parser.add_argument("--config", default="", help="Arquivo TOML")
    parser.add_argument("--tui", action="store_true", help="Abre a Central Operacional FGV em terminal (prompt_toolkit)")
    parser.add_argument("--gui", action="store_true", help="Abre a interface gráfica FGV de atividades acadêmicas")
    parser.add_argument("--init-toml", action="store_true", help="Abre o gerador interativo completo de TOML")
    parser.add_argument("--toml-profile", default="", help="Preset inicial para --init-toml, ex.: atividade_local_fgv")
    parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela entre etapas do --init-toml")
    parser.add_argument("--list-toml-profiles", action="store_true", help="Lista presets do gerador de TOML")
    parser.add_argument("--list-institutions", action="store_true", help="Lista perfis institucionais disponíveis")
    parser.add_argument("--list-layouts", action="store_true", help="Lista layouts disponíveis do perfil institucional informado no TOML")
    parser.add_argument("--explain-profile", default="", nargs="?", const="fgv", help="Explica um perfil institucional, ex.: --explain-profile fgv")
    parser.add_argument("--show-prompts", action="store_true", help="Mostra os prompts/diretivas ativos para o TOML informado")
    parser.add_argument("--write-prompt-lock", action="store_true", help="Gera prompt_lock.json/md para o TOML e encerra")
    parser.add_argument("--check-institution-compliance", action="store_true", help="Valida conformidade institucional de artefatos já gerados")
    parser.add_argument("--doctor", action="store_true", help="Diagnostica ambiente, ferramentas, arquivos FGV e estilo bibliográfico")
    parser.add_argument("--check-config", action="store_true", help="Valida preventivamente o TOML e encerra")
    parser.add_argument("--recompile", action="store_true", help="Recompila um .org existente sem chamar IA")
    parser.add_argument("--org", default="", help="Arquivo .org para --recompile")
    parser.add_argument("--academic-writing", default="", help="Override do academic-writing.el para --recompile")
    parser.add_argument("--latex-extra-path", default="", help="Override do latex_extra_path para --recompile")
    parser.add_argument("--pdf-engine", default="", help="Override do pdf_engine para --recompile")
    parser.add_argument("--no-clean", action="store_true", help="Não remove auxiliares no --recompile")
    parser.add_argument("--somente-renderizar", action="store_true", help="Usa document.json existente e só renderiza saídas")
    parser.add_argument("--somente-mapa-mental", action="store_true", help="Usa document.json existente e gera/renderiza apenas o mapa mental")
    parser.add_argument("--reusar-mapa-mental", action="store_true", help="Reaproveita imagem de mapa mental existente quando disponível, sem chamar IA/PlantUML")
    parser.add_argument("--forcar-regeneracao-mapa-mental", action="store_true", help="Remove mapa mental existente e recria PlantUML/imagem quando a etapa de mapa mental for executada")
    parser.add_argument("--document-json", default="", help="Caminho de document.json existente")
    parser.add_argument("--init-project", default="", help="Cria app_bundle/projetos/<nome> com TOML, ZIPs placeholder e doi_manifest.csv")
    parser.add_argument("--project-type", default="paper", choices=["paper", "atividade", "prisma", "atividade_prisma", "paper_prisma"], help="Tipo de projeto para --init-project")
    parser.add_argument("--institution", default="fgv", help="Perfil institucional usado por --init-project, ex.: fgv")
    parser.add_argument("--base-dir", default="", help="Raiz do academic_pipeline ou app_bundle para --init-project")
    parser.add_argument("--overwrite-project", action="store_true", help="Permite sobrescrever arquivos seguros criados por --init-project")
    parser.add_argument("--make-doi-manifest", action="store_true", help="Gera doi_manifest.csv a partir de --input-zip ou --input-dir")
    parser.add_argument("--input-zip", default="", help="ZIP de documentos para --make-doi-manifest")
    parser.add_argument("--input-dir", default="", help="Pasta de documentos para --make-doi-manifest")
    parser.add_argument("--output", default="", help="Arquivo de saída para --make-doi-manifest")
    parser.add_argument("--output-dir", default="", help="Override de [paths].document_output_dir para a geração/renderização do documento")
    parser.add_argument("--work-dir", default="", help="Override de [paths].work_dir para extrações temporárias")
    parser.add_argument("--cache-dir", default="", help="Override de [paths].cache_dir para fulltext_cache")
    parser.add_argument("--research-output-dir", default="", help="Override de [paths].research_output_dir para relatório PRISMA")
    parser.add_argument("--output-prefix", default="", help="Override de [paths].document_prefix")
    parser.add_argument("--layout", default="", help="Override de [documento].layout, ex.: atividade_fgv")
    parser.add_argument("--tipo-conteudo", default="", help="Override de [documento].tipo_conteudo, ex.: resumo_artigos")
    parser.add_argument("--genero-academico", default="", help="Override de [documento].genero_academico, ex.: atividade")
    parser.add_argument("--no-output-subdir", action="store_true", help="Não cria subdiretório com document_prefix dentro de --output-dir/[paths].document_output_dir")
    parser.add_argument("--inspect-bib", default="", help="Inspeciona arquivo .bib e gera relatório .md/.json")
    parser.add_argument("--quality-report", action="store_true", help="Gera quality_report.md a partir de --document-json e opcionalmente --org")
    parser.add_argument("--bib", default="", help="Arquivo .bib opcional para --quality-report ou --check-institution-compliance")
    parser.add_argument("--docx", default="", help="Arquivo .docx opcional para --check-institution-compliance")
    parser.add_argument("--pdf", default="", help="Arquivo .pdf opcional para --check-institution-compliance")
    args = parser.parse_args()

    if args.gui:
        from academic_pipeline_gui import run_gui
        return run_gui()

    if args.tui:
        from academic_pipeline_tui import run_tui
        return run_tui(no_clear=bool(args.no_clear))

    if args.list_toml_profiles:
        from academic_pipeline_toml_generator_interativo import print_profiles
        print_profiles()
        return 0

    if args.init_toml:
        from academic_pipeline_toml_generator_interativo import generate_interactive
        generate_interactive(non_interactive_profile=args.toml_profile or None, no_clear=bool(args.no_clear))
        return 0

    if args.list_institutions:
        print(describe_institution_profiles())
        return 0

    if args.list_layouts:
        if not args.config:
            raise RuntimeError("--list-layouts exige --config caminho.toml")
        cfg_layouts = load_config(Path(args.config).expanduser().resolve())
        layouts = available_layouts(cfg_layouts)
        if not layouts:
            print("Nenhum layout declarado no perfil institucional.")
        else:
            print("Layouts disponíveis:")
            for layout_id, spec in layouts.items():
                desc = str(spec.get("description") or spec.get("descricao") or "").strip()
                genero = str(spec.get("genero_academico") or "").strip()
                print(f"- {layout_id}" + (f" ({genero})" if genero else "") + (f": {desc}" if desc else ""))
            resolved = resolve_layout_spec(cfg_layouts)
            print(f"Layout resolvido para este TOML: {resolved.id}")
        return 0

    if args.explain_profile:
        print(explain_profile(args.explain_profile))
        return 0

    if args.show_prompts:
        if not args.config:
            raise RuntimeError("--show-prompts exige --config caminho.toml")
        cfg_preview = load_config(Path(args.config).expanduser().resolve())
        print(json.dumps(prompt_report_for_cfg(cfg_preview), ensure_ascii=False, indent=2))
        return 0

    if args.init_project:
        base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else None
        result = init_project(args.init_project, project_type=args.project_type, base_dir=base_dir, overwrite=bool(args.overwrite_project), institution=args.institution)
        print("Projeto criado:")
        print(f"- Diretório: {result.project_dir}")
        print(f"- TOML: {result.config_path}")
        print(f"- DOI manifest: {result.doi_manifest_path}")
        print(f"- Documentos ZIP: {result.documentos_zip_path}")
        print(f"- Orientações ZIP: {result.orientacoes_zip_path}")
        print(f"- README: {result.readme_path}")
        return 0

    if args.make_doi_manifest:
        input_zip = Path(args.input_zip).expanduser().resolve() if args.input_zip else None
        input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
        if args.output:
            output = Path(args.output).expanduser().resolve()
        else:
            if input_zip:
                output = input_zip.parent / "doi_manifest.csv"
            elif input_dir:
                output = input_dir / "doi_manifest.csv"
            else:
                raise RuntimeError("Use --make-doi-manifest com --input-zip ou --input-dir.")
        result = make_doi_manifest(input_zip, input_dir, output, overwrite=True)
        print("DOI manifest gerado:")
        print(f"- Fonte: {result['source']}")
        print(f"- Saída: {result['output']}")
        print(f"- Arquivos listados: {result['total_files']}")
        return 0

    if args.inspect_bib:
        bib = Path(args.inspect_bib).expanduser().resolve()
        prefix = bib.with_name(bib.name + "_inspection")
        report = inspect_bib(bib, output_prefix=prefix)
        print(render_bib_inspection_markdown(report))
        print(f"Relatórios: {str(prefix)}.md e {str(prefix)}.json")
        return 0 if report.get("ok") else 1

    if args.quality_report:
        if not args.document_json:
            raise RuntimeError("--quality-report exige --document-json caminho/document.json")
        document_json = Path(args.document_json).expanduser().resolve()
        document = load_existing_document_json(document_json)
        org = Path(args.org).expanduser().resolve() if args.org else None
        bib_keys: list[str] = []
        if args.bib:
            from bibliography_manager import split_bib_entries, bib_entry_key
            bib_path = Path(args.bib).expanduser().resolve()
            if bib_path.exists():
                bib_keys = [k for e in split_bib_entries(bib_path.read_text(encoding='utf-8', errors='ignore')) if (k := bib_entry_key(e))]
        report = build_quality_report(document, org_path=org, bib_keys=bib_keys or list(document.bibliography.entries_used or []))
        out = document_json.with_suffix(".quality_report.md")
        write_quality_report(report, out)
        print(f"Relatório de qualidade: {out}")
        return 0 if report.get("ok") else 1

    cfg = _load_optional_config(args.config) if args.config else None
    if cfg:
        cfg = apply_cli_path_overrides(cfg, args)

    if args.somente_renderizar and args.somente_mapa_mental:
        raise RuntimeError("Use apenas um entre --somente-renderizar e --somente-mapa-mental.")
    if args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental:
        raise RuntimeError("Use apenas um entre --reusar-mapa-mental e --forcar-regeneracao-mapa-mental.")

    if args.write_prompt_lock:
        if not cfg:
            raise RuntimeError("--write-prompt-lock exige --config caminho.toml")
        out_dir, prefix = output_paths(cfg)
        lock_path = out_dir / f"{prefix}.prompt_lock.json"
        lock_md = out_dir / f"{prefix}.prompt_lock.md"
        lock = write_prompt_lock(cfg, lock_path)
        write_prompt_lock_markdown(lock, lock_md)
        print(f"Prompt lock gerado: {lock_path}")
        print(f"Prompt lock markdown: {lock_md}")
        return 0

    if args.check_institution_compliance:
        if not cfg:
            raise RuntimeError("--check-institution-compliance exige --config caminho.toml")
        out_dir, prefix = output_paths(cfg)
        org = Path(args.org).expanduser().resolve() if args.org else out_dir / f"{prefix}.org"
        bib = Path(args.bib).expanduser().resolve() if args.bib else out_dir / f"{prefix}.bib"
        docx = Path(args.docx).expanduser().resolve() if args.docx else out_dir / f"{prefix}.docx"
        pdf = Path(args.pdf).expanduser().resolve() if args.pdf else out_dir / f"{prefix}.pdf"
        report = run_institution_compliance(cfg, org_path=org, bib_path=bib, docx_path=docx, pdf_path=pdf)
        md_path, json_path = write_compliance_reports(report, out_dir / prefix)
        print(render_compliance_markdown(report))
        print(f"Relatórios: {md_path} e {json_path}")
        return 0 if report.get("ok") else 2

    if args.doctor:
        report = run_doctor(cfg)
        print_doctor_report(report)
        if cfg:
            out_dir, prefix = output_paths(cfg)
            write_json(out_dir / f"{prefix}.doctor_report.json", report)
        return 0 if report.get("ok") else 2

    if args.check_config:
        if not cfg:
            raise RuntimeError("--check-config exige --config caminho.toml")
        report = check_config(cfg)
        print_check_config_report(report)
        out_dir, prefix = output_paths(cfg)
        write_json(out_dir / f"{prefix}.check_config_report.json", report)
        return 0 if report.get("ok") else 2

    if args.recompile:
        return run_recompile(args, cfg)

    if not cfg:
        raise RuntimeError("Informe --config, ou use --doctor sem config.")

    cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
    out_dir, prefix = output_paths(cfg)
    work_dir, cache_dir = work_cache_paths(cfg, prefix)
    doc_cfg = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
    latex_cfg = cfg.get("latex", {}) if isinstance(cfg.get("latex"), dict) else {}
    config_dir = Path(str(cfg.get("__config_dir__"))).resolve()
    warnings: list[str] = []
    client: Any | None = None
    model = _openai_model_from_cfg(cfg)

    # Validação preventiva leve; não bloqueia warnings.
    stage("Validando configuração preventiva")
    precheck = check_config(cfg)
    if precheck.get("warnings"):
        warnings.extend(precheck["warnings"])
    if precheck.get("errors"):
        raise RuntimeError("Configuração inválida:\n- " + "\n- ".join(precheck["errors"]))

    document_json_path = Path(args.document_json).expanduser().resolve() if args.document_json else out_dir / f"{prefix}.document.json"

    if args.somente_mapa_mental:
        if not document_json_path.exists():
            raise FileNotFoundError(f"document.json não encontrado para --somente-mapa-mental: {document_json_path}")
        if not should_generate_mindmap(cfg):
            raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --somente-mapa-mental.")
        stage("Carregando document.json existente")
        document = load_existing_document_json(document_json_path)
        removed_mindmap_files: list[str] = []
        if args.forcar_regeneracao_mapa_mental:
            stage("Removendo mapa mental existente")
            removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
        mm_diag = None
        if args.reusar_mapa_mental:
            stage("Tentando reutilizar mapa mental existente")
            mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
            if not mm_diag:
                warnings.append("Mapa mental existente não encontrado; gerando novo mapa mental.")
        if not mm_diag:
            stage("Inicializando cliente OpenAI")
            client, model = make_client(model)
            stage("Gerando/renderizando apenas o mapa mental")
            mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
        if removed_mindmap_files:
            mm_diag = dict(mm_diag or {})
            mm_diag["removed_before_regeneration"] = removed_mindmap_files
        document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
        stage("Salvando document.json atualizado")
        write_json(document_json_path, document.model_dump())
        outputs = {
            "output_dir": str(out_dir),
            "document_json": str(document_json_path),
            "mindmap_puml": (mm_diag or {}).get("puml_path") if mm_diag else None,
            "mindmap_image": (mm_diag or {}).get("image_path") if mm_diag else None,
            "mindmap_reused": bool((mm_diag or {}).get("reused")),
            "mindmap_removed": removed_mindmap_files,
        }
        report = make_run_report(
            cfg=cfg,
            config_path=Path(str(cfg.get("__config_path__"))),
            out_dir=out_dir,
            prefix=prefix,
            model=model,
            outputs=outputs,
            warnings=warnings,
            extra={"mode": "somente_mapa_mental"},
        )
        write_json(out_dir / f"{prefix}.run_report.json", report)
        write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
        print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — mapa mental renderizado")
        if warnings:
            print("Avisos:")
            for w in warnings:
                print(f"- {w}")
        return 0

    prisma_outputs = None
    source_info: dict[str, Any] | None = None

    if args.somente_renderizar:
        if not document_json_path.exists():
            raise FileNotFoundError(f"document.json não encontrado para --somente-renderizar: {document_json_path}")
        stage("Carregando document.json existente")
        document = load_existing_document_json(document_json_path)
        stage("Resolvendo bibliografia para renderização")
        bib_path, bib_keys = resolve_bib_for_existing_document(document, document_json_path, out_dir, prefix)
        stage("Saneando document_model existente")
        document, leak_repairs = sanitize_document_model_technical_leaks(document)
        if leak_repairs:
            warnings.append("Menções técnicas removidas/reescritas no document_model existente: " + ", ".join(leak_repairs[:20]))
        document, raw_key_repairs = sanitize_document_model_raw_bibkeys(document, bib_keys)
        if raw_key_repairs:
            warnings.append("Chaves BibTeX cruas convertidas em citações LaTeX no document_model existente: " + ", ".join(raw_key_repairs[:20]))
        if args.forcar_regeneracao_mapa_mental:
            if not should_generate_mindmap(cfg):
                raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --forcar-regeneracao-mapa-mental.")
            stage("Removendo mapa mental existente")
            removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
            if removed_mindmap_files:
                warnings.append("Arquivos de mapa mental removidos antes da regeneração: " + ", ".join(removed_mindmap_files[:10]))
            stage("Inicializando cliente OpenAI para regenerar mapa mental")
            client, model = make_client(model)
            stage("Regenerando mapa mental antes da renderização")
            mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
            document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
            stage("Salvando document.json atualizado com novo mapa mental")
            write_json(document_json_path, document.model_dump())
        elif args.reusar_mapa_mental:
            stage("Tentando reutilizar mapa mental existente")
            mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
            if mm_diag:
                document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
                write_json(document_json_path, document.model_dump())
            else:
                warnings.append("--reusar-mapa-mental informado, mas nenhum mapa existente foi encontrado; renderização seguirá com figuras já registradas no document.json.")
    else:
        pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline"), dict) else {}
        executar_documento = bool(pipeline_cfg.get("executar_documento", True))
        stage("Inicializando cliente OpenAI")
        client, model = make_client(model)
        stage("Descobrindo e extraindo documentos locais")
        docs, source_info = discover_local_documents(cfg, work_dir)
        clean_cache = bool((cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}).get("limpar_cache_anterior", True))
        stage("Copiando documentos para fulltext_cache")
        copy_documents_to_fulltext_cache(docs, cache_dir, clean=clean_cache)
        stage("Carregando orientações do projeto")
        orientations = collect_orientation_docs(cfg, work_dir)
        stage("Gerando e validando bibliografia")
        bib_result = build_bibliography(cfg, docs, out_dir, prefix, client, model)
        bib_path = bib_result.bib_path
        bib_keys = bib_result.keys
        stage("Verificando geração de relatório PRISMA")
        prisma_outputs = run_prisma_report_outputs(cfg, docs, orientations, bib_result, out_dir, prefix) if prisma_enabled(cfg) else None

        if not executar_documento:
            outputs = {
                "output_dir": str(out_dir),
        "work_dir": str(work_dir),
        "cache_dir": str(cache_dir),
                "work_dir": str(work_dir),
                "cache_dir": str(cache_dir),
                "document_json": None,
                "org": None,
                "bib": str(bib_path),
                "pdf": None,
                "docx": None,
                "relatorio_pesquisa": prisma_outputs,
            }
            prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
            prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
            stage("Registrando prompt_lock")
            prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
            write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
            outputs["prompt_lock"] = str(prompt_lock_path)
            report = make_run_report(
                cfg=cfg,
                config_path=Path(str(cfg.get("__config_path__"))),
                out_dir=out_dir,
                prefix=prefix,
                model=model,
                outputs=outputs,
                warnings=warnings,
                extra={"mode": "research_only", "source_info": source_info, "work_dir": str(work_dir), "cache_dir": str(cache_dir)},
            )
            write_json(out_dir / f"{prefix}.run_report.json", report)
            write_json(out_dir / f"{prefix}.rc10_report.json", outputs)  # compatibilidade
            write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
            print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} concluído sem documento acadêmico")
            return 0

        resumo_cfg_for_stage = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
        if resumo_cfg_for_stage.get("ativo") and bool(resumo_cfg_for_stage.get("geracao_em_etapas", True)):
            stage("Gerando document.json canônico com IA em etapas")
        else:
            stage("Gerando document.json canônico com IA")
        document = build_document_model(
            client,
            model,
            cfg,
            docs,
            orientations,
            bib_keys,
            bib_path,
            progress=stage,
            checkpoint_dir=out_dir,
            prefix=prefix,
        )
        if not document.bibliography.entries_used:
            document.bibliography.entries_used = bib_keys
        if not document.bibliography.bib_path:
            document.bibliography.bib_path = bib_path.name
        stage("Saneando linguagem técnica do document_model")
        document, leak_repairs = sanitize_document_model_technical_leaks(document)
        if leak_repairs:
            warnings.append("Menções técnicas removidas/reescritas no document_model: " + ", ".join(leak_repairs[:20]))
        document, raw_key_repairs = sanitize_document_model_raw_bibkeys(document, bib_keys)
        if raw_key_repairs:
            warnings.append("Chaves BibTeX cruas convertidas em citações LaTeX no document_model: " + ", ".join(raw_key_repairs[:20]))
        stage("Validando document_model")
        raise_if_errors(validate_document_model(document, bib_keys), "Validação do document_model falhou")
        stage("Gerando/anexando mapa mental, se configurado")
        mm_diag = None
        if args.forcar_regeneracao_mapa_mental:
            removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
            if removed_mindmap_files:
                warnings.append("Arquivos de mapa mental removidos antes da regeneração: " + ", ".join(removed_mindmap_files[:10]))
        if args.reusar_mapa_mental:
            mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
            if not mm_diag:
                warnings.append("Mapa mental existente não encontrado; gerando novo mapa mental.")
        if not mm_diag:
            mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
        document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
        document.diagnostics.source_info_json = json.dumps(source_info or {}, ensure_ascii=False)
        if prisma_outputs:
            document.diagnostics.relatorio_pesquisa_json = json.dumps(prisma_outputs, ensure_ascii=False)
        stage("Salvando document.json")
        write_json(document_json_path, document.model_dump())

    stage("Renderizando ORG/LaTeX")
    org_path = out_dir / f"{prefix}.org"
    org_text = render_org_latex(document, org_path, bib_path.name if 'bib_path' in locals() else f"{prefix}.bib", cfg=cfg, bib_keys=bib_keys if 'bib_keys' in locals() else None)
    stage("Validando ORG renderizado")
    raise_if_errors(validate_org_text(org_text, bib_keys), "Validação do ORG renderizado falhou")

    pdf_path = None
    if bool(doc_cfg.get("exportar_pdf", True)):
        academic_writing = resolve_path(latex_cfg.get("org_latex_class_init"), config_dir)
        latex_extra = resolve_path(latex_cfg.get("latex_extra_path"), config_dir)
        pdf_engine = str(latex_cfg.get("pdf_engine") or "lualatex")
        stage("Compilando PDF via Emacs/LaTeX")
        pdf_path = run_compile_sequence(org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)

    docx_path = None
    docx_validation: dict[str, Any] | None = None
    if bool(doc_cfg.get("exportar_docx", True)):
        docx_cfg = cfg.get("docx", {}) if isinstance(cfg.get("docx"), dict) else {}
        ref = resolve_path(docx_cfg.get("reference_docx") or doc_cfg.get("docx_reference"), config_dir)
        stage("Renderizando DOCX")
        docx_path = render_docx(document, out_dir / f"{prefix}.docx", bib_path=bib_path, reference_docx=ref, cfg=cfg)
        docx_validation = validate_docx_file(docx_path, expected_title=document.metadata.titulo, require_references=bool(document.bibliography.entries_used))
        if docx_validation and docx_validation.get("warnings"):
            warnings.extend([f"DOCX: {w}" for w in docx_validation.get("warnings", [])])

    outputs = {
        "output_dir": str(out_dir),
        "document_json": str(document_json_path),
        "org": str(org_path),
        "bib": str(bib_path),
        "pdf": str(pdf_path) if pdf_path else None,
        "docx": str(docx_path) if docx_path else None,
        "relatorio_pesquisa": _json_or_none(getattr(document.diagnostics, "relatorio_pesquisa_json", "")) if getattr(document, "diagnostics", None) else prisma_outputs,
    }

    # Prompt lock: rastreabilidade exata dos prompts/diretivas usados.
    prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
    prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
    prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
    write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
    outputs["prompt_lock"] = str(prompt_lock_path)

    # Conformidade institucional: valida artefatos contra o perfil escolhido.
    stage("Executando conformidade institucional")
    compliance_report = run_institution_compliance(
        cfg,
        org_path=org_path,
        bib_path=bib_path,
        docx_path=docx_path,
        pdf_path=pdf_path,
    )
    compliance_md, compliance_json = write_compliance_reports(compliance_report, out_dir / prefix)
    outputs["compliance_report"] = str(compliance_md)
    if compliance_report.get("warnings"):
        warnings.extend([f"CONFORMIDADE: {w.get('message')}" for w in compliance_report.get("warnings", [])])
    if not compliance_report.get("ok"):
        warnings.extend([f"CONFORMIDADE CRÍTICA: {e.get('message')}" for e in compliance_report.get("errors", [])])

    stage("Gerando relatório de qualidade")
    quality = build_quality_report(document, org_path=org_path, bib_keys=bib_keys)
    quality_path = out_dir / f"{prefix}.quality_report.md"
    write_quality_report(quality, quality_path)
    if quality.get("warnings"):
        warnings.extend([f"QUALIDADE: {w}" for w in quality.get("warnings", [])])
    outputs["quality_report"] = str(quality_path)

    report = make_run_report(
        cfg=cfg,
        config_path=Path(str(cfg.get("__config_path__"))),
        out_dir=out_dir,
        prefix=prefix,
        model=model,
        outputs=outputs,
        warnings=warnings,
        extra={
            "mode": "somente_renderizar" if args.somente_renderizar else "full",
            "work_dir": str(work_dir),
            "cache_dir": str(cache_dir),
            "precheck": precheck,
            "docx_validation": docx_validation,
        },
    )
    write_json(out_dir / f"{prefix}.run_report.json", report)
    write_json(out_dir / f"{prefix}.rc10_report.json", outputs)  # compatibilidade com scripts antigos
    write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
    print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} concluído")
    if warnings:
        print("Avisos:")
        for w in warnings:
            print(f"- {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

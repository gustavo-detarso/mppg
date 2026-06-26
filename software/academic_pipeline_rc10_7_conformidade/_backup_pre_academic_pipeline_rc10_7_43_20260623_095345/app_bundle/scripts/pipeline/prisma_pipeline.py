#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from bibliography_manager import BibBuildResult
from corpus_manager import SourceDoc
from latex_compile import run_compile_sequence
from prisma_builder import build_prisma_report
from prisma_validator import raise_if_prisma_errors, validate_prisma_report
from render_prisma_docx import render_prisma_docx
from render_prisma_flow import render_prisma_flow_svg
from render_prisma_org import render_prisma_org
from render_prisma_xlsx import render_prisma_xlsx
from utils import resolve_path, write_json


def prisma_enabled(cfg: dict[str, Any]) -> bool:
    rel = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}
    return bool(rel.get("ativo", False))


def prisma_output_paths(cfg: dict[str, Any], document_out_dir: Path, document_prefix: str) -> tuple[Path, str]:
    rel = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    prefix = str(paths.get("research_prefix") or rel.get("titulo_slug") or f"relatorio_prisma_{document_prefix}").strip()
    base = resolve_path(paths.get("research_output_dir") or "../../output/pesquisa", config_dir)
    out_dir = base / prefix if bool(paths.get("create_research_subdir", True)) else base
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, prefix


def _copy_bib_for_prisma(bib_result: BibBuildResult, out_dir: Path) -> Path | None:
    if not bib_result.bib_path or not bib_result.bib_path.exists():
        return None
    dest = out_dir / bib_result.bib_path.name
    if bib_result.bib_path.resolve() != dest.resolve():
        shutil.copy2(bib_result.bib_path, dest)
    return dest


def run_prisma_report_outputs(
    cfg: dict[str, Any],
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    bib_result: BibBuildResult,
    document_out_dir: Path,
    document_prefix: str,
) -> dict[str, Any] | None:
    if not prisma_enabled(cfg):
        return None
    rel = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}
    latex_cfg = cfg.get("latex", {}) if isinstance(cfg.get("latex"), dict) else {}
    docx_cfg = cfg.get("docx", {}) if isinstance(cfg.get("docx"), dict) else {}
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()

    out_dir, prefix = prisma_output_paths(cfg, document_out_dir, document_prefix)
    local_bib = _copy_bib_for_prisma(bib_result, out_dir)
    report = build_prisma_report(cfg, docs, orientations, bib_result, out_dir, prefix)
    messages = validate_prisma_report(report, strict=bool(rel.get("validar", True)))
    if bool(rel.get("falhar_se_invalido", True)):
        raise_if_prisma_errors(messages)
    else:
        for msg in messages:
            report.diagnostics.avisos.append(msg)

    paths: dict[str, Any] = {"output_dir": str(out_dir)}
    json_path = out_dir / f"{prefix}.prisma_report.json"
    if bool(rel.get("exportar_json", True)):
        write_json(json_path, report.model_dump())
        paths["json"] = str(json_path)

    flow_svg_path = None
    if bool(rel.get("exportar_fluxograma", True)):
        flow_svg_path = render_prisma_flow_svg(report, out_dir / f"{prefix}_fluxo_prisma.svg")
        paths["fluxograma_svg"] = str(flow_svg_path)

    org_path = out_dir / f"{prefix}.org"
    if bool(rel.get("exportar_org", True)) or bool(rel.get("exportar_pdf", True)):
        render_prisma_org(report, org_path, local_bib.name if local_bib else None, cfg=cfg)
        paths["org"] = str(org_path)
        if local_bib:
            paths["bib"] = str(local_bib)

    if bool(rel.get("exportar_pdf", True)):
        academic_writing = resolve_path(latex_cfg.get("org_latex_class_init"), config_dir)
        latex_extra = resolve_path(latex_cfg.get("latex_extra_path"), config_dir)
        pdf_engine = str(latex_cfg.get("pdf_engine") or "lualatex")
        pdf_path = run_compile_sequence(org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
        paths["pdf"] = str(pdf_path)

    if bool(rel.get("exportar_docx", True)):
        ref = resolve_path(rel.get("reference_docx") or docx_cfg.get("reference_docx"), config_dir)
        docx_path = render_prisma_docx(report, out_dir / f"{prefix}.docx", reference_docx=ref, flow_svg_path=flow_svg_path)
        paths["docx"] = str(docx_path)

    if bool(rel.get("exportar_xlsx", True)):
        xlsx_path = render_prisma_xlsx(report, out_dir / f"{prefix}.xlsx")
        paths["xlsx"] = str(xlsx_path)

    summary_path = out_dir / f"{prefix}.prisma_outputs.json"
    write_json(summary_path, paths)
    paths["summary"] = str(summary_path)
    return paths

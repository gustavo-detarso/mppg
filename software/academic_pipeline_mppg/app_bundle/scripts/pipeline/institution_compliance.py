#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformidade institucional auditável para academic_pipeline rc10.7.

A validação aqui é deliberadamente pragmática: ela checa a configuração e os
artefatos gerados contra regras institucionais registradas em
app_bundle/institutions/<perfil>/validators/. Nem toda regra visual de PDF/DOCX
é verificável por texto, mas o relatório explicita o que foi aprovado, advertido
ou marcado como pendência.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .bibliography_manager import split_bib_entries, bib_entry_key
else:
    from bibliography_manager import split_bib_entries, bib_entry_key
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .diagnostics import now_iso, PIPELINE_VERSION
else:
    from diagnostics import now_iso, PIPELINE_VERSION
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .institution_profiles import find_app_bundle
else:
    from institution_profiles import find_app_bundle
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import normalize_title_loose, resolve_path, write_json, write_text
else:
    from utils import normalize_title_loose, resolve_path, write_json, write_text


@dataclass
class ComplianceItem:
    id: str
    status: str  # pass | warn | fail | info
    message: str
    detail: str = ""


def _cfg_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = cfg.get(name, {})
    return sec if isinstance(sec, dict) else {}


def _config_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()


def _doc_type(cfg: dict[str, Any]) -> str:
    v = str(_cfg_section(cfg, "documento").get("tipo_documento") or "paper").strip().lower()
    return {"dissertação": "dissertacao", "dissertation": "dissertacao"}.get(v, v)


def _profile_dir(cfg: dict[str, Any]) -> Path | None:
    raw = cfg.get("__institution_profile_path__")
    if not raw:
        return None
    p = Path(str(raw)).expanduser().resolve()
    return p.parent if p.exists() else None


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_institution_rules(cfg: dict[str, Any]) -> dict[str, Any]:
    pdir = _profile_dir(cfg)
    if not pdir:
        return {}
    validators = pdir / "validators"
    merged: dict[str, Any] = {}
    for name in ["fgv_rules.toml", f"{_doc_type(cfg)}_rules.toml"]:
        p = validators / name
        data = _load_toml(p)
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
    return merged


def _style_from_cfg(cfg: dict[str, Any]) -> str:
    bib = _cfg_section(cfg, "bibliografia")
    doc = _cfg_section(cfg, "documento")
    return str(bib.get("latex_style") or bib.get("estilo_citacao") or doc.get("estilo_citacao") or "").strip().lower()


def _read(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _add(items: list[ComplianceItem], status: str, id_: str, message: str, detail: str = "") -> None:
    items.append(ComplianceItem(id=id_, status=status, message=message, detail=detail))


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text or "", flags=re.IGNORECASE | re.MULTILINE) is not None


def _strip_nonvisible_org_regions(org_text: str) -> str:
    """Retorna apenas o conteúdo aproximadamente visível do ORG.

    A conformidade institucional deve evitar falso positivo em metadados,
    cabeçalhos LaTeX, comentários, caminhos de imagem e caminhos do .bib.
    Essas regiões não aparecem no PDF/DOCX e podem conter nomes internos do
    software, como academic_pipeline.
    """
    text = org_text or ""
    text = re.sub(r"(?ims)^\s*#\+begin_comment\b.*?^\s*#\+end_comment\s*$", "\n", text)
    text = re.sub(r"(?m)%.*$", "", text)
    text = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#+") and not line.lstrip().startswith("# ")
    )
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", r"\\includegraphics{}", text)
    text = re.sub(r"\\addbibresource\{[^}]*\}", r"\\addbibresource{}", text)
    return text


def _find_technical_terms_visible(org_text: str) -> list[str]:
    visible = _strip_nonvisible_org_regions(org_text)
    patterns = {
        "fulltext_cache": r"(?i)(?<![A-Za-z0-9_])fulltext_cache(?![A-Za-z0-9_])",
        "metadados incompletos": r"(?i)\bmetadados\s+incompletos\b",
        "pipeline": r"(?i)(?<![A-Za-z0-9_])pipeline(?![A-Za-z0-9_])",
        "<empty citation>": r"(?i)<empty citation>",
    }
    return [name for name, pat in patterns.items() if re.search(pat, visible)]


def _check_config(cfg: dict[str, Any], rules: dict[str, Any], items: list[ComplianceItem]) -> None:
    doc = _cfg_section(cfg, "documento")
    latex = _cfg_section(cfg, "latex")
    bib = _cfg_section(cfg, "bibliografia")
    fmt = rules.get("layout", {}) if isinstance(rules.get("layout"), dict) else {}
    rbib = rules.get("bibliografia", {}) if isinstance(rules.get("bibliografia"), dict) else {}
    doc_type = _doc_type(cfg)

    if cfg.get("__institution_profile_name__"):
        _add(items, "pass", "institution.profile", f"Perfil institucional carregado: {cfg.get('__institution_profile_name__')}", str(cfg.get("__institution_profile_path__")))
    else:
        _add(items, "warn", "institution.profile", "Nenhum perfil institucional foi carregado.")

    # Margens/fonte são verificadas por declaração de regra/perfil; a renderização final depende do .sty/template.
    if fmt:
        _add(items, "info", "layout.rules", "Regras de layout institucionais carregadas.", str(fmt))
    else:
        _add(items, "warn", "layout.rules", "Regras de layout institucionais não foram encontradas.")

    expected_system = normalize_title_loose(str(rbib.get("sistema_citacao") or "autor-data"))
    if expected_system and "autor" in expected_system:
        style = _style_from_cfg(cfg)
        if style in {"abnt", "apa", "authoryear", "chicago"}:
            _add(items, "pass", "bibliography.author_date", f"Estilo bibliográfico compatível com autor-data: {style}")
        else:
            _add(items, "fail", "bibliography.author_date", f"Estilo bibliográfico pode não ser autor-data: {style or 'não informado'}")

    expected_style = str(rbib.get("estilo_padrao") or "").strip().lower()
    if expected_style:
        actual = _style_from_cfg(cfg)
        if actual == expected_style:
            _add(items, "pass", "bibliography.default_style", f"Estilo bibliográfico padrão da instituição aplicado: {actual}")
        else:
            _add(items, "warn", "bibliography.default_style", f"Estilo bibliográfico diferente do padrão institucional: {actual or 'não informado'}", f"Padrão: {expected_style}")

    if bool(rbib.get("notas_referencia") is False):
        _add(items, "pass", "bibliography.reference_notes", "Perfil institucional não adota notas de referência como padrão.")

    if doc_type in {"paper", "atividade"}:
        program = normalize_title_loose(str(doc.get("program_name") or ""))
        course = normalize_title_loose(str(doc.get("course_name") or _cfg_section(cfg, "atividade").get("curso") or ""))
        if program and course and program == course:
            _add(items, "fail", "cover.duplicate_program_course", "program_name e course_name estão iguais; isso pode duplicar a capa.")
        else:
            _add(items, "pass", "cover.duplicate_program_course", "Não foi detectada duplicidade program_name/course_name.")

    aw = resolve_path(latex.get("org_latex_class_init"), _config_dir(cfg)) if latex.get("org_latex_class_init") else None
    if aw and aw.exists():
        _add(items, "pass", "latex.academic_writing", "academic-writing.el encontrado.", str(aw))
    else:
        _add(items, "warn", "latex.academic_writing", "academic-writing.el não encontrado ou não informado.", str(aw or ""))


def _check_org(cfg: dict[str, Any], rules: dict[str, Any], org_path: Path | None, items: list[ComplianceItem]) -> None:
    text = _read(org_path)
    if not text:
        _add(items, "warn", "org.exists", "ORG não informado ou não encontrado.", str(org_path or ""))
        return
    _add(items, "pass", "org.exists", "ORG encontrado.", str(org_path))
    lower = normalize_title_loose(text)
    doc_type = _doc_type(cfg)
    style = _style_from_cfg(cfg)

    if "<empty citation>" in text or "[cite:" in text or "[cite/" in text:
        _add(items, "fail", "org.citations_clean", "ORG contém citações não saneadas ou <empty citation>.")
    else:
        _add(items, "pass", "org.citations_clean", "Não há <empty citation> nem Org Cite cru no ORG.")

    if "\\printbibliography" in text or "#+PRINT_BIBLIOGRAPHY" in text.upper() or "referencias" in lower:
        _add(items, "pass", "org.references", "Referências/bibliografia identificadas no ORG.")
    else:
        _add(items, "fail", "org.references", "Não identifiquei seção de referências/bibliografia no ORG.")

    if style:
        if re.search(r"style\s*=\s*" + re.escape(style), text, flags=re.IGNORECASE) or f"biblatex {style}" in lower:
            _add(items, "pass", "org.biblatex_style", f"Estilo BibLaTeX/Org Cite `{style}` identificado no ORG.")
        else:
            _add(items, "warn", "org.biblatex_style", f"Não identifiquei claramente o estilo `{style}` no ORG.")

    found = _find_technical_terms_visible(text)
    if found:
        _add(items, "warn", "org.technical_leaks", "Possíveis menções técnicas no texto visível do ORG.", ", ".join(found))
    else:
        _add(items, "pass", "org.technical_leaks", "Não foram detectadas menções técnicas proibidas no texto visível.")

    if doc_type == "atividade":
        if "ficha tecnica" in lower:
            # A ficha técnica pode aparecer como heading Org não numerado OU como bloco
            # LaTeX visual (tcolorbox). Neste último caso, ela não gera numeração de
            # seção e portanto também está correta.
            has_unnumbered_heading = bool(re.search(r"(?is)^\*+\s+Ficha\s+T[eé]cnica.*?:UNNUMBERED:\s*t", text))
            has_raw_visual_box = bool(re.search(r"(?is)\\begin\{tcolorbox\}\[[^\]]*title\s*=\s*Ficha\s+T[eé]cnica", text))
            if has_unnumbered_heading or has_raw_visual_box:
                _add(items, "pass", "activity.ficha_tecnica", "Ficha Técnica identificada como elemento não numerado/visual.")
            else:
                _add(items, "warn", "activity.ficha_tecnica", "Ficha Técnica identificada, mas a forma não numerada não pôde ser confirmada.")
        else:
            _add(items, "warn", "activity.ficha_tecnica", "Ficha Técnica não identificada no ORG de atividade.")

    if doc_type == "dissertacao":
        checks = {
            "dissertation.cover": r"\\capa\b",
            "dissertation.title_page": r"\\folhaderosto\b",
            "dissertation.approval": r"\\folhadeaprovacao\b|folha de aprova",
            "dissertation.resumo": r"\bRESUMO\b|\*+\s+Resumo",
            "dissertation.abstract": r"\bABSTRACT\b|\*+\s+Abstract",
            "dissertation.summary": r"\\tableofcontents|\bSUMARIO\b|\bSUMÁRIO\b",
        }
        for id_, pat in checks.items():
            if _contains(text, pat):
                _add(items, "pass", id_, f"{id_} identificado.")
            else:
                # ficha catalográfica pode ser pendência de versão final
                status = "warn" if id_.endswith("approval") else "fail"
                _add(items, status, id_, f"{id_} não identificado no ORG.")

    # Mapa mental depois das referências
    if "mapa mental" in lower:
        ref_pos = lower.rfind("referencias") if "referencias" in lower else lower.rfind("printbibliography")
        mm_pos = lower.rfind("mapa mental")
        if ref_pos != -1 and mm_pos > ref_pos:
            _add(items, "pass", "mindmap.after_references", "Mapa mental aparece depois das referências.")
        else:
            _add(items, "warn", "mindmap.after_references", "Mapa mental foi detectado, mas sua posição depois das referências não pôde ser confirmada.")


def _check_bib(bib_path: Path | None, items: list[ComplianceItem]) -> None:
    text = _read(bib_path)
    if not text:
        _add(items, "warn", "bib.exists", "BIB não informado ou não encontrado.", str(bib_path or ""))
        return
    entries = split_bib_entries(text)
    keys = [k for e in entries if (k := bib_entry_key(e))]
    if entries:
        _add(items, "pass", "bib.entries", f"BIB contém {len(entries)} entrada(s).")
    else:
        _add(items, "fail", "bib.entries", "BIB não contém entradas identificáveis.")
    bad_terms = ["jourarticle", "<scp>", "Metadados bibliográficos", "Material fornecido pelo professor"]
    found = [t for t in bad_terms if t.lower() in text.lower()]
    if found:
        _add(items, "warn", "bib.noise", "BIB contém ruídos conhecidos.", ", ".join(found))
    else:
        _add(items, "pass", "bib.noise", "Não detectei ruídos bibliográficos conhecidos.")
    if len(keys) != len(set(keys)):
        _add(items, "warn", "bib.duplicate_keys", "Há chaves BibTeX duplicadas.")
    else:
        _add(items, "pass", "bib.duplicate_keys", "Não há chaves BibTeX duplicadas.")


def _check_docx(docx_path: Path | None, items: list[ComplianceItem]) -> None:
    if not docx_path or not docx_path.exists():
        _add(items, "info", "docx.exists", "DOCX não informado ou não gerado.", str(docx_path or ""))
        return
    try:
        from docx import Document
        doc = Document(str(docx_path))
        _add(items, "pass", "docx.exists", "DOCX encontrado.", str(docx_path))
        section = doc.sections[0]
        # python-docx usa EMU; converter para cm aproximado
        margins = {
            "top": round(section.top_margin.cm, 2),
            "left": round(section.left_margin.cm, 2),
            "right": round(section.right_margin.cm, 2),
            "bottom": round(section.bottom_margin.cm, 2),
        }
        if abs(margins["top"] - 3) <= 0.15 and abs(margins["left"] - 3) <= 0.15 and abs(margins["right"] - 2) <= 0.15 and abs(margins["bottom"] - 2) <= 0.15:
            _add(items, "pass", "docx.margins", "Margens DOCX compatíveis com 3/3/2/2 cm.", str(margins))
        else:
            _add(items, "warn", "docx.margins", "Margens DOCX diferem de 3/3/2/2 cm.", str(margins))
    except Exception as exc:
        _add(items, "warn", "docx.inspect", f"Não foi possível inspecionar DOCX: {exc}")


def run_institution_compliance(
    cfg: dict[str, Any],
    *,
    org_path: Path | None = None,
    bib_path: Path | None = None,
    docx_path: Path | None = None,
    pdf_path: Path | None = None,
) -> dict[str, Any]:
    rules = load_institution_rules(cfg)
    items: list[ComplianceItem] = []
    _check_config(cfg, rules, items)
    _check_org(cfg, rules, org_path, items)
    _check_bib(bib_path, items)
    _check_docx(docx_path, items)
    if pdf_path:
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            _add(items, "pass", "pdf.exists", "PDF encontrado e não vazio.", str(pdf_path))
        else:
            _add(items, "warn", "pdf.exists", "PDF informado, mas não encontrado ou vazio.", str(pdf_path))

    errors = [asdict(i) for i in items if i.status == "fail"]
    warnings = [asdict(i) for i in items if i.status == "warn"]
    return {
        "version": PIPELINE_VERSION,
        "generated_at": now_iso(),
        "ok": not errors,
        "institution_profile": cfg.get("__institution_profile_name__"),
        "institution_profile_path": cfg.get("__institution_profile_path__"),
        "document_type": _doc_type(cfg),
        "rules_loaded": bool(rules),
        "artifacts": {
            "org": str(org_path) if org_path else None,
            "bib": str(bib_path) if bib_path else None,
            "docx": str(docx_path) if docx_path else None,
            "pdf": str(pdf_path) if pdf_path else None,
        },
        "errors": errors,
        "warnings": warnings,
        "items": [asdict(i) for i in items],
    }


def render_compliance_markdown(report: dict[str, Any]) -> str:
    status = "OK" if report.get("ok") else "ATENÇÃO"
    lines = [
        "# Relatório de conformidade institucional",
        "",
        f"- Status geral: **{status}**",
        f"- Versão do pipeline: `{report.get('version')}`",
        f"- Gerado em: `{report.get('generated_at')}`",
        f"- Perfil institucional: `{report.get('institution_profile') or 'nenhum'}`",
        f"- Tipo de documento: `{report.get('document_type')}`",
        "",
        "## Artefatos avaliados",
        "",
    ]
    for k, v in (report.get("artifacts") or {}).items():
        lines.append(f"- {k}: `{v}`")
    groups = [("fail", "Pendências críticas"), ("warn", "Avisos"), ("pass", "Itens aprovados"), ("info", "Informações")]
    items = report.get("items") or []
    for status_key, title in groups:
        subset = [i for i in items if i.get("status") == status_key]
        lines.extend(["", f"## {title}", ""])
        if not subset:
            lines.append("Nenhum item.")
            continue
        for item in subset:
            detail = f" — {item.get('detail')}" if item.get("detail") else ""
            lines.append(f"- `{item.get('id')}`: {item.get('message')}{detail}")
    return "\n".join(lines).strip() + "\n"


def write_compliance_reports(report: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    json_path = output_prefix.with_suffix(".compliance_report.json")
    md_path = output_prefix.with_suffix(".compliance_report.md")
    write_json(json_path, report)
    write_text(md_path, render_compliance_markdown(report))
    return md_path, json_path

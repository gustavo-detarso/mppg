#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relatório de qualidade textual para academic_pipeline rc10.4."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .document_model import AcademicDocument, Block, Citation, TextSpan
else:
    from document_model import AcademicDocument, Block, Citation, TextSpan
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import normalize_title_loose, write_json, write_text
else:
    from utils import normalize_title_loose, write_json, write_text

TECHNICAL_TERMS = [
    "metadados incompletos",
    "metadados inferidos",
    "cache local",
    "fulltext_cache",
    "limitação de acesso",
    "extração textual",
    "OCR",
    "pipeline",
    "material fornecido pelo professor",
]


def _strip_non_visible_org_regions(text: str) -> str:
    """Remove regiões técnicas do Org antes da varredura de qualidade."""
    text = re.sub(r"(?ims)^\s*#\+begin_comment\b.*?^\s*#\+end_comment\s*$", "\n", text or "")
    text = re.sub(r"(?ims)^\s*#\+begin_export\s+latex\b.*?^\s*#\+end_export\s*$", "\n", text)
    text = re.sub(r"(?im)^\s*#.*$", "", text)
    return text


def _technical_terms_found(text: str) -> list[str]:
    """Detecta termos técnicos proibidos evitando falso positivo por substring.

    O caso que motivou a correção foi `ocr` dentro de `democracia`.
    Termos são buscados com fronteira alfanumérica depois de normalização.
    """
    visible = _strip_non_visible_org_regions(text)
    normalized = normalize_title_loose(visible)
    found: list[str] = []
    for term in TECHNICAL_TERMS:
        raw = str(term)
        term_norm = normalize_title_loose(raw)
        if not term_norm:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(raw)
    return found


def _words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or "", flags=re.UNICODE))


def _block_text(block: Block) -> str:
    if block.text:
        return block.text
    parts: list[str] = []
    for span in block.content or []:
        if isinstance(span, TextSpan):
            parts.append(span.text)
        elif isinstance(span, Citation):
            parts.append(" ".join(span.keys))
    if block.items:
        parts.extend(block.items)
    if block.table:
        parts.extend(block.table.headers)
        for row in block.table.rows:
            parts.extend(row)
    return " ".join(parts)


def _section_text(section) -> str:
    return "\n".join(_block_text(block) for block in section.blocks)


def _collect_citations(document: AcademicDocument) -> list[str]:
    keys: list[str] = []
    for section in document.sections:
        for block in section.blocks:
            for span in block.content or []:
                if isinstance(span, Citation):
                    for key in span.keys:
                        if key and key not in keys:
                            keys.append(key)
    return keys


def _org_scan(org_path: Path | None) -> dict[str, Any]:
    if not org_path or not org_path.exists():
        return {}
    text = org_path.read_text(encoding="utf-8", errors="ignore")
    visible = _strip_non_visible_org_regions(text)
    return {
        "path": str(org_path),
        "contains_empty_citation": "<empty citation>" in text,
        "contains_org_cite": bool(re.search(r"\[cite[:/]", text)),
        "technical_terms_found": _technical_terms_found(text),
        "raw_biblike_tokens": sorted(set(re.findall(r"\b[a-z]+\d{4}_[a-z0-9_]+\b", visible)))[:50],
    }


def build_quality_report(document: AcademicDocument, *, org_path: Path | None = None, bib_keys: list[str] | None = None) -> dict[str, Any]:
    bib_keys = bib_keys or list(document.bibliography.entries_used or [])
    section_stats = []
    total_words = 0
    warnings: list[str] = []
    sections = list(document.sections or [])
    for idx, section in enumerate(sections):
        text = _section_text(section)
        wc = _words(text)
        total_words += wc
        section_stats.append({"id": section.id, "title": section.title, "level": section.level, "words": wc, "blocks": len(section.blocks)})
        has_child_after = any((getattr(s, "level", 1) or 1) > (section.level or 1) for s in sections[idx + 1:])
        is_container_heading = len(section.blocks or []) == 0 and has_child_after
        if wc < 80 and not is_container_heading:
            warnings.append(f"Seção possivelmente curta: {section.title} ({wc} palavras)")
    cited = _collect_citations(document)
    missing_in_bib = sorted(k for k in cited if bib_keys and k not in set(bib_keys))
    if missing_in_bib:
        warnings.append("Há citações sem chave no .bib: " + ", ".join(missing_in_bib))
    if document.bibliography.entries_used:
        not_cited = sorted(k for k in document.bibliography.entries_used if k not in set(cited))
    else:
        not_cited = []
    if not any("conclus" in normalize_title_loose(s.title) or "consideracoes finais" in normalize_title_loose(s.title) for s in document.sections):
        warnings.append("Não identifiquei seção de conclusão/considerações finais.")
    if total_words < 900 and document.metadata.tipo_documento in {"paper", "atividade", "pesquisa"}:
        warnings.append(f"Texto possivelmente curto para {document.metadata.tipo_documento}: {total_words} palavras.")
    org_diagnostics = _org_scan(org_path)
    if org_diagnostics.get("contains_empty_citation"):
        warnings.append("ORG contém <empty citation>.")
    if org_diagnostics.get("contains_org_cite"):
        warnings.append("ORG contém [cite:] ou [cite/...].")
    if org_diagnostics.get("technical_terms_found"):
        warnings.append("ORG contém termos técnicos proibidos: " + ", ".join(org_diagnostics["technical_terms_found"]))
    return {
        "title": document.metadata.titulo,
        "type": document.metadata.tipo_documento,
        "total_words": total_words,
        "sections": section_stats,
        "citations_total": len(cited),
        "cited_keys": cited,
        "bibliography_entries_used_total": len(document.bibliography.entries_used or []),
        "bibliography_entries_not_cited": not_cited,
        "missing_in_bib": missing_in_bib,
        "org_diagnostics": org_diagnostics,
        "warnings": warnings,
        "ok": not warnings,
    }


def render_quality_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Relatório de qualidade textual",
        "",
        f"- Título: {report.get('title', '')}",
        f"- Tipo: {report.get('type', '')}",
        f"- Total de palavras: {report.get('total_words', 0)}",
        f"- Citações únicas: {report.get('citations_total', 0)}",
        f"- Status: {'OK' if report.get('ok') else 'REVISAR'}",
        "",
        "## Palavras por seção",
        "",
        "| Seção | Palavras | Blocos |",
        "|---|---:|---:|",
    ]
    for section in report.get("sections") or []:
        lines.append(f"| {section.get('title')} | {section.get('words')} | {section.get('blocks')} |")
    warnings = report.get("warnings") or []
    lines += ["", "## Alertas", ""]
    if warnings:
        lines += [f"- {w}" for w in warnings]
    else:
        lines.append("- Nenhum alerta relevante.")
    not_cited = report.get("bibliography_entries_not_cited") or []
    if not_cited:
        lines += ["", "## Referências previstas mas não citadas", ""]
        lines += [f"- `{k}`" for k in not_cited]
    return "\n".join(lines).rstrip() + "\n"


def write_quality_report(report: dict[str, Any], output_md: Path) -> None:
    write_text(output_md, render_quality_markdown(report))
    write_json(output_md.with_suffix(".json"), report)

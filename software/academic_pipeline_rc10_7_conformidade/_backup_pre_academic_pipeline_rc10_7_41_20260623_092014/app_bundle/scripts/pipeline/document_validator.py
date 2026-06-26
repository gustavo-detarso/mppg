#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any
from document_model import AcademicDocument, Citation, TextSpan, Block
from citation_renderer import extract_cited_keys_from_model_inline, extract_latex_cited_keys
from utils import normalize_title_loose

TECHNICAL_LEAK_TERMS = [
    "metadados incompletos", "metadados inferidos", "metadados bibliográficos",
    "cache local", "fulltext_cache", "limitação de acesso", "extração textual", "OCR",
    "pipeline", "documentos processados", "material fornecido pelo professor", "fornecido pelo professor",
]

# Substituições seguras para quando a IA deixa escapar linguagem operacional.
# A regra é preservar o sentido acadêmico sem expor bastidores técnicos.
TECHNICAL_LEAK_REPLACEMENTS = [
    (r"(?i)\bOCR\b", "leitura do documento"),
    (r"(?i)\bpipeline\b", "processo de elaboração"),
    (r"(?i)fulltext_cache", "corpus local"),
    (r"(?i)cache\s+local", "corpus local"),
    (r"(?i)metadados\s+bibliogr[aá]ficos", "informações bibliográficas"),
    (r"(?i)metadados\s+incompletos", "informações bibliográficas parciais"),
    (r"(?i)metadados\s+inferidos", "informações bibliográficas estimadas"),
    (r"(?i)limita[cç][aã]o\s+de\s+acesso", "limitação do material disponível"),
    (r"(?i)extra[cç][aã]o\s+textual", "leitura do material"),
    (r"(?i)documentos\s+processados", "textos analisados"),
    (r"(?i)material\s+fornecido\s+pelo\s+professor", "corpus da atividade"),
    (r"(?i)fornecido\s+pelo\s+professor", "integrante do corpus"),
]


def _technical_term_pattern(term: str) -> str:
    """Regex de termo técnico sobre texto já normalizado.

    A validação anterior usava `term in texto`, o que gerava falso positivo:
    a sigla OCR aparecia dentro de palavras acadêmicas como "democracia".
    Esta função exige fronteiras de palavra para termos curtos e fronteiras
    de frase para termos compostos.
    """
    normalized = normalize_title_loose(term)
    tokens = [t for t in normalized.split() if t]
    if not tokens:
        return r"a^"
    body = r"\s+".join(re.escape(t) for t in tokens)
    return rf"(?<![a-z0-9]){body}(?![a-z0-9])"


def find_technical_leaks_in_text(value: str) -> list[str]:
    """Retorna termos técnicos realmente presentes no texto visível.

    Usa matching por fronteira de palavra/frase para evitar falsos positivos,
    especialmente OCR dentro de "democracia".
    """
    low = normalize_title_loose(value or "")
    found: list[str] = []
    for term in TECHNICAL_LEAK_TERMS:
        if re.search(_technical_term_pattern(term), low):
            found.append(term)
    return found


def _clean_visible_text(value: str) -> str:
    text = str(value or "")
    for pattern, replacement in TECHNICAL_LEAK_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


def _replace_raw_bibkeys_with_latex_cites(value: str, bib_keys: list[str]) -> str:
    r"""Converte chaves BibTeX cruas deixadas pela IA em comandos \parencite.

    A IA deve gerar objetos Citation, mas em alguns casos ela escreve a chave
    literal no texto (ex.: geet2019_policy_design). Isso quebra a validação do
    ORG. Esta rotina é conservadora: não mexe em chaves já dentro de comandos
    \parencite/\textcite e trata também o caso comum "(chave)".
    """
    text = str(value or "")
    if not text or not bib_keys:
        return text
    for key in sorted({str(k).strip().lstrip("@") for k in bib_keys if str(k).strip()}, key=len, reverse=True):
        if not key:
            continue
        # Primeiro, substitui ocorrências isoladas entre parênteses.
        text = re.sub(
            rf"(?<![\\{{,])\(\s*{re.escape(key)}\s*\)",
            rf"\\parencite{{{key}}}",
            text,
        )
        # Depois, substitui a chave nua, desde que não esteja dentro de comando de citação.
        text = re.sub(
            rf"(?<![\\{{,])\b{re.escape(key)}\b(?![}}])",
            rf"\\parencite{{{key}}}",
            text,
        )
    return text


def sanitize_document_model_raw_bibkeys(doc: AcademicDocument, bib_keys: list[str]) -> tuple[AcademicDocument, list[str]]:
    """Saneia chaves BibTeX cruas no conteúdo visível do document_model."""
    changed: list[str] = []

    def clean(label: str, value: str) -> str:
        new = _replace_raw_bibkeys_with_latex_cites(value, bib_keys)
        if new != value:
            changed.append(label)
        return new

    if doc.abstract:
        doc.abstract.texto = clean("abstract.texto", doc.abstract.texto)
        doc.abstract.palavras_chave = [clean(f"abstract.palavras_chave[{i}]", kw) for i, kw in enumerate(doc.abstract.palavras_chave)]

    for si, section in enumerate(doc.sections):
        section.title = clean(f"sections[{si}].title", section.title)
        for bi, block in enumerate(section.blocks):
            for attr in ["text", "title"]:
                old = getattr(block, attr, "")
                setattr(block, attr, clean(f"sections[{si}].blocks[{bi}].{attr}", old))
            for ii, inline in enumerate(block.content or []):
                if isinstance(inline, TextSpan):
                    inline.text = clean(f"sections[{si}].blocks[{bi}].content[{ii}].text", inline.text)
                elif isinstance(inline, Citation):
                    inline.prefix = clean(f"sections[{si}].blocks[{bi}].content[{ii}].citation.prefix", inline.prefix)
                    inline.suffix = clean(f"sections[{si}].blocks[{bi}].content[{ii}].citation.suffix", inline.suffix)
            if getattr(block, "items", None):
                block.items = [clean(f"sections[{si}].blocks[{bi}].items[{ii}]", item) for ii, item in enumerate(block.items or [])]
            if getattr(block, "table", None):
                tbl = block.table
                tbl.caption = clean(f"sections[{si}].blocks[{bi}].table.caption", tbl.caption)
                tbl.headers = [clean(f"sections[{si}].blocks[{bi}].table.headers[{hi}]", h) for hi, h in enumerate(tbl.headers or [])]
                tbl.rows = [
                    [clean(f"sections[{si}].blocks[{bi}].table.rows[{ri}][{ci}]", cell) for ci, cell in enumerate(row or [])]
                    for ri, row in enumerate(tbl.rows or [])
                ]

    for fi, fig in enumerate(doc.figures):
        fig.title = clean(f"figures[{fi}].title", fig.title)

    # Remove duplicatas preservando ordem.
    unique: list[str] = []
    seen: set[str] = set()
    for item in changed:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return doc, unique


def sanitize_document_model_technical_leaks(doc: AcademicDocument) -> tuple[AcademicDocument, list[str]]:
    """Remove linguagem operacional do conteúdo visível do documento canônico.

    Importante: não examina nem altera diagnostics.*, porque esses campos podem
    conter paths, hashes e nomes internos como "academic_pipeline". Eles são
    artefatos de auditoria, não texto acadêmico renderizado.
    """
    changed: list[str] = []

    # Metadata visível em capa/ficha técnica.
    for attr in [
        "titulo", "subtitulo", "tipo_trabalho", "nota_capa", "disciplina", "professor",
        "curso", "programa", "instituicao", "cidade", "polo", "turma",
    ]:
        old = getattr(doc.metadata, attr, "")
        new = _clean_visible_text(old)
        if new != old:
            setattr(doc.metadata, attr, new)
            changed.append(f"metadata.{attr}")

    if doc.abstract:
        old = doc.abstract.texto
        new = _clean_visible_text(old)
        if new != old:
            doc.abstract.texto = new
            changed.append("abstract.texto")
        cleaned_kw = []
        for kw in doc.abstract.palavras_chave:
            cleaned_kw.append(_clean_visible_text(kw))
        if cleaned_kw != doc.abstract.palavras_chave:
            doc.abstract.palavras_chave = cleaned_kw
            changed.append("abstract.palavras_chave")

    for si, section in enumerate(doc.sections):
        old_title = section.title
        section.title = _clean_visible_text(section.title)
        if section.title != old_title:
            changed.append(f"sections[{si}].title")
        for bi, block in enumerate(section.blocks):
            for attr in ["text", "title", "id"]:
                old = getattr(block, attr, "")
                new = _clean_visible_text(old)
                if new != old:
                    setattr(block, attr, new)
                    changed.append(f"sections[{si}].blocks[{bi}].{attr}")
            # Conteúdo inline de parágrafos.
            for ii, inline in enumerate(block.content or []):
                if isinstance(inline, TextSpan):
                    old = inline.text
                    inline.text = _clean_visible_text(inline.text)
                    if inline.text != old:
                        changed.append(f"sections[{si}].blocks[{bi}].content[{ii}].text")
                elif isinstance(inline, Citation):
                    # prefix/suffix são texto visível em torno da citação.
                    # A validação também olha esses campos; portanto eles precisam
                    # ser saneados para evitar vazamento de termos como OCR/pipeline.
                    old_prefix = inline.prefix
                    old_suffix = inline.suffix
                    inline.prefix = _clean_visible_text(inline.prefix)
                    inline.suffix = _clean_visible_text(inline.suffix)
                    if inline.prefix != old_prefix:
                        changed.append(f"sections[{si}].blocks[{bi}].content[{ii}].citation.prefix")
                    if inline.suffix != old_suffix:
                        changed.append(f"sections[{si}].blocks[{bi}].content[{ii}].citation.suffix")
            # Itens de listas.
            if block.items:
                new_items = [_clean_visible_text(item) for item in block.items]
                if new_items != block.items:
                    block.items = new_items
                    changed.append(f"sections[{si}].blocks[{bi}].items")
            # Tabelas.
            if block.table:
                old = block.table.caption
                block.table.caption = _clean_visible_text(block.table.caption)
                if block.table.caption != old:
                    changed.append(f"sections[{si}].blocks[{bi}].table.caption")
                new_headers = [_clean_visible_text(h) for h in block.table.headers]
                if new_headers != block.table.headers:
                    block.table.headers = new_headers
                    changed.append(f"sections[{si}].blocks[{bi}].table.headers")
                new_rows = [[_clean_visible_text(cell) for cell in row] for row in block.table.rows]
                if new_rows != block.table.rows:
                    block.table.rows = new_rows
                    changed.append(f"sections[{si}].blocks[{bi}].table.rows")

    for fi, fig in enumerate(doc.figures):
        old = fig.title
        fig.title = _clean_visible_text(fig.title)
        if fig.title != old:
            changed.append(f"figures[{fi}].title")

    return doc, changed


def _visible_strings_from_block(block: Block) -> list[str]:
    values: list[str] = []
    for attr in ["text", "title"]:
        v = str(getattr(block, attr, "") or "")
        if v:
            values.append(v)
    for inline in block.content or []:
        if isinstance(inline, TextSpan):
            values.append(inline.text or "")
        # Citation prefix/suffix can be user-visible, but keys are not prose.
        elif isinstance(inline, Citation):
            values.append(inline.prefix or "")
            values.append(inline.suffix or "")
    values.extend(block.items or [])
    if block.table:
        values.append(block.table.caption or "")
        values.extend(block.table.headers or [])
        for row in block.table.rows or []:
            values.extend(str(cell) for cell in row)
    return [v for v in values if str(v).strip()]


def visible_text_from_document_model(doc: AcademicDocument) -> str:
    """Texto acadêmico visível, excluindo diagnostics e paths internos."""
    values: list[str] = []
    meta = doc.metadata
    for attr in [
        "titulo", "subtitulo", "autor", "instituicao", "programa", "curso", "turma",
        "polo", "disciplina", "professor", "cidade", "tipo_trabalho", "nota_capa",
    ]:
        v = str(getattr(meta, attr, "") or "")
        if v:
            values.append(v)
    if doc.abstract:
        values.append(doc.abstract.titulo or "")
        values.append(doc.abstract.texto or "")
        values.extend(doc.abstract.palavras_chave or [])
    for section in doc.sections:
        values.append(section.title)
        for block in section.blocks:
            values.extend(_visible_strings_from_block(block))
    for fig in doc.figures:
        values.append(fig.title)
    return "\n".join(v for v in values if str(v).strip())


def cited_keys_from_document_model(doc: AcademicDocument) -> list[str]:
    keys: list[str] = []
    for section in doc.sections:
        for block in section.blocks:
            for key in extract_cited_keys_from_model_inline(block.content):
                if key not in keys:
                    keys.append(key)
    for key in doc.bibliography.entries_used:
        if key not in keys:
            keys.append(key)
    return keys


def _find_technical_leak_locations_in_model(doc: AcademicDocument) -> list[str]:
    """Retorna locais aproximados de vazamento no conteúdo visível.

    Ajuda a diagnosticar casos residuais sem inspecionar diagnostics/paths internos.
    """
    locations: list[str] = []

    def check(label: str, value: str) -> None:
        low = normalize_title_loose(value or "")
        found = find_technical_leaks_in_text(value or "")
        if found:
            locations.append(f"{label}: " + ", ".join(found))

    meta = doc.metadata
    for attr in [
        "titulo", "subtitulo", "autor", "instituicao", "programa", "curso", "turma",
        "polo", "disciplina", "professor", "cidade", "tipo_trabalho", "nota_capa",
    ]:
        check(f"metadata.{attr}", str(getattr(meta, attr, "") or ""))

    if doc.abstract:
        check("abstract.texto", doc.abstract.texto or "")
        for i, kw in enumerate(doc.abstract.palavras_chave or []):
            check(f"abstract.palavras_chave[{i}]", kw)

    for si, section in enumerate(doc.sections):
        check(f"sections[{si}].title", section.title or "")
        for bi, block in enumerate(section.blocks):
            check(f"sections[{si}].blocks[{bi}].text", getattr(block, "text", "") or "")
            check(f"sections[{si}].blocks[{bi}].title", getattr(block, "title", "") or "")
            for ii, inline in enumerate(block.content or []):
                if isinstance(inline, TextSpan):
                    check(f"sections[{si}].blocks[{bi}].content[{ii}].text", inline.text or "")
                elif isinstance(inline, Citation):
                    check(f"sections[{si}].blocks[{bi}].content[{ii}].citation.prefix", inline.prefix or "")
                    check(f"sections[{si}].blocks[{bi}].content[{ii}].citation.suffix", inline.suffix or "")
            for ii, item in enumerate(block.items or []):
                check(f"sections[{si}].blocks[{bi}].items[{ii}]", item)
            if block.table:
                check(f"sections[{si}].blocks[{bi}].table.caption", block.table.caption or "")
                for hi, h in enumerate(block.table.headers or []):
                    check(f"sections[{si}].blocks[{bi}].table.headers[{hi}]", h)
                for ri, row in enumerate(block.table.rows or []):
                    for ci, cell in enumerate(row):
                        check(f"sections[{si}].blocks[{bi}].table.rows[{ri}][{ci}]", str(cell))

    for fi, fig in enumerate(doc.figures):
        check(f"figures[{fi}].title", fig.title or "")

    return locations


def validate_document_model(doc: AcademicDocument, bib_keys: list[str], *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    bib_set = set(bib_keys)
    cited = cited_keys_from_document_model(doc)
    missing = [k for k in cited if k not in bib_set]
    if missing:
        errors.append("Chaves citadas ausentes no .bib: " + ", ".join(missing))

    # Só avalia conteúdo visível. Diagnostics podem conter paths/hashes internos.
    visible = visible_text_from_document_model(doc)
    low = normalize_title_loose(visible)
    leaks = find_technical_leaks_in_text(visible)
    if leaks:
        locs = _find_technical_leak_locations_in_model(doc)
        detail = ""
        if locs:
            detail = " | locais: " + "; ".join(locs[:12])
        errors.append("Documento canônico contém menções técnicas proibidas: " + ", ".join(leaks) + detail)

    if str(doc.metadata.tipo_documento).lower() in {"paper", "atividade"}:
        if doc.metadata.programa and normalize_title_loose(doc.metadata.programa) == normalize_title_loose(doc.metadata.curso):
            errors.append("metadata.programa e metadata.curso estão iguais; isso duplicaria a capa.")
    return errors


def _strip_nonvisible_org_regions(org_text: str) -> str:
    """Remove regiões não visíveis antes da validação de vazamento técnico.

    A validação de citações continua sendo feita sobre o ORG integral em outro
    trecho. Para detectar termos proibidos, porém, devemos ignorar cabeçalhos,
    diretivas e comentários técnicos. Caso contrário, expressões como
    "academic_pipeline" em caminhos, comentários LaTeX ou headers do Org geram
    falso positivo embora não apareçam no PDF/DOCX.
    """
    text = org_text or ""
    # Blocos de comentário Org não são renderizados.
    text = re.sub(r"(?ims)^\s*#\+begin_comment\b.*?^\s*#\+end_comment\s*$", "\n", text)
    # Comentários LaTeX em headers ou blocos de exportação não aparecem no documento.
    text = re.sub(r"(?m)%.*$", "", text)
    # Diretivas Org (#+TITLE, #+LATEX_HEADER, #+OPTIONS etc.) são metadados.
    text = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#+") and not line.lstrip().startswith("# ")
    )
    # Remove caminhos de imagens/bibliografia que podem conter nomes de diretório do programa.
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", r"\\includegraphics{}", text)
    text = re.sub(r"\\addbibresource\{[^}]*\}", r"\\addbibresource{}", text)
    return text


def validate_org_text(org_text: str, bib_keys: list[str]) -> list[str]:
    errors: list[str] = []
    if "<empty citation>" in org_text:
        errors.append("ORG contém <empty citation>.")
    if "[cite:" in org_text or "[cite/" in org_text:
        errors.append("ORG contém citações Org Cite não renderizadas.")

    visible_org = _strip_nonvisible_org_regions(org_text)
    low = normalize_title_loose(visible_org)
    leaks = find_technical_leaks_in_text(visible_org)
    if leaks:
        errors.append("ORG contém menções técnicas proibidas: " + ", ".join(leaks))

    cited = extract_latex_cited_keys(org_text)
    missing = [k for k in cited if k not in set(bib_keys)]
    if missing:
        errors.append("ORG cita chaves ausentes no .bib: " + ", ".join(missing))
    # A procura de chaves cruas deve mirar apenas conteúdo visível. Headers,
    # comentários, addbibresource, nomes técnicos e caminhos podem conter strings
    # parecidas com chaves, mas não aparecem como texto do documento. Isso evita
    # falsos positivos e preserva a função principal: impedir que a chave BibTeX
    # literal apareça no corpo do PDF/DOCX.
    raw_scan_text = _strip_nonvisible_org_regions(org_text)
    raw_scan_text = re.sub(
        r"\\(?:paren|text|auto|foot|smart|cite|citeauthor|citeyear)cite\*?(?:\[[^\]]*\])*\{[^}]+\}",
        "",
        raw_scan_text,
    )
    raw = [k for k in bib_keys if re.search(rf"\b{re.escape(k)}\b", raw_scan_text)]
    if raw:
        details: list[str] = []
        for lineno, line in enumerate((org_text or "").splitlines(), start=1):
            visible_line = _strip_nonvisible_org_regions(line)
            if not visible_line.strip():
                continue
            masked_line = re.sub(
                r"\\(?:paren|text|auto|foot|smart|cite|citeauthor|citeyear)cite\*?(?:\[[^\]]*\])*\{[^}]+\}",
                "",
                visible_line,
            )
            hits = [k for k in raw if re.search(rf"\b{re.escape(k)}\b", masked_line)]
            if hits:
                details.append(f"L{lineno}: " + ", ".join(hits))
            if len(details) >= 8:
                break
        msg = "ORG contém chaves BibTeX cruas fora de comandos de citação: " + ", ".join(raw[:20])
        if details:
            msg += " | ocorrências: " + " ; ".join(details)
        errors.append(msg)
    numeric_citation_lines: list[str] = []
    numeric_scan_text = re.sub(
        r"\\(?:paren|text|auto|foot|smart|cite|citeauthor|citeyear)cite\*?(?:\[[^\]]*\])*\{[^}]+\}",
        "",
        _strip_nonvisible_org_regions(org_text),
    )
    for lineno, line in enumerate((numeric_scan_text or "").splitlines(), start=1):
        if re.search(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", line):
            numeric_citation_lines.append(f"L{lineno}: {line.strip()[:120]}")
        if len(numeric_citation_lines) >= 8:
            break
    if numeric_citation_lines:
        errors.append(
            "ORG contém citações numéricas cruas em colchetes; use citações autor-data via BibLaTeX "
            "(ex.: \\parencite{chave} ou \\textcite{chave}). Ocorrências: "
            + " ; ".join(numeric_citation_lines)
        )

    pm = re.search(r"\\programname\{([^}]*)\}", org_text)
    cm = re.search(r"\\coursename\{([^}]*)\}", org_text)
    if pm and cm and pm.group(1).strip() and normalize_title_loose(pm.group(1)) == normalize_title_loose(cm.group(1)):
        errors.append("\\programname e \\coursename estão iguais.")
    return errors


def raise_if_errors(errors: list[str], title: str = "Validação falhou") -> None:
    if errors:
        raise RuntimeError(title + ":\n- " + "\n- ".join(errors))

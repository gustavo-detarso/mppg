#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from document_model import AcademicDocument, Block
from citation_renderer import render_latex_inlines
from utils import latex_escape, normalize_snippet_placeholders, write_text, resolve_path
from institution_layouts import resolve_layout_spec


def clean_heading_title(title: str) -> str:
    """Remove numeração que a IA às vezes coloca dentro do título.

    O Org/LaTeX já numera seções automaticamente. Se a IA devolve títulos como
    "1 INTRODUÇÃO" ou "2.1 Artigo X", o PDF fica com duplicação visual
    (ex.: "1 1 INTRODUÇÃO").
    """
    text = str(title or "").strip()
    text = re.sub(r"^\s*\d+(?:\.\d+)*(?:[\.)])?\s+", "", text).strip()
    return text or str(title or "").strip()


def org_heading(level: int, title: str) -> str:
    return "*" * max(1, min(int(level or 1), 6)) + " " + clean_heading_title(title)


def _strip_numeric_citation_markers(text: str) -> str:
    cleaned = re.sub(r"\s*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", "", str(text or ""))
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _is_citation_only_latex(text: str) -> bool:
    if not str(text or "").strip():
        return True
    masked = re.sub(
        r"\\(?:paren|text|auto|foot|smart|cite|citeauthor|citeyear)cite\*?(?:\[[^\]]*\])*\{[^}]+\}",
        "",
        str(text or ""),
    )
    masked = re.sub(r"[\s,.;:()\\]+", "", masked)
    return not masked


def _plain_cell_text(value: Any) -> str:
    """Normaliza texto de célula de tabela para saída LaTeX/Org."""
    text = str(value if value is not None else "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _latex_table_cell(value: Any) -> str:
    """Escapa conteúdo textual de célula, preservando quebras controladas."""
    text = _plain_cell_text(value)
    if not text:
        return ""
    return latex_escape(text)


def _latex_table_colspec(ncols: int) -> str:
    """Define colspec responsivo para longtblr."""
    n = max(1, int(ncols or 1))
    if n == 1:
        weights = [1]
    elif n == 2:
        weights = [1, 2]
    elif n == 3:
        weights = [1.0, 1.4, 1.4]
    elif n == 4:
        weights = [1.0, 1.25, 1.25, 1.25]
    elif n == 5:
        weights = [1.0, 1.15, 1.15, 1.15, 1.2]
    elif n == 6:
        weights = [1.0, 1.2, 1.15, 1.15, 1.25, 1.15]
    else:
        weights = [1] * n
    return " ".join(f"X[{w},l]" for w in weights[:n])


def _should_render_table_as_latex(block: Block) -> bool:
    """Usa LaTeX responsivo quando a tabela tem muitas colunas/células longas."""
    if not block.table:
        return False
    headers = list(block.table.headers or [])
    rows = list(block.table.rows or [])
    ncols = max([len(headers)] + [len(r) for r in rows] + [0])
    if ncols >= 4:
        return True
    max_cell = 0
    for row in rows:
        for cell in row:
            max_cell = max(max_cell, len(_plain_cell_text(cell)))
    return max_cell >= 90 or len(rows) >= 8


def _render_table_as_org_pipe(block: Block) -> str:
    lines = []
    if block.table.caption:
        lines.append(f"#+CAPTION: {block.table.caption}")
    if block.id:
        lines.append(f"#+NAME: {block.id}")
    if block.table.headers:
        lines.append("| " + " | ".join(_plain_cell_text(h) for h in block.table.headers) + " |")
        lines.append("|" + "---|" * len(block.table.headers))
    for row in block.table.rows:
        lines.append("| " + " | ".join(_plain_cell_text(x) for x in row) + " |")
    return "\n".join(lines)


def _render_table_as_latex(block: Block) -> str:
    headers = list(block.table.headers or []) if block.table else []
    rows = list(block.table.rows or []) if block.table else []
    ncols = max([len(headers)] + [len(r) for r in rows] + [1])
    caption = _plain_cell_text(block.table.caption if block.table else "")
    label = re.sub(r"[^A-Za-z0-9:_-]+", "_", str(block.id or "tab:auto")).strip("_") or "tab:auto"
    colspec = _latex_table_colspec(ncols)
    landscape = ncols >= 5

    def row_values(row: list[Any]) -> list[Any]:
        values = list(row[:ncols])
        if len(values) < ncols:
            values += [""] * (ncols - len(values))
        return values

    lines: list[str] = ["#+begin_export latex"]
    if landscape:
        lines.append(r"\begin{landscape}")
    lines += [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
    ]
    opts = []
    if caption:
        opts.append(f"caption={{{latex_escape(caption)}}}")
    if label:
        opts.append(f"label={{{label}}}")
    opt_block = "[" + ",".join(opts) + "]" if opts else ""
    lines.append(rf"\begin{{longtblr}}{opt_block}{{%")
    lines.append(r"  width=\linewidth,")
    lines.append(rf"  colspec={{{colspec}}},")
    lines.append(r"  rowhead=1,")
    lines.append(r"  rows={valign=t},")
    lines.append(r"  row{1}={font=\bfseries},")
    lines.append(r"  hline{1,Z}={0.08em},")
    lines.append(r"  hline{2}={0.04em},")
    lines.append(r"}")
    if headers:
        lines.append(" & ".join(_latex_table_cell(h) for h in row_values(headers)) + r" \\")
    for row in rows:
        lines.append(" & ".join(_latex_table_cell(x) for x in row_values(row)) + r" \\")
    lines += [
        r"\end{longtblr}",
        r"\endgroup",
    ]
    if landscape:
        lines.append(r"\end{landscape}")
    lines.append("#+end_export")
    return "\n".join(lines)


def render_table_block_org(block: Block) -> str:
    """Renderiza tabela com fallback responsivo para LaTeX.

    Tabelas largas são convertidas para longtblr em landscape para evitar corte
    lateral no PDF. Tabelas pequenas permanecem em formato Org nativo.
    """
    if not block.table:
        return ""
    if _should_render_table_as_latex(block):
        return _render_table_as_latex(block)
    return _render_table_as_org_pipe(block)


def _cfg_section(cfg: dict[str, Any] | None, name: str) -> dict[str, Any]:
    return cfg.get(name, {}) if cfg and isinstance(cfg.get(name), dict) else {}


def _layout_spec(cfg: dict[str, Any] | None, doc: AcademicDocument | None = None):
    try:
        return resolve_layout_spec(cfg or {}, doc)
    except Exception:
        return None


def _latex_class_from_layout(cfg: dict[str, Any] | None, doc: AcademicDocument | None, default: str) -> str:
    spec = _layout_spec(cfg, doc)
    if spec and spec.classe_latex:
        return spec.classe_latex
    doc_cfg = _cfg_section(cfg, "documento")
    return str(doc_cfg.get("classe_latex") or default)


def _first_nonempty(*values: Any, default: str = "") -> str:
    for v in values:
        s = str(v or "").strip()
        if s:
            return s
    return default


def _as_latex_graphics_path(path_value: Any, fallback: str = "fgv.png", cfg: dict[str, Any] | None = None) -> str:
    r"""Retorna caminho seguro para \includegraphics no LaTeX.

    A compilação ocorre no diretório de saída, enquanto o TOML normalmente usa
    caminhos relativos ao diretório do próprio TOML. Por isso, o caminho do logo
    precisa ser resolvido contra ``__config_dir__`` e injetado como caminho
    absoluto sempre que possível.
    """
    raw = str(path_value or "").strip() or fallback
    raw = raw.replace("\\", "/")
    base = None
    if cfg and cfg.get("__config_dir__"):
        base = Path(str(cfg.get("__config_dir__"))).resolve()
    resolved = resolve_path(raw, base)
    if resolved and resolved.exists():
        return resolved.as_posix()
    return raw


def normalize_biblatex_style(style: str | None) -> str:
    value = str(style or "apa").strip().lower().replace("-", "_")
    aliases = {
        "apa7": "apa",
        "apa_7": "apa",
        "abnt2": "abnt",
        "abnt_6023": "abnt",
        "nbr6023": "abnt",
        "nbr_6023": "abnt",
        "chicago_author_date": "authoryear-chicago",
        "chicago": "authoryear-chicago",
    }
    value = aliases.get(value, value)
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
        return "apa"
    return value


def bibliography_style_from_cfg(cfg: dict[str, Any] | None, doc: AcademicDocument | None = None) -> str:
    bib_cfg = _cfg_section(cfg, "bibliografia")
    doc_cfg = _cfg_section(cfg, "documento")
    raw = (
        bib_cfg.get("latex_style")
        or bib_cfg.get("estilo_biblatex")
        or bib_cfg.get("estilo_citacao")
        or doc_cfg.get("estilo_citacao")
        or (doc.bibliography.style if doc and getattr(doc, "bibliography", None) else None)
        or "apa"
    )
    return normalize_biblatex_style(str(raw))


def biblatex_options_for_style(style: str, cfg: dict[str, Any] | None = None) -> str:
    bib_cfg = _cfg_section(cfg, "bibliografia")
    override = str(bib_cfg.get("latex_options") or "").strip()
    if override:
        return override
    style = normalize_biblatex_style(style)
    if style == "abnt":
        return "backend=biber,style=abnt,sorting=nty,giveninits=true"
    if style == "apa":
        return "backend=biber,style=apa,sorting=nyt"
    return f"backend=biber,style={style}"




def strip_org_cite_export_lines(org: str) -> str:
    """Remove diretivas #+CITE_EXPORT herdadas de templates antigos.

    A rc10 renderiza citações como LaTeX direto (\\parencite/\\textcite)
    e injeta biblatex via #+LATEX_HEADER. Em Org 9.5, #+CITE_EXPORT:
    biblatex pode acionar erro de processor em batch.
    """
    return re.sub(r"(?im)^\s*#\+cite_export:.*(?:\n|$)", "", org)

def render_bibliography_preamble(bib_filename: str, style: str, cfg: dict[str, Any] | None = None) -> list[str]:
    bib_name = Path(str(bib_filename)).name
    options = biblatex_options_for_style(style, cfg)
    return [
        "#+LATEX_HEADER: % Bibliografia controlada pelo TOML; os estilos FGV não carregam biblatex.",
        f"#+LATEX_HEADER: \\usepackage[{options}]{{biblatex}}",
        f"#+LATEX_HEADER: \\addbibresource{{{bib_name}}}",
    ]


def render_block_org(block: Block, base_level: int = 1) -> str:
    if block.type == "paragraph":
        rendered = render_latex_inlines(block.content).strip() if block.content else ""
        raw = _strip_numeric_citation_markers(block.text or "")
        # Fallback defensivo: em algumas respostas estruturadas a IA colocou o texto
        # em block.text e deixou em block.content apenas a citação. Sem isso, a seção
        # aparece no PDF como uma sequência de [1], [2] etc.
        if raw and (not rendered or _is_citation_only_latex(rendered) or len(rendered) < max(30, len(raw) // 5)):
            return (raw + (" " + rendered if rendered else "")).strip()
        return rendered
    if block.type == "heading":
        return org_heading(block.level, block.text or block.title)
    if block.type == "quote":
        return "#+begin_quote\n" + (block.text or render_latex_inlines(block.content)).strip() + "\n#+end_quote"
    if block.type == "bullet_list":
        return "\n".join(f"- {item}" for item in block.items)
    if block.type == "numbered_list":
        return "\n".join(f"{i}. {item}" for i, item in enumerate(block.items, start=1))
    if block.type == "table" and block.table:
        return render_table_block_org(block)
    if block.type == "figure":
        prefix = "#+LATEX: \\clearpage\n" if block.page_break_before else ""
        caption = f"#+CAPTION: {block.title}\n" if block.title else ""
        return prefix + caption + f"[[file:{block.path}]]"
    if block.type == "page_break":
        return "#+LATEX: \\clearpage"
    return block.text.strip()


PAPER_ABSTRACT_NATIVE_MARKER = "academic_pipeline:paper_abstracts:native"


def _normalise_language_code(value: Any) -> str:
    raw = str(value or "").strip().casefold().replace("_", "-")
    aliases = {
        "portugues": "pt-br", "português": "pt-br", "pt": "pt-br", "pt-br": "pt-br",
        "ingles": "en", "inglês": "en", "english": "en",
        "espanhol": "es", "español": "es", "spanish": "es",
    }
    return aliases.get(raw, raw or "pt-br")


def _org_language_name(code: str) -> str:
    normalized = _normalise_language_code(code)
    if normalized == "pt-br":
        return "pt_BR"
    return normalized.split("-", 1)[0] or "pt_BR"


def _render_output_language(cfg: dict[str, Any] | None, output_path: Path) -> str:
    parts = list(output_path.parts)
    for index, item in enumerate(parts[:-1]):
        if item == "idiomas" and index + 1 < len(parts):
            return _normalise_language_code(parts[index + 1])
    section = _cfg_section(cfg, "idiomas_saida")
    return _normalise_language_code(section.get("principal") or "pt-BR")


def _paper_abstract_sidecar_path(output_path: Path, cfg: dict[str, Any] | None) -> Path | None:
    """Localiza o JSON de resumos no diretório principal, inclusive em cópias traduzidas."""
    paths = _cfg_section(cfg, "paths")
    prefix = str(paths.get("document_prefix") or "").strip() or output_path.stem
    seen: set[Path] = set()
    candidates = [output_path.parent, *output_path.parents]
    for directory in candidates[:5]:
        candidate = directory / f"{prefix}.resumos_paper.json"
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _paper_abstract_rows(
    cfg: dict[str, Any] | None,
    output_path: Path,
    output_language: str,
) -> list[dict[str, Any]]:
    section = _cfg_section(cfg, "resumos_paper")
    if not bool(section.get("ativo", False)) or not bool(section.get("gerar_resumo_principal", True)):
        return []
    sidecar = _paper_abstract_sidecar_path(output_path, cfg)
    if sidecar is None:
        return []
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    if not isinstance(items, dict):
        return []

    is_translation = "idiomas" in output_path.parts
    requested: list[str]
    if is_translation:
        requested = [output_language]
    else:
        requested = [
            _normalise_language_code(
                section.get("principal")
                or _cfg_section(cfg, "idiomas_saida").get("principal")
                or "pt-BR"
            )
        ]
        if bool(section.get("gerar_resumo_adicional", False)):
            raw_items = section.get("idiomas_adicionais", [])
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            if isinstance(raw_items, list):
                requested.extend(_normalise_language_code(item) for item in raw_items)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for code in requested:
        if code in seen:
            continue
        seen.add(code)
        row = items.get(code)
        if isinstance(row, dict) and str(row.get("abstract") or "").strip():
            rows.append(row)
    return rows


def _render_paper_abstract_rows(rows: list[dict[str, Any]]) -> str:
    """Renderiza resumos no front matter sem criar headings numerados no corpo."""
    if not rows:
        return ""
    parts = [
        "#+BEGIN_COMMENT",
        PAPER_ABSTRACT_NATIVE_MARKER,
        "#+END_COMMENT",
        "",
        "#+begin_export latex",
        r"\begingroup",
        r"\small",
    ]
    for index, row in enumerate(rows):
        if index:
            parts.append(r"\vspace{1.1em}")
        heading = latex_escape(str(row.get("heading") or "Resumo").strip())
        abstract = latex_escape(re.sub(r"\s+", " ", str(row.get("abstract") or "").strip()))
        parts.extend([
            r"\begin{center}",
            rf"\textbf{{{heading}}}",
            r"\end{center}",
            abstract,
        ])
        if bool(row.get("include_keywords", True)):
            raw_keywords = row.get("keywords", [])
            keywords = [str(item).strip() for item in raw_keywords if str(item).strip()] if isinstance(raw_keywords, list) else []
            if keywords:
                label = latex_escape(str(row.get("keywords_heading") or "Palavras-chave").strip())
                rendered_keywords = "; ".join(latex_escape(item) for item in keywords)
                parts.extend(["", rf"\noindent\textbf{{{label}:}} {rendered_keywords}."])
    parts.extend([r"\endgroup", "#+end_export"])
    return "\n".join(parts)


def _common_headers(meta: Any, bib_filename: str, cfg: dict[str, Any] | None) -> list[str]:
    style = bibliography_style_from_cfg(cfg, None)
    output_language = str((cfg or {}).get("__render_output_language__") or "pt-br")
    return [
        f"#+TITLE: {meta.titulo}",
        f"#+AUTHOR: {meta.autor}",
        f"#+LANGUAGE: {_org_language_name(output_language)}",
        "#+OPTIONS: toc:nil num:t title:nil html-postamble:nil ^:{}",
        *render_bibliography_preamble(bib_filename, style, cfg),
    ]


def render_paper_front_matter(doc: AcademicDocument, bib_filename: str, cfg: dict[str, Any] | None = None) -> str:
    meta = doc.metadata
    doc_cfg = _cfg_section(cfg, "documento")
    program = str(doc_cfg.get("program_name") if "program_name" in doc_cfg else meta.programa or "").strip()
    course = _first_nonempty(doc_cfg.get("course_name"), meta.curso)
    lines = [
        *_common_headers(meta, bib_filename, cfg)[:4],
        f"#+LATEX_CLASS: {_latex_class_from_layout(cfg, doc, 'fgv-paper')}",
        *_common_headers(meta, bib_filename, cfg)[4:],
        "#+LATEX_HEADER: \\usepapercover",
        f"#+LATEX_HEADER: \\institution{{{latex_escape(meta.instituicao)}}}",
        f"#+LATEX_HEADER: \\programname{{{latex_escape(program)}}}",
        f"#+LATEX_HEADER: \\coursename{{{latex_escape(course)}}}",
        f"#+LATEX_HEADER: \\disciplinename{{{latex_escape(meta.disciplina)}}}",
        f"#+LATEX_HEADER: \\professorname{{{latex_escape(meta.professor)}}}",
        f"#+LATEX_HEADER: \\cityname{{{latex_escape(meta.cidade)}}}",
        f"#+LATEX_HEADER: \\papertype{{{latex_escape(meta.tipo_trabalho or 'Paper acadêmico')}}}",
        f"#+LATEX_HEADER: \\covernote{{{latex_escape(meta.nota_capa)}}}",
        "",
        "#+LATEX: \\makemytitle",
    ]
    native_rows = (cfg or {}).get("__paper_abstract_rows__", [])
    if isinstance(native_rows, list) and native_rows:
        lines += ["", _render_paper_abstract_rows(native_rows)]
    elif doc.abstract and doc.abstract.texto:
        # Compatibilidade: usa o resumo do document_model somente quando não
        # existe o sidecar auditável de resumos do paper.
        lines += ["", "#+begin_abstract", doc.abstract.texto.strip()]
        if doc.abstract.palavras_chave:
            lines += ["", "Palavras-chave: " + "; ".join(doc.abstract.palavras_chave) + "."]
        lines.append("#+end_abstract")
    return "\n".join(lines).strip() + "\n\n"


def render_activity_front_matter(doc: AcademicDocument, bib_filename: str, cfg: dict[str, Any] | None = None) -> str:
    r"""Renderiza atividade FGV com cabeçalho institucional e Ficha Técnica visual.

    Reproduz o padrão do template_atividade_fgv_v5_2_7.org:
    - sem \maketitle automático do Org;
    - logo FGV no cabeçalho;
    - barra fina em degradê azul escuro → azul claro;
    - Ficha Técnica em tcolorbox/tblr;
    - quebra de página antes do corpo textual.
    """
    meta = doc.metadata
    atividade = _cfg_section(cfg, "atividade")
    latex_cfg = _cfg_section(cfg, "latex")
    logo_path = _as_latex_graphics_path(latex_cfg.get("fgv_logo_path"), "fgv.png", cfg=cfg)

    curso = _first_nonempty(meta.curso, atividade.get("curso"))
    turma = _first_nonempty(meta.turma, atividade.get("turma"))
    polo = _first_nonempty(meta.polo, atividade.get("polo"))
    disciplina = _first_nonempty(meta.disciplina, atividade.get("disciplina"))
    professor = _first_nonempty(meta.professor, atividade.get("professor"))
    aluno = _first_nonempty(meta.autor, atividade.get("aluno"))
    data = _first_nonempty(meta.data, atividade.get("data"), meta.ano)
    titulo = _first_nonempty(meta.titulo, atividade.get("titulo"))

    lines = [
        *_common_headers(meta, bib_filename, cfg)[:4],
        "#+STARTUP: indent",
        "#+LATEX_COMPILER: lualatex",
        f"#+LATEX_CLASS: {_latex_class_from_layout(cfg, doc, 'fgv-paper')}",
        "#+LATEX_CLASS_OPTIONS: [12pt,a4paper]",
        *_common_headers(meta, bib_filename, cfg)[4:],
        "#+LATEX_HEADER: \\geometry{a4paper,left=3cm,right=2cm,top=3cm,bottom=2cm,headheight=58pt,headsep=18pt,footskip=1.2cm}",
        "#+LATEX_HEADER: \\usepackage{float}",
        "#+LATEX_HEADER: \\usepackage{tikz}",
        "#+LATEX_HEADER: \\usepackage{tcolorbox}",
        "#+LATEX_HEADER: \\usepackage{tabularray}",
        "#+LATEX_HEADER: \\usepackage{ltablex}",
        "#+LATEX_HEADER: \\keepXColumns",
        "#+LATEX_HEADER: \\setlength{\\emergencystretch}{3em}",
        "#+LATEX_HEADER: \\sloppy",
        "#+LATEX_HEADER: \\definecolor{FGVHeaderBlueDark}{HTML}{003B71}",
        "#+LATEX_HEADER: \\definecolor{FGVHeaderBlueMid}{HTML}{006BB6}",
        "#+LATEX_HEADER: \\definecolor{FGVHeaderBlueLight}{HTML}{8FD3F4}",
        "#+LATEX_HEADER: \\setlength{\\headheight}{58pt}",
        "#+LATEX_HEADER: \\setlength{\\headsep}{18pt}",
        f"#+LATEX_HEADER: \\newcommand{{\\FGVHeaderLogoInclude}}{{\\IfFileExists{{{logo_path}}}{{\\includegraphics[height=1.05cm]{{{logo_path}}}}}{{}}}}",
        "#+LATEX_HEADER: \\newcommand{\\FGVHeaderGradientRule}{\\begin{tikzpicture}[baseline=(current bounding box.center)]\\shade[left color=FGVHeaderBlueDark,middle color=FGVHeaderBlueMid,right color=FGVHeaderBlueLight] (0pt,0pt) rectangle (\\headwidth,2pt);\\end{tikzpicture}}",
        "#+LATEX_HEADER: \\newcommand{\\FGVHeaderBlock}{\\makebox[\\headwidth][l]{\\begin{minipage}[t]{\\headwidth}\\FGVHeaderLogoInclude\\\\[-1pt]\\FGVHeaderGradientRule\\end{minipage}}}",
        "#+LATEX_HEADER: \\fancypagestyle{fgvreportstyle}{%",
        "#+LATEX_HEADER: \\fancyhf{}%",
        "#+LATEX_HEADER: \\fancyhead[L]{\\FGVHeaderBlock}%",
        "#+LATEX_HEADER: \\renewcommand{\\headrulewidth}{0pt}%",
        "#+LATEX_HEADER: \\fancyfoot[C]{\\thepage}%",
        "#+LATEX_HEADER: }",
        "#+LATEX_HEADER: \\fancypagestyle{plain}{%",
        "#+LATEX_HEADER: \\fancyhf{}%",
        "#+LATEX_HEADER: \\fancyhead[L]{\\FGVHeaderBlock}%",
        "#+LATEX_HEADER: \\renewcommand{\\headrulewidth}{0pt}%",
        "#+LATEX_HEADER: \\fancyfoot[C]{\\thepage}%",
        "#+LATEX_HEADER: }",
        "#+LATEX_HEADER: \\AtBeginDocument{\\pagestyle{fgvreportstyle}\\thispagestyle{fgvreportstyle}}",
        f"#+LATEX_HEADER: \\institution{{{latex_escape(meta.instituicao)}}}",
        "#+LATEX_HEADER: \\programname{}",
        f"#+LATEX_HEADER: \\coursename{{{latex_escape(curso)}}}",
        f"#+LATEX_HEADER: \\disciplinename{{{latex_escape(disciplina)}}}",
        f"#+LATEX_HEADER: \\professorname{{{latex_escape(professor)}}}",
        f"#+LATEX_HEADER: \\cityname{{{latex_escape(meta.cidade)}}}",
        "",
        "#+begin_export latex",
        "\\begingroup",
        "\\linespread{1}\\selectfont",
        "\\begin{tcolorbox}[title=Ficha Técnica,",
        "  colback=gray!5,colframe=gray!40,boxrule=0.4pt,sharp corners]",
        "\\begin{tblr}{rowsep=1pt,stretch=1, rows={t}, colspec={Q[l,2.8cm] X[l]}}",
        rf"\textbf{{Curso}}                   & {latex_escape(curso)} \\",
        rf"\textbf{{Turma}}                   & {latex_escape(turma)} \\",
        rf"\textbf{{Pólo}}                    & {latex_escape(polo)} \\",
        rf"\textbf{{Disciplina}}              & {latex_escape(disciplina)} \\",
        rf"\textbf{{Professor}}               & {latex_escape(professor)} \\",
        rf"\textbf{{Aluno(s)}}                & {latex_escape(aluno)} \\",
        rf"\textbf{{Data}}                    & {latex_escape(data)} \\",
        rf"\textbf{{Título do trabalho}}      & {latex_escape(titulo)} \\",
        "\\end{tblr}",
        "\\end{tcolorbox}",
        "\\endgroup",
        "#+end_export",
        "",
        "#+LATEX: \\clearpage",
        "",
    ]
    return "\n".join(lines).strip() + "\n\n"


def render_dissertation_front_matter(doc: AcademicDocument, bib_filename: str, cfg: dict[str, Any] | None = None) -> str:
    meta = doc.metadata
    documento = _cfg_section(cfg, "documento")
    atividade = _cfg_section(cfg, "atividade")
    title_main = meta.titulo
    subtitle = meta.subtitulo
    if not subtitle and ":" in title_main:
        title_main, subtitle = [p.strip() for p in title_main.split(":", 1)]
    area = _first_nonempty(documento.get("area_de_concentracao"), "Políticas Públicas e Governo")
    natureza = _first_nonempty(documento.get("natureza_trabalho"), f"Dissertação apresentada à Fundação Getúlio Vargas, como requisito para obtenção do título de Mestre em {area}.")
    lines = [
        *_common_headers(meta, bib_filename, cfg)[:4],
        f"#+LATEX_CLASS: {_latex_class_from_layout(cfg, doc, 'fgv-dissertacao')}",
        "#+LATEX_CLASS_OPTIONS: [12pt,a4paper,oneside]",
        *_common_headers(meta, bib_filename, cfg)[4:],
        f"#+LATEX_HEADER: \\autor{{{latex_escape(meta.autor)}}}",
        f"#+LATEX_HEADER: \\titulo{{{latex_escape(title_main)}}}",
        f"#+LATEX_HEADER: \\subtitulo{{{latex_escape(subtitle)}}}",
        f"#+LATEX_HEADER: \\cidade{{{latex_escape(meta.cidade or 'Brasília')}}}",
        f"#+LATEX_HEADER: \\ano{{{latex_escape(meta.ano or meta.data)}}}",
        f"#+LATEX_HEADER: \\instituicao{{{latex_escape(meta.instituicao.upper())}}}",
        f"#+LATEX_HEADER: \\programa{{{latex_escape(_first_nonempty(documento.get('program_name'), meta.programa, meta.curso))}}}",
        f"#+LATEX_HEADER: \\curso{{{latex_escape(meta.curso)}}}",
        f"#+LATEX_HEADER: \\areadeconcentracao{{{latex_escape(area)}}}",
        f"#+LATEX_HEADER: \\linhapesquisa{{{latex_escape(str(documento.get('linha_pesquisa') or ''))}}}",
        f"#+LATEX_HEADER: \\orientador{{{latex_escape(_first_nonempty(documento.get('orientador'), meta.professor, atividade.get('professor')))}}}",
        f"#+LATEX_HEADER: \\coorientador{{{latex_escape(str(documento.get('coorientador') or documento.get('co_orientador') or ''))}}}",
        f"#+LATEX_HEADER: \\naturezatrabalho{{{latex_escape(natureza)}}}",
        f"#+LATEX_HEADER: \\dataaprovacao{{{latex_escape(str(documento.get('data_aprovacao') or ''))}}}",
        "",
        "#+LATEX: \\capa",
        "#+LATEX: \\folhaderosto",
        "#+LATEX: \\elementospretextuais",
    ]
    if doc.abstract and doc.abstract.texto:
        lines += [
            "#+begin_export latex",
            "\\begin{resumo}",
            doc.abstract.texto.strip(),
            "\\end{resumo}",
            "#+end_export",
        ]
    lines += [
        "#+LATEX: \\tableofcontents",
        "#+LATEX: \\elementostextuais",
        "",
    ]
    return "\n".join(lines).strip() + "\n\n"


def render_front_matter(doc: AcademicDocument, bib_filename: str, cfg: dict[str, Any] | None = None) -> str:
    spec = _layout_spec(cfg, doc)
    front = str(spec.front_matter if spec else "").strip().lower().replace("-", "_")
    genero = str(spec.genero_academico if spec else getattr(doc.metadata, "tipo_documento", "paper")).strip().lower()

    if front in {"atividade_fgv", "ficha_tecnica_fgv", "activity_fgv"} or genero == "atividade":
        return render_activity_front_matter(doc, bib_filename, cfg)
    if front in {"dissertacao_fgv", "dissertation_fgv"} or genero == "dissertacao":
        return render_dissertation_front_matter(doc, bib_filename, cfg)
    if front in {"paper_fgv", "paper"} or genero == "paper":
        return render_paper_front_matter(doc, bib_filename, cfg)

    doc_type = str(getattr(doc.metadata, "tipo_documento", "paper") or "paper").lower()
    if doc_type == "atividade":
        return render_activity_front_matter(doc, bib_filename, cfg)
    if doc_type == "dissertacao":
        return render_dissertation_front_matter(doc, bib_filename, cfg)
    return render_paper_front_matter(doc, bib_filename, cfg)


REFERENCE_SECTION_TITLES = {
    "referencias",
    "referencia",
    "bibliografia",
    "bibliografias",
    "references",
    "reference",
    "works cited",
}


def _normalize_heading_for_skip(title: str) -> str:
    text = str(title or "").strip().lower()
    replacements = {
        "á": "a", "à": "a", "â": "a", "ã": "a", "é": "e", "ê": "e", "í": "i",
        "ó": "o", "ô": "o", "õ": "o", "ú": "u", "ü": "u", "ç": "c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove numeração que a IA às vezes incorpora no título.
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text).strip()
    return text


def is_ai_generated_reference_section_title(title: str) -> bool:
    r"""Detecta seção de referências criada pela IA.

    O pipeline já injeta a bibliografia de modo determinístico via
    ``\printbibliography``. Logo, qualquer seção textual chamada
    REFERÊNCIAS/BIBLIOGRAFIA dentro do ``document_model`` deve ser ignorada
    pelos renderizadores para evitar duplicidade e vazamento de chaves BibTeX
    cruas como texto visível.
    """
    return _normalize_heading_for_skip(title) in REFERENCE_SECTION_TITLES


def render_document_body(doc: AcademicDocument) -> str:
    out: list[str] = []
    for section in doc.sections:
        if is_ai_generated_reference_section_title(section.title):
            continue
        out.append(org_heading(section.level, section.title))
        for block in section.blocks:
            rendered = render_block_org(block, base_level=section.level)
            if rendered:
                out.append(rendered)
        out.append("")
    return "\n\n".join(part for part in out if part is not None).rstrip() + "\n"



def _mask_existing_citation_commands(text: str) -> tuple[str, list[str]]:
    """Mascara comandos LaTeX de citação já válidos antes de saneamentos."""
    placeholders: list[str] = []

    def repl(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"@@CITE_PLACEHOLDER_{len(placeholders) - 1}@@"

    masked = re.sub(
        r"\\(?:paren|text|auto|foot|smart|cite|citeauthor|citeyear)cite\*?(?:\[[^\]]*\])*\{[^}]+\}",
        repl,
        text or "",
    )
    return masked, placeholders


def _restore_existing_citation_commands(text: str, placeholders: list[str]) -> str:
    for idx, original in enumerate(placeholders):
        text = text.replace(f"@@CITE_PLACEHOLDER_{idx}@@", original)
    return text


def _sanitize_raw_bibkeys_in_visible_line(line: str, keys: list[str]) -> str:
    """Converte chaves BibTeX cruas em uma linha visível de ORG."""
    if not line.strip() or not keys:
        return line

    masked, placeholders = _mask_existing_citation_commands(line)
    text = masked
    for key in keys:
        # Caso comum: (chave) vira \parencite{chave}
        text = re.sub(
            rf"\(\s*{re.escape(key)}\s*\)",
            rf"\\parencite{{{key}}}",
            text,
        )
        # Chave nua em prosa, lista ou célula de tabela.
        text = re.sub(
            rf"(?<![\\{{,])\b{re.escape(key)}\b(?![}}])",
            rf"\\parencite{{{key}}}",
            text,
        )
    return _restore_existing_citation_commands(text, placeholders)


def sanitize_raw_bibkeys_in_org(org: str, doc: AcademicDocument, bib_keys: list[str] | None = None) -> str:
    r"""Converte chaves BibTeX cruas residuais em \parencite no ORG final.

    A versão rc10.7.14 ainda deixava passar chaves em listas/tabelas ou em
    trechos que não vinham de ``block.content``. Esta rotina trabalha linha a
    linha, saneando apenas linhas visíveis do corpo do ORG. Linhas técnicas
    (#+TITLE, #+LATEX_HEADER, #+NAME etc.) são preservadas para não quebrar
    metadados, labels, nomes de tabela, caminhos de imagem ou addbibresource.
    """
    keys = sorted(
        {str(k).strip().lstrip("@") for k in list(doc.bibliography.entries_used or []) + list(bib_keys or []) if str(k).strip()},
        key=len,
        reverse=True,
    )
    if not org or not keys:
        return org or ""

    out: list[str] = []
    in_comment = False
    in_src = False
    for line in (org or "").splitlines():
        stripped = line.lstrip()
        low = stripped.lower()

        if low.startswith("#+begin_comment"):
            in_comment = True
            out.append(line)
            continue
        if low.startswith("#+end_comment"):
            in_comment = False
            out.append(line)
            continue
        if low.startswith("#+begin_src") or low.startswith("#+begin_example"):
            in_src = True
            out.append(line)
            continue
        if low.startswith("#+end_src") or low.startswith("#+end_example"):
            in_src = False
            out.append(line)
            continue

        if in_comment or in_src:
            out.append(line)
            continue

        # Diretivas Org são metadados/controle e não devem receber \parencite.
        # Exceção: #+CAPTION é conteúdo visível para figuras/tabelas.
        if stripped.startswith("#+") and not low.startswith("#+caption:"):
            out.append(line)
            continue
        if stripped.startswith("# "):
            out.append(line)
            continue

        out.append(_sanitize_raw_bibkeys_in_visible_line(line, keys))

    return "\n".join(out) + ("\n" if org.endswith("\n") else "")


def _visible_org_lines_transform(org: str, transform) -> str:
    """Aplica transformação apenas em linhas visíveis do ORG.

    Preserva diretivas técnicas, blocos src/example/comment e comentários.
    """
    if not org:
        return org or ""
    out: list[str] = []
    in_comment = False
    in_src = False
    for line in (org or "").splitlines():
        stripped = line.lstrip()
        low = stripped.lower()

        if low.startswith("#+begin_comment"):
            in_comment = True
            out.append(line)
            continue
        if low.startswith("#+end_comment"):
            in_comment = False
            out.append(line)
            continue
        if low.startswith("#+begin_src") or low.startswith("#+begin_example"):
            in_src = True
            out.append(line)
            continue
        if low.startswith("#+end_src") or low.startswith("#+end_example"):
            in_src = False
            out.append(line)
            continue

        if in_comment or in_src:
            out.append(line)
            continue

        if stripped.startswith("#+") and not low.startswith("#+caption:"):
            out.append(line)
            continue
        if stripped.startswith("# "):
            out.append(line)
            continue

        out.append(transform(line))
    return "\n".join(out) + ("\n" if org.endswith("\n") else "")


def _sanitize_numeric_citation_markers_in_line(line: str, bib_keys: list[str]) -> str:
    r"""Converte citações numéricas cruas do tipo [1] ou [1, 2] em \parencite.

    A IA às vezes usa a ordem da bibliografia para escrever citações visíveis
    como ``[1]``. Em documentos FGV/ABNT/APA isso fica incorreto quando o
    objetivo é autor-data. Esta rotina mapeia 1->primeira chave do .bib,
    2->segunda chave etc., preservando comandos de citação LaTeX já válidos.
    """
    if not line or not bib_keys:
        return line
    masked, placeholders = _mask_existing_citation_commands(line)

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        nums = [n.strip() for n in re.split(r"[,;]", raw) if n.strip()]
        if not nums:
            return match.group(0)
        keys: list[str] = []
        for n in nums:
            if not re.fullmatch(r"\d+", n):
                return match.group(0)
            idx = int(n) - 1
            if idx < 0 or idx >= len(bib_keys):
                return match.group(0)
            key = str(bib_keys[idx]).strip().lstrip("@")
            if key and key not in keys:
                keys.append(key)
        if not keys:
            return match.group(0)
        return r"\parencite{" + ",".join(keys) + "}"

    # Não captura [p. 10], [ver], links markdown etc.; apenas listas numéricas.
    text = re.sub(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]", repl, masked)
    return _restore_existing_citation_commands(text, placeholders)


def sanitize_numeric_citation_markers_in_org(org: str, bib_keys: list[str] | None = None) -> str:
    """Saneia citações numéricas visíveis geradas indevidamente pela IA."""
    keys = [str(k).strip().lstrip("@") for k in (bib_keys or []) if str(k).strip()]
    if not org or not keys:
        return org or ""
    return _visible_org_lines_transform(org, lambda line: _sanitize_numeric_citation_markers_in_line(line, keys))


def render_after_references_figures(doc: AcademicDocument) -> str:
    blocks: list[str] = []
    for fig in doc.figures:
        if fig.placement != "after_references":
            continue
        blocks.append("# MINDMAP_AUTO_START: " + fig.id)
        blocks.append("#+LATEX: \\clearpage")
        blocks.append(org_heading(1, fig.title))
        blocks.append("#+begin_export latex")
        blocks.append(r"\vspace{0.8em}")
        blocks.append(r"\begingroup")
        blocks.append(r"\centering")
        blocks.append(r"\includegraphics[width=\textwidth,height=0.78\textheight,keepaspectratio]{" + fig.path.replace("\\", "/") + r"}")
        blocks.append(r"\par")
        blocks.append(r"\endgroup")
        blocks.append(r"\clearpage")
        blocks.append("#+end_export")
        blocks.append("# MINDMAP_AUTO_END: " + fig.id)
        blocks.append("")
    return "\n".join(blocks).rstrip() + ("\n" if blocks else "")


def render_org_latex(doc: AcademicDocument, output_path: Path, bib_filename: str, cfg: dict[str, Any] | None = None, bib_keys: list[str] | None = None) -> str:
    # Contexto de renderização não modifica o TOML carregado nem o document_model.
    render_cfg = dict(cfg or {})
    output_language = _render_output_language(cfg, output_path)
    render_cfg["__render_output_language__"] = output_language
    render_cfg["__paper_abstract_rows__"] = _paper_abstract_rows(cfg, output_path, output_language)
    org = render_front_matter(doc, bib_filename, cfg=render_cfg)
    org += render_document_body(doc)
    org = org.rstrip() + "\n\n#+LATEX: \\printbibliography\n\n"
    org += render_after_references_figures(doc)
    org = normalize_snippet_placeholders(org)
    org = strip_org_cite_export_lines(org)
    org = sanitize_raw_bibkeys_in_org(org, doc, bib_keys=bib_keys)
    org = sanitize_numeric_citation_markers_in_org(org, bib_keys=bib_keys or list(doc.bibliography.entries_used or []))
    write_text(output_path, org)
    return org

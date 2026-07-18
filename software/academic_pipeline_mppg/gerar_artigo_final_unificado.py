#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

BIBLATEX_ABNT_OPTIONS = (
    "backend=biber,"
    "style=abnt,"
    "language=brazil,"
    "sorting=nyt,"
    "giveninits=true,"
    "uniquename=false,"
    "uniquelist=false"
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path: Path, tag: str, quiet: bool = False) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_name(path.name + f".bak_{tag}_{now_stamp()}")
    shutil.copy2(path, bak)
    if not quiet:
        print(f"[OK] Backup: {bak}")
    return bak


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, quiet: bool = False) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    out = (p.stdout or b"").decode("utf-8", errors="replace")
    if not quiet:
        print("\n$ " + " ".join(cmd))
        print(out[-4000:])
    return p.returncode, out


def call_module_main(module_name: str, argv: list[str], quiet: bool = False) -> str:
    old_argv = sys.argv[:]
    buf = io.StringIO()
    try:
        sys.argv = [module_name.rsplit(".", 1)[-1] + ".py"] + argv
        cm_stdout = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
        cm_stderr = contextlib.redirect_stderr(buf) if quiet else contextlib.nullcontext()
        with cm_stdout, cm_stderr:
            mod = importlib.import_module(module_name)
            if not hasattr(mod, "main"):
                raise RuntimeError(f"O módulo {module_name} não possui função main().")
            try:
                ret = mod.main()
                code = 0 if ret is None else int(ret)
            except SystemExit as e:
                code = 0 if e.code is None else int(e.code)
        captured = buf.getvalue()
        if code != 0:
            raise RuntimeError(f"Falha ao executar {module_name} com código {code}.\n{captured[-4000:]}")
        return captured
    finally:
        sys.argv = old_argv


def ensure_cfg(cfg: Path, template_cfg: Path | None, quiet: bool = False) -> None:
    if cfg.exists():
        return
    if template_cfg is None:
        raise SystemExit(f"ERRO: TOML não encontrado: {cfg}\nInforme --template-cfg para criar o TOML a partir de um modelo.")
    if not template_cfg.exists():
        raise SystemExit(f"ERRO: template TOML não encontrado: {template_cfg}")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_cfg, cfg)
    if not quiet:
        print(f"[OK] TOML criado a partir do template: {cfg}")


def set_toml_value_if_present(txt: str, key: str, value: str) -> str:
    pat = re.compile(rf"(?m)^({re.escape(key)}\s*=\s*).*$")
    if not pat.search(txt):
        return txt
    return pat.sub(lambda m: m.group(1) + value, txt)


def prepare_cfg(cfg: Path, target_n: int, min_palavras: int, quiet: bool = False) -> None:
    backup(cfg, "cfg_unificado", quiet=quiet)
    txt = cfg.read_text(encoding="utf-8", errors="ignore")
    for key, value in [
        ("target_n", str(target_n)),
        ("target_n_fulltext", str(target_n)),
        ("min_palavras", str(min_palavras)),
        ("min_palavras_total", str(min_palavras)),
        ("usar_biber", "true"),
        ("usar_biblatex", "true"),
        ("estilo_bibliografia", '"abnt"'),
    ]:
        txt = set_toml_value_if_present(txt, key, value)
    cfg.write_text(txt, encoding="utf-8")
    if not quiet:
        print(f"[OK] TOML preparado: {cfg}")


def parse_bib_keys(bib: Path) -> list[str]:
    if not bib.exists():
        return []
    txt = bib.read_text(encoding="utf-8", errors="ignore")
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)", txt)


def sanitize_bib(bib: Path, quiet: bool = False) -> None:
    if not bib.exists():
        raise SystemExit(f"ERRO: BIB não encontrado: {bib}")
    backup(bib, "bib_unificado", quiet=quiet)
    txt = bib.read_text(encoding="utf-8", errors="ignore")
    field_names = ["author", "editor", "title", "journaltitle", "journal", "publisher", "institution", "organization"]
    for field in field_names:
        pat = re.compile(rf"(?im)^(\s*{field}\s*=\s*[\{{\"])(.*?)([\}}\"],?\s*)$")
        def repl(m: re.Match[str]) -> str:
            start, value, end = m.groups()
            if field in {"author", "editor"}:
                value = value.replace(";", " and ")
            value = value.replace(" & ", r" \& ")
            return start + value + end
        txt = pat.sub(repl, txt)
    bib.write_text(txt, encoding="utf-8")
    if not quiet:
        print(f"[OK] BIB saneado: {bib}")


def convert_literal_citations(txt: str, keys: list[str]) -> str:
    if not keys:
        return txt
    keyset = set(keys)
    def repl_key(m: re.Match[str]) -> str:
        key = m.group(1)
        return rf"\parencite{{{key}}}" if key in keyset else m.group(0)
    txt = re.sub(r"\[([A-Za-z][A-Za-z0-9:_\-.]{4,})\]", repl_key, txt)
    def repl_num(m: re.Match[str]) -> str:
        raw = m.group(1)
        nums: list[int] = []
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece.isdigit():
                return m.group(0)
            nums.append(int(piece))
        if nums and all(1 <= n <= len(keys) for n in nums):
            return rf"\parencite{{{','.join(keys[n-1] for n in nums)}}}"
        return m.group(0)
    return re.sub(r"(?<![A-Za-z0-9])\[([0-9]+(?:\s*,\s*[0-9]+)*)\](?![A-Za-z0-9])", repl_num, txt)


def generic_text_fixes(txt: str) -> str:
    novo = "evitando tabela extensa de difícil leitura e preservando a rastreabilidade das citações"
    txt = re.sub(
        r"evitando\s+tabela\s+larga\s+no\s*(?:LATEX|LaTeX|\\LaTeX(?:\{\})?)\s*e\s+preservando\s+a\s+rastreabilidade\s+das\s+citações",
        novo,
        txt,
        flags=re.I,
    )
    for old in [
        "evitando tabela larga no LATEX e preservando a rastreabilidade das citações",
        "evitando tabela larga no LaTeX e preservando a rastreabilidade das citações",
        r"evitando tabela larga no \LaTeX{} e preservando a rastreabilidade das citações",
        r"evitando tabela larga no \LaTeX e preservando a rastreabilidade das citações",
    ]:
        txt = txt.replace(old, novo)
    txt = txt.replace(r"$\backslash$%", r"\%")
    txt = txt.replace(r"\textbackslash{}", "")
    return txt


def remove_org_bib_headers(txt: str) -> str:
    txt = re.sub(r"(?m)^#\+LATEX_HEADER:\s*\\usepackage(?:\[[^\]]*\])?\{biblatex\}\s*$", "", txt)
    txt = re.sub(r"(?m)^#\+LATEX_HEADER:\s*\\addbibresource\{[^}]+\}\s*$", "", txt)
    txt = re.sub(r"(?m)^#\+LATEX_HEADER:\s*\\usepackage(?:\[[^\]]*\])?\{abntex2cite\}\s*$", "", txt)
    txt = re.sub(r"(?m)^#\+LATEX_HEADER:\s*\\bibliographystyle\{[^}]+\}\s*$", "", txt)
    txt = re.sub(r"(?m)^#\+LATEX_HEADER:\s*\\bibliography\{[^}]+\}\s*$", "", txt)
    return txt


def insert_org_bib_headers(txt: str, bib_name: str) -> str:
    lines = txt.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#+"):
            insert_at = i + 1
    headers = [
        rf"#+LATEX_HEADER: \usepackage[{BIBLATEX_ABNT_OPTIONS}]{{biblatex}}",
        rf"#+LATEX_HEADER: \addbibresource{{{bib_name}}}",
    ]
    for h in reversed(headers):
        lines.insert(insert_at, h)
    return "\n".join(lines).rstrip() + "\n"


def ensure_org_references(txt: str) -> str:
    txt = re.sub(r"(?m)^\\bibliographystyle\{[^}]+\}\s*$", "", txt)
    txt = re.sub(r"(?m)^\\bibliography\{[^}]+\}\s*$", "", txt)
    if r"\printbibliography" not in txt:
        txt = txt.rstrip() + "\n\n* Referências\n\\nocite{*}\n\\printbibliography[heading=none]\n"
    return txt


def patch_org(org: Path, bib_name: str, keys: list[str], quiet: bool = False) -> None:
    if not org.exists():
        raise SystemExit(f"ERRO: ORG não encontrado: {org}")
    backup(org, "org_unificado", quiet=quiet)
    txt = org.read_text(encoding="utf-8", errors="ignore")
    txt = remove_org_bib_headers(txt)
    txt = insert_org_bib_headers(txt, bib_name)
    txt = convert_literal_citations(txt, keys)
    txt = txt.replace(r"\cite{", r"\parencite{")
    txt = generic_text_fixes(txt)
    txt = ensure_org_references(txt)
    org.write_text(txt, encoding="utf-8")
    if not quiet:
        print(f"[OK] ORG corrigido: {org}")


def _lisp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def export_org_to_tex(org: Path, out: Path, project_root: Path, quiet: bool = False) -> None:
    if not org.exists():
        raise SystemExit(f"ERRO: ORG não encontrado: {org}")
    if shutil.which("emacs") is None:
        raise SystemExit("ERRO: emacs não encontrado no PATH.")

    misc_dir = project_root / "app_bundle" / "misc"
    academic_el = misc_dir / "academic-writing.el"
    fallback_class = "\\documentclass[12pt,a4paper]{article}\n[NO-DEFAULT-PACKAGES]\n[PACKAGES]\n[EXTRA]"
    section = "\\section{%s}"
    section_star = "\\section*{%s}"
    subsection = "\\subsection{%s}"
    subsection_star = "\\subsection*{%s}"
    subsubsection = "\\subsubsection{%s}"
    subsubsection_star = "\\subsubsection*{%s}"

    el_code = "\n".join([
        "(require 'org)",
        "(require 'ox-latex)",
        f"(add-to-list 'load-path {_lisp_string(str(misc_dir))})",
        f"(let ((academic-el {_lisp_string(str(academic_el))})) (when (file-exists-p academic-el) (load-file academic-el)))",
        "(unless (assoc \"fgv-paper\" org-latex-classes)",
        "  (add-to-list 'org-latex-classes",
        "    '(\"fgv-paper\"",
        f"      {_lisp_string(fallback_class)}",
        f"      ({_lisp_string(section)} . {_lisp_string(section_star)})",
        f"      ({_lisp_string(subsection)} . {_lisp_string(subsection_star)})",
        f"      ({_lisp_string(subsubsection)} . {_lisp_string(subsubsection_star)}))))",
        f"(find-file {_lisp_string(str(org))})",
        "(org-latex-export-to-latex)",
        "",
    ])

    with tempfile.NamedTemporaryFile("w", suffix=".el", encoding="utf-8", delete=False) as fh:
        script_path = Path(fh.name)
        fh.write(el_code)

    try:
        code, output = run(["emacs", "--batch", "-l", str(script_path)], cwd=out, quiet=quiet)
    finally:
        try:
            script_path.unlink()
        except OSError:
            pass

    if code != 0:
        raise SystemExit(f"ERRO: falha no export Org -> TeX.\n{output[-4000:]}")


def remove_tex_bib_config(tex: str) -> str:
    tex = re.sub(r"(?m)^\\PassOptionsToPackage\{[^}]*\}\{biblatex\}\s*$", "", tex)
    tex = re.sub(r"(?m)^\\usepackage(?:\[[^\]]*\])?\{biblatex\}\s*$", "", tex)
    tex = re.sub(r"(?m)^\\addbibresource\{[^}]+\}\s*$", "", tex)
    tex = re.sub(r"(?m)^\\usepackage(?:\[[^\]]*\])?\{abntex2cite\}\s*$", "", tex)
    tex = re.sub(r"(?m)^\\bibliographystyle\{[^}]+\}\s*$", "", tex)
    tex = re.sub(r"(?m)^\\bibliography\{[^}]+\}\s*$", "", tex)
    return tex


def insert_tex_bib_config(tex: str, bib_name: str) -> str:
    insert = rf"\usepackage[{BIBLATEX_ABNT_OPTIONS}]{{biblatex}}" + "\n" + rf"\addbibresource{{{bib_name}}}" + "\n"
    if r"\begin{document}" not in tex:
        raise SystemExit("ERRO: não encontrei \\begin{document} no TEX.")
    return tex.replace(r"\begin{document}", insert + r"\begin{document}", 1)


def ensure_tex_references(tex: str) -> str:
    tex = re.sub(r"(?m)^\\bibliographystyle\{[^}]+\}\s*$", "", tex)
    tex = re.sub(r"(?m)^\\bibliography\{[^}]+\}\s*$", "", tex)
    if r"\printbibliography" not in tex:
        tex = tex.replace(r"\end{document}", r"\section{Referências}" + "\n" + r"\nocite{*}" + "\n" + r"\printbibliography[heading=none]" + "\n" + r"\end{document}", 1)
    return tex


def patch_tex(tex_path: Path, bib_name: str, keys: list[str], quiet: bool = False) -> None:
    if not tex_path.exists():
        raise SystemExit(f"ERRO: TEX não encontrado: {tex_path}")
    backup(tex_path, "tex_unificado", quiet=quiet)
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    tex = remove_tex_bib_config(tex)
    tex = insert_tex_bib_config(tex, bib_name)
    tex = convert_literal_citations(tex, keys)
    tex = tex.replace(r"\cite{", r"\parencite{")
    tex = generic_text_fixes(tex)
    tex = ensure_tex_references(tex)
    tex_path.write_text(tex, encoding="utf-8")
    if not quiet:
        print(f"[OK] TEX corrigido: {tex_path}")


def compile_pdf(out: Path, prefix: str, project_root: Path, quiet: bool = False) -> Path:
    tex = out / f"{prefix}.tex"
    pdf = out / f"{prefix}.pdf"
    if not tex.exists():
        raise SystemExit(f"ERRO: TEX não encontrado: {tex}")
    env = os.environ.copy()
    tex_paths = [".", str(out), str(project_root), str(project_root / "app_bundle" / "misc")]
    env["TEXINPUTS"] = ":".join(tex_paths) + ":" + env.get("TEXINPUTS", "")
    env["BIBINPUTS"] = f".:{out}:" + env.get("BIBINPUTS", "")
    for ext in ["pdf", "aux", "bbl", "bcf", "blg", "log", "out", "run.xml", "toc"]:
        p = out / f"{prefix}.{ext}"
        if p.exists():
            p.unlink()
    for cmd in [
        ["pdflatex", "-file-line-error", "-interaction=nonstopmode", f"{prefix}.tex"],
        ["biber", prefix],
        ["pdflatex", "-file-line-error", "-interaction=nonstopmode", f"{prefix}.tex"],
        ["pdflatex", "-file-line-error", "-interaction=nonstopmode", f"{prefix}.tex"],
    ]:
        code, output = run(cmd, cwd=out, env=env, quiet=quiet)
        if code != 0:
            raise SystemExit(f"ERRO: falha em {' '.join(cmd)}\n{output[-4000:]}")
    if not pdf.exists():
        raise SystemExit(f"ERRO: PDF não foi gerado: {pdf}")
    return pdf


def validate_pdf(pdf: Path, quiet: bool = False) -> None:
    if not pdf.exists():
        raise SystemExit(f"ERRO: PDF não encontrado para validação: {pdf}")
    if shutil.which("pdftotext") is None:
        if not quiet:
            print("[AVISO] pdftotext não encontrado; validação textual ignorada.")
        return
    code, txt = run(["pdftotext", str(pdf), "-"], quiet=True)
    if code != 0:
        raise SystemExit("ERRO: pdftotext falhou na validação.")
    bad_patterns = [
        r"LATEX",
        r"LaTeX",
        r"\[[0-9]+(?:,\s*[0-9]+)*\]",
        r"Soares, Ninin e Lima \(Soares",
        r"Gabbay et al\. \(Gabbay",
        r"Dijkstra \(Dijkstra",
    ]
    hits: list[str] = []
    for i, line in enumerate(txt.splitlines(), start=1):
        if any(re.search(pat, line) for pat in bad_patterns):
            hits.append(f"{i}:{line}")
    if hits:
        raise SystemExit("ERRO: validação encontrou problemas no PDF final:\n" + "\n".join(hits[:40]))
    if not quiet:
        print("[OK] Validação textual do PDF sem problemas.")



def _strip_braces_quotes(value: str) -> str:
    value = value.strip().rstrip(',').strip()
    changed = True
    while changed and len(value) >= 2:
        changed = False
        if (value[0] == '{' and value[-1] == '}') or (value[0] == '"' and value[-1] == '"'):
            value = value[1:-1].strip()
            changed = True
    value = value.replace(r"\&", "&")
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def parse_bib_entries_for_docx(bib: Path) -> dict[str, dict[str, str]]:
    """Extrai metadados suficientes para materializar citações e referências no DOCX."""
    if not bib.exists():
        return {}
    txt = bib.read_text(encoding="utf-8", errors="ignore")
    entries: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt):
        key = m.group(2).strip()
        start = m.end()
        level = 1
        i = start
        while i < len(txt) and level > 0:
            if txt[i] == "{":
                level += 1
            elif txt[i] == "}":
                level -= 1
            i += 1
        body = txt[start:max(start, i - 1)]
        fields: dict[str, str] = {"ENTRYTYPE": m.group(1), "ID": key}
        field_pat = re.compile(
            r"(?is)\b([A-Za-z][A-Za-z0-9_:-]*)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\")\s*,?"
        )
        for fm in field_pat.finditer(body):
            fields[fm.group(1).lower()] = _strip_braces_quotes(fm.group(2))
        if "year" not in fields:
            date = fields.get("date", "")
            ym = re.search(r"(19|20)\d{2}", date)
            if ym:
                fields["year"] = ym.group(0)
        entries[key] = fields
    return entries


def _split_authors(author_field: str) -> list[str]:
    author_field = _strip_braces_quotes(author_field or "")
    if not author_field:
        return []
    if " and " in author_field:
        parts = re.split(r"\s+and\s+", author_field)
    elif ";" in author_field:
        parts = author_field.split(";")
    else:
        parts = [author_field]
    return [p.strip() for p in parts if p.strip()]


def _author_surname(author: str) -> str:
    author = _strip_braces_quotes(author)
    if not author:
        return ""
    if "," in author:
        return author.split(",", 1)[0].strip()
    tokens = author.split()
    if len(tokens) == 1:
        return tokens[0]
    # Autor institucional ou já abreviado em maiúsculas.
    if author.upper() == author and len(tokens) <= 4:
        return author.title()
    return tokens[-1]


def _citation_authors(fields: dict[str, str], textual: bool = False) -> str:
    authors = [_author_surname(a) for a in _split_authors(fields.get("author", ""))]
    authors = [a for a in authors if a]
    if not authors:
        title = fields.get("title", fields.get("ID", "Fonte"))
        return title.split(":", 1)[0][:60]
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} e {authors[1]}" if textual else f"{authors[0]}; {authors[1]}"
    if len(authors) == 3:
        return f"{authors[0]}, {authors[1]} e {authors[2]}" if textual else f"{authors[0]}; {authors[1]}; {authors[2]}"
    return f"{authors[0]} et al."


def _cite_text_for_key(key: str, entries: dict[str, dict[str, str]], textual: bool = False) -> str:
    fields = entries.get(key.strip(), {"ID": key.strip()})
    year = fields.get("year", "s.d.") or "s.d."
    authors = _citation_authors(fields, textual=textual)
    return f"{authors} ({year})" if textual else f"{authors}, {year}"


def materialize_citations_for_docx(txt: str, entries: dict[str, dict[str, str]]) -> str:
    def repl_textcite(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        return "; ".join(_cite_text_for_key(k, entries, textual=True) for k in keys)

    def repl_parencite(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        return "(" + "; ".join(_cite_text_for_key(k, entries, textual=False) for k in keys) + ")"

    txt = re.sub(r"\\textcite\{([^}]+)\}", repl_textcite, txt)
    txt = re.sub(r"\\parencite\{([^}]+)\}", repl_parencite, txt)
    txt = re.sub(r"\\cite\{([^}]+)\}", repl_parencite, txt)
    return txt


def _reference_line(fields: dict[str, str]) -> str:
    authors = fields.get("author", "").replace(" and ", "; ")
    authors = _strip_braces_quotes(authors)
    if authors:
        authors = authors.upper()
    else:
        authors = fields.get("ID", "REFERÊNCIA").upper()
    title = fields.get("title", "Sem título")
    year = fields.get("year", "s.d.") or "s.d."
    venue = fields.get("journaltitle") or fields.get("journal") or fields.get("publisher") or fields.get("institution") or ""
    parts = [f"{authors}. {title}."]
    if venue:
        parts.append(f"{venue}, {year}.")
    else:
        parts.append(f"{year}.")
    if fields.get("doi"):
        parts.append(f"DOI: {fields['doi']}.")
    if fields.get("url"):
        parts.append(f"Disponível em: {fields['url']}.")
    return " ".join(parts)


def render_references_for_docx(entries: dict[str, dict[str, str]]) -> str:
    if not entries:
        return ""
    lines = [_reference_line(fields) for _, fields in entries.items()]
    return "\n\n".join(lines)


def make_docx_source_org(org: Path, bib: Path, out: Path, prefix: str) -> Path:
    entries = parse_bib_entries_for_docx(bib)
    txt = org.read_text(encoding="utf-8", errors="ignore")
    txt = generic_text_fixes(txt)
    txt = materialize_citations_for_docx(txt, entries)

    cleaned_lines: list[str] = []
    for line in txt.splitlines():
        if re.match(r"^#\+(LATEX|LATEX_HEADER|LATEX_CLASS|LATEX_CLASS_OPTIONS):", line, flags=re.I):
            continue
        if re.match(r"^\\(makemytitle|clearpage|newpage|tableofcontents)\b", line.strip()):
            continue
        cleaned_lines.append(line)
    txt = "\n".join(cleaned_lines)

    refs = render_references_for_docx(entries)
    txt = re.sub(r"(?m)^\\nocite\{\*\}\s*$", "", txt)
    if refs:
        txt, n = re.subn(r"(?m)^\\printbibliography(?:\[[^\]]*\])?\s*$", refs, txt, count=1)
        if n == 0 and "* Referências" not in txt:
            txt = txt.rstrip() + "\n\n* Referências\n" + refs + "\n"
    else:
        txt = re.sub(r"(?m)^\\printbibliography(?:\[[^\]]*\])?\s*$", "", txt)

    txt = txt.replace(r"\%", "%").replace(r"\&", "&")
    txt = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", txt)

    tmp = out / f"{prefix}.docx_source.org"
    tmp.write_text(txt.rstrip() + "\n", encoding="utf-8")
    return tmp


def ensure_reference_docx(reference_docx: Path, quiet: bool = False) -> Path:
    if reference_docx.exists():
        return reference_docx
    try:
        from docx import Document
        from docx.shared import Cm, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
    except Exception as exc:
        raise SystemExit(
            "ERRO: python-docx não está disponível para criar o reference DOCX.\n"
            "Instale no ambiente do projeto com: pipenv install python-docx\n"
            f"Detalhe: {exc}"
        )

    def set_style_font(style, name: str = "Times New Roman", size_pt: int = 12, bold: bool | None = None) -> None:
        style.font.name = name
        style.font.size = Pt(size_pt)
        if bold is not None:
            style.font.bold = bold
        rpr = style._element.get_or_add_rPr()
        rfonts = rpr.rFonts
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        for attr in ["w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"]:
            rfonts.set(qn(attr), name)

    reference_docx.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.0)

    styles = doc.styles
    normal = styles["Normal"]
    set_style_font(normal, size_pt=12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.first_line_indent = Cm(1.25)

    for name, size, bold, before, after in [
        ("Title", 14, True, 0, 12),
        ("Heading 1", 12, True, 12, 6),
        ("Heading 2", 12, True, 12, 6),
        ("Heading 3", 12, True, 12, 6),
    ]:
        if name in styles:
            st = styles[name]
            set_style_font(st, size_pt=size, bold=bold)
            st.paragraph_format.line_spacing = 1.5
            st.paragraph_format.space_before = Pt(before)
            st.paragraph_format.space_after = Pt(after)
            if name.startswith("Heading"):
                st.paragraph_format.first_line_indent = Cm(0)

    for name in ["Body Text", "Body Text 2", "Body Text 3"]:
        if name in styles:
            st = styles[name]
            set_style_font(st, size_pt=12)
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            st.paragraph_format.line_spacing = 1.5
            st.paragraph_format.space_before = Pt(0)
            st.paragraph_format.space_after = Pt(0)
            st.paragraph_format.first_line_indent = Cm(1.25)

    if "Bibliography" in styles:
        st = styles["Bibliography"]
        set_style_font(st, size_pt=12)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        st.paragraph_format.line_spacing = 1.0
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.first_line_indent = Cm(0)

    doc.add_paragraph("Modelo ABNT/FGV para exportação DOCX.")
    doc.save(reference_docx)
    if not quiet:
        print(f"[OK] reference DOCX criado: {reference_docx}")
    return reference_docx


def validate_docx(docx_path: Path) -> None:
    if not docx_path.exists() or docx_path.stat().st_size < 1000:
        raise SystemExit(f"ERRO: DOCX não foi gerado corretamente: {docx_path}")
    try:
        with zipfile.ZipFile(docx_path) as zf:
            names = set(zf.namelist())
            if "word/document.xml" not in names:
                raise SystemExit(f"ERRO: DOCX inválido, word/document.xml ausente: {docx_path}")
    except zipfile.BadZipFile:
        raise SystemExit(f"ERRO: arquivo DOCX inválido: {docx_path}")


def emit_docx_stage(art: Path, prefix: str, project_root: Path, reference_docx: Path | None = None, quiet: bool = False) -> Path:
    """Gera DOCX canônico por python-docx, usando ORG/document.json/BIB finais.

    Versão v11: remove resíduos de blocos LaTeX/Org, extrai Resumo/Abstract
    dos blocos exportados e prioriza os metadados de capa do ORG final.
    O argumento reference_docx é mantido apenas por compatibilidade com a CLI v8/v9.
    """
    from app_bundle.scripts.pipeline.render_docx_canonico import render_docx_for_article

    return render_docx_for_article(art=art, prefix=prefix, cfg=None, output=None, quiet=quiet)

def prepare_fulltext_stage(art: Path, cfg: Path, target_n: int, min_palavras: int, chars_por_pdf: int, quiet: bool = False) -> None:
    if not quiet:
        print("[ETAPA] Preparando corpus full text...")
    call_module_main(
        "app_bundle.scripts.pipeline.preparar_artigo_longo_fulltext",
        ["--art-dir", str(art), "--cfg-art", str(cfg), "--target-n", str(target_n), "--min-palavras", str(min_palavras), "--chars-por-pdf", str(chars_por_pdf)],
        quiet=quiet,
    )


def generate_article_stage(art: Path, cfg: Path, quiet: bool = False) -> None:
    if not quiet:
        print("[ETAPA] Gerando artigo longo seccional...")
    call_module_main(
        "app_bundle.scripts.pipeline.gerar_artigo_longo_fulltext_secional",
        ["--art-dir", str(art), "--cfg-art", str(cfg)],
        quiet=quiet,
    )


def final_fix_stage(art: Path, prefix: str, project_root: Path, quiet: bool = False) -> Path:
    out = art / "output"
    org = out / f"{prefix}.org"
    tex = out / f"{prefix}.tex"
    bib = out / f"{prefix}.bib"
    if not out.exists():
        raise SystemExit(f"ERRO: pasta output não encontrada: {out}")
    if not org.exists():
        raise SystemExit(f"ERRO: ORG não encontrado: {org}")
    if not bib.exists():
        raise SystemExit(f"ERRO: BIB não encontrado: {bib}")
    sanitize_bib(bib, quiet=quiet)
    keys = parse_bib_keys(bib)
    patch_org(org, bib.name, keys, quiet=quiet)
    export_org_to_tex(org, out, project_root=project_root, quiet=quiet)
    if not tex.exists():
        raise SystemExit(f"ERRO: TEX não foi gerado: {tex}")
    patch_tex(tex, bib.name, keys, quiet=quiet)
    pdf = compile_pdf(out, prefix, project_root=project_root, quiet=quiet)
    validate_pdf(pdf, quiet=quiet)
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera artigo final em PDF com fluxo unificado.")
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--cfg-art", required=True)
    ap.add_argument("--template-cfg", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--target-n", type=int, default=20)
    ap.add_argument("--min-palavras", type=int, default=8500)
    ap.add_argument("--chars-por-pdf", type=int, default=18000)
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--emit-docx", action="store_true", help="Também gera DOCX a partir do ORG final, com reference DOCX ABNT/FGV.")
    ap.add_argument("--docx-only", action="store_true", help="Gera somente o DOCX a partir do ORG/BIB existentes ou recém-gerados, sem recompilar PDF.")
    ap.add_argument("--reference-docx", default=None, help="Modelo DOCX de referência. Se omitido, cria/usa app_bundle/templates/docx/reference_abnt_fgv.docx.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    art = Path(args.art_dir).resolve()
    cfg = Path(args.cfg_art).resolve()
    template_cfg = Path(args.template_cfg).resolve() if args.template_cfg else None
    prefix = args.prefix or cfg.stem

    if not art.exists():
        raise SystemExit(f"ERRO: diretório do artigo não encontrado: {art}")

    ensure_cfg(cfg, template_cfg, quiet=args.quiet)
    if not (args.skip_prepare and args.skip_generate):
        prepare_cfg(cfg, args.target_n, args.min_palavras, quiet=args.quiet)
    elif not args.quiet:
        print("[OK] Modo recompilação: TOML preservado.")

    if not args.skip_prepare:
        prepare_fulltext_stage(art, cfg, args.target_n, args.min_palavras, args.chars_por_pdf, quiet=args.quiet)
    if not args.skip_generate:
        generate_article_stage(art, cfg, quiet=args.quiet)

    reference_docx = Path(args.reference_docx).resolve() if args.reference_docx else None

    if args.docx_only:
        docx = emit_docx_stage(art, prefix, project_root=project_root, reference_docx=reference_docx, quiet=args.quiet)
        print(f"[OK] DOCX final: {docx}")
        return 0

    pdf = final_fix_stage(art, prefix, project_root=project_root, quiet=args.quiet)
    print(f"[OK] PDF final: {pdf}")

    if args.emit_docx:
        docx = emit_docx_stage(art, prefix, project_root=project_root, reference_docx=reference_docx, quiet=args.quiet)
        print(f"[OK] DOCX final: {docx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

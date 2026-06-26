#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

_ALLOWED_ENGINES = {"lualatex", "xelatex", "pdflatex"}


def _safe_engine(raw: str | None) -> str:
    engine = str(raw or "lualatex").strip().lower()
    if engine not in _ALLOWED_ENGINES:
        raise ValueError(f"pdf_engine inválido: {engine}. Use um de: {', '.join(sorted(_ALLOWED_ENGINES))}")
    if not shutil.which(engine):
        raise RuntimeError(f"Engine LaTeX não encontrado no PATH: {engine}")
    return engine


def run_compile_sequence(
    org_path: Path,
    academic_writing: Path | None = None,
    latex_extra_path: Path | None = None,
    pdf_engine: str = "lualatex",
) -> Path:
    """Exporta ORG para PDF via Emacs batch + LaTeX/Biber.

    `pdf_engine` agora vem do TOML ([latex].pdf_engine) em vez de ficar
    fixo em lualatex. O padrão continua lualatex porque é o fluxo validado.
    """
    org_path = org_path.resolve()
    if not org_path.exists():
        raise FileNotFoundError(org_path)
    emacs = shutil.which("emacs")
    if not emacs:
        raise RuntimeError("Emacs não encontrado no PATH.")
    engine = _safe_engine(pdf_engine)

    env = os.environ.copy()
    if latex_extra_path and latex_extra_path.exists():
        p = latex_extra_path
        if p.is_file():
            p = p.parent
        env["TEXINPUTS"] = str(p.resolve()) + "//:" + env.get("TEXINPUTS", "")

    elisp = org_path.with_name(org_path.stem + "_export_pdf.el")
    load_line = f'(load-file "{academic_writing}")' if academic_writing and academic_writing.exists() else ''
    if academic_writing and not academic_writing.exists():
        print(f"AVISO: academic-writing.el não encontrado: {academic_writing}")

    latex_cmd = f"{engine} -interaction nonstopmode -shell-escape -output-directory %o %f"
    elisp.write_text(f'''
(require 'org)
(require 'ox-latex)
(require 'oc)
(ignore-errors (require 'oc-biblatex))
{load_line}
;; Evita erro do Org 9.5 com forma antiga/dotted de org-cite-export-processors.
;; O pipeline usa citações LaTeX diretas (\\parencite/\\textcite) e injeta BibLaTeX via LATEX_HEADER.
(setq org-cite-export-processors '((latex biblatex) (t basic)))
(setq org-confirm-babel-evaluate nil)
(setq org-latex-pdf-process '("{latex_cmd}" "biber %b" "{latex_cmd}" "{latex_cmd}"))
(find-file "{org_path}")
(org-latex-export-to-pdf)
''', encoding="utf-8")

    proc = subprocess.run([emacs, "--batch", "-l", str(elisp)], cwd=str(org_path.parent), text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        err = org_path.with_name(org_path.stem + "_pdf_erro.txt")
        err.write_text(proc.stdout + "\n\nSTDERR:\n" + proc.stderr, encoding="utf-8")
        raise RuntimeError(f"Falha ao exportar PDF. Log: {err}")
    pdf = org_path.with_suffix(".pdf")
    if not pdf.exists():
        raise RuntimeError(f"PDF não foi criado: {pdf}")
    return pdf

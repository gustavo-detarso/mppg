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




# MPPG_FGV_RECOMPILE_AUTO_RESOURCES_V1
# Para o comando --recompile sem --config, reutilizar os assets da distribuição.
# Não reescreve Org/TOML e não se aplica a classes não institucionais.
def _auto_fgv_org_resources(
    org_path: Path,
    academic_writing: Path | None,
    latex_extra_path: Path | None,
) -> tuple[Path | None, Path | None]:
    if academic_writing is not None and latex_extra_path is not None:
        return academic_writing, latex_extra_path

    latex_class = None
    # Apenas o cabeçalho do Org; nunca interpretar exemplos no corpo do texto.
    with org_path.open("r", encoding="utf-8", errors="replace") as stream:
        for index, line in enumerate(stream):
            if index >= 256 or line.startswith("* "):
                break
            match = re.match(r"^\s*#\+LATEX_CLASS:\s*([^\s#]+)", line, flags=re.IGNORECASE)
            if match:
                latex_class = match.group(1).strip().lower()
                break

    if latex_class not in {"fgv-paper", "fgv-dissertacao"}:
        return academic_writing, latex_extra_path

    # latex_compile.py => pipeline/scripts/app_bundle/<raiz do software>.
    root = Path(__file__).resolve().parents[3]
    if academic_writing is None:
        candidate = root / "app_bundle" / "misc" / "academic-writing.el"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Classe {latex_class}: academic-writing.el não encontrado: {candidate}. "
                "Informe --academic-writing explicitamente."
            )
        academic_writing = candidate
    if latex_extra_path is None:
        candidate_dir = root / "app_bundle" / "institutions" / "fgv" / "latex"
        candidate_style = candidate_dir / f"{latex_class}.sty"
        if not candidate_style.is_file():
            raise FileNotFoundError(
                f"Classe {latex_class}: estilo não encontrado: {candidate_style}. "
                "Informe --latex-extra-path explicitamente."
            )
        latex_extra_path = candidate_dir

    print(
        f"[MPPG/FGV] Recursos LaTeX: classe={latex_class}; "
        f"academic-writing={academic_writing}; latex-extra-path={latex_extra_path}"
    )
    return academic_writing, latex_extra_path

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
    academic_writing, latex_extra_path = _auto_fgv_org_resources(
        org_path, academic_writing, latex_extra_path
    )
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

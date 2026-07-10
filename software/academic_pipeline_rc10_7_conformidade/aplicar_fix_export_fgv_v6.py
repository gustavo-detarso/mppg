#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path

NEW_FUNCTION = r'''def make_export_el(project_root: Path, org: Path) -> str:
    """
    Gera um arquivo Emacs Lisp temporário para exportar ORG -> TEX.

    Importante: todos os trechos LaTeX são serializados com json.dumps.
    Isso evita que o Emacs leia sequências como \usepackage como escape Unicode
    inválido (erro: Non-hex character used for Unicode escape).
    """
    root_s = json.dumps(str(project_root))
    org_s = json.dumps(str(org))

    class_body = json.dumps(
        "\\documentclass[12pt,a4paper]{article}\n"
        "[NO-DEFAULT-PACKAGES]\n"
        "[PACKAGES]\n"
        "\\usepackage{fgv-paper}\n"
        "[EXTRA]"
    )

    section = json.dumps("\\section{%s}")
    section_star = json.dumps("\\section*{%s}")
    subsection = json.dumps("\\subsection{%s}")
    subsection_star = json.dumps("\\subsection*{%s}")
    subsubsection = json.dumps("\\subsubsection{%s}")
    subsubsection_star = json.dumps("\\subsubsection*{%s}")
    paragraph = json.dumps("\\paragraph{%s}")
    paragraph_star = json.dumps("\\paragraph*{%s}")

    return f"""(setq debug-on-error t)
(require 'org)
(require 'ox-latex)
(let* ((project-root {root_s})
       (org-file {org_s})
       (misc-dir (expand-file-name \"app_bundle/misc\" project-root))
       (academic-writing (expand-file-name \"app_bundle/misc/academic-writing.el\" project-root)))
  (add-to-list 'load-path misc-dir)
  (when (file-exists-p academic-writing)
    (load-file academic-writing))
  (when (fboundp 'gm/academic-setup)
    (gm/academic-setup))
  (unless (boundp 'org-latex-classes)
    (setq org-latex-classes nil))
  (unless (assoc \"fgv-paper\" org-latex-classes)
    (add-to-list
     'org-latex-classes
     (list \"fgv-paper\"
           {class_body}
           (cons {section} {section_star})
           (cons {subsection} {subsection_star})
           (cons {subsubsection} {subsubsection_star})
           (cons {paragraph} {paragraph_star}))))
  (find-file org-file)
  (org-latex-export-to-latex))
"""
'''


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def replace_function(text: str) -> tuple[str, int]:
    pattern = re.compile(
        r"^def make_export_el\(project_root: Path, org: Path\) -> str:\n"
        r".*?"
        r"(?=^def export_org_to_tex\()",
        flags=re.S | re.M,
    )
    return pattern.subn(lambda m: NEW_FUNCTION.rstrip() + "\n\n", text, count=1)


def main() -> int:
    root = Path.cwd().resolve()
    if not (root / "app_bundle").exists():
        raise SystemExit("ERRO: execute este aplicador na raiz do projeto, onde existe app_bundle/.")

    targets = [
        root / "gerar_artigo_final_unificado.py",
        root / "app_bundle" / "scripts" / "pipeline" / "gerar_artigo_final_unificado.py",
    ]

    for p in targets:
        if not p.exists():
            print(f"[AVISO] Não encontrado: {p}")
            continue

        backup = p.with_name(p.name + ".bak_fix_export_fgv_v6_" + stamp())
        shutil.copy2(p, backup)

        text = p.read_text(encoding="utf-8", errors="ignore")
        new_text, n = replace_function(text)
        if n != 1:
            raise SystemExit(f"ERRO: não consegui substituir make_export_el em {p}. Substituições={n}")

        p.write_text(new_text, encoding="utf-8")
        py_compile.compile(str(p), doraise=True)

        print(f"[OK] Corrigido: {p}")
        print(f"[OK] Backup: {backup}")
        print(f"[OK] py_compile: {p}")

    print("\n[OK] Exportação FGV corrigida na versão v6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

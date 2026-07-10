#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil

ROOT = Path.cwd().resolve()
TARGETS = [
    ROOT / "gerar_artigo_final_unificado.py",
    ROOT / "app_bundle/scripts/pipeline/gerar_artigo_final_unificado.py",
]

NEW_EXPORT_FUNC = r'''def export_org_to_tex(org: Path, out: Path, quiet: bool = False) -> None:
    """
    Exporta ORG -> TEX via Emacs carregando a configuração acadêmica da pipeline.

    A opção mais segura é criar um arquivo .el temporário, em vez de montar um
    --eval grande na linha de comando. Isso evita erros de quoting como
    invalid-read-syntax e garante o registro da classe Org/LaTeX fgv-paper.
    """
    import json
    import tempfile

    if not org.exists():
        raise SystemExit(f"ERRO: ORG não encontrado: {org}")

    def _project_root() -> Path:
        candidates = [Path.cwd().resolve()]
        try:
            here = Path(__file__).resolve()
            candidates.extend([here.parent, *here.parents])
        except Exception:
            pass
        for c in candidates:
            if (c / "app_bundle" / "misc" / "academic-writing.el").exists():
                return c
            if (c / "app_bundle" / "scripts" / "pipeline").exists():
                return c
        return Path.cwd().resolve()

    root = _project_root()
    academic_el = root / "app_bundle" / "misc" / "academic-writing.el"

    # Fallback: se academic-writing.el não registrar fgv-paper, registramos uma classe mínima.
    # A compilação LaTeX continuará usando a classe/documento fgv-paper quando disponível no TEXINPUTS.
    lisp = f"""
(setq debug-on-error t)
(require 'org)
(require 'ox-latex)
(let ((academic-el {json.dumps(str(academic_el))}))
  (when (file-exists-p academic-el)
    (load-file academic-el)))
(unless (assoc "fgv-paper" org-latex-classes)
  (add-to-list 'org-latex-classes
               '("fgv-paper"
                 "\\\\documentclass[12pt,a4paper]{{fgv-paper}}"
                 ("\\\\section{{%s}}" . "\\\\section*{{%s}}")
                 ("\\\\subsection{{%s}}" . "\\\\subsection*{{%s}}")
                 ("\\\\subsubsection{{%s}}" . "\\\\subsubsection*{{%s}}"))))
(find-file {json.dumps(str(org))})
(let ((default-directory {json.dumps(str(out) + '/') }))
  (org-latex-export-to-latex))
"""

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".el", prefix="export_org_", encoding="utf-8", delete=False) as fh:
        fh.write(lisp)
        el_path = Path(fh.name)

    try:
        code, output = run(["emacs", "--batch", "-l", str(el_path)], cwd=out, quiet=quiet)
    finally:
        try:
            el_path.unlink()
        except OSError:
            pass

    if code != 0:
        raise SystemExit(f"ERRO: falha no export Org -> TeX.\n{output[-4000:]}")
'''


def replace_function(src: str, name: str, new_code: str) -> tuple[str, bool]:
    """Replace a top-level function by name, from def line until next top-level def/class or EOF."""
    pat = re.compile(
        rf"^def\s+{re.escape(name)}\s*\([^\n]*\):\n.*?(?=^(?:def|class)\s+|\Z)",
        flags=re.S | re.M,
    )
    if not pat.search(src):
        return src, False
    return pat.sub(lambda m: new_code.rstrip() + "\n\n", src, count=1), True


def ensure_import(src: str, import_line: str) -> str:
    if import_line in src:
        return src
    lines = src.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("from __future__"):
            insert_at = i + 1
    lines.insert(insert_at, import_line)
    return "\n".join(lines) + ("\n" if src.endswith("\n") else "")


def main() -> int:
    changed_any = False
    for path in TARGETS:
        if not path.exists():
            label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            print(f"[AVISO] Não encontrado: {label}")
            continue

        text = path.read_text(encoding="utf-8")
        new_text, ok = replace_function(text, "export_org_to_tex", NEW_EXPORT_FUNC)
        if not ok:
            raise SystemExit(f"ERRO: não encontrei export_org_to_tex em {path}")

        if "from pathlib import Path" not in new_text:
            new_text = ensure_import(new_text, "from pathlib import Path")

        if new_text != text:
            bak = path.with_name(path.name + ".bak_fix_export_fgv_v4_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            shutil.copy2(path, bak)
            path.write_text(new_text, encoding="utf-8")
            label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            print(f"[OK] Corrigido: {label}")
            print(f"[OK] Backup: {bak}")
            changed_any = True
        else:
            print(f"[OK] Sem alterações necessárias: {path}")

    if not changed_any:
        print("[OK] Nenhum arquivo precisou ser alterado.")
    print("[OK] Patch v4 aplicado. Agora rode py_compile e recompile o PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

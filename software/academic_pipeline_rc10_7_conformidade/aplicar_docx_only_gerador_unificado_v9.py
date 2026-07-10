#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import py_compile

ROOT = Path.cwd()
TARGETS = [
    ROOT / "gerar_artigo_final_unificado.py",
    ROOT / "app_bundle" / "scripts" / "pipeline" / "gerar_artigo_final_unificado.py",
]

ADD_ARG_AFTER = '    ap.add_argument("--emit-docx", action="store_true", help="Também gera DOCX a partir do ORG final, com reference DOCX ABNT/FGV.")\n'
ADD_ARG = '    ap.add_argument("--docx-only", action="store_true", help="Gera somente o DOCX a partir do ORG/BIB existentes ou recém-gerados, sem recompilar PDF.")\n'

OLD_BLOCK = '''    if not args.skip_prepare:
        prepare_fulltext_stage(art, cfg, args.target_n, args.min_palavras, args.chars_por_pdf, quiet=args.quiet)
    if not args.skip_generate:
        generate_article_stage(art, cfg, quiet=args.quiet)

    pdf = final_fix_stage(art, prefix, project_root=project_root, quiet=args.quiet)
    print(f"[OK] PDF final: {pdf}")

    if args.emit_docx:
        reference_docx = Path(args.reference_docx).resolve() if args.reference_docx else None
        docx = emit_docx_stage(art, prefix, project_root=project_root, reference_docx=reference_docx, quiet=args.quiet)
        print(f"[OK] DOCX final: {docx}")

    return 0
'''

NEW_BLOCK = '''    if not args.skip_prepare:
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
'''


def patch_file(path: Path) -> None:
    if not path.exists():
        print(f"[AVISO] Não encontrado: {path}")
        return

    bak = path.with_name(path.name + ".bak_docx_only_v9_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(path, bak)

    txt = path.read_text(encoding="utf-8")

    if "--docx-only" not in txt:
        if ADD_ARG_AFTER not in txt:
            raise SystemExit(f"ERRO: ponto de inserção de --docx-only não encontrado em {path}")
        txt = txt.replace(ADD_ARG_AFTER, ADD_ARG_AFTER + ADD_ARG, 1)

    if OLD_BLOCK in txt:
        txt = txt.replace(OLD_BLOCK, NEW_BLOCK, 1)
    elif "if args.docx_only:" in txt:
        print(f"[OK] {path} já contém lógica --docx-only.")
    else:
        raise SystemExit(f"ERRO: bloco principal para DOCX não encontrado em {path}")

    path.write_text(txt, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)

    print(f"[OK] Corrigido: {path}")
    print(f"[OK] Backup: {bak}")


def main() -> int:
    for target in TARGETS:
        patch_file(target)
    print("[OK] Patch v9 aplicado. Use --docx-only para gerar somente DOCX.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

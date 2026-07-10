#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PATCH_NAME = "robustez_fluxo_artigo_v1"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "app_bundle" / "scripts" / "pipeline").exists():
        raise SystemExit(
            "ERRO: execute este aplicador na raiz do projeto academic_pipeline_rc10_7_conformidade.\n"
            f"Diretório atual: {root}"
        )
    return root


def backup(path: Path) -> None:
    if not path.exists():
        return
    bak = path.with_name(path.name + f".bak_{PATCH_NAME}_{stamp()}")
    shutil.copy2(path, bak)
    print(f"[OK] Backup: {bak}")


def copy_tree(src: Path, dst_root: Path) -> None:
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        backup(dst)
        shutil.copy2(item, dst)
        print(f"[OK] Instalado: {dst}")


def run(cmd: list[str], cwd: Path) -> int:
    print("\n$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"[ERRO] Comando terminou com código {proc.returncode}")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aplica robustez estrutural ao fluxo Artigo PRISMA.")
    parser.add_argument("--skip-tests", action="store_true", help="Instala os arquivos sem rodar testes.")
    args = parser.parse_args()

    root = project_root()
    here = Path(__file__).resolve().parent
    files = here / "files"
    if not files.exists():
        raise SystemExit(f"ERRO: pasta files não encontrada ao lado do aplicador: {files}")

    copy_tree(files, root)

    if not args.skip_tests:
        py = sys.executable
        rc = run([
            py, "-m", "py_compile",
            "app_bundle/scripts/pipeline/academic_pipeline_tui.py",
            "app_bundle/scripts/pipeline/artigo_prisma_workflow.py",
            "app_bundle/scripts/pipeline/article_workflow/__init__.py",
            "app_bundle/scripts/pipeline/article_workflow/state.py",
            "app_bundle/scripts/pipeline/article_workflow/validators.py",
        ], root)
        if rc != 0:
            raise SystemExit(rc)

        # pytest pode não estar instalado em todos os ambientes. Se estiver, roda.
        pipeline_path = str(root / "app_bundle" / "scripts" / "pipeline")
        os.environ["PYTHONPATH"] = pipeline_path + os.pathsep + os.environ.get("PYTHONPATH", "")
        if shutil.which("pytest"):
            rc = run([
                py, "-m", "pytest", "-q",
                "app_bundle/tests/test_article_workflow_smoke.py",
                "app_bundle/tests/test_rc10_smoke.py",
            ], root)
            if rc != 0:
                raise SystemExit(rc)
        else:
            print("[AVISO] pytest não encontrado no PATH; testes pytest ignorados.")

    print("\n[OK] Patch aplicado: robustez estrutural do fluxo Artigo PRISMA v1")
    print("Abra a TUI com:")
    print("pipenv run python app_bundle/scripts/pipeline/academic_pipeline_tui.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

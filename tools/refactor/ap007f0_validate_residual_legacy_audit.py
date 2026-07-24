#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import traceback
from datetime import datetime

EXPECTED_HEAD = "ba43b7d606378501d6faafa62ad8c8a6697665e5"
EXPECTED_TREE = "078326090dd64572fb12a026e8d92968bf106d0f"
EXPECTED_RUNTIME_SHA256 = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
EXPECTED_MASTER_HEAD = "6adc5e7c6ce510a49eba13266eabfa227fbeae31"
EXPECTED_MASTER_TREE = "6822a3347ce2ddc771b2b8a965e8a1b9eb416b74"
EXPECTED_MASTER_NUL_SHA256 = "b9ed392d49c6ab8ee19739a0e5249d34ae9bb651a25912a4d85903e0755af367"
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py']
RUNTIME_REL = "software/academic_pipeline_mppg/academic_pipeline/runtime.py"
INVENTORY_REL = "docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json"
MASTER_REPO = pathlib.Path("/home/gustavodetarso/Documentos/mppg")


def run(args: list[str], cwd: pathlib.Path | None = None) -> str:
    completed = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"comando falhou ({completed.returncode}): {args!r}\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[2]


def main() -> int:
    root = repo_root()
    head = run(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    tree = run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"]).strip()
    if head != EXPECTED_HEAD or tree != EXPECTED_TREE:
        raise RuntimeError(f"baseline divergente: head={head} tree={tree}")
    if sha256(root / RUNTIME_REL) != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("runtime SHA-256 divergente")

    data = json.loads((root / INVENTORY_REL).read_text(encoding="utf-8"))
    if data.get("status") != "residual_legacy_audit_complete":
        raise RuntimeError("status do inventário divergente")
    if data.get("debt_catalog", {}).get("count") != 70:
        raise RuntimeError("catálogo não possui 70 dívidas")
    if len(data.get("direct_source_cases", [])) != 4:
        raise RuntimeError("não há exatamente quatro casos direct-source")

    raw = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    if not all(item.startswith("?? ") for item in entries):
        raise RuntimeError(f"mudança rastreada ou staged detectada: {entries!r}")
    actual_paths = sorted(item[3:] for item in entries)
    if actual_paths != sorted(EXPECTED_PATHS):
        raise RuntimeError(f"escopo divergente: {actual_paths!r}")
    if run(["git", "-C", str(root), "diff", "--cached", "--name-only"]) != "":
        raise RuntimeError("staging não vazio")

    master_head = run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD"]).strip()
    master_tree = run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD^{tree}"]).strip()
    if master_head != EXPECTED_MASTER_HEAD or master_tree != EXPECTED_MASTER_TREE:
        raise RuntimeError("master protegida divergente")
    raw_untracked = subprocess.check_output(["git", "-C", str(MASTER_REPO), "ls-files", "--others", "--exclude-standard", "-z"])
    paths = sorted(part for part in raw_untracked.split(b"\0") if part)
    nul = b"\0".join(paths) + (b"\0" if paths else b"")
    if hashlib.sha256(nul).hexdigest() != EXPECTED_MASTER_NUL_SHA256:
        raise RuntimeError("snapshot NUL da master divergente")

    print("COMPLETENESS_MARKER=COMPLETE")
    print("INTERNAL_CODE=0")
    print("FAILURES=0")
    print("WARNINGS=0")
    print("AP007F0_STATUS=residual_legacy_audit_complete")
    print("[GATE] AP-007F.0: contrato materializado válido, escopo exato, runtime e master preservados.")
    return 0


if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path.home() / "Downloads" / "mppg-logs" / f"ap007f0_validator_{stamp}_{os.getpid()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ap007f0_validator_{stamp}_{os.getpid()}.log"
    code = 1
    with log_path.open("w", encoding="utf-8") as log:
        class Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, data):
                for stream in self.streams: stream.write(data); stream.flush()
                return len(data)
            def flush(self):
                for stream in self.streams: stream.flush()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, log)
        sys.stderr = Tee(old_err, log)
        try:
            code = main()
        except Exception:
            traceback.print_exc()
            print("COMPLETENESS_MARKER=INCOMPLETE")
            print("INTERNAL_CODE=1")
            print("FAILURES=1")
            print("WARNINGS=0")
            print("AP007F0_STATUS=failed_closed")
            print("[GATE] AP-007F.0: validação bloqueada em modo fail-closed.")
        finally:
            print("=== ARQUIVO DE LOG PARA ANEXAR ===")
            print(str(log_path))
            sys.stdout, sys.stderr = old_out, old_err
    raise SystemExit(code)

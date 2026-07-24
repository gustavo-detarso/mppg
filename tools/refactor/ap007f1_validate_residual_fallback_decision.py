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
EXPECTED_RUNTIME_SHA = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
EXPECTED_MASTER_HEAD = "6adc5e7c6ce510a49eba13266eabfa227fbeae31"
EXPECTED_MASTER_TREE = "6822a3347ce2ddc771b2b8a965e8a1b9eb416b74"
EXPECTED_MASTER_NUL_SHA = "b9ed392d49c6ab8ee19739a0e5249d34ae9bb651a25912a4d85903e0755af367"
MANIFEST_REL = "docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json"
RUNTIME_REL = "software/academic_pipeline_mppg/academic_pipeline/runtime.py"
EXPECTED_ALL_UNTRACKED = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F1_RESIDUAL_FALLBACK_DECISION.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f1_residual_fallback_decision_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py', 'tools/refactor/ap007f1_validate_residual_fallback_decision.py']
MASTER_REPO = pathlib.Path("/home/gustavodetarso/Documentos/mppg")

def run(args: list[str], cwd: pathlib.Path | None = None) -> str:
    completed = subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"comando falhou rc={completed.returncode}: {args!r}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout

def root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[2]

def main() -> int:
    repo = root()
    assert run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip() == EXPECTED_HEAD
    assert run(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"]).strip() == EXPECTED_TREE
    assert hashlib.sha256((repo / RUNTIME_REL).read_bytes()).hexdigest() == EXPECTED_RUNTIME_SHA

    data = json.loads((repo / MANIFEST_REL).read_text(encoding="utf-8"))
    assert data["status"] == "residual_fallback_preserved_no_productive_edit"
    assert data["runtime_analysis"]["actual_fallback_return_count"] == 6
    assert data["runtime_analysis"]["cli_injection_count"] == 1
    assert data["runtime_analysis"]["run_legacy_runner_call_count"] == 1
    assert data["decisions"]["run_legacy"]["decision"] == "preserve_published_compatibility"
    assert data["decisions"]["direct_source_execution"]["decision"] == "supersede_test_contract_only"

    raw = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    assert all(item.startswith("?? ") for item in entries), entries
    assert sorted(item[3:] for item in entries) == EXPECTED_ALL_UNTRACKED
    assert run(["git", "-C", str(repo), "diff", "--cached", "--name-only"]) == ""

    assert run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD"]).strip() == EXPECTED_MASTER_HEAD
    assert run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD^{tree}"]).strip() == EXPECTED_MASTER_TREE
    raw_master = subprocess.check_output(
        ["git", "-C", str(MASTER_REPO), "ls-files", "--others", "--exclude-standard", "-z"]
    )
    paths = sorted(part for part in raw_master.split(b"\0") if part)
    nul = b"\0".join(paths) + (b"\0" if paths else b"")
    assert hashlib.sha256(nul).hexdigest() == EXPECTED_MASTER_NUL_SHA

    print("COMPLETENESS_MARKER=COMPLETE")
    print("INTERNAL_CODE=0")
    print("FAILURES=0")
    print("WARNINGS=0")
    print("AP007F1_STATUS=residual_fallback_preserved_no_productive_edit")
    print("[GATE] AP-007F.1: decisão residual válida; seis retornos reais, injeção e escopo confirmados.")
    return 0

if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path.home() / "Downloads" / "mppg-logs" / f"ap007f1_validator_{stamp}_{os.getpid()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ap007f1_validator_{stamp}_{os.getpid()}.log"
    code = 1
    with log_path.open("w", encoding="utf-8") as log:
        class Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush()
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
            print("AP007F1_STATUS=failed_closed")
            print("[GATE] AP-007F.1: validação bloqueada em modo fail-closed.")
        finally:
            print("=== ARQUIVO DE LOG PARA ANEXAR ===")
            print(str(log_path))
            sys.stdout, sys.stderr = old_out, old_err
    raise SystemExit(code)

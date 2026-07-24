from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import traceback
from datetime import datetime

REPO = pathlib.Path(__file__).parents[2]
MASTER_REPO = pathlib.Path("/home/gustavodetarso/Documentos/mppg")
EXPECTED_HEAD = "ba43b7d606378501d6faafa62ad8c8a6697665e5"
EXPECTED_TREE = "078326090dd64572fb12a026e8d92968bf106d0f"
EXPECTED_RUNTIME_SHA = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
EXPECTED_MASTER_HEAD = "6adc5e7c6ce510a49eba13266eabfa227fbeae31"
EXPECTED_MASTER_TREE = "6822a3347ce2ddc771b2b8a965e8a1b9eb416b74"
EXPECTED_MASTER_NUL_SHA = "b9ed392d49c6ab8ee19739a0e5249d34ae9bb651a25912a4d85903e0755af367"
MANIFEST_REL = "docs/refactor/academic-pipeline/AP-007/ap007f3_global_closure_manifest.json"
FINAL_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F1_RESIDUAL_FALLBACK_DECISION.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F2_FINAL_INTEGRATED_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F3_GLOBAL_CLOSURE.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json', 'docs/refactor/academic-pipeline/AP-007/ap007f2_final_integrated_validation.json', 'docs/refactor/academic-pipeline/AP-007/ap007f3_global_closure_manifest.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f1_residual_fallback_decision_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f2_final_integrated_validation_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f3_global_closure_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py', 'tools/refactor/ap007f1_validate_residual_fallback_decision.py', 'tools/refactor/ap007f2_validate_final_integrated_validation.py', 'tools/refactor/ap007f3_validate_global_closure.py']

def run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()

def main() -> int:
    data = json.loads((REPO / MANIFEST_REL).read_text(encoding="utf-8"))
    assert data["status"] == "ap007_global_closure_ready_for_authorized_commit"
    assert data["gate"] == "explicit_commit_authorization_required"
    assert data["closure_evidence"]["final_scope_count"] == 16
    assert data["commit_decision"]["authorization_required"] is True
    assert data["commit_decision"]["authorized"] is False
    assert data["project_state"]["ap007_formally_closed"] is True
    assert data["project_state"]["ap007_committed"] is False

    assert run(["git", "-C", str(REPO), "rev-parse", "HEAD"]) == EXPECTED_HEAD
    assert run(["git", "-C", str(REPO), "rev-parse", "HEAD^{tree}"]) == EXPECTED_TREE
    assert hashlib.sha256((REPO / "software/academic_pipeline_mppg/academic_pipeline/runtime.py").read_bytes()).hexdigest() == EXPECTED_RUNTIME_SHA
    assert run(["git", "-C", str(REPO), "diff", "--cached", "--name-only"]) == ""
    assert run(["git", "-C", str(REPO), "diff", "--name-only"]) == ""

    raw = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    assert all(item.startswith("?? ") for item in entries), entries
    assert sorted(item[3:] for item in entries) == FINAL_PATHS

    assert run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD"]) == EXPECTED_MASTER_HEAD
    assert run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD^{tree}"]) == EXPECTED_MASTER_TREE
    master_raw = subprocess.check_output(
        ["git", "-C", str(MASTER_REPO), "ls-files", "--others", "--exclude-standard", "-z"]
    )
    paths = sorted(item for item in master_raw.split(b"\0") if item)
    payload = b"\0".join(paths) + (b"\0" if paths else b"")
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_MASTER_NUL_SHA

    print("COMPLETENESS_MARKER=COMPLETE")
    print("INTERNAL_CODE=0")
    print("FAILURES=0")
    print("WARNINGS=0")
    print("AP007F3_STATUS=ap007_global_closure_ready_for_authorized_commit")
    print("[GATE] AP-007F.3: fechamento global válido; 16 caminhos exatos e autorização de commit ainda pendente.")
    return 0

if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path.home() / "Downloads" / "mppg-logs" / f"ap007f3_validator_{stamp}_{os.getpid()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ap007f3_validator_{stamp}_{os.getpid()}.log"
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
                for stream in self.streams:
                    stream.flush()
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
            print("AP007F3_STATUS=failed_closed")
        finally:
            print("=== ARQUIVO DE LOG PARA ANEXAR ===")
            print(str(log_path))
            sys.stdout, sys.stderr = old_out, old_err
    raise SystemExit(code)

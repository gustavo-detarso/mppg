#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, sys, traceback
from datetime import datetime

EXPECTED_HEAD = "ba43b7d606378501d6faafa62ad8c8a6697665e5"
EXPECTED_TREE = "078326090dd64572fb12a026e8d92968bf106d0f"
EXPECTED_RUNTIME_SHA = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
EXPECTED_MASTER_HEAD = "6adc5e7c6ce510a49eba13266eabfa227fbeae31"
EXPECTED_MASTER_TREE = "6822a3347ce2ddc771b2b8a965e8a1b9eb416b74"
EXPECTED_MASTER_NUL_SHA = "b9ed392d49c6ab8ee19739a0e5249d34ae9bb651a25912a4d85903e0755af367"
MANIFEST_REL = "docs/refactor/academic-pipeline/AP-007/ap007f2_final_integrated_validation.json"
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F1_RESIDUAL_FALLBACK_DECISION.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F2_FINAL_INTEGRATED_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json', 'docs/refactor/academic-pipeline/AP-007/ap007f2_final_integrated_validation.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f1_residual_fallback_decision_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f2_final_integrated_validation_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py', 'tools/refactor/ap007f1_validate_residual_fallback_decision.py', 'tools/refactor/ap007f2_validate_final_integrated_validation.py']
MASTER_REPO = pathlib.Path("/home/gustavodetarso/Documentos/mppg")

def run(args: list[str]) -> str:
    cp = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"comando falhou rc={cp.returncode}: {args!r}\n{cp.stdout}\n{cp.stderr}")
    return cp.stdout

def root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[2]

def main() -> int:
    repo = root()
    assert run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip() == EXPECTED_HEAD
    assert run(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"]).strip() == EXPECTED_TREE
    runtime = repo / "software/academic_pipeline_mppg/academic_pipeline/runtime.py"
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == EXPECTED_RUNTIME_SHA

    data = json.loads((repo / MANIFEST_REL).read_text(encoding="utf-8"))
    assert data["status"] == "final_integrated_validation_complete"
    assert data["runtime_equivalence"]["execution_count"] == 30
    assert data["runtime_equivalence"]["comparison_count"] == 24
    assert data["runtime_equivalence"]["non_equivalent_comparison_count"] == 0
    assert len(data["runtime_equivalence"]["surfaces"]) == 5
    assert all(
        isinstance(item["cwd"], str)
        for item in data["runtime_equivalence"]["surfaces"]
    )
    assert data["regression"]["historical_debt_count"] == 70
    assert data["regression"]["phase_local_scope_deselections"] == 2
    assert data["regression"]["total_deselection_arguments"] == 72
    assert data["regression"]["phase_local_nodeids"] == [
        "tests/characterization/test_ap007e0_distribution_isolation_inventory_contract.py::test_ap007e0_validator_executes_successfully",
        "tests/characterization/test_ap007f0_residual_legacy_audit_contract.py::test_ap007f0_scope_is_exact_and_unstaged",
    ]
    assert data["regression"]["productive_return_code"] == 0
    assert data["regression"]["failed"] == 0
    assert data["regression"]["errors"] == 0
    assert data["regression"]["xpassed"] == 0
    assert len(data["direct_source_reproduction"]) == 4
    assert all(item["signature_exact"] for item in data["direct_source_reproduction"])
    assert data["canonical_environment"]["preserved"] is True

    raw = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    assert all(item.startswith("?? ") for item in entries), entries
    assert sorted(item[3:] for item in entries) == EXPECTED_PATHS
    assert run(["git", "-C", str(repo), "diff", "--cached", "--name-only"]) == ""

    assert run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD"]).strip() == EXPECTED_MASTER_HEAD
    assert run(["git", "-C", str(MASTER_REPO), "rev-parse", "HEAD^{tree}"]).strip() == EXPECTED_MASTER_TREE
    master_raw = subprocess.check_output(["git", "-C", str(MASTER_REPO), "ls-files", "--others", "--exclude-standard", "-z"])
    paths = sorted(item for item in master_raw.split(b"\0") if item)
    payload = b"\0".join(paths) + (b"\0" if paths else b"")
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_MASTER_NUL_SHA

    print("COMPLETENESS_MARKER=COMPLETE")
    print("INTERNAL_CODE=0")
    print("FAILURES=0")
    print("WARNINGS=0")
    print("AP007F2_STATUS=final_integrated_validation_complete")
    print("[GATE] AP-007F.2: contrato final integrado válido e escopo exato de 12 caminhos.")
    return 0

if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path.home() / "Downloads" / "mppg-logs" / f"ap007f2_validator_{stamp}_{os.getpid()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ap007f2_validator_{stamp}_{os.getpid()}.log"
    code = 1
    with log_path.open("w", encoding="utf-8") as log:
        class Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, data):
                for stream in self.streams:
                    stream.write(data); stream.flush()
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
            print("AP007F2_STATUS=failed_closed")
        finally:
            print("=== ARQUIVO DE LOG PARA ANEXAR ===")
            print(str(log_path))
            sys.stdout, sys.stderr = old_out, old_err
    raise SystemExit(code)

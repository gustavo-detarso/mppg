from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

EXPECTED_HEAD = "ba43b7d606378501d6faafa62ad8c8a6697665e5"
EXPECTED_TREE = "078326090dd64572fb12a026e8d92968bf106d0f"
EXPECTED_RUNTIME_SHA = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
MANIFEST_REL = "docs/refactor/academic-pipeline/AP-007/ap007f3_global_closure_manifest.json"
FINAL_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F1_RESIDUAL_FALLBACK_DECISION.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F2_FINAL_INTEGRATED_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F3_GLOBAL_CLOSURE.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json', 'docs/refactor/academic-pipeline/AP-007/ap007f2_final_integrated_validation.json', 'docs/refactor/academic-pipeline/AP-007/ap007f3_global_closure_manifest.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f1_residual_fallback_decision_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f2_final_integrated_validation_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f3_global_closure_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py', 'tools/refactor/ap007f1_validate_residual_fallback_decision.py', 'tools/refactor/ap007f2_validate_final_integrated_validation.py', 'tools/refactor/ap007f3_validate_global_closure.py']

def _root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[4]

def _manifest() -> dict:
    return json.loads((_root() / MANIFEST_REL).read_text(encoding="utf-8"))

def test_ap007f3_global_closure_status() -> None:
    data = _manifest()
    assert data["schema_version"] == "ap007f3-global-closure-v1"
    assert data["status"] == "ap007_global_closure_ready_for_authorized_commit"
    assert data["project_state"]["ap007_formally_closed"] is True
    assert data["project_state"]["ap007_committed"] is False
    assert data["commit_decision"]["authorization_required"] is True
    assert data["commit_decision"]["authorized"] is False

def test_ap007f3_consolidated_validation() -> None:
    final = _manifest()["final_integrated_validation"]
    assert final["executions"] == 30
    assert final["comparisons"] == 24
    assert final["divergences"] == 0
    assert final["passed"] == 759
    assert final["deselected"] == 72
    assert final["xfailed"] == 3
    assert final["failed"] == 0
    assert final["errors"] == 0
    assert final["xpassed"] == 0
    assert final["module_hash_parity"] is True
    assert final["critical_resource_hash_parity"] is True
    assert final["canonical_environment_preserved"] is True

def test_ap007f3_residual_decision_and_mutation_control() -> None:
    data = _manifest()
    residual = data["residual_compatibility_decision"]
    assert residual["legacy_fallback_return_count"] == 6
    assert residual["direct_source_contract_count"] == 4
    assert residual["run_legacy"] == "preserve_published_compatibility"
    assert residual["direct_source_contracts"] == "supersede_test_contract_only"
    assert all(value is False for value in data["mutation_control"].values())

def test_ap007f3_scope_is_exact_and_unstaged() -> None:
    root = _root()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == EXPECTED_HEAD
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip() == EXPECTED_TREE
    assert hashlib.sha256((root / "software/academic_pipeline_mppg/academic_pipeline/runtime.py").read_bytes()).hexdigest() == EXPECTED_RUNTIME_SHA
    raw = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    assert all(item.startswith("?? ") for item in entries), entries
    assert sorted(item[3:] for item in entries) == FINAL_PATHS
    assert subprocess.check_output(["git", "-C", str(root), "diff", "--cached", "--name-only"], text=True) == ""

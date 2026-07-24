from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

EXPECTED_HEAD = "ba43b7d606378501d6faafa62ad8c8a6697665e5"
EXPECTED_RUNTIME_SHA = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
MANIFEST_REL = "docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json"
EXPECTED_ALL_UNTRACKED = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F1_RESIDUAL_FALLBACK_DECISION.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f1_residual_fallback_decision_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py', 'tools/refactor/ap007f1_validate_residual_fallback_decision.py']

def _root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[4]

def _manifest() -> dict:
    return json.loads((_root() / MANIFEST_REL).read_text(encoding="utf-8"))

def test_ap007f1_decision_and_cardinalities() -> None:
    data = _manifest()
    assert data["schema_version"] == "ap007f1-residual-fallback-decision-v1"
    assert data["status"] == "residual_fallback_preserved_no_productive_edit"
    assert data["productive_edit"] is False
    analysis = data["runtime_analysis"]
    assert analysis["ap007f0_ancestor_condition_records"] == 8
    assert analysis["actual_fallback_return_count"] == 6
    assert analysis["cli_injection_count"] == 1
    assert analysis["run_legacy_runner_call_count"] == 1

def test_ap007f1_preserves_runtime_and_legacy_adapter() -> None:
    root = _root()
    assert hashlib.sha256((root / "software/academic_pipeline_mppg/academic_pipeline/runtime.py").read_bytes()).hexdigest() == EXPECTED_RUNTIME_SHA
    data = _manifest()
    assert data["decisions"]["run_legacy"]["decision"] == "preserve_published_compatibility"
    assert data["decisions"]["fallback_returns"]["decision"] == "preserve_published_compatibility"
    assert data["decisions"]["direct_source_execution"]["decision"] == "supersede_test_contract_only"
    assert len(data["decisions"]["direct_source_execution"]["nodeids"]) == 4

def test_ap007f1_dynamic_matrix_is_complete() -> None:
    data = _manifest()
    matrix = data["runtime_analysis"]["dynamic_matrix"]
    names = {item["name"] for item in matrix["route_cases"]}
    assert {
        "institution_exact", "institution_mixed_doctor",
        "doi_exact", "doi_mixed_doctor",
        "doctor_exact", "doctor_preceding_guard",
        "check_exact", "check_preceding_guard",
        "profiles_exact", "profiles_unrelated_guard",
        "empty_default", "unknown_default",
    } <= names
    assert all(item["actual"] == item["expected"] for item in matrix["route_cases"])
    assert len(matrix["route_cases"]) == len(matrix["dispatch_cases"])

def test_ap007f1_scope_is_exact_and_unstaged() -> None:
    root = _root()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == EXPECTED_HEAD
    raw = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    assert all(item.startswith("?? ") for item in entries), entries
    assert sorted(item[3:] for item in entries) == EXPECTED_ALL_UNTRACKED
    assert subprocess.check_output(
        ["git", "-C", str(root), "diff", "--cached", "--name-only"], text=True
    ) == ""

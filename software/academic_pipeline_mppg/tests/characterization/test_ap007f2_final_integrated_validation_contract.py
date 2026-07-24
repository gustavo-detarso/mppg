from __future__ import annotations
import hashlib, json, pathlib, subprocess

EXPECTED_HEAD = "ba43b7d606378501d6faafa62ad8c8a6697665e5"
EXPECTED_RUNTIME_SHA = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
MANIFEST_REL = "docs/refactor/academic-pipeline/AP-007/ap007f2_final_integrated_validation.json"
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F1_RESIDUAL_FALLBACK_DECISION.md', 'docs/refactor/academic-pipeline/AP-007/AP-007F2_FINAL_INTEGRATED_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007f1_residual_fallback_decision.json', 'docs/refactor/academic-pipeline/AP-007/ap007f2_final_integrated_validation.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f1_residual_fallback_decision_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f2_final_integrated_validation_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py', 'tools/refactor/ap007f1_validate_residual_fallback_decision.py', 'tools/refactor/ap007f2_validate_final_integrated_validation.py']

def _root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[4]

def _manifest() -> dict:
    return json.loads((_root() / MANIFEST_REL).read_text(encoding="utf-8"))

def test_ap007f2_integrated_runtime_matrix() -> None:
    data = _manifest()
    assert data["schema_version"] == "ap007f2-final-integrated-validation-v1"
    assert data["status"] == "final_integrated_validation_complete"
    matrix = data["runtime_equivalence"]
    assert matrix["case_count"] == 6
    assert matrix["surface_count"] == 5
    assert matrix["execution_count"] == 30
    assert matrix["comparison_count"] == 24
    assert matrix["non_equivalent_comparison_count"] == 0
    assert len(matrix["surfaces"]) == 5
    assert all(isinstance(item["cwd"], str) for item in matrix["surfaces"])

def test_ap007f2_regression_and_debt_contract() -> None:
    data = _manifest()
    regression = data["regression"]
    assert regression["historical_debt_count"] == 70
    assert regression["phase_local_scope_deselections"] == 2
    assert regression["phase_local_nodeids"] == [
        "tests/characterization/test_ap007e0_distribution_isolation_inventory_contract.py::test_ap007e0_validator_executes_successfully",
        "tests/characterization/test_ap007f0_residual_legacy_audit_contract.py::test_ap007f0_scope_is_exact_and_unstaged",
    ]
    assert regression["productive_return_code"] == 0
    assert regression["failed"] == 0
    assert regression["errors"] == 0
    assert regression["xpassed"] == 0
    assert len(data["direct_source_reproduction"]) == 4
    assert all(item["signature_exact"] for item in data["direct_source_reproduction"])

def test_ap007f2_distribution_and_isolation() -> None:
    data = _manifest()
    assert data["build"]["wheel_count"] == 2
    assert data["build"]["sdist_count"] == 1
    assert data["build"]["network_allowed"] is False
    assert data["build"]["dependency_installation"] is False
    assert data["canonical_environment"]["preserved"] is True
    assert data["module_hash_parity"] is True
    assert data["critical_resource_hash_parity"] is True

def test_ap007f2_scope_is_exact_and_unstaged() -> None:
    root = _root()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == EXPECTED_HEAD
    assert hashlib.sha256((root / "software/academic_pipeline_mppg/academic_pipeline/runtime.py").read_bytes()).hexdigest() == EXPECTED_RUNTIME_SHA
    raw = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    assert all(item.startswith("?? ") for item in entries), entries
    assert sorted(item[3:] for item in entries) == EXPECTED_PATHS
    assert subprocess.check_output(["git", "-C", str(root), "diff", "--cached", "--name-only"], text=True) == ""

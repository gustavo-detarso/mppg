from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

EXPECTED_HEAD = "ba43b7d606378501d6faafa62ad8c8a6697665e5"
EXPECTED_RUNTIME_SHA256 = "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007F0_RESIDUAL_LEGACY_AUDIT.md', 'docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007f0_residual_legacy_audit_contract.py', 'tools/refactor/ap007f0_validate_residual_legacy_audit.py']


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[4]


def _load_inventory() -> dict:
    path = _repo_root() / "docs/refactor/academic-pipeline/AP-007/ap007f0_residual_legacy_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_ap007f0_inventory_schema_and_gate() -> None:
    data = _load_inventory()
    assert data["schema_version"] == "ap007f0-residual-legacy-inventory-v1"
    assert data["status"] == "residual_legacy_audit_complete"
    assert data["gate"] == "residual_legacy_audit_complete"
    assert data["productive_edit"] is False
    assert data["dependency_installation"] is False
    assert data["git_write_operations"] is False
    assert data["materialized_paths"] == EXPECTED_PATHS


def test_ap007f0_exact_debt_and_direct_case_cardinality() -> None:
    data = _load_inventory()
    catalog = data["debt_catalog"]
    assert catalog["count"] == 70
    assert len(catalog["items"]) == 70
    assert len(data["direct_source_cases"]) == 4
    assert all(item["signature_exact"] for item in data["direct_source_cases"])
    assert all(item["execution_returncode"] != 4 for item in data["direct_source_cases"])


def test_ap007f0_runtime_baseline_preserved() -> None:
    root = _repo_root()
    runtime = root / "software/academic_pipeline_mppg/academic_pipeline/runtime.py"
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == EXPECTED_RUNTIME_SHA256
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert head == EXPECTED_HEAD


def test_ap007f0_scope_is_exact_and_unstaged() -> None:
    root = _repo_root()
    raw = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    entries = [item for item in raw.decode("utf-8", "surrogateescape").split("\0") if item]
    assert all(item.startswith("?? ") for item in entries), entries
    assert sorted(item[3:] for item in entries) == sorted(EXPECTED_PATHS)
    staged = subprocess.check_output(["git", "-C", str(root), "diff", "--cached", "--name-only"], text=True)
    assert staged == ""

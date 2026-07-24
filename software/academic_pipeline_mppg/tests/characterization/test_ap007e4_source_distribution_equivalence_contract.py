from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e4_source_distribution_equivalence.json').read_text(encoding="utf-8"))

def test_ap007e4_scope_and_baseline_contract() -> None:
    assert MANIFEST["schema"] == "ap007e4_source_distribution_equivalence.v1"
    assert MANIFEST["phase"] == "AP-007E.4"
    assert MANIFEST["baseline"]["commit"] == '766956710435f1c338d2e0332d24e55106b981b7'
    assert MANIFEST["baseline"]["tree"] == '1d673e7c324b74f1fef033578aa995e836da1014'
    assert len(MANIFEST["scope"]["materialized_paths"]) == 4
    assert MANIFEST["scope"]["dependency_installation_executed"] is False
    assert MANIFEST["scope"]["network_allowed"] is False
    assert MANIFEST["scope"]["git_write_executed"] is False

def test_ap007e4_runtime_matrix_equivalence_contract() -> None:
    summary = MANIFEST["summary"]
    assert summary["case_count"] == 6
    assert summary["surface_count"] == 5
    assert summary["execution_count"] == 30
    assert summary["comparison_count"] == 24
    assert summary["equivalent_comparison_count"] == 24
    assert summary["non_equivalent_comparison_count"] == 0
    assert summary["all_expected_return_codes"] is True
    assert summary["all_no_timeout"] is True
    assert all(item["equivalent"] is True for item in MANIFEST["runtime_matrix"]["comparisons"])

def test_ap007e4_isolation_and_origin_contract() -> None:
    summary = MANIFEST["summary"]
    assert summary["pythonpath_removed"] is True
    assert summary["installed_source_leak_count"] == 0
    assert summary["canonical_environment_preserved"] is True
    assert summary["dependency_installation_executed"] is False
    surfaces = {item["id"] for item in MANIFEST["runtime_matrix"]["surfaces"]}
    assert surfaces == {"source_python_m","wheel_python_m","wheel_console","sdist_python_m","sdist_console"}

def test_ap007e4_resource_and_module_parity_contract() -> None:
    assert MANIFEST["content_parity"]["critical_resource_hash_parity"] is True
    assert MANIFEST["content_parity"]["module_hash_parity"] is True
    assert len(MANIFEST["content_parity"]["critical_resources"]) == 8
    assert MANIFEST["summary"]["blocking_finding_count"] == 0

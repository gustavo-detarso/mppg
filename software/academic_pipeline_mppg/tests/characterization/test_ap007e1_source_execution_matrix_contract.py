from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DATA = json.loads((REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e1_source_execution_matrix.json').read_text(encoding="utf-8"))


def test_ap007e1_source_matrix_core_equivalence() -> None:
    assert DATA["schema"] == "ap007e1_source_execution_matrix.v1"
    assert DATA["status"] in ('matrix_approved', 'matrix_approved_with_classified_historical_debt')
    assert len(DATA["results"]) == 36
    assert all(item["equivalent"] for item in DATA["core_equivalence"])
    assert all(item["expected_return_code_ok"] for item in DATA["core_equivalence"])
    assert all(item["no_timeout"] for item in DATA["core_equivalence"])


def test_ap007e1_process_isolation_contract() -> None:
    assert DATA["isolation"]["PYTHONPATH_REMOVED"] is True
    assert DATA["isolation"]["fresh_subprocess_per_case"] is True
    assert DATA["process_contracts"]
    assert all(item["argv_preserved"] for item in DATA["process_contracts"])
    assert all(item["sys_path_preserved"] for item in DATA["process_contracts"])
    assert all(item["cwd_preserved"] for item in DATA["process_contracts"])
    assert all(item["module_in_source_tree"] for item in DATA["process_contracts"])
    assert all(not item["legacy_import_attempts"] for item in DATA["process_contracts"])


def test_ap007e1_historical_compatibility_and_scope() -> None:
    assert len(DATA["historical_compatibility_tests"]) == 4
    assert all(item["status"] in ('passed', 'classified_failure') for item in DATA["historical_compatibility_tests"])
    assert all(item["blocking"] is False for item in DATA["historical_compatibility_tests"])
    assert all(item["execution_mode"] == "individual_exact_nodeid" for item in DATA["historical_compatibility_tests"])
    assert DATA["failures"] == []
    assert DATA["scope"]["build_executed"] is False
    assert DATA["scope"]["installation_executed"] is False
    assert DATA["scope"]["git_write_executed"] is False

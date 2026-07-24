#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JSON_PATH = REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e1_source_execution_matrix.json'
EXPECTED_SCHEMA = "ap007e1_source_execution_matrix.v1"
EXPECTED_PATHS = {'docs/refactor/academic-pipeline/AP-007/AP-007E1_SOURCE_EXECUTION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-007/ap007e1_source_execution_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e1_source_execution_matrix_contract.py', 'tools/refactor/ap007e1_validate_source_execution_matrix.py'}
EXPECTED_CASES = {"help", "list_institutions", "list_profiles", "check_config_missing_config", "institution_compliance_missing_config", "doi_manifest_missing_input"}
EXPECTED_CORE = {"direct_import_call_source_root", "python_minus_m_source_root", "official_target_source_root"}


def main() -> int:
    failures = []
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    checks = [
        (data.get("schema") == EXPECTED_SCHEMA, "schema"),
        (data.get("phase") == "AP-007E.1", "phase"),
        (data.get("status") in ('matrix_approved', 'matrix_approved_with_classified_historical_debt'), "status"),
        (data.get("baseline", {}).get("commit") == '766956710435f1c338d2e0332d24e55106b981b7', "commit"),
        (data.get("baseline", {}).get("tree") == '1d673e7c324b74f1fef033578aa995e836da1014', "tree"),
        (data.get("baseline", {}).get("runtime_sha256") == 'b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c', "runtime hash"),
        (data.get("entrypoint", {}).get("target") == "academic_pipeline.cli:main", "entrypoint"),
        (data.get("isolation", {}).get("PYTHONPATH_REMOVED") is True, "PYTHONPATH"),
        (data.get("isolation", {}).get("fresh_subprocess_per_case") is True, "fresh subprocess"),
        (set(item["id"] for item in data.get("commands", [])) == EXPECTED_CASES, "cases"),
        (len(data.get("surfaces", [])) == 6, "surfaces"),
        (len(data.get("results", [])) == 36, "results"),
        (all(item.get("equivalent") for item in data.get("core_equivalence", [])), "core equivalence"),
        (all(item.get("expected_return_code_ok") for item in data.get("core_equivalence", [])), "return codes"),
        (all(item.get("no_timeout") for item in data.get("core_equivalence", [])), "timeouts"),
        (all(item.get("argv_preserved") and item.get("sys_path_preserved") and item.get("cwd_preserved") for item in data.get("process_contracts", [])), "process state"),
        (all(item.get("module_in_source_tree") for item in data.get("process_contracts", [])), "module origin"),
        (all(not item.get("legacy_import_attempts") for item in data.get("process_contracts", [])), "legacy import"),
        (len(data.get("historical_compatibility_tests", [])) == 4, "historical count"),
        (all(item.get("status") in ('passed', 'classified_failure') for item in data.get("historical_compatibility_tests", [])), "historical classifications"),
        (all(item.get("blocking") is False for item in data.get("historical_compatibility_tests", [])), "historical non-blocking"),
        (all(item.get("execution_mode") == "individual_exact_nodeid" for item in data.get("historical_compatibility_tests", [])), "historical individual execution"),
        (data.get("failures") == [], "failures"),
        (set(data.get("scope", {}).get("materialized_paths", [])) == EXPECTED_PATHS, "scope"),
        (data.get("scope", {}).get("build_executed") is False, "build"),
        (data.get("scope", {}).get("installation_executed") is False, "install"),
        (data.get("scope", {}).get("git_write_executed") is False, "git write"),
    ]
    for ok, label in checks:
        if not ok:
            failures.append(label)
    print("AP007E1_VALIDATION_FAILURES=" + str(len(failures)))
    for item in failures:
        print("[FAIL] " + item)
    if failures:
        return 1
    print(data["gate"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

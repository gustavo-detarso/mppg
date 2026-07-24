#!/usr/bin/env python3
from __future__ import annotations
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e4_source_distribution_equivalence.json'
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007E0_DISTRIBUTION_ISOLATION_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-007/AP-007E1_SOURCE_EXECUTION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-007/AP-007E2_CONTROLLED_BUILD_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-007/AP-007E3_ISOLATED_INSTALLATION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-007/AP-007E4_SOURCE_DISTRIBUTION_EQUIVALENCE.md', 'docs/refactor/academic-pipeline/AP-007/ap007e0_distribution_isolation_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007e1_source_execution_matrix.json', 'docs/refactor/academic-pipeline/AP-007/ap007e2_controlled_build_inventory.json', 'docs/refactor/academic-pipeline/AP-007/ap007e3_isolated_installation_matrix.json', 'docs/refactor/academic-pipeline/AP-007/ap007e4_source_distribution_equivalence.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e0_distribution_isolation_inventory_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e1_source_execution_matrix_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e2_controlled_build_inventory_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e3_isolated_installation_matrix_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e4_source_distribution_equivalence_contract.py', 'tools/refactor/ap007e0_validate_distribution_isolation_inventory.py', 'tools/refactor/ap007e1_validate_source_execution_matrix.py', 'tools/refactor/ap007e2_validate_controlled_build_inventory.py', 'tools/refactor/ap007e3_validate_isolated_installation_matrix.py', 'tools/refactor/ap007e4_validate_source_distribution_equivalence.py']
EXPECTED_COMMIT = '766956710435f1c338d2e0332d24e55106b981b7'
EXPECTED_TREE = '1d673e7c324b74f1fef033578aa995e836da1014'

failures = 0
def check(condition: bool, message: str) -> None:
    global failures
    if condition:
        print("[OK] " + message)
    else:
        print("[FAIL] " + message)
        failures += 1

data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
check(data.get("schema") == "ap007e4_source_distribution_equivalence.v1", "schema")
check(data.get("phase") == "AP-007E.4", "phase")
check(data.get("status") in {"equivalence_approved", "equivalence_approved_with_classified_findings"}, "status")
check(data.get("baseline", {}).get("commit") == EXPECTED_COMMIT, "baseline commit")
check(data.get("baseline", {}).get("tree") == EXPECTED_TREE, "baseline tree")
summary = data.get("summary", {})
check(summary.get("case_count") == 6, "six cases")
check(summary.get("surface_count") == 5, "five surfaces")
check(summary.get("execution_count") == 30, "thirty executions")
check(summary.get("comparison_count") == 24, "twenty-four comparisons")
check(summary.get("non_equivalent_comparison_count") == 0, "zero divergences")
check(summary.get("equivalent_comparison_count") == 24, "all comparisons equivalent")
check(summary.get("all_expected_return_codes") is True, "expected return codes")
check(summary.get("all_no_timeout") is True, "no timeouts")
check(summary.get("pythonpath_removed") is True, "PYTHONPATH removed")
check(summary.get("installed_source_leak_count") == 0, "no source leaks")
check(summary.get("critical_resource_hash_parity") is True, "resource parity")
check(summary.get("module_hash_parity") is True, "module parity")
check(summary.get("canonical_environment_preserved") is True, "canonical environment preserved")
check(summary.get("dependency_installation_executed") is False, "no dependency installation")
scope = data.get("scope", {})
check(scope.get("network_allowed") is False, "network disabled")
check(scope.get("git_write_executed") is False, "no git write")
check(scope.get("productive_modules_modified") == [], "no productive module edits")
check(len(scope.get("materialized_paths", [])) == 4, "four materialized paths")
comparisons = data.get("runtime_matrix", {}).get("comparisons", [])
check(len(comparisons) == 24 and all(item.get("equivalent") is True for item in comparisons), "comparison matrix")
executions = data.get("runtime_matrix", {}).get("executions", [])
check(len(executions) == 30 and all(item.get("no_timeout") is True for item in executions), "execution matrix")

status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
actual = sorted(line for line in status.stdout.splitlines() if line)
expected = sorted("?? " + path for path in EXPECTED_PATHS)
check(status.returncode == 0, "git status")
check(actual == expected, "exact twenty untracked paths")
print(f"AP007E4_VALIDATION_FAILURES={failures}")
if failures:
    sys.exit(1)
print("[GATE] AP-007E.4: EQUIVALÊNCIA FONTE/WHEEL/SDIST VALIDADA EM 30 EXECUÇÕES E 24 COMPARAÇÕES; ZERO DIVERGÊNCIAS, SEM REUSO DA FONTE, REDE, DEPENDÊNCIAS ADICIONAIS OU GIT WRITE.")

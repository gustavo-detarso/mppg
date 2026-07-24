#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
from pathlib import Path

REPO = Path('/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline-ap005')
MANIFEST = REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e5_closure_manifest.json'
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007E0_DISTRIBUTION_ISOLATION_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-007/ap007e0_distribution_isolation_inventory.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e0_distribution_isolation_inventory_contract.py', 'tools/refactor/ap007e0_validate_distribution_isolation_inventory.py', 'docs/refactor/academic-pipeline/AP-007/AP-007E1_SOURCE_EXECUTION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-007/ap007e1_source_execution_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e1_source_execution_matrix_contract.py', 'tools/refactor/ap007e1_validate_source_execution_matrix.py', 'docs/refactor/academic-pipeline/AP-007/AP-007E2_CONTROLLED_BUILD_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-007/ap007e2_controlled_build_inventory.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e2_controlled_build_inventory_contract.py', 'tools/refactor/ap007e2_validate_controlled_build_inventory.py', 'docs/refactor/academic-pipeline/AP-007/AP-007E3_ISOLATED_INSTALLATION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-007/ap007e3_isolated_installation_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e3_isolated_installation_matrix_contract.py', 'tools/refactor/ap007e3_validate_isolated_installation_matrix.py', 'docs/refactor/academic-pipeline/AP-007/AP-007E4_SOURCE_DISTRIBUTION_EQUIVALENCE.md', 'docs/refactor/academic-pipeline/AP-007/ap007e4_source_distribution_equivalence.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e4_source_distribution_equivalence_contract.py', 'tools/refactor/ap007e4_validate_source_distribution_equivalence.py', 'docs/refactor/academic-pipeline/AP-007/AP-007E5_CLOSURE_REPORT.md', 'docs/refactor/academic-pipeline/AP-007/ap007e5_closure_manifest.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e5_closure_contract.py', 'tools/refactor/ap007e5_validate_closure.py']
failures = 0

def check(condition: bool, label: str) -> None:
    global failures
    if condition:
        print(f"[OK] {label}")
    else:
        print(f"[FAIL] {label}")
        failures += 1

data = json.loads(MANIFEST.read_text(encoding="utf-8"))
check(data.get("schema") == "ap007e5_closure.v1", "schema")
check(data.get("phase") == "AP-007E.5", "phase")
check(data.get("status") == "ready_for_isolated_commit_decision", "status")
check(data.get("baseline", {}).get("commit") == '766956710435f1c338d2e0332d24e55106b981b7', "baseline commit")
check(data.get("baseline", {}).get("tree") == '1d673e7c324b74f1fef033578aa995e836da1014', "baseline tree")
check(data.get("scope", {}).get("candidate_path_count") == 24, "candidate path count")
check(data.get("scope", {}).get("candidate_paths") == EXPECTED_PATHS, "candidate paths")
check(data.get("scope", {}).get("productive_modules_modified") == [], "no productive module edits")
check(data.get("scope", {}).get("git_write_executed") is False, "no git write")
check(data.get("canonical_environment", {}).get("preserved") is True, "canonical environment preserved")
check(data.get("stable_contracts", {}).get("count") == 16, "sixteen stable contracts")
check(data.get("stable_contracts", {}).get("return_code") == 0, "stable contracts passed")
check(data.get("phase_local_baselines", {}).get("replayed") is False, "phase-local baselines separated")
census = data.get("integrated_regression_census", {})
check(census.get("return_code") == 1, "integrated census expected return code")
check(census.get("status") == "exact_historical_phase_debt_confirmed", "integrated census status")
check(census.get("failure_count") == 70, "exact seventy classified failures")
check(census.get("failed_nodeids") == census.get("expected_failed_nodeids"), "exact integrated failure node IDs")
check(census.get("classification_counts") == census.get("expected_classification_counts"), "exact integrated classification counts")
check(census.get("missing_required_output_markers") == [], "integrated failure signatures preserved")
check(census.get("blocking") is False, "integrated debt nonblocking")
regression = data.get("productive_regression", {})
check(regression.get("return_code") == 0, "current productive suite return code")
check(regression.get("status") == "passed_after_exact_historical_phase_debt_deselection", "current productive suite status")
check(regression.get("summary", {}).get("failed") == 0, "productive suite zero failures")
check(regression.get("summary", {}).get("xpassed") == 0, "productive suite zero xpasses")
check(regression.get("xfail_nodeids") == regression.get("expected_xfail_nodeids"), "exact frozen xfails")
historical = data.get("historical_compatibility", {})
check(len(historical.get("tests", [])) == 4, "four historical tests")
check(historical.get("blocking_failure_count") == 0, "no blocking historical failure")
check(all(not item.get("blocking") for item in historical.get("tests", [])), "all historical classifications nonblocking")
evidence = data.get("source_distribution_evidence", {})
check(evidence.get("runtime_execution_count") == 30, "thirty runtime executions")
check(evidence.get("runtime_comparison_count") == 24, "twenty-four runtime comparisons")
check(evidence.get("non_equivalent_comparison_count") == 0, "zero runtime divergences")
check(evidence.get("critical_resource_hash_parity") is True, "critical resource parity")
check(evidence.get("module_hash_parity") is True, "module parity")
check(len(data.get("approved_corrections", [])) >= 10, "approved corrections consolidated")
decision = data.get("commit_decision", {})
check(decision.get("ready") is True, "commit decision ready")
check(decision.get("requires_explicit_user_authorization") is True, "explicit authorization required")
check(decision.get("staging_performed") is False, "no staging")
check(decision.get("commit_performed") is False, "no commit")
summary = data.get("summary", {})
check(summary.get("ready_for_isolated_commit_decision") is True, "summary ready")
check(summary.get("blocking_finding_count") == 0, "zero blocking findings")
status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
actual = sorted(line for line in status.stdout.splitlines() if line)
expected = sorted(f"?? {path}" for path in EXPECTED_PATHS)
check(status.returncode == 0, "git status")
check(actual == expected, "exact twenty-four untracked paths")
print(f"AP007E5_VALIDATION_FAILURES={failures}")
raise SystemExit(1 if failures else 0)

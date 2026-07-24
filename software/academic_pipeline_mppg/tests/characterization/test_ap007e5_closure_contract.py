from __future__ import annotations
import json
from pathlib import Path

REPO = Path('/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline-ap005')
MANIFEST = json.loads((REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e5_closure_manifest.json').read_text(encoding="utf-8"))

def test_ap007e5_scope_and_baseline_contract() -> None:
    assert MANIFEST["schema"] == "ap007e5_closure.v1"
    assert MANIFEST["phase"] == "AP-007E.5"
    assert MANIFEST["status"] == "ready_for_isolated_commit_decision"
    assert MANIFEST["baseline"]["commit"] == '766956710435f1c338d2e0332d24e55106b981b7'
    assert MANIFEST["baseline"]["tree"] == '1d673e7c324b74f1fef033578aa995e836da1014'
    assert MANIFEST["scope"]["candidate_path_count"] == 24
    assert len(MANIFEST["scope"]["candidate_paths"]) == 24
    assert MANIFEST["scope"]["productive_modules_modified"] == []
    assert MANIFEST["scope"]["git_write_executed"] is False
    assert MANIFEST["canonical_environment"]["preserved"] is True

def test_ap007e5_regression_and_phase_local_separation_contract() -> None:
    stable = MANIFEST["stable_contracts"]
    assert stable["count"] == 16
    assert stable["return_code"] == 0
    assert stable["no_timeout"] is True
    assert MANIFEST["phase_local_baselines"]["replayed"] is False
    census = MANIFEST["integrated_regression_census"]
    assert census["return_code"] == 1
    assert census["status"] == "exact_historical_phase_debt_confirmed"
    assert census["failure_count"] == 70
    assert census["failed_nodeids"] == census["expected_failed_nodeids"]
    assert census["classification_counts"] == census["expected_classification_counts"]
    assert census["missing_required_output_markers"] == []
    assert census["blocking"] is False
    regression = MANIFEST["productive_regression"]
    assert regression["return_code"] == 0
    assert regression["status"] == "passed_after_exact_historical_phase_debt_deselection"
    assert regression["summary"]["failed"] == 0
    assert regression["summary"]["errors"] == 0
    assert regression["summary"]["xpassed"] == 0
    assert regression["xfail_nodeids"] == regression["expected_xfail_nodeids"]

def test_ap007e5_historical_and_distribution_evidence_contract() -> None:
    historical = MANIFEST["historical_compatibility"]
    assert historical["execution_mode"] == "individual_exact_nodeid"
    assert len(historical["tests"]) == 4
    assert historical["blocking_failure_count"] == 0
    assert all(item["classification"] in {"contract_currently_satisfied", "legacy_direct_source_bridge_absent"} for item in historical["tests"])
    evidence = MANIFEST["source_distribution_evidence"]
    assert evidence["runtime_execution_count"] == 30
    assert evidence["runtime_comparison_count"] == 24
    assert evidence["non_equivalent_comparison_count"] == 0
    assert evidence["critical_resource_hash_parity"] is True
    assert evidence["module_hash_parity"] is True

def test_ap007e5_commit_decision_and_corrections_contract() -> None:
    assert len(MANIFEST["approved_corrections"]) >= 10
    decision = MANIFEST["commit_decision"]
    assert decision["ready"] is True
    assert decision["decision"] == "ready_for_isolated_commit_decision"
    assert decision["requires_explicit_user_authorization"] is True
    assert decision["candidate_path_count"] == 24
    assert decision["staging_performed"] is False
    assert decision["commit_performed"] is False
    assert decision["tag_performed"] is False
    assert decision["push_performed"] is False
    assert MANIFEST["summary"]["blocking_finding_count"] == 0

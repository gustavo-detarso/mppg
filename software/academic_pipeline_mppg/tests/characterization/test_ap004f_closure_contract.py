"""Contrato de encerramento da AP-004F.

Gerado por tools/refactor/ap004f_generate_closure.py.
Não editar manualmente: regenere após repetir o gate da AP-004F.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_SCHEMA = 'ap004f.closure-manifest.v1'
EXPECTED_HEAD = 'b5f924ae2b55c961f251a8d65f3405eb3cea35b8'
EXPECTED_FINGERPRINT = '924865e01241083a03ddfb5d152a3eaa4972ecb2c514258a0ff99fdedd0684c0'
EXPECTED_PHASES = [('AP-004A', '6de61fc9741035187836460d97da6d672708998a'), ('AP-004B', 'aa9829f09a5c1b9e69c634637c311b03f360b07e'), ('AP-004C', '81293d79e86da8b4d0407b483fc3dedaf27768cb'), ('AP-004D', '389f0ae526d12327a58ce23937225cf05b032566'), ('AP-004E', 'b5f924ae2b55c961f251a8d65f3405eb3cea35b8')]
EXPECTED_XFAILS = ['app_bundle/tests/test_article_workflow_characterization.py::test_refresh_from_files_should_keep_downstream_stages_blocked_after_first_failure', 'app_bundle/tests/test_canonical_docx_characterization.py::test_extract_resumos_should_separate_inline_keywords_from_heading_abstract', 'app_bundle/tests/test_rc10_configuration_characterization.py::test_reference_strip_should_remove_parenthetical_citations']
EXPECTED_INTEGRATION_TARGET = 'origin/refactor/academic-pipeline'
EXPECTED_INTEGRATION_MODE = 'fast-forward'
EXPECTED_INTEGRATION_READY = True


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        marker = parent / 'docs/refactor/academic-pipeline/AP-004/ap004f_closure_manifest.json'
        if marker.is_file():
            return parent
    raise AssertionError("não foi possível localizar a raiz do repositório")


def _load_manifest() -> dict:
    path = _repo_root() / 'docs/refactor/academic-pipeline/AP-004/ap004f_closure_manifest.json'
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_basis(data: dict) -> str:
    basis = {
        key: value
        for key, value in data.items()
        if key not in {"generated_at", "contract_fingerprint"}
    }
    encoded = json.dumps(
        basis,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_ap004f_manifest_contract_is_frozen() -> None:
    data = _load_manifest()
    assert data["schema_version"] == EXPECTED_SCHEMA
    assert data["baseline"]["head"] == EXPECTED_HEAD
    assert data["baseline"]["remote_head"] == EXPECTED_HEAD
    assert data["baseline"]["divergence"] == [0, 0]
    assert [(item["phase"], item["commit"]) for item in data["phases"]] == EXPECTED_PHASES
    assert data["contract_fingerprint"] == EXPECTED_FINGERPRINT
    assert _fingerprint_basis(data) == EXPECTED_FINGERPRINT


def test_ap004f_final_validation_contract() -> None:
    data = _load_manifest()
    contracts = data["validation"]["contract_tests"]
    suite = data["validation"]["canonical_suite_before_ap004f_contract"]
    assert contracts["passed"] == 7
    assert contracts["failed"] == 0
    assert contracts["errors"] == 0
    assert suite["passed"] == 489
    assert suite["xfailed"] == 3
    assert suite["xpassed"] == 0
    assert suite["failed"] == 0
    assert suite["errors"] == 0
    assert data["validation"]["expected_xfails"] == EXPECTED_XFAILS


def test_ap004f_closes_ap004_without_productive_applicator() -> None:
    data = _load_manifest()
    closure = data["closure_decision"]
    ap004e = data["ap004e_inventory_contract"]
    assert closure["ap004_status"] == "technically_closed"
    assert closure["productive_applicator_required"] is False
    assert closure["productive_changes_in_ap004f"] is False
    assert closure["remaining_manual_inventory_decisions"] == 0
    assert closure["safe_removal_candidates"] == 0
    assert closure["residual_known_defects"] == 3
    assert ap004e["summary"]["item_count"] == 64
    assert ap004e["summary"]["removal_candidates"] == 0
    assert ap004e["summary"]["blocked_items"] == 0


def test_ap004f_integration_remains_explicitly_blocked() -> None:
    data = _load_manifest()
    integration = data["integration_assessment"]
    gate = data["gate"]
    assert integration["target_ref"] == EXPECTED_INTEGRATION_TARGET
    assert integration["integration_mode"] == EXPECTED_INTEGRATION_MODE
    assert integration["technically_ready"] is EXPECTED_INTEGRATION_READY
    assert integration["integration_executed"] is False
    assert gate["productive_changes_allowed"] is False
    assert gate["merge_allowed"] is False
    assert gate["rebase_allowed"] is False
    assert gate["cherry_pick_allowed"] is False
    assert gate["commit_allowed_before_review"] is False
    assert gate["push_allowed_before_review"] is False
    assert gate["integration_allowed_before_explicit_approval"] is False
    assert gate["integration_executed"] is False

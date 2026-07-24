from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

EXPECTED_HEAD = 'ab066e68947ac5f33f1c12c9a7db5086d0f93790'
EXPECTED_TREE = 'e8bebd0b88a719a765ba6cc78150b01e024de9dd'
EXPECTED_RUNTIME_SHA256 = '7a2fd63ae060c74fe3f06e2eaf7457f176dc5059c0ac7aa95fc23805e810b1e6'
EXPECTED_CANDIDATES = ['--check-institution-compliance', '--list-profiles', '--make-doi-manifest']
EXPECTED_NATIVE = ['--check-config', '--doctor', '--explain-profile', '--help', '--list-institutions', '--list-layouts', '--list-toml-profiles']
DEFAULT_REL = 'docs/refactor/academic-pipeline/AP-007/ap007d1_first_operational_wave_decision.json'


def _path() -> Path:
    override = os.environ.get("AP007D1_DECISION_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / DEFAULT_REL


def _payload() -> dict:
    return json.loads(_path().read_text(encoding="utf-8"))


def test_candidate_set_and_scope() -> None:
    payload = _payload()
    assert payload["schema"] == 'ap007d1-first-operational-wave-decision/v2'
    assert [item["flag"] for item in payload["candidates_by_input_order"]] == EXPECTED_CANDIDATES
    assert payload["observed_state"]["head"] == EXPECTED_HEAD
    assert payload["observed_state"]["tree"] == EXPECTED_TREE
    assert payload["observed_state"]["runtime_sha256"] == EXPECTED_RUNTIME_SHA256
    assert payload["scope"]["production_files_modified"] == []
    assert payload["scope"]["public_routes_changed"] == []
    assert payload["scope"]["git_mutations"] == []


def test_selection_is_single_and_fail_closed() -> None:
    payload = _payload()
    selected = payload["decision"]["selected_flags"]
    assert len(selected) <= 1
    by_flag = {item["flag"]: item for item in payload["candidates"]}
    assert set(selected).isdisjoint(EXPECTED_NATIVE)
    for item in by_flag.values():
        assert item["resolved_seed_handlers"]
        assert "handler_not_resolved" not in item["hard_exclusions"]
        assert all(record["method"] != "unresolved" for record in item["seed_resolution_evidence"])
    for flag in selected:
        item = by_flag[flag]
        assert item["first_wave_eligible"] is True
        assert item["hard_exclusions"] == []
        effects = item["transitive_effects"]
        assert not effects["write_calls"]
        assert not effects["destructive_calls"]
        assert not effects["network_markers"]
        assert not effects["ui_markers"]
        assert not effects["subprocess_markers"]
        assert not effects["credential_markers"]
        assert not effects["dynamic_markers"]
        assert not effects["has_input"]
        assert not effects["has_chdir"]
        assert not effects["has_global"]
        assert not effects["mutates_sys_argv"]
        assert not effects["mutates_sys_path"]
        assert item["tests"]
        assert item["historical_stages"]


def test_generating_command_is_not_selected() -> None:
    payload = _payload()
    selected = set(payload["decision"]["selected_flags"])
    assert "--make-doi-manifest" not in selected
    item = next(record for record in payload["candidates"] if record["flag"] == "--make-doi-manifest")
    assert "mutating_or_generating_command_semantics" in item["hard_exclusions"]


def test_integrity() -> None:
    payload = _payload()
    clone = dict(payload)
    clone["integrity"] = {"decision_payload_sha256": ""}
    observed = hashlib.sha256(json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert payload["integrity"]["decision_payload_sha256"] == observed


def main() -> int:
    test_candidate_set_and_scope()
    test_selection_is_single_and_fail_closed()
    test_generating_command_is_not_selected()
    test_integrity()
    print("[OK] test_ap007d1_first_operational_wave_contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

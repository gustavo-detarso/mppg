from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

EXPECTED_HEAD = 'ab066e68947ac5f33f1c12c9a7db5086d0f93790'
EXPECTED_TREE = 'e8bebd0b88a719a765ba6cc78150b01e024de9dd'
EXPECTED_RUNTIME_SHA256 = '7a2fd63ae060c74fe3f06e2eaf7457f176dc5059c0ac7aa95fc23805e810b1e6'
EXPECTED_NATIVE = {'--help': 'native_first_wave', '--list-toml-profiles': 'native_first_wave', '--list-institutions': 'native_first_wave', '--list-layouts': 'native_first_wave', '--explain-profile': 'native_first_wave', '--doctor': 'native_doctor', '--check-config': 'native_check_config'}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _inventory_path() -> Path:
    override = os.environ.get("AP007D0_INVENTORY_PATH")
    if override:
        return Path(override)
    return _repo_root() / "docs/refactor/academic-pipeline/AP-007/ap007d0_operational_command_inventory.json"


def _payload() -> dict:
    return json.loads(_inventory_path().read_text(encoding="utf-8"))


def test_native_routes_are_not_legacy_candidates() -> None:
    payload = _payload()
    assert payload["runtime_dispatch"]["expected_native_routes"] == EXPECTED_NATIVE
    assert payload["runtime_dispatch"]["observed_native_routes"] == EXPECTED_NATIVE
    legacy = {item["flag"] for item in payload["legacy_commands"]}
    recommended = set(payload["first_wave_recommendation"]["flags"])
    assert legacy.isdisjoint(EXPECTED_NATIVE)
    assert recommended.isdisjoint(EXPECTED_NATIVE)


def test_recommendation_is_conservative() -> None:
    payload = _payload()
    by_flag = {item["flag"]: item for item in payload["legacy_commands"]}
    for flag in payload["first_wave_recommendation"]["flags"]:
        command = by_flag[flag]
        assert command["first_wave_eligible"] is True
        assert command["hard_exclusions"] == []
        assert command["score_total"] >= 24
        assert command["tests"]
        assert command["historical_stages"]
        effects = command["effects"]
        assert not effects["write_calls"]
        assert not effects["destructive_calls"]
        assert not effects["network_markers"]
        assert not effects["ui_markers"]
        assert not effects["subprocess_markers"]
        assert not effects["credential_markers"]
        assert not effects["volatile_markers"]
        assert not effects["has_input"]
        assert not effects["has_chdir"]
        assert not effects["has_global"]
        assert not effects["mutates_sys_argv"]
        assert not effects["mutates_sys_path"]


def test_scope_and_integrity_contract() -> None:
    payload = _payload()
    assert payload["schema"] == "ap007d0-operational-command-inventory/v2"
    assert payload["observed_state"]["head"] == EXPECTED_HEAD
    assert payload["observed_state"]["tree"] == EXPECTED_TREE
    assert payload["observed_state"]["runtime_sha256"] == EXPECTED_RUNTIME_SHA256
    assert payload["scope"]["production_files_modified"] == []
    assert payload["scope"]["git_mutations"] == []
    clone = dict(payload)
    clone["integrity"] = {"inventory_payload_sha256": ""}
    observed = hashlib.sha256(json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert payload["integrity"]["inventory_payload_sha256"] == observed


def main() -> int:
    test_native_routes_are_not_legacy_candidates()
    test_recommendation_is_conservative()
    test_scope_and_integrity_contract()
    print("[OK] test_ap007d0_operational_inventory_contract v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

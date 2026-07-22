from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VALIDATOR = REPO / "tools/refactor/ap007a_validate_runtime_legacy_inventory.py"
SPEC = importlib.util.spec_from_file_location("ap007a_validator", VALIDATOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ap007a_materialized_contract_is_valid() -> None:
    result = MODULE.validate(REPO)
    assert result["ok"] is True
    assert result["phase"] == "AP-007A"


def test_ap007a_preserves_raw_inventory_and_corrects_semantics() -> None:
    result = MODULE.validate(REPO)
    assert result["raw_references"] == 7506
    assert result["canonical_lines"] == 47
    assert result["operational_lines"] == 20


def test_ap007a_freezes_seven_productive_surfaces_and_public_contract() -> None:
    result = MODULE.validate(REPO)
    assert result["productive_surfaces"] == 7
    assert result["public_options"] == 63


def test_ap007a_materialization_does_not_claim_runtime_changes() -> None:
    result = MODULE.validate(REPO)
    assert result["runtime_productive_files_modified"] == []

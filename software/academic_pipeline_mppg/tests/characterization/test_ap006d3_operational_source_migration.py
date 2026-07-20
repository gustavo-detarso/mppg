from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = (
    REPO_ROOT
    / "tools/refactor/"
    "ap006d3_validate_operational_source_migration.py"
)

spec = importlib.util.spec_from_file_location(
    "ap006d3_validator",
    VALIDATOR_PATH,
)
assert spec is not None and spec.loader is not None

validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_ap006d3_validator_accepts_materialized_tree() -> None:
    assert validator.validate(REPO_ROOT)["status"] == "valid"


@pytest.mark.parametrize(
    ("relative", "old_count", "new_count"),
    [
        (validator.README, 0, 3),
        (validator.SETUP, 1, 1),
        (validator.UPDATER, 0, 1),
        (validator.TOML, 0, 7),
    ],
)
def test_ap006d3_reference_counts(
    relative: Path,
    old_count: int,
    new_count: int,
) -> None:
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert text.count(validator.OLD_NAME) == old_count
    assert text.count(validator.NEW_NAME) == new_count


def test_ap006d3_setup_preserves_one_legacy_reference() -> None:
    text = (REPO_ROOT / validator.SETUP).read_text(encoding="utf-8")
    assert sum(
        validator.OLD_NAME in line
        for line in text.splitlines()
    ) == 1


def test_ap006d3_external_regeneration_is_deferred() -> None:
    contract = json.loads(
        (REPO_ROOT / validator.CONTRACT).read_text(encoding="utf-8")
    )
    regeneration = contract["external_regeneration"]
    assert regeneration["phase"] == "AP-006D.4"
    assert len(regeneration["entries"]) == 3
    assert regeneration["separate_commit_required"] is True


def test_ap006d3_bridge_remains_relative() -> None:
    bridge = REPO_ROOT / validator.OLD_REL
    assert bridge.is_symlink()
    assert bridge.readlink() == Path(validator.NEW_NAME)

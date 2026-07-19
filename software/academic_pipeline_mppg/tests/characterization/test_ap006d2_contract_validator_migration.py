from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = (
    REPO_ROOT
    / "tools/refactor/ap006d2_validate_contract_validator_migration.py"
)

spec = importlib.util.spec_from_file_location("ap006d2_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None

validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_ap006d2_validator_accepts_materialized_tree() -> None:
    result = validator.validate(REPO_ROOT)
    assert result["status"] == "valid"


@pytest.mark.parametrize(
    ("relative", "old_count", "new_count"),
    [
        ("tools/refactor/ap004e_inventory_compatibility.py", 2, 3),
        ("tools/refactor/ap004f_generate_closure.py", 0, 2),
        ("tools/refactor/ap005d_inventory_facades.py", 0, 2),
        (
            "tools/refactor/ap005e1_inventory_installation_entrypoints.py",
            4,
            13,
        ),
        (
            "tools/refactor/ap005e2_characterize_isolated_build_installation.py",
            0,
            3,
        ),
    ],
)
def test_ap006d2_reference_counts(
    relative: str,
    old_count: int,
    new_count: int,
) -> None:
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert text.count(validator.OLD_NAME) == old_count
    assert text.count(validator.NEW_NAME) == new_count


def test_ap006d2_bridge_remains_relative() -> None:
    bridge = REPO_ROOT / validator.OLD_REL
    assert bridge.is_symlink()
    assert bridge.readlink() == Path(validator.NEW_NAME)


def test_ap006d2_dual_root_markers_are_explicit() -> None:
    text = (
        REPO_ROOT
        / "tools/refactor/ap004e_inventory_compatibility.py"
    ).read_text(encoding="utf-8")

    assert "project_root_names" in text
    assert "markers = (" in text
    assert validator.NEW_NAME in text
    assert validator.OLD_NAME in text

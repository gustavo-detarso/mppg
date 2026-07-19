from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = (
    REPO_ROOT
    / "tools"
    / "refactor"
    / "ap006d1_validate_runtime_consumer_migration.py"
)

_spec = importlib.util.spec_from_file_location(
    "ap006d1_validator",
    VALIDATOR_PATH,
)
assert _spec is not None and _spec.loader is not None
_validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validator)


def test_ap006d1_validator_accepts_materialized_tree() -> None:
    assert _validator.validate(REPO_ROOT) == []


def test_runtime_files_no_longer_name_legacy_root() -> None:
    for relative in _validator.MIGRATED:
        text = (REPO_ROOT / relative).read_text(
            encoding="utf-8",
            errors="surrogateescape",
        )
        assert _validator.OLD_NAME not in text


def test_tui_state_uses_four_canonical_paths() -> None:
    state_path = REPO_ROOT / _validator.MIGRATED[4]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    serialized = json.dumps(state, ensure_ascii=False)
    assert serialized.count(_validator.NEW_NAME) == 4
    assert _validator.OLD_NAME not in serialized


def test_historical_report_keeps_legacy_provenance() -> None:
    report_path = REPO_ROOT / _validator.REPORT
    report_text = report_path.read_text(
        encoding="utf-8",
        errors="surrogateescape",
    )
    assert report_text.count(_validator.OLD_NAME) == 2


def test_compatibility_bridge_remains_relative() -> None:
    bridge = REPO_ROOT / _validator.OLD_REL
    assert bridge.is_symlink()
    assert bridge.readlink().as_posix() == _validator.NEW_NAME

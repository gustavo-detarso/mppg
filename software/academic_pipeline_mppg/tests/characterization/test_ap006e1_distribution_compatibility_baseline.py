from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_validator():
    path = repo_root() / "tools/refactor/ap006e1_validate_distribution_compatibility_baseline.py"
    spec = importlib.util.spec_from_file_location("ap006e1_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ap006e1_distribution_compatibility_baseline() -> None:
    validator = load_validator()
    summary = validator.validate(repo_root())
    assert summary["phase"] == "AP-006E.1"
    assert summary["status"] == "ok"
    assert summary["contract_owned_path_count"] == 4
    assert summary["relevant_pth_file_count"] == 0
    assert summary["environment_mode"] in {
        "source_tree_only_uninstalled_distribution",
        "installed_distribution",
    }


def test_ap006e1_contract_does_not_classify_itself() -> None:
    validator = load_validator()
    data = json.loads((repo_root() / validator.JSON_REL).read_text(encoding="utf-8"))
    owned = set(data["deterministic"]["contract_owned_paths"])
    candidates = set(data["deterministic"]["reference_partition"]["recorded_candidate_paths"])
    assert owned == set(validator.CONTRACT_OWNED_PATHS)
    assert owned.isdisjoint(candidates)
    assert data["deterministic"]["reference_partition"]["groups_are_disjoint"] is True
    assert data["deterministic"]["reference_partition"]["contract_owned_paths_excluded"] is True

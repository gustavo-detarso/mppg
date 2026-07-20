from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
MANIFEST = (
    REPO
    / "docs/refactor/academic-pipeline/AP-006"
    / "ap006d4e_backup_evidence_preservation_contract.json"
)
VALIDATOR = (
    REPO
    / "tools/refactor"
    / "ap006d4e_validate_backup_evidence_preservation.py"
)


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "ap006d4e_validator",
        VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_preserves_all_backup_evidence_in_place() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4E"
    assert data["summary"]["candidate_count"] == 177
    assert data["summary"]["productive_reference_count"] == 0
    assert data["summary"]["preserve_in_place_count"] == 177
    assert data["constraints"]["delete_forbidden"] is True
    assert data["constraints"]["move_forbidden"] is True
    assert data["constraints"]["rename_forbidden"] is True
    assert data["constraints"]["content_modification_forbidden"] is True
    assert len(data["candidates"]) == 177


def test_validator_confirms_candidate_blobs_and_paths() -> None:
    result = load_validator().validate(REPO, MANIFEST)
    assert result["status"] == "ok"
    assert result["candidate_count"] == 177
    assert result["productive_reference_count"] == 0
    assert sum(result["classification_counts"].values()) == 177


def test_contract_artifacts_do_not_expand_candidate_set() -> None:
    validator = load_validator()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tree = validator.parse_tree(REPO)
    observed = {
        path
        for path, entry in tree.items()
        if (
            entry["kind"] == "blob"
            and validator.is_candidate_path(path)
            and path not in validator.CONTRACT_OWNED_PATHS
        )
    }
    recorded = {item["path"] for item in data["candidates"]}
    assert validator.CONTRACT_OWNED_PATHS <= set(tree)
    assert validator.CONTRACT_OWNED_PATHS.isdisjoint(recorded)
    assert observed == recorded

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DATA = json.loads((REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e3_isolated_installation_matrix.json').read_text(encoding="utf-8"))


def test_ap007e3_scope_and_canonical_environment_contract() -> None:
    assert DATA["schema"] == "ap007e3_isolated_installation_matrix.v1"
    assert DATA["phase"] == "AP-007E.3"
    assert DATA["scope"]["artifact_installation_executed"] is True
    assert DATA["scope"]["dependency_installation_executed"] is False
    assert DATA["scope"]["network_allowed"] is False
    assert DATA["scope"]["pythonpath_removed"] is True
    assert DATA["scope"]["canonical_environment_modified"] is False
    assert DATA["scope"]["productive_modules_modified"] == []
    assert DATA["scope"]["git_write_executed"] is False
    assert DATA["canonical_environment"]["preserved"] is True


def test_ap007e3_artifact_provenance_contract() -> None:
    reconstruction = DATA["artifact_reconstruction"]
    assert all(item["equivalent"] for item in reconstruction["wheel_equivalence_to_ap007e2"])
    assert all(item["equivalent"] for item in reconstruction["sdist_equivalence_to_ap007e2"])
    assert reconstruction["sdist_derived_wheel_equivalence_to_direct_wheel"]["equivalent"] is True
    assert all(item["equivalent"] for item in reconstruction["sdist_derived_wheel_equivalence_to_ap007e2"])
    assert DATA["summary"]["normalized_artifact_provenance_validated"] is True


def test_ap007e3_installation_isolation_contract() -> None:
    assert len(DATA["installations"]) == 2
    assert {item["origin"] for item in DATA["installations"]} == {"wheel", "sdist"}
    for installation in DATA["installations"]:
        runtime = installation["runtime"]
        assert runtime["summary"]["module_locations_inside_venv"] is True
        assert runtime["summary"]["distribution_location_inside_venv"] is True
        assert runtime["summary"]["pythonpath_removed"] is True
        assert runtime["summary"]["source_leaks"] == 0
        assert runtime["console_script"]["exists"] is True
        assert runtime["console_script"]["executable"] is True
        assert runtime["summary"]["blocking_finding_count"] == 0


def test_ap007e3_resource_and_dependency_classification_contract() -> None:
    assert DATA["resources"]["critical_resources_present_in_both_installations"] is True
    assert DATA["summary"]["critical_resources_present"] is True
    assert DATA["summary"]["blocking_finding_count"] == 0
    assert all(item.get("blocking") is False for item in DATA["findings"])
    assert DATA["dependencies"]["installation_policy"] == [
        "--no-index",
        "--no-deps",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--no-compile",
    ]

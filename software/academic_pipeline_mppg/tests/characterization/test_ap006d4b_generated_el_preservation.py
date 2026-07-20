from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
MANIFEST = (
    REPO
    / "docs/refactor/academic-pipeline/AP-006"
    / "ap006d4b_generated_el_preservation.json"
)
VALIDATOR = (
    REPO
    / "tools/refactor"
    / "ap006d4b_validate_generated_el_preservation.py"
)


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "ap006d4b_validator",
        VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_locks_the_preservation_decision() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4B"
    assert data["decision"] == (
        "preserve_current_generated_artifacts_with_documented_provenance"
    )
    assert data["summary"] == {
        "current_generated_artifact_count": 6,
        "current_old_reference_line_count": 9,
        "historical_evidence_count": 1,
        "historical_old_reference_line_count": 2,
        "preserved_artifact_count": 7,
    }
    assert data["constraints"]["manual_el_edit_forbidden"] is True
    assert data["constraints"]["historical_el_regeneration_forbidden"] is True


def test_validator_confirms_hashes_bridge_and_generator_role() -> None:
    result = load_validator().validate(REPO, MANIFEST)
    assert result["status"] == "ok"
    assert result["artifact_count"] == 7
    assert result["generator_classification"] == (
        "pdf_export_executor_not_el_writer"
    )

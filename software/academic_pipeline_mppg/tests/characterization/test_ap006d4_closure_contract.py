from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
MANIFEST = (
    REPO
    / "docs/refactor/academic-pipeline/AP-006"
    / "ap006d4_closure_contract.json"
)
VALIDATOR = REPO / "tools/refactor/ap006d4_validate_closure.py"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "ap006d4_closure_validator",
        VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closure_manifest_records_five_verified_waves() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4"
    assert data["status"] == "closure_materialized"
    assert len(data["waves"]) == 5
    assert [item["wave"] for item in data["waves"]] == [
        "AP-006D.4A",
        "AP-006D.4B",
        "AP-006D.4C",
        "AP-006D.4D",
        "AP-006D.4E",
    ]
    assert data["integrated_suite"]["passed"] == 624
    assert data["integrated_suite"]["xfailed"] == 3
    assert data["constraints"]["productive_code_modification_performed"] is False


def test_closure_validator_confirms_chain_and_contract() -> None:
    result = load_validator().validate(REPO, MANIFEST)
    assert result["status"] == "ok"
    assert result["wave_count"] == 5
    assert result["validator_count"] == 4
    assert result["integrated_passed"] == 624
    assert result["integrated_xfailed"] == 3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
MANIFEST = (
    REPO
    / "docs/refactor/academic-pipeline/AP-006"
    / "ap006d4d_source_csv_provenance_contract.json"
)
VALIDATOR = (
    REPO
    / "tools/refactor"
    / "ap006d4d_validate_source_csv_provenance.py"
)


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "ap006d4d_validator",
        VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_preserves_source_csv_as_historical_provenance() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4D"
    assert data["summary"]["pair_count"] == 4
    assert data["summary"]["total_row_count"] == 308
    assert data["summary"]["unique_source_csv_value_count"] == 1
    assert data["summary"]["read_sink_count"] == 0
    assert data["summary"]["existence_sink_count"] == 0
    assert data["constraints"]["source_csv_must_remain_verbatim"] is True
    assert data["constraints"][
        "runtime_path_resolver_forbidden_without_new_evidence"
    ] is True


def test_validator_confirms_no_runtime_dereference() -> None:
    result = load_validator().validate(REPO, MANIFEST)
    assert result["status"] == "ok"
    assert result["pair_count"] == 4
    assert result["total_row_count"] == 308
    assert result["read_sink_count"] == 0
    assert result["existence_sink_count"] == 0
    assert result["serialization_sink_count"] >= 1

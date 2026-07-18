from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VALIDATOR = REPO / "tools/refactor/ap006a_validate_scope.py"


def _module():
    spec = importlib.util.spec_from_file_location("ap006a_validate_scope", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ap006a_scope_contract() -> None:
    _module().validate(REPO)


def test_ap006a_confirms_six_subphases() -> None:
    module = _module()
    manifest = json.loads((REPO / module.MANIFEST_REL).read_text(encoding="utf-8"))
    assert manifest["decision"]["recommended_subphases"] == 6
    assert manifest["subphases"][-1] == "AP-006F"


def test_ap006a_defers_physical_target_to_ap006b() -> None:
    module = _module()
    manifest = json.loads((REPO / module.MANIFEST_REL).read_text(encoding="utf-8"))
    decision = manifest["decision"]
    assert decision["excluded_direct_destination"] == "software/academic_pipeline"
    assert decision["deferred_candidate_destinations"] == [
        "software/academic_pipeline_mppg",
        "software/academic-pipeline",
    ]

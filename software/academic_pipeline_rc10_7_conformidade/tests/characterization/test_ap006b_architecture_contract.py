from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VALIDATOR = REPO / "tools/refactor/ap006b_validate_architecture.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "ap006b_validate_architecture",
        VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ap006b_architecture_contract() -> None:
    _module().validate(REPO)


def test_ap006b_selects_noncolliding_underscore_target() -> None:
    module = _module()
    data = json.loads((REPO / module.DECISION_REL).read_text(encoding="utf-8"))
    architecture = data["target_architecture"]
    assert architecture["selected_physical_target"] == "software/academic_pipeline_mppg"
    assert architecture["old_physical_path"].endswith("rc10_7_conformidade")


def test_ap006b_preserves_public_surfaces() -> None:
    module = _module()
    data = json.loads((REPO / module.DECISION_REL).read_text(encoding="utf-8"))
    public = data["public_contract"]
    assert public["distribution_name"] == "academic-pipeline-mppg"
    assert public["console_script"]["name"] == "academic-pipeline"
    assert public["python_import_surfaces"] == ["academic_pipeline", "app_bundle"]


def test_ap006b_defers_physical_change_and_bridge_removal() -> None:
    module = _module()
    data = json.loads((REPO / module.DECISION_REL).read_text(encoding="utf-8"))
    gates = data["phase_gates"]
    assert gates["AP-006C"] == "materialize_physical_target_bridge_and_resolver"
    assert gates["AP-006F"] == "decide_bridge_removal_and_close"

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_validator():
    root = repo_root()
    path = root / "tools/refactor/ap006f4_validate_comparative_source_distribution.py"
    spec = importlib.util.spec_from_file_location("ap006f4_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ap006f4_validator_accepts_current_state() -> None:
    result = load_validator().validate(repo_root(), "auto")
    assert result["status"] == "ok"
    assert result["gate_ap006f5"] == "PASS"
    assert result["logical_suite"] == {"passed": 636, "xfailed": 3}
    assert result["descendant_state"] in {
        "ap006f4_precommit",
        "ap006f5_precommit",
        "ap006f4_postcommit",
        "ap006f5_or_later_postcommit",
    }


def test_ap006f4_contract_records_exact_evidence() -> None:
    root = repo_root()
    payload = json.loads((
        root / "docs/refactor/academic-pipeline/AP-006/ap006f4_comparative_source_distribution_validation.json"
    ).read_text(encoding="utf-8"))
    assert payload["phase"] == "AP-006F.4"
    assert payload["gate_ap006f5"] == "PASS"
    assert payload["materialized_state"]["bridge"] == "removed"
    assert payload["materialized_state"]["fallback"] == "preserved_active_run_legacy"
    assert payload["wheel_comparison"]["member_count"] == 110
    assert payload["functional_comparison"]["operations"] == [
        "list_institutions", "list_layouts", "list_toml_profiles"
    ]
    assert payload["test_results"]["current_nonhistorical"] == {
        "passed": 603, "xfailed": 3
    }
    assert payload["test_results"]["historical"] == {"passed": 33}
    assert payload["test_results"]["logical"] == {"passed": 636, "xfailed": 3}


def test_ap006f4_dispatch_repair_is_distributive() -> None:
    root = repo_root()
    dispatch = root / "software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py"
    tree = ast.parse(dispatch.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "app_bundle.scripts.pipeline.academic_pipeline_gui" in imported
    assert "app_bundle.scripts.pipeline.academic_pipeline_tui" in imported
    assert "app_bundle.scripts.pipeline.academic_pipeline_toml_generator_interativo" in imported
    text = dispatch.read_text(encoding="utf-8")
    for obsolete in (
        "runtime['run_gui']", "runtime['run_tui']",
        "runtime['print_profiles']", "runtime['generate_interactive']",
    ):
        assert obsolete not in text

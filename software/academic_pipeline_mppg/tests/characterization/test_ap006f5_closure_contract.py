from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_validator():
    root = repo_root()
    path = root / "tools/refactor/ap006f5_validate_closure.py"
    spec = importlib.util.spec_from_file_location("ap006f5_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ap006f5_validator_accepts_precommit_closure() -> None:
    result = load_validator().validate(repo_root(), "auto")
    assert result["status"] == "ok"
    assert result["gate_ap006f_commit"] == "PASS"
    assert result["commit_candidate_path_count"] == 18
    assert result["closure_state"] in {
        "ready_for_explicit_commit_authorization", "closed_postcommit"
    }


def test_ap006f5_manifest_is_exact() -> None:
    root = repo_root()
    payload = json.loads((
        root / "docs/refactor/academic-pipeline/AP-006/ap006f5_closure_manifest.json"
    ).read_text(encoding="utf-8"))
    assert payload["phase"] == "AP-006F.5"
    assert payload["status"] == "closure_materialized_precommit"
    assert payload["gate_ap006f_commit"] == "PASS"
    assert payload["test_results"]["logical"] == {"passed": 636, "xfailed": 3}
    assert payload["commit_readiness"]["candidate_path_count"] == 18
    assert payload["commit_readiness"]["explicit_authorization_required"] is True
    assert len(payload["phase_artifacts"]["AP-006F.5"]) == 4


def test_ap006f5_final_topology_and_runtime_adapter() -> None:
    root = repo_root()
    assert not (root / "software/academic_pipeline_rc10_7_conformidade").exists()
    dispatch = root / "software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py"
    legacy = root / "software/academic_pipeline_mppg/academic_pipeline/legacy.py"
    assert dispatch.is_file() and legacy.is_file()
    names = {
        node.name for node in ast.parse(legacy.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run_legacy" in names
    text = dispatch.read_text(encoding="utf-8")
    assert "app_bundle.scripts.pipeline.academic_pipeline_toml_generator_interativo" in text
    assert "runtime['print_profiles']" not in text

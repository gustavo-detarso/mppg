from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
JSON_PATH = REPO / 'docs/refactor/academic-pipeline/AP-006/ap006f3_minimal_materialization.json'
BRIDGE = REPO / 'software/academic_pipeline_rc10_7_conformidade'
CANONICAL = REPO / 'software/academic_pipeline_mppg'
LEGACY = REPO / 'software/academic_pipeline_mppg/academic_pipeline/legacy.py'
EXPECTED_LEGACY_SHA256 = 'f11ddffc30f60ac0c5e0856e8bf00ffaae866a8df806fd3c2b99f1afaa09e6b9'


def test_ap006f3_minimal_materialization_contract() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert payload["phase"] == "AP-006F.3"
    assert payload["status"] == "ok"
    assert payload["gate_ap006f4"] == "PASS"
    assert payload["decisions"] == {
        "bridge": "removed",
        "fallback": "preserved_active_run_legacy",
    }
    assert not BRIDGE.exists()
    assert not BRIDGE.is_symlink()
    assert CANONICAL.is_dir()
    assert hashlib.sha256(LEGACY.read_bytes()).hexdigest() == (
        EXPECTED_LEGACY_SHA256
    )
    names = {
        node.name
        for node in ast.parse(
            LEGACY.read_text(encoding="utf-8")
        ).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run_legacy" in names
    assert payload["wheel_comparison"]["identical"] is True
    assert payload["integrity"]["master_unchanged"] is True
    assert payload["integrity"]["pth_unchanged"] is True

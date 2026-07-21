from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = ROOT / "docs/refactor/academic-pipeline/AP-006/ap006e3_consumer_stabilization.json"
VALIDATOR = ROOT / "tools/refactor/ap006e3_validate_consumer_stabilization.py"
SETUP = ROOT / "software/academic_pipeline_mppg/app_bundle/docs/SETUP_PIPENV.md"

def load() -> dict:
    return json.loads(CONTRACT.read_text())

def test_ap006e3_has_no_productive_or_tracked_materialization() -> None:
    data = load()
    assert data["phase"] == "AP-006E.3"
    assert data["materialized_paths"] == []
    assert data["productive_code_change_count"] == 0
    assert data["setup_documentation_decision"] == (
        "preserve_ap006d3_nonoperational_compatibility_reference"
    )

def test_ap006e3_setup_remains_ap006d3_compatible() -> None:
    text = SETUP.read_text()
    assert text.count("academic_pipeline_rc10_7_conformidade") == 1
    assert text.count("academic_pipeline_mppg") == 1

def test_ap006e3_validator_accepts_current_repository_state() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--mode", "auto"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = ROOT / "docs/refactor/academic-pipeline/AP-006/ap006e5_closure.json"
VALIDATOR = ROOT / "tools/refactor/ap006e5_validate_closure.py"

def load() -> dict:
    return json.loads(CONTRACT.read_text())

def test_ap006e5_closes_without_productive_change() -> None:
    data = load()
    assert data["phase"] == "AP-006E.5"
    assert data["status"] == "closed"
    assert data["candidate_path_count"] == 12
    assert data["productive_code_change_count"] == 0
    assert data["compatibility_bridge"]["decision"] == "preserve_until_ap006f"
    assert data["fallback_decision"] == "preserve_until_ap006f"

def test_ap006e5_records_integrated_validation_gate() -> None:
    validation = load()["validation"]
    assert validation["baseline_passed"] == 626
    assert validation["baseline_xfailed"] == 3
    assert validation["candidate_main_passed"] == 628
    assert validation["historical_isolated_passed"] == 3
    assert validation["candidate_total_passed"] == 631
    assert validation["candidate_xfailed"] == 3
    assert validation["delta_passed"] == 5
    assert validation["regression_count"] == 0
    assert validation["wheel_member_count"] == 110
    assert validation["non_record_changed_member_count"] == 0
    assert validation["legacy_physical_path_count"] == 0
    assert validation["console_help_rc"] == 0
    assert validation["module_help_rc"] == 0

def test_ap006e5_validator_accepts_current_repository_state() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--mode", "auto"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

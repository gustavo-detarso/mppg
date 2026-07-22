from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    cp = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return Path(cp.stdout.strip())


def test_ap006f1_dependency_decision_matrix_contract() -> None:
    repo = _repo_root()
    data = repo / "docs/refactor/academic-pipeline/AP-006/ap006f1_dependency_decision_matrix.json"
    payload = json.loads(data.read_text(encoding="utf-8"))
    assert payload["schema"] == "ap006f1-dependency-decision-matrix-v1"
    assert payload["gates"]["ap006f2"] == "PASS"
    decisions = {item["surface"]: item["decision"] for item in payload["decisions"]}
    assert decisions["bridge_symlink"] == "preserve_pending_ap006f2_no_bridge_trial"
    assert decisions["academic_pipeline.legacy:run_legacy"] == "preserve_as_active_runtime_adapter_pending_replacement_trial"

    validator = repo / "tools/refactor/ap006f1_validate_dependency_decision_matrix.py"
    cp = subprocess.run(
        [sys.executable, str(validator), "--repo", str(repo), "--mode", "auto"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert cp.returncode == 0, cp.stdout
    assert '"status": "ok"' in cp.stdout

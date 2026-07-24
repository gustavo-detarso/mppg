from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
JSON_REL = 'docs/refactor/academic-pipeline/AP-007/ap007e0_distribution_isolation_inventory.json'
DOC_REL = 'docs/refactor/academic-pipeline/AP-007/AP-007E0_DISTRIBUTION_ISOLATION_INVENTORY.md'
VALIDATOR_REL = 'tools/refactor/ap007e0_validate_distribution_isolation_inventory.py'
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007E0_DISTRIBUTION_ISOLATION_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-007/ap007e0_distribution_isolation_inventory.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e0_distribution_isolation_inventory_contract.py', 'tools/refactor/ap007e0_validate_distribution_isolation_inventory.py']
EXPECTED_COMMANDS = ['--help', '--list-institutions', '--list-profiles', '--check-config', '--check-institution-compliance', '--make-doi-manifest']
EXPECTED_GATE = '[GATE] AP-007E.0: INVENTÁRIO DE DISTRIBUIÇÃO, ISOLAMENTO E COMPATIBILIDADE MATERIALIZADO E VALIDADO; NENHUM BUILD, INSTALAÇÃO OU MODIFICAÇÃO PRODUTIVA, SEM STAGING, COMMIT, TAG OU PUSH.'


def load_inventory() -> dict:
    return json.loads((REPO / JSON_REL).read_text(encoding="utf-8"))


def test_ap007e0_inventory_contract_exact_scope_and_baseline() -> None:
    data = load_inventory()
    assert data["schema"] == "ap007e0_distribution_isolation_inventory.v1"
    assert data["phase"] == "AP-007E.0"
    assert data["baseline"]["commit"] == '766956710435f1c338d2e0332d24e55106b981b7'
    assert data["baseline"]["tree"] == '1d673e7c324b74f1fef033578aa995e836da1014'
    assert data["baseline"]["runtime_sha256"] == 'b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c'
    assert data["baseline"]["ap007d_commit_paths_count"] == 48
    assert data["scope"]["materialized_paths"] == EXPECTED_PATHS
    assert data["scope"]["productive_modules_modified"] == []
    assert data["scope"]["build_executed"] is False
    assert data["scope"]["installation_executed"] is False
    assert data["scope"]["git_write_executed"] is False


def test_ap007e0_inventory_contract_distribution_and_matrix() -> None:
    data = load_inventory()
    assert data["distribution"]["tracked_paths_total"] >= data["distribution"]["tracked_paths_audited"]
    assert isinstance(data["distribution"]["tracked_residual_paths_excluded"], list)
    assert isinstance(data["distribution"]["build_files"], list)
    assert isinstance(data["distribution"]["entrypoints"], list)
    assert data["runtime_surfaces"]["selected_commands"] == EXPECTED_COMMANDS
    assert data["runtime_surfaces"]["official_target"] == "academic_pipeline.cli:main"
    assert len(data["runtime_surfaces"]["source_matrix_ap007e1"]) >= 4
    assert "PYTHONPATH_removed" in data["isolation"]["requirements_ap007e1"]
    assert data["gate"] == EXPECTED_GATE


def test_ap007e0_validator_executes_successfully() -> None:
    cp = subprocess.run(
        [sys.executable, str(REPO / VALIDATOR_REL)],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "AP007E0_VALIDATION_FAILURES=0" in cp.stdout

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import academic_pipeline
from academic_pipeline import cli, institution_compliance_runtime, runtime

TEST_FILE = Path(__file__).resolve()
REPO = TEST_FILE.parents[4]
MANIFEST = REPO / "docs/refactor/academic-pipeline/AP-007/ap007d4_institution_compliance_public_integration.json"


def test_manifest_identity_and_scope() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d4-institution-compliance-public-integration/v1"
    assert data["status"] == "institution_compliance_publicly_integrated"
    assert data["candidate_path_count"] == 31
    assert data["integration_origin"] in {"fresh_transactional_patch", "reconciled_preexisting_exact_write"}
    assert len(data["candidate_paths"]) == 31
    assert len(set(data["candidate_paths"])) == 31


def test_exact_command_and_config_are_native() -> None:
    assert runtime.select_runtime_route(("--check-institution-compliance",)) is runtime.RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE
    assert runtime.select_runtime_route(("--config", "x.toml", "--check-institution-compliance")) is runtime.RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE
    assert runtime.select_runtime_route(("--config=x.toml", "--check-institution-compliance")) is runtime.RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE


def test_precedence_and_competing_commands_are_conservative() -> None:
    assert runtime.select_runtime_route(("--help", "--check-institution-compliance")) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE
    assert runtime.select_runtime_route(("--doctor", "--check-institution-compliance")) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(("--check-config", "--check-institution-compliance")) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(("--list-profiles", "--check-institution-compliance")) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(("--tui", "--check-institution-compliance")) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_existing_list_profiles_route_is_preserved() -> None:
    assert runtime.select_runtime_route(("--list-profiles",)) is runtime.RuntimeRoute.NATIVE_LIST_PROFILES


def test_public_runner_preserves_semantic_codes(monkeypatch) -> None:
    captured: list[list[str]] = []
    def fake(argv: Any) -> int:
        captured.append(list(argv))
        return 2
    def forbidden(argv: Any) -> int:
        raise AssertionError(f"fallback indevido: {argv}")
    monkeypatch.setattr(institution_compliance_runtime, "run_institution_compliance_command", fake)
    monkeypatch.setattr(cli, "run_legacy", forbidden)
    argv = ["--config", "x.toml", "--check-institution-compliance"]
    assert academic_pipeline.main(argv) == 2
    assert captured == [argv]


def test_public_runner_maps_adapter_usage_error_to_one(monkeypatch, capsys) -> None:
    def fake(argv: Any) -> int:
        raise institution_compliance_runtime.InstitutionComplianceRuntimeError("--check-institution-compliance exige --config caminho.toml")
    monkeypatch.setattr(institution_compliance_runtime, "run_institution_compliance_command", fake)
    assert academic_pipeline.main(["--check-institution-compliance"]) == 1
    assert "exige --config caminho.toml" in capsys.readouterr().err


def test_explicit_argv_preserves_process_state(monkeypatch) -> None:
    before_argv = list(sys.argv)
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    monkeypatch.setattr(institution_compliance_runtime, "run_institution_compliance_command", lambda argv: 0)
    assert academic_pipeline.main(["--config", "x.toml", "--check-institution-compliance"]) == 0
    assert sys.argv == before_argv
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


def test_hashes_match_manifest() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected


def test_runtime_has_no_implicit_legacy_bridge() -> None:
    source = (REPO / "software/academic_pipeline_mppg/academic_pipeline/runtime.py").read_text(encoding="utf-8")
    for forbidden in ("globals(", "locals(", "sys.path", "importlib", "academic_pipeline_rc10", "LEGACY_MODULE_NAME"):
        assert forbidden not in source

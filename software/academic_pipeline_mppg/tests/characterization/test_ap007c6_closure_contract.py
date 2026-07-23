from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from academic_pipeline import runtime

TEST_FILE = Path(__file__).resolve()
SOFTWARE_ROOT = TEST_FILE.parents[2]
REPO_ROOT = TEST_FILE.parents[4]
MANIFEST = (
    REPO_ROOT
    / "docs/refactor/academic-pipeline/AP-007/"
    "ap007c6_closure_manifest.json"
)


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_closure_manifest_identity() -> None:
    payload = manifest()
    assert payload["phase"] == "AP-007C.6"
    assert payload["status"] == "ready_for_isolated_commit_decision"
    assert payload["candidate_path_count"] == 23


def test_public_routes_are_final() -> None:
    assert runtime.select_runtime_route(
        ("--doctor",)
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(
        ("--check-config",)
    ) is runtime.RuntimeRoute.NATIVE_CHECK_CONFIG


def test_first_wave_route_is_preserved() -> None:
    assert runtime.select_runtime_route(
        ("--list-layouts",)
    ) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE


def test_doctor_precedes_check_config() -> None:
    assert runtime.select_runtime_route(
        ("--doctor", "--check-config")
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR


def test_earlier_stage_precedes_check_config() -> None:
    assert runtime.select_runtime_route(
        ("--check-institution-compliance", "--check-config")
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_unrelated_operational_command_remains_fallback() -> None:
    assert runtime.select_runtime_route(
        ("--tui",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_process_exit_contract_is_recorded() -> None:
    assert manifest()["process_exit_codes"] == {
        "check_config_missing_config": 1,
        "diagnostic_ok": 0,
        "diagnostic_not_ok": 2,
    }


def test_native_command_inventory_is_complete() -> None:
    assert manifest()["native_public_commands"] == [
        "--check-config",
        "--doctor",
        "--explain-profile",
        "--help",
        "--list-institutions",
        "--list-layouts",
        "--list-toml-profiles",
    ]


def test_runtime_source_has_no_legacy_bridge() -> None:
    source = (
        SOFTWARE_ROOT / "academic_pipeline/runtime.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "globals(",
        "locals(",
        "sys.path",
        "importlib",
        "academic_pipeline_rc10",
        "LEGACY_MODULE_NAME",
    ):
        assert forbidden not in source


def test_next_phase_is_operational_migration() -> None:
    assert manifest()["next_phase"] == "AP-007D"


def test_commit_is_not_performed_by_closure() -> None:
    assert manifest()["commit_authorized"] is False


def test_candidate_paths_are_unique() -> None:
    paths = manifest()["candidate_paths"]
    assert len(paths) == 23
    assert len(set(paths)) == 23

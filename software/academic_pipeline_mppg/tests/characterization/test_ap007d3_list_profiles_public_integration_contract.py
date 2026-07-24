from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import academic_pipeline
from academic_pipeline import cli, list_profiles_runtime, runtime

TEST_FILE = Path(__file__).resolve()
SOFTWARE_ROOT = TEST_FILE.parents[2]
REPO_ROOT = TEST_FILE.parents[4]
MANIFEST = (
    REPO_ROOT
    / "docs/refactor/academic-pipeline/AP-007/"
    "ap007d3_list_profiles_public_integration.json"
)


def payload() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_identity_and_scope() -> None:
    data = payload()
    assert data["schema"] == "ap007d3-list-profiles-public-integration/v1"
    assert data["phase"] == "AP-007D.3"
    assert data["status"] == "list_profiles_publicly_integrated"
    assert data["candidate_path_count"] == 18
    assert len(data["candidate_paths"]) == 18
    assert len(set(data["candidate_paths"])) == 18


def test_native_parser_accepts_list_profiles() -> None:
    parser = runtime._build_parser()
    parsed = parser.parse_args(["--list-profiles"])
    assert parsed.list_profiles is True
    matches = [
        action
        for action in parser._actions
        if "--list-profiles" in action.option_strings
    ]
    assert len(matches) == 1


def test_public_route_is_native() -> None:
    assert runtime.select_runtime_route(
        ("--list-profiles",)
    ) is runtime.RuntimeRoute.NATIVE_LIST_PROFILES


def test_existing_native_precedence_is_preserved() -> None:
    assert runtime.select_runtime_route(
        ("--help", "--list-profiles")
    ) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE
    assert runtime.select_runtime_route(
        ("--doctor", "--list-profiles")
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(
        ("--check-config", "--list-profiles")
    ) is runtime.RuntimeRoute.NATIVE_CHECK_CONFIG


def test_unrelated_combination_remains_legacy() -> None:
    assert runtime.select_runtime_route(
        ("--tui", "--list-profiles")
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_native_runner_receives_explicit_list(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake(argv: Any) -> int:
        captured["argv"] = list(argv)
        return 37

    def forbidden(argv: Any) -> int:
        raise AssertionError(f"fallback indevido: {argv}")

    monkeypatch.setattr(
        list_profiles_runtime,
        "run_list_profiles_command",
        fake,
    )
    monkeypatch.setattr(cli, "run_legacy", forbidden)
    assert academic_pipeline.main(["--list-profiles"]) == 37
    assert captured == {"argv": ["--list-profiles"]}


def test_process_state_is_preserved(monkeypatch) -> None:
    before_argv = list(sys.argv)
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    monkeypatch.setattr(
        list_profiles_runtime,
        "run_list_profiles_command",
        lambda argv: 0,
    )
    assert academic_pipeline.main(["--list-profiles"]) == 0
    assert sys.argv == before_argv
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


def test_runtime_and_adapter_hashes_match_manifest() -> None:
    data = payload()
    for relative, expected in data["artifact_sha256"].items():
        actual = hashlib.sha256(
            (REPO_ROOT / relative).read_bytes()
        ).hexdigest()
        assert actual == expected


def test_runtime_has_no_implicit_legacy_bridge() -> None:
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

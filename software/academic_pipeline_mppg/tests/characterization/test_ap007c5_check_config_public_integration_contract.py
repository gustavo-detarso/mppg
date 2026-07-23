from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from academic_pipeline import (
    check_config_runtime,
    cli,
    doctor_runtime,
    runtime,
)

TEST_FILE = Path(__file__).resolve()
SOFTWARE_ROOT = TEST_FILE.parents[2]
REPO_ROOT = TEST_FILE.parents[4]
LEGACY_SCRIPT = (
    SOFTWARE_ROOT
    / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)
MANIFEST = (
    REPO_ROOT
    / "docs/refactor/academic-pipeline/AP-007/"
    "ap007c5_check_config_public_integration.json"
)


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_runtime_routes_public_diagnostics_natively() -> None:
    assert runtime.select_runtime_route(
        ("--doctor",)
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(
        ("--check-config",)
    ) is runtime.RuntimeRoute.NATIVE_CHECK_CONFIG


def test_runtime_check_config_never_calls_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_config_runtime,
        "run_check_config_command",
        lambda argv: 23,
    )

    def forbidden(_argv: Any) -> int:
        raise AssertionError("fallback não deve ser chamado")

    assert runtime.run(
        ["--check-config"],
        legacy_runner=forbidden,
    ) == 23


def test_cli_check_config_never_calls_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_config_runtime,
        "run_check_config_command",
        lambda argv: 19,
    )
    monkeypatch.setattr(
        cli,
        "run_legacy",
        lambda argv: (_ for _ in ()).throw(
            AssertionError("fallback indevido")
        ),
    )
    assert cli.main(["--check-config"]) == 19


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (2, 2)],
)
def test_runtime_preserves_diagnostic_return_codes(
    monkeypatch: pytest.MonkeyPatch,
    value: int,
    expected: int,
) -> None:
    monkeypatch.setattr(
        check_config_runtime,
        "run_check_config_command",
        lambda argv: value,
    )
    assert runtime.run(
        ["--check-config"],
        legacy_runner=lambda argv: 99,
    ) == expected


def test_doctor_precedes_check_config() -> None:
    assert runtime.select_runtime_route(
        ("--doctor", "--check-config")
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR


def test_first_wave_precedes_check_config() -> None:
    assert runtime.select_runtime_route(
        ("--list-layouts", "--check-config")
    ) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE


def test_earlier_legacy_stage_precedes_check_config() -> None:
    assert runtime.select_runtime_route(
        tuple(manifest()["precedence_probe_argv"])
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_unrelated_command_remains_list_argv_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fallback(argv: Any) -> int:
        captured["argv"] = list(argv)
        return 17

    monkeypatch.setattr(cli, "run_legacy", fallback)
    assert cli.main(["--tui"]) == 17
    assert captured == {"argv": ["--tui"]}


def test_public_missing_config_matches_historical() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOFTWARE_ROOT)
    public = subprocess.run(
        [
            sys.executable,
            "-m",
            "academic_pipeline",
            "--check-config",
        ],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    historical = subprocess.run(
        [
            sys.executable,
            str(LEGACY_SCRIPT),
            "--check-config",
        ],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    message = "--check-config exige --config caminho.toml"
    assert public.returncode == historical.returncode == 1
    assert message in public.stderr
    assert message in historical.stderr


def test_explicit_argv_preserves_process_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_config_runtime,
        "run_check_config_command",
        lambda argv: 0,
    )
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    monkeypatch.setattr(
        sys,
        "argv",
        ["process", "--doctor"],
    )
    assert runtime.run(
        ["--check-config"],
        legacy_runner=lambda argv: 99,
    ) == 0
    assert sys.argv == ["process", "--doctor"]
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


def test_runtime_source_remains_free_of_legacy_bridge() -> None:
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


def test_ap007c4_phase_local_route_is_superseded() -> None:
    c4 = json.loads(
        (
            REPO_ROOT
            / "docs/refactor/academic-pipeline/AP-007/"
            "ap007c4_check_config_native_adapter.json"
        ).read_text(encoding="utf-8")
    )
    assert c4["public_route"]["integration_phase"] == (
        "AP-007C.5"
    )
    assert manifest()["supersedes_phase_local_route"] == (
        "AP-007C.4"
    )


def test_manifest_records_public_integration() -> None:
    payload = manifest()
    assert payload["phase"] == "AP-007C.5"
    assert payload["status"] == (
        "check_config_publicly_integrated"
    )
    assert payload["public_route"] == {
        "doctor": "native_doctor",
        "check_config": "native_check_config",
    }
    assert payload["process_exit_codes"] == {
        "missing_config": 1,
        "report_ok": 0,
        "report_not_ok": 2,
    }

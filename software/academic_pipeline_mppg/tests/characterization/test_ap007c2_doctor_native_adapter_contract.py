from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from academic_pipeline import doctor_runtime, runtime

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
    "ap007c2_doctor_native_adapter.json"
)


def _context(
    *,
    report: dict[str, Any],
    events: list[tuple[str, Any]],
    output_dir: Path,
) -> doctor_runtime.DoctorRuntimeContext:
    def load_config(path: Path) -> dict[str, Any]:
        events.append(("load_config", path))
        return {
            "__config_path__": str(path),
            "__config_dir__": str(path.parent),
            "paths": {},
        }

    def apply(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
        events.append(("apply", args.doctor))
        return cfg

    def output_paths(
        cfg: dict[str, Any],
    ) -> tuple[Path, str]:
        events.append(("output_paths", cfg))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, "doctor_test"

    def research_paths(
        cfg: dict[str, Any],
    ) -> tuple[Path, str]:
        events.append(("research_paths", cfg))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, "research_doctor_test"

    def external(cfg: dict[str, Any]) -> bool:
        events.append(("external", cfg))
        return False

    def run_doctor(
        cfg: dict[str, Any] | None,
    ) -> dict[str, Any]:
        events.append(("run_doctor", cfg))
        return report

    def print_report(value: dict[str, Any]) -> None:
        events.append(("print_report", value))

    def write_json(path: Path, value: Any) -> None:
        events.append(("write_json", path))
        path.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return doctor_runtime.DoctorRuntimeContext(
        load_config=load_config,
        apply_cli_path_overrides=apply,
        output_paths=output_paths,
        research_output_paths=research_paths,
        external_search_enabled=external,
        run_doctor=run_doctor,
        print_doctor_report=print_report,
        write_json=write_json,
    )


def test_context_is_frozen_slotted_and_exact() -> None:
    assert dataclasses.is_dataclass(
        doctor_runtime.DoctorRuntimeContext
    )
    assert (
        doctor_runtime.DoctorRuntimeContext
        .__dataclass_params__.frozen
    )
    assert hasattr(
        doctor_runtime.DoctorRuntimeContext,
        "__slots__",
    )
    assert {
        field.name
        for field in dataclasses.fields(
            doctor_runtime.DoctorRuntimeContext
        )
    } == {
        "load_config",
        "apply_cli_path_overrides",
        "output_paths",
        "research_output_paths",
        "external_search_enabled",
        "run_doctor",
        "print_doctor_report",
        "write_json",
    }


def test_source_has_no_legacy_dependency_container() -> None:
    source = (
        SOFTWARE_ROOT / "academic_pipeline/doctor_runtime.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "globals(",
        "locals(",
        "sys.path",
        "importlib",
        "academic_pipeline_rc10",
        "run_legacy",
        "LEGACY_MODULE_NAME",
    ):
        assert forbidden not in source


def test_default_context_exposes_callable_dependencies() -> None:
    context = doctor_runtime.default_doctor_runtime_context()
    for field in dataclasses.fields(context):
        assert callable(getattr(context, field.name))


@pytest.mark.parametrize(
    ("ok_value", "expected"),
    [(True, 0), (False, 2)],
)
def test_no_config_preserves_semantic_exit_codes(
    tmp_path: Path,
    ok_value: bool,
    expected: int,
) -> None:
    events: list[tuple[str, Any]] = []
    context = _context(
        report={"ok": ok_value, "warnings": []},
        events=events,
        output_dir=tmp_path,
    )
    assert doctor_runtime.run_doctor_command(
        ["--doctor"],
        context=context,
    ) == expected
    assert ("run_doctor", None) in events
    assert not any(
        name == "write_json"
        for name, _value in events
    )


def test_config_path_writes_expected_report(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Any]] = []
    context = _context(
        report={"ok": True, "warnings": []},
        events=events,
        output_dir=tmp_path,
    )
    config = tmp_path / "project.toml"
    config.write_text(
        "[projeto]\nnome='test'\n",
        encoding="utf-8",
    )
    assert doctor_runtime.run_doctor_command(
        ["--config", str(config), "--doctor"],
        context=context,
    ) == 0
    target = tmp_path / "doctor_test.doctor_report.json"
    assert target.is_file()
    assert json.loads(
        target.read_text(encoding="utf-8")
    )["ok"] is True


def test_public_route_is_intentionally_deferred() -> None:
    assert runtime.select_runtime_route(
        ("--doctor",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(
        ("--check-config",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_real_no_config_adapter_matches_historical() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOFTWARE_ROOT)
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from academic_pipeline.doctor_runtime import "
                "run_doctor_command; "
                "raise SystemExit("
                "run_doctor_command(['--doctor']))"
            ),
        ],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=90,
    )
    historical = subprocess.run(
        [sys.executable, str(LEGACY_SCRIPT), "--doctor"],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=90,
    )
    assert native.returncode == historical.returncode
    assert native.returncode in {0, 2}
    assert native.stdout.rstrip() == historical.stdout.rstrip()
    assert native.stderr == historical.stderr


def test_explicit_argv_preserves_process_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Any]] = []
    context = _context(
        report={"ok": True},
        events=events,
        output_dir=tmp_path,
    )
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    monkeypatch.setattr(
        sys,
        "argv",
        ["process", "--check-config"],
    )
    assert doctor_runtime.run_doctor_command(
        ["--doctor"],
        context=context,
    ) == 0
    assert sys.path == before_path
    assert os.getcwd() == before_cwd
    assert sys.argv == ["process", "--check-config"]


def test_manifest_records_non_integrated_state() -> None:
    payload = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )
    assert payload["phase"] == "AP-007C.2"
    assert payload["status"] == (
        "doctor_adapter_materialized_not_integrated"
    )
    assert payload["public_route"]["doctor"] == (
        "legacy_fallback"
    )
    assert payload["public_route"]["integration_phase"] == (
        "AP-007C.3"
    )
    assert payload["next_wave"] == "--check-config"

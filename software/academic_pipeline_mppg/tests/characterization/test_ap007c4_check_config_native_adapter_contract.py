from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from academic_pipeline import check_config_runtime, runtime

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
    "ap007c4_check_config_native_adapter.json"
)


def _context(
    *,
    report: dict[str, Any],
    events: list[tuple[str, Any]],
    output_dir: Path,
) -> check_config_runtime.CheckConfigRuntimeContext:
    def load_config(path: Path) -> dict[str, Any]:
        events.append(("load_config", path))
        return {
            "__config_path__": str(path),
            "__config_dir__": str(path.parent),
            "paths": {},
        }

    def apply(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
        events.append(("apply", args.check_config))
        return cfg

    def output_paths(
        cfg: dict[str, Any],
    ) -> tuple[Path, str]:
        events.append(("output_paths", cfg))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, "check_test"

    def research_paths(
        cfg: dict[str, Any],
    ) -> tuple[Path, str]:
        events.append(("research_paths", cfg))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, "research_check_test"

    def external(cfg: dict[str, Any]) -> bool:
        events.append(("external", cfg))
        return False

    def check(cfg: dict[str, Any]) -> dict[str, Any]:
        events.append(("check", cfg))
        return report

    def print_report(value: dict[str, Any]) -> None:
        events.append(("print_report", value))

    def write_json(path: Path, value: Any) -> None:
        events.append(("write_json", path))
        path.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return check_config_runtime.CheckConfigRuntimeContext(
        load_config=load_config,
        apply_cli_path_overrides=apply,
        output_paths=output_paths,
        research_output_paths=research_paths,
        external_search_enabled=external,
        check_config=check,
        print_check_config_report=print_report,
        write_json=write_json,
    )


def test_context_is_frozen_slotted_and_exact() -> None:
    cls = check_config_runtime.CheckConfigRuntimeContext
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen
    assert hasattr(cls, "__slots__")
    assert {
        field.name
        for field in dataclasses.fields(cls)
    } == {
        "load_config",
        "apply_cli_path_overrides",
        "output_paths",
        "research_output_paths",
        "external_search_enabled",
        "check_config",
        "print_check_config_report",
        "write_json",
    }


def test_source_has_no_legacy_dependency_container() -> None:
    source = (
        SOFTWARE_ROOT
        / "academic_pipeline/check_config_runtime.py"
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
    context = (
        check_config_runtime
        .default_check_config_runtime_context()
    )
    for field in dataclasses.fields(context):
        assert callable(getattr(context, field.name))


def test_missing_config_preserves_required_config_error(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Any]] = []
    context = _context(
        report={"ok": True},
        events=events,
        output_dir=tmp_path,
    )
    with pytest.raises(
        RuntimeError,
        match="--check-config exige --config caminho.toml",
    ):
        check_config_runtime.run_check_config_command(
            ["--check-config"],
            context=context,
        )
    assert not any(
        name in {"load_config", "apply", "check", "write_json"}
        for name, _value in events
    )


@pytest.mark.parametrize(
    ("ok_value", "expected"),
    [(True, 0), (False, 2)],
)
def test_semantic_exit_codes_and_report(
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
    config = tmp_path / "project.toml"
    config.write_text(
        "[projeto]\nnome='test'\n",
        encoding="utf-8",
    )
    assert (
        check_config_runtime.run_check_config_command(
            [
                "--config",
                str(config),
                "--check-config",
            ],
            context=context,
        )
        == expected
    )
    target = tmp_path / "check_test.check_config_report.json"
    assert target.is_file()
    assert json.loads(
        target.read_text(encoding="utf-8")
    )["ok"] is ok_value


def test_explicit_config_is_loaded_and_overridden(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Any]] = []
    context = _context(
        report={"ok": True},
        events=events,
        output_dir=tmp_path,
    )
    config = tmp_path / "project.toml"
    config.write_text(
        "[projeto]\nnome='test'\n",
        encoding="utf-8",
    )
    assert (
        check_config_runtime.run_check_config_command(
            [
                "--config",
                str(config),
                "--check-config",
            ],
            context=context,
        )
        == 0
    )
    assert any(name == "load_config" for name, _ in events)
    assert any(name == "apply" for name, _ in events)


def test_public_route_is_intentionally_deferred() -> None:
    assert runtime.select_runtime_route(
        ("--check-config",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(
        ("--doctor",)
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR


def test_real_adapter_matches_historical_with_config(
    tmp_path: Path,
) -> None:
    config = (
        SOFTWARE_ROOT
        / "app_bundle/config/examples/paper_rc10_exemplo.toml"
    )
    if not config.is_file():
        pytest.skip("configuração de exemplo ausente")

    out_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    research_dir = tmp_path / "research"
    argv = [
        "--config",
        str(config),
        "--output-dir",
        str(out_dir),
        "--work-dir",
        str(work_dir),
        "--cache-dir",
        str(cache_dir),
        "--research-output-dir",
        str(research_dir),
        "--output-prefix",
        "check_equivalence",
        "--no-output-subdir",
        "--check-config",
    ]
    report = (
        out_dir
        / "check_equivalence.check_config_report.json"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOFTWARE_ROOT)

    native = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys;"
                "from academic_pipeline.check_config_runtime "
                "import run_check_config_command;"
                "raise SystemExit(run_check_config_command("
                "json.loads(sys.argv[1])))"
            ),
            json.dumps(argv),
        ],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert native.returncode in {0, 2}
    assert report.is_file()
    native_payload = json.loads(
        report.read_text(encoding="utf-8")
    )
    report.unlink()

    historical = subprocess.run(
        [sys.executable, str(LEGACY_SCRIPT), *argv],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert historical.returncode in {0, 2}
    assert report.is_file()
    historical_payload = json.loads(
        report.read_text(encoding="utf-8")
    )

    native_payload.pop("generated_at", None)
    historical_payload.pop("generated_at", None)
    assert native.returncode == historical.returncode
    assert native.stdout == historical.stdout
    assert native.stderr == historical.stderr
    assert native_payload == historical_payload


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
    config = tmp_path / "process_state.toml"
    config.write_text(
        "[projeto]\\nnome='state-test'\\n",
        encoding="utf-8",
    )
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    monkeypatch.setattr(
        sys,
        "argv",
        ["process", "--doctor"],
    )
    assert (
        check_config_runtime.run_check_config_command(
            [
                "--config",
                str(config),
                "--check-config",
            ],
            context=context,
        )
        == 0
    )
    assert sys.argv == ["process", "--doctor"]
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


def test_manifest_records_non_integrated_state() -> None:
    payload = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )
    assert payload["phase"] == "AP-007C.4"
    assert payload["status"] == (
        "check_config_adapter_materialized_not_integrated"
    )
    assert payload["public_route"] == {
        "doctor": "native_doctor",
        "check_config": "legacy_fallback",
        "integration_phase": "AP-007C.5",
    }
    assert payload["semantic_exit_codes"] == [0, 2]

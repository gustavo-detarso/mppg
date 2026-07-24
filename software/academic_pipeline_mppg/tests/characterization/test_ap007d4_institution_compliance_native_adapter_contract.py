from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from academic_pipeline import (
    institution_compliance_runtime,
    runtime,
)
from academic_pipeline.command_dispatch import dispatch_stage_015

TEST_FILE = Path(__file__).resolve()
SOFTWARE_ROOT = TEST_FILE.parents[2]
REPO_ROOT = TEST_FILE.parents[4]
MANIFEST = (
    REPO_ROOT
    / "docs/refactor/academic-pipeline/AP-007/"
    "ap007d4_institution_compliance_native_adapter.json"
)


def _context(
    *,
    report: dict[str, Any],
    events: list[tuple[str, Any]],
    output_dir: Path,
) -> institution_compliance_runtime.InstitutionComplianceRuntimeContext:
    def load_config(path: Path) -> dict[str, Any]:
        events.append(("load_config", path))
        return {
            "__config_path__": str(path),
            "__config_dir__": str(path.parent),
            "paths": {},
        }

    def apply(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
        events.append(("apply", bool(args.check_institution_compliance)))
        return cfg

    def output_paths(cfg: dict[str, Any]) -> tuple[Path, str]:
        events.append(("output_paths", cfg))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, "compliance_test"

    def run_compliance(
        cfg: dict[str, Any],
        **paths: Any,
    ) -> dict[str, Any]:
        events.append(("run", paths))
        return report

    def write_reports(
        value: dict[str, Any],
        prefix: Path,
    ) -> tuple[Path, Path]:
        events.append(("write", prefix))
        md = prefix.with_suffix(".compliance_report.md")
        js = prefix.with_suffix(".compliance_report.json")
        md.write_text("report\n", encoding="utf-8")
        js.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return md, js

    def render(value: dict[str, Any]) -> str:
        events.append(("render", value))
        return "COMPLIANCE=" + ("OK" if value.get("ok") else "FAIL")

    return institution_compliance_runtime.InstitutionComplianceRuntimeContext(
        load_config=load_config,
        apply_cli_path_overrides=apply,
        output_paths=output_paths,
        run_institution_compliance=run_compliance,
        write_compliance_reports=write_reports,
        render_compliance_markdown=render,
    )


def _mapping(
    cfg: dict[str, Any],
    context: institution_compliance_runtime.InstitutionComplianceRuntimeContext,
) -> MappingProxyType[str, Any]:
    return MappingProxyType(
        {
            "Path": Path,
            "cfg": cfg,
            "output_paths": context.output_paths,
            "run_institution_compliance": context.run_institution_compliance,
            "write_compliance_reports": context.write_compliance_reports,
            "render_compliance_markdown": context.render_compliance_markdown,
        }
    )


def test_context_is_frozen_slotted_and_exact() -> None:
    cls = institution_compliance_runtime.InstitutionComplianceRuntimeContext
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen
    assert hasattr(cls, "__slots__")
    assert {field.name for field in dataclasses.fields(cls)} == {
        "load_config",
        "apply_cli_path_overrides",
        "output_paths",
        "run_institution_compliance",
        "write_compliance_reports",
        "render_compliance_markdown",
    }


def test_source_has_no_implicit_legacy_bridge() -> None:
    source = (
        SOFTWARE_ROOT
        / "academic_pipeline/institution_compliance_runtime.py"
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
        institution_compliance_runtime
        .default_institution_compliance_runtime_context()
    )
    for field in dataclasses.fields(context):
        assert callable(getattr(context, field.name))


def test_missing_config_preserves_historical_error() -> None:
    with pytest.raises(
        RuntimeError,
        match="--check-institution-compliance exige --config caminho.toml",
    ):
        institution_compliance_runtime.run_institution_compliance_command(
            ["--check-institution-compliance"]
        )


@pytest.mark.parametrize(("report_ok", "expected"), [(True, 0), (False, 2)])
def test_semantic_exit_codes_and_reports(
    tmp_path: Path,
    report_ok: bool,
    expected: int,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[paths]\n", encoding="utf-8")
    events: list[tuple[str, Any]] = []
    output = tmp_path / "out"
    context = _context(
        report={"ok": report_ok, "items": []},
        events=events,
        output_dir=output,
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        observed = (
            institution_compliance_runtime
            .run_institution_compliance_command(
                [
                    "--config",
                    str(config),
                    "--check-institution-compliance",
                ],
                context=context,
            )
        )
    assert observed == expected
    assert f"COMPLIANCE={'OK' if report_ok else 'FAIL'}" in buffer.getvalue()
    assert "Relatórios:" in buffer.getvalue()
    assert (output / "compliance_test.compliance_report.md").is_file()
    assert (output / "compliance_test.compliance_report.json").is_file()
    assert [event[0] for event in events] == [
        "load_config",
        "apply",
        "output_paths",
        "run",
        "write",
        "render",
    ]


def test_adapter_matches_canonical_stage_with_controlled_context(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[paths]\n", encoding="utf-8")
    report = {"ok": True, "items": []}

    stage_events: list[tuple[str, Any]] = []
    stage_out = tmp_path / "stage"
    stage_context = _context(
        report=report,
        events=stage_events,
        output_dir=stage_out,
    )
    parser = institution_compliance_runtime._build_parser()
    args = parser.parse_args(
        ["--config", str(config), "--check-institution-compliance"]
    )
    cfg = institution_compliance_runtime._prepare_config(args, stage_context)
    stage_stdout = io.StringIO()
    with contextlib.redirect_stdout(stage_stdout):
        stage_result = dispatch_stage_015(args, _mapping(cfg, stage_context))

    adapter_events: list[tuple[str, Any]] = []
    adapter_out = tmp_path / "adapter"
    adapter_context = _context(
        report=report,
        events=adapter_events,
        output_dir=adapter_out,
    )
    adapter_stdout = io.StringIO()
    with contextlib.redirect_stdout(adapter_stdout):
        adapter_result = (
            institution_compliance_runtime
            .run_institution_compliance_command(
                ["--config", str(config), "--check-institution-compliance"],
                context=adapter_context,
            )
        )
    assert stage_result.handled is True
    assert int(stage_result.value) == adapter_result == 0
    assert stage_stdout.getvalue().replace(str(stage_out), "<OUT>") == (
        adapter_stdout.getvalue().replace(str(adapter_out), "<OUT>")
    )
    stage_payload = json.loads(
        (stage_out / "compliance_test.compliance_report.json").read_text(
            encoding="utf-8"
        )
    )
    adapter_payload = json.loads(
        (adapter_out / "compliance_test.compliance_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert stage_payload == adapter_payload


def test_explicit_argv_preserves_process_state(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[paths]\n", encoding="utf-8")
    context = _context(
        report={"ok": True, "items": []},
        events=[],
        output_dir=tmp_path / "out",
    )
    before_argv = list(sys.argv)
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    assert (
        institution_compliance_runtime
        .run_institution_compliance_command(
            ["--config", str(config), "--check-institution-compliance"],
            context=context,
        )
        == 0
    )
    assert sys.argv == before_argv
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


def test_route_remains_legacy_in_adapter_phase() -> None:
    assert runtime.select_runtime_route(
        ("--check-institution-compliance",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_manifest_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d4-institution-compliance-native-adapter/v1"
    assert data["status"] == "materialized_route_still_legacy"
    assert data["command"] == "--check-institution-compliance"
    assert data["exit_contract"] == {"ok_true": 0, "ok_false": 2}
    assert data["public_route_changed"] is False

\
from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from academic_pipeline import cli
from academic_pipeline import runtime

ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = (
    ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)
CONFIG = ROOT / "app_bundle/config/examples/atividade_rc10_exemplo.toml"

FIRST_WAVE_CASES = (
    ("help", ("--help",)),
    ("list_profiles", ("--list-toml-profiles",)),
    ("list_institutions", ("--list-institutions",)),
    ("list_layouts", ("--config", str(CONFIG), "--list-layouts")),
    ("explain_profile", ("--explain-profile", "fgv")),
)


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _normalize(text: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in text.replace(str(CONFIG), "<CONFIG>").splitlines()
        if line.strip()
    ).strip()


def _option_set(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)--[a-z0-9][a-z0-9-]*", text))


def test_runtime_context_is_frozen_slotted_and_explicit() -> None:
    assert dataclasses.is_dataclass(runtime.RuntimeContext)
    assert runtime.RuntimeContext.__dataclass_params__.frozen is True
    assert hasattr(runtime.RuntimeContext, "__slots__")

    names = {
        field.name for field in dataclasses.fields(runtime.RuntimeContext)
    }
    assert names == {
        "path_type",
        "load_config",
        "describe_institution_profiles",
        "available_layouts",
        "resolve_layout_spec",
        "explain_profile",
    }


def test_dispatch_result_contract_is_handled_value() -> None:
    from academic_pipeline.command_dispatch import DispatchResult

    assert [
        field.name for field in dataclasses.fields(DispatchResult)
    ] == ["handled", "value"]

    handled = DispatchResult(True, 0)
    not_handled = DispatchResult(False, None)

    assert handled.handled is True
    assert handled.value == 0
    assert not_handled.handled is False
    assert not_handled.value is None


def test_runtime_context_maps_exact_dispatch_dependencies() -> None:
    context = runtime.default_runtime_context()
    mapping = context.as_dispatch_mapping()

    assert set(mapping) == {
        "Path",
        "available_layouts",
        "describe_institution_profiles",
        "explain_profile",
        "load_config",
        "resolve_layout_spec",
    }
    assert mapping["Path"] is Path
    assert callable(mapping["load_config"])
    assert callable(mapping["resolve_layout_spec"])


def test_native_load_config_adds_canonical_metadata() -> None:
    context = runtime.default_runtime_context()
    payload = context.load_config(CONFIG.resolve())

    assert payload["__config_path__"] == str(CONFIG.resolve())
    assert payload["__config_dir__"] == str(CONFIG.resolve().parent)


def test_runtime_reuses_existing_cli_parser_builder() -> None:
    import inspect

    from academic_pipeline import cli_parser
    from app_bundle.scripts.pipeline.diagnostics import PIPELINE_VERSION

    assert str(inspect.signature(cli_parser.build_parser)) == (
        "(*, pipeline_version: 'str') -> 'argparse.ArgumentParser'"
    )
    parser = cli_parser.build_parser(
        pipeline_version=PIPELINE_VERSION,
    )
    assert parser.prog
    registered_long_options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    assert len(registered_long_options) == 63

    source = (ROOT / "academic_pipeline/runtime.py").read_text(
        encoding="utf-8"
    )
    assert "cli_parser.build_parser(" in source
    assert "pipeline_version=PIPELINE_VERSION" in source


def test_route_selection_is_explicit() -> None:
    for _label, argv in FIRST_WAVE_CASES:
        assert (
            runtime.select_runtime_route(argv)
            is runtime.RuntimeRoute.NATIVE_FIRST_WAVE
        )

    assert (
        runtime.select_runtime_route(("--doctor",))
        is runtime.RuntimeRoute.LEGACY_FALLBACK
    )
    assert (
        runtime.select_runtime_route(("--explain-profile=fgv",))
        is runtime.RuntimeRoute.NATIVE_FIRST_WAVE
    )


def test_cli_main_routes_non_native_command_to_injected_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_legacy(argv: Any) -> int:
        captured["argv"] = list(argv)
        return 37

    monkeypatch.setattr(cli, "run_legacy", fake_legacy)

    assert cli.main(["--doctor"]) == 37
    assert captured == {"argv": ["--doctor"]}


def test_explicit_argv_does_not_read_process_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_legacy(argv: Any) -> int:
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli, "run_legacy", fake_legacy)
    monkeypatch.setattr(sys, "argv", ["process", "--doctor"])

    assert cli.main(["--check-config"]) == 0
    assert captured["argv"] == ["--check-config"]


@pytest.mark.parametrize(("_label", "argv"), FIRST_WAVE_CASES)
def test_first_wave_never_invokes_legacy_and_preserves_process_state(
    _label: str,
    argv: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_legacy(_argv: Any) -> int:
        raise AssertionError("fallback legado não pode ser chamado")

    before_argv = list(sys.argv)
    before_path = list(sys.path)
    before_cwd = os.getcwd()

    if "--help" in argv:
        with pytest.raises(SystemExit) as caught:
            runtime.run(argv, legacy_runner=forbidden_legacy)
        assert caught.value.code == 0
    else:
        assert runtime.run(argv, legacy_runner=forbidden_legacy) == 0

    capsys.readouterr()
    assert sys.argv == before_argv
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


@pytest.mark.parametrize(("_label", "argv"), FIRST_WAVE_CASES)
def test_public_module_matches_historical_first_wave(
    _label: str,
    argv: tuple[str, ...],
) -> None:
    public = _run_python("-m", "academic_pipeline", *argv)
    historical = _run_python(str(LEGACY_SCRIPT), *argv)

    assert public.returncode == 0, public.stderr
    assert historical.returncode == 0, historical.stderr
    assert public.stderr == historical.stderr == ""

    if "--help" in argv:
        assert _option_set(public.stdout) == _option_set(historical.stdout)
        assert len(_option_set(public.stdout)) == 66
        assert public.stdout.startswith("usage: academic-pipeline ")
        assert historical.stdout.startswith(
            "usage: academic_pipeline_rc10.py "
        )
    else:
        assert _normalize(public.stdout) == _normalize(historical.stdout)


def test_runtime_source_has_no_forbidden_dependency_container() -> None:
    source = (ROOT / "academic_pipeline/runtime.py").read_text(
        encoding="utf-8"
    )

    assert "globals(" not in source
    assert "locals(" not in source
    assert "sys.path" not in source
    assert "importlib" not in source
    assert "academic_pipeline_rc10" not in source
    assert "LEGACY_MODULE_NAME" not in source


def test_first_wave_set_is_exact() -> None:
    assert runtime.FIRST_WAVE_OPTIONS == {
        "--help",
        "--list-toml-profiles",
        "--list-institutions",
        "--list-layouts",
        "--explain-profile",
    }

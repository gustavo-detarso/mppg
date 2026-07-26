from __future__ import annotations

import ast
import contextlib
import io
import pathlib
import sys

import pytest

from academic_pipeline import default_runtime, runtime
from academic_pipeline import prisma_generic_orchestration


def forbidden_legacy(argv):
    raise AssertionError(f"legacy_runner alcançado: {argv!r}")


@pytest.mark.parametrize(
    "argv, token",
    [
        (["--ap008d2-unknown"], "--ap008d2-unknown"),
        (["--ap008d2-unknown=value"], "--ap008d2-unknown=value"),
        (["-Z"], "-Z"),
        (["unexpected-positional"], "unexpected-positional"),
        (["--list-profiles-old"], "--list-profiles-old"),
    ],
)
def test_unknown_inputs_use_argparse_without_legacy(argv, token):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as exc:
        runtime.run(argv, legacy_runner=forbidden_legacy)
    assert exc.value.code == 2
    assert token in stderr.getvalue()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--config", "x.toml"],
        ["--prisma-curadoria-max-incluir", "1"],
        ["--prisma-curadoria-top-n-candidatos", "1"],
        ["--prisma-curadoria-limiar-minimo", "1"],
    ],
)
def test_parse_valid_default_inputs_use_native_default(monkeypatch, argv):
    calls = []

    def fake_run_default(forwarded):
        calls.append(list(forwarded))
        return 37

    monkeypatch.setattr(default_runtime, "run_default", fake_run_default)
    assert runtime.run(argv, legacy_runner=forbidden_legacy) == 37
    assert calls == [argv]


def test_list_profile_argparse_abbreviation_uses_native_list_profiles(monkeypatch):
    calls = []

    def fake_list_profiles(argv):
        calls.append(list(argv))
        return 41

    monkeypatch.setattr(runtime, "_run_native_list_profiles", fake_list_profiles)
    assert runtime.run(["--list-profile"], legacy_runner=forbidden_legacy) == 41
    assert calls == [["--list-profiles"]]


@pytest.mark.parametrize(
    "option",
    [
        "--prisma-exportar-bib",
        "--prisma-congelar-artigo",
        "--prisma-gerar-toml-artigo",
        "--prisma-gerar-artigo-final",
    ],
)
def test_prisma_wrapper_options_bypass_base_parser(monkeypatch, option):
    calls = []

    def fake_run_default(argv):
        calls.append(list(argv))
        return 43

    monkeypatch.setattr(default_runtime, "run_default", fake_run_default)
    assert runtime.run([option], legacy_runner=forbidden_legacy) == 43
    assert calls == [[option]]


def test_default_runtime_uses_canonical_prisma_entrypoint_and_restores_argv(monkeypatch):
    original = list(sys.argv)
    observed = {}

    def fake_entrypoint(mapping):
        observed["argv"] = list(sys.argv)
        observed["has_core"] = "_ap003f_pipeline_core" in mapping
        return 47

    monkeypatch.setattr(
        prisma_generic_orchestration,
        "run_prisma_generic_entrypoint",
        fake_entrypoint,
    )
    assert default_runtime.run_default(["--config", "x.toml"]) == 47
    assert observed["argv"][1:] == ["--config", "x.toml"]
    assert observed["has_core"] is True
    assert sys.argv == original


def test_runtime_source_has_no_productive_legacy_fallback():
    source = pathlib.Path(runtime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "LEGACY_FALLBACK" not in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "legacy_runner"
        for node in ast.walk(tree)
    )
    assert runtime.RuntimeRoute.NATIVE_DEFAULT.value == "native_default"


def test_default_runtime_is_independent_from_historical_module():
    source = pathlib.Path(default_runtime.__file__).read_text(encoding="utf-8")
    assert "app_bundle.scripts.pipeline.academic_pipeline_rc10" not in source
    assert "from .bibliography_manager" not in source
    assert "from app_bundle.scripts.pipeline.bibliography_manager" in source
    assert "run_legacy" not in source
    assert "if __name__" not in source
    assert default_runtime._HISTORICAL_SOURCE_SHA256 == (
        "f385b32fed0445dde90a596440903a7c174e42eac2e1675251ddbd0ce516288f"
    )

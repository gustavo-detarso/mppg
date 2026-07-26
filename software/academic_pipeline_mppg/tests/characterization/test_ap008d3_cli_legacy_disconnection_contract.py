"""Contrato AP-008D.3 para desconexão da fachada CLI do legado residual."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import academic_pipeline
from academic_pipeline import cli, runtime


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = PACKAGE_ROOT / "academic_pipeline" / "cli.py"


def test_cli_source_has_no_productive_legacy_import_or_run_legacy_reference() -> None:
    source = CLI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CLI_PATH))

    legacy_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "legacy"
        and any(alias.name == "run_legacy" for alias in node.names)
    ]
    run_legacy_names = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "run_legacy"
    ]

    assert legacy_imports == []
    assert run_legacy_names == []


def test_cli_main_delegates_to_runtime_without_legacy_runner(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv):
        observed["argv"] = argv
        return 37

    monkeypatch.setattr(cli, "run", fake_run)

    argv = ["--help"]
    assert cli.main(argv) == 37
    assert observed == {"argv": argv}


def test_runtime_preserves_optional_keyword_only_legacy_runner_parameter() -> None:
    parameter = inspect.signature(runtime.run).parameters["legacy_runner"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_runtime_signature_accepts_explicit_legacy_runner_injection() -> None:
    def injected(argv):
        return 99

    bound = inspect.signature(runtime.run).bind(
        ["--help"],
        legacy_runner=injected,
    )

    assert bound.arguments["legacy_runner"] is injected


def test_importing_cli_does_not_import_academic_pipeline_legacy() -> None:
    sentinel = object()
    previous_cli_module = sys.modules.pop("academic_pipeline.cli", sentinel)
    previous_legacy_module = sys.modules.pop("academic_pipeline.legacy", sentinel)
    previous_cli_attribute = getattr(academic_pipeline, "cli", sentinel)

    try:
        if previous_cli_attribute is not sentinel:
            delattr(academic_pipeline, "cli")

        imported_cli = importlib.import_module("academic_pipeline.cli")

        assert imported_cli.__name__ == "academic_pipeline.cli"
        assert "academic_pipeline.legacy" not in sys.modules
    finally:
        sys.modules.pop("academic_pipeline.cli", None)
        sys.modules.pop("academic_pipeline.legacy", None)

        if previous_cli_module is not sentinel:
            sys.modules["academic_pipeline.cli"] = previous_cli_module
        if previous_legacy_module is not sentinel:
            sys.modules["academic_pipeline.legacy"] = previous_legacy_module

        if previous_cli_attribute is sentinel:
            if hasattr(academic_pipeline, "cli"):
                delattr(academic_pipeline, "cli")
        else:
            setattr(academic_pipeline, "cli", previous_cli_attribute)


def test_legacy_module_remains_physically_available_for_later_retirement() -> None:
    spec = importlib.util.find_spec("academic_pipeline.legacy")

    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).name == "legacy.py"

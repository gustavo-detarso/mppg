from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest


def _rc10_module() -> Any:
    return importlib.import_module(
        "app_bundle.scripts.pipeline."
        "academic_pipeline_rc10"
    )


def test_package_main_preserves_cli_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module(
        "academic_pipeline"
    )
    cli = importlib.import_module(
        "academic_pipeline.cli"
    )

    calls: list[object] = []
    argv = ["--doctor", "--no-clear"]

    def fake_main(received_argv: object = None) -> int:
        calls.append(received_argv)
        return 73

    monkeypatch.setattr(cli, "main", fake_main)

    assert package.main(argv) == 73
    assert calls == [argv]


def test_module_entrypoint_preserves_cli_main_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module(
        "academic_pipeline.cli"
    )

    calls: list[object] = []

    def fake_main(received_argv: object = None) -> int:
        calls.append(received_argv)
        return 74

    monkeypatch.setattr(cli, "main", fake_main)

    monkeypatch.delitem(
        sys.modules,
        "academic_pipeline.__main__",
        raising=False,
    )

    with pytest.raises(SystemExit) as captured:
        runpy.run_module(
            "academic_pipeline.__main__",
            run_name="__main__",
        )

    assert captured.value.code == 74
    assert calls == [None]


def test_rc10_load_config_preserves_extracted_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rc10 = _rc10_module()
    orchestration = importlib.import_module(
        "academic_pipeline.document_orchestration"
    )

    config_path = tmp_path / "config.toml"
    sentinel = object()
    calls: list[tuple[dict[str, Any], Path]] = []

    def fake_load_config_impl(
        runtime: dict[str, Any],
        path: Path,
    ) -> object:
        calls.append((runtime, path))
        return sentinel

    monkeypatch.setattr(
        orchestration,
        "load_config_impl",
        fake_load_config_impl,
    )

    result = rc10.load_config(config_path)

    assert result is sentinel
    assert len(calls) == 1

    runtime, received_path = calls[0]

    assert received_path == config_path
    assert runtime["_refs_original_load_config"] is (
        rc10._refs_original_load_config
    )
    assert runtime["_refs_apply_runtime_policy"] is (
        rc10._refs_apply_runtime_policy
    )


def test_rc10_document_loader_preserves_extracted_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rc10 = _rc10_module()
    orchestration = importlib.import_module(
        "academic_pipeline.document_orchestration"
    )

    document_path = tmp_path / "document.json"
    sentinel = object()
    calls: list[tuple[dict[str, Any], Path]] = []

    def fake_loader(
        runtime: dict[str, Any],
        path: Path,
    ) -> object:
        calls.append((runtime, path))
        return sentinel

    monkeypatch.setattr(
        orchestration,
        "load_existing_document_json_impl",
        fake_loader,
    )

    result = rc10.load_existing_document_json(
        document_path
    )

    assert result is sentinel
    assert len(calls) == 1

    runtime, received_path = calls[0]

    assert received_path == document_path
    assert runtime["AcademicDocument"] is (
        rc10.AcademicDocument
    )


def test_preserved_compatibility_symbols_remain_available() -> None:
    cli = importlib.import_module(
        "academic_pipeline.cli"
    )
    orchestration = importlib.import_module(
        "academic_pipeline.document_orchestration"
    )

    assert callable(cli.main)
    assert callable(orchestration.load_config_impl)
    assert callable(
        orchestration.load_existing_document_json_impl
    )

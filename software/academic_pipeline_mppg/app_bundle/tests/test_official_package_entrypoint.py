from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOFTWARE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = SOFTWARE_ROOT / "academic_pipeline"
LEGACY_SCRIPT = (
    SOFTWARE_ROOT
    / "app_bundle"
    / "scripts"
    / "pipeline"
    / "academic_pipeline_rc10.py"
)


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["OPENAI_API_KEY"] = ""
    return subprocess.run(
        [sys.executable, *args],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _option_names(help_text: str) -> set[str]:
    return set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))


def test_official_package_files_exist() -> None:
    assert (PACKAGE_DIR / "__init__.py").is_file()
    assert (PACKAGE_DIR / "__main__.py").is_file()
    assert (PACKAGE_DIR / "cli.py").is_file()
    assert (PACKAGE_DIR / "legacy.py").is_file()


def test_importing_official_package_is_lazy() -> None:
    proc = _run_python(
        "-c",
        (
            "import sys, academic_pipeline; "
            "print('academic_pipeline_rc10' in sys.modules)"
        ),
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_official_module_help_succeeds_offline() -> None:
    proc = _run_python("-m", "academic_pipeline", "--help")

    assert proc.returncode == 0, proc.stderr
    assert "document_model canônico" in proc.stdout
    assert proc.stdout.startswith("usage: academic-pipeline ")


def test_legacy_script_help_remains_supported() -> None:
    proc = _run_python(str(LEGACY_SCRIPT), "--help")

    assert proc.returncode == 0, proc.stderr
    assert "document_model canônico" in proc.stdout
    assert proc.stdout.startswith("usage: academic_pipeline_rc10.py ")


def test_official_and_legacy_help_expose_same_options() -> None:
    official = _run_python("-m", "academic_pipeline", "--help")
    legacy = _run_python(str(LEGACY_SCRIPT), "--help")

    assert official.returncode == 0, official.stderr
    assert legacy.returncode == 0, legacy.stderr
    assert _option_names(official.stdout) == _option_names(legacy.stdout)


def test_official_and_legacy_list_institutions_match() -> None:
    official = _run_python("-m", "academic_pipeline", "--list-institutions")
    legacy = _run_python(str(LEGACY_SCRIPT), "--list-institutions")

    assert official.returncode == 0, official.stderr
    assert legacy.returncode == 0, legacy.stderr
    assert official.stdout == legacy.stdout


def test_ensure_legacy_path_normalizes_preexisting_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from academic_pipeline import legacy

    target = str(legacy.LEGACY_PIPELINE_DIR.resolve())
    original_path = sys.path[:]

    monkeypatch.setattr(
        sys,
        "path",
        [
            target,
            str(legacy.LEGACY_PIPELINE_DIR / "."),
            *original_path,
            target,
        ],
    )

    path = legacy.ensure_legacy_path()
    legacy.ensure_legacy_path()

    assert path == legacy.LEGACY_PIPELINE_DIR.resolve()
    assert path.is_dir()
    assert sys.path[0] == target
    assert sys.path.count(target) == 1


def test_load_legacy_module_exposes_main() -> None:
    from academic_pipeline.legacy import LEGACY_PIPELINE_DIR, load_legacy_module

    module = load_legacy_module()

    assert callable(module.main)
    assert Path(module.__file__).resolve().parent == LEGACY_PIPELINE_DIR.resolve()


def test_run_legacy_forwards_arguments_and_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from academic_pipeline import legacy

    captured: dict[str, list[str]] = {}

    def fake_main() -> int:
        captured["argv"] = sys.argv[:]
        return 7

    monkeypatch.setattr(
        legacy,
        "load_legacy_module",
        lambda: SimpleNamespace(main=fake_main),
    )

    original = sys.argv[:]
    result = legacy.run_legacy(["--doctor"], program_name="academic-pipeline-test")

    assert result == 7
    assert captured["argv"] == ["academic-pipeline-test", "--doctor"]
    assert sys.argv == original


def test_run_legacy_restores_argv_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from academic_pipeline import legacy

    class ExpectedError(RuntimeError):
        pass

    def fake_main() -> int:
        raise ExpectedError("falha simulada")

    monkeypatch.setattr(
        legacy,
        "load_legacy_module",
        lambda: SimpleNamespace(main=fake_main),
    )

    original = sys.argv[:]
    with pytest.raises(ExpectedError, match="falha simulada"):
        legacy.run_legacy(["--doctor"])

    assert sys.argv == original


def test_run_legacy_rejects_invalid_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from academic_pipeline import legacy

    monkeypatch.setattr(
        legacy,
        "load_legacy_module",
        lambda: SimpleNamespace(main=lambda: "invalido"),
    )

    with pytest.raises(legacy.LegacyRuntimeError, match="código de saída inválido"):
        legacy.run_legacy([])


def test_public_main_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    import academic_pipeline
    from academic_pipeline import cli

    captured: dict[str, object] = {}

    def fake_run(argv):
        captured["argv"] = argv
        return 9

    monkeypatch.setattr(cli, "run_legacy", fake_run)

    assert academic_pipeline.main(["--doctor"]) == 9
    assert captured["argv"] == ["--doctor"]

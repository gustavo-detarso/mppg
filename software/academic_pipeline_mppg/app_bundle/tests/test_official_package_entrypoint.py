from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

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
    assert not (PACKAGE_DIR / "legacy.py").exists()


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


def test_academic_pipeline_legacy_is_physically_absent() -> None:
    assert not (PACKAGE_DIR / "legacy.py").exists()


def test_academic_pipeline_legacy_is_not_importable() -> None:
    assert importlib.util.find_spec("academic_pipeline.legacy") is None
    with pytest.raises(ModuleNotFoundError) as captured:
        importlib.import_module("academic_pipeline.legacy")
    assert captured.value.name == "academic_pipeline.legacy"


def test_run_legacy_is_not_exported() -> None:
    import academic_pipeline
    from academic_pipeline import cli, runtime

    assert not hasattr(academic_pipeline, "run_legacy")
    assert not hasattr(cli, "run_legacy")
    assert not hasattr(runtime, "run_legacy")



def test_public_main_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    import academic_pipeline
    from academic_pipeline import cli

    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 9

    monkeypatch.setattr(cli, "main", fake_main)
    assert academic_pipeline.main(["--doctor"]) == 9
    assert captured["argv"] == ["--doctor"]

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SOFTWARE_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = SOFTWARE_ROOT / "pyproject.toml"
PIPELINE_DIR = SOFTWARE_ROOT / "app_bundle" / "scripts" / "pipeline"

DIST_NAME = "academic-pipeline-mppg"
DIST_VERSION = "0.1.0"
CONSOLE_TARGET = "academic_pipeline.cli:main"


def _metadata() -> dict[str, object]:
    return tomllib.loads(
        PYPROJECT_PATH.read_text(encoding="utf-8")
    )


def _run_python(
    *args: str,
    cwd: Path = SOFTWARE_ROOT,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["OPENAI_API_KEY"] = ""
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )


def test_pyproject_uses_setuptools_build_backend() -> None:
    data = _metadata()
    build = data["build-system"]

    assert build["build-backend"] == "setuptools.build_meta"
    assert "setuptools>=68" in build["requires"]
    assert "wheel" in build["requires"]


def test_project_metadata_is_stable_and_versioned() -> None:
    project = _metadata()["project"]

    assert project["name"] == DIST_NAME
    assert project["version"] == DIST_VERSION
    assert project["requires-python"] == ">=3.11"
    assert "rc10" not in project["name"].lower()
    assert "rc10" not in project["version"].lower()


def test_console_script_targets_official_cli() -> None:
    project = _metadata()["project"]

    assert project["scripts"]["academic-pipeline"] == CONSOLE_TARGET


def test_setuptools_discovers_only_project_packages() -> None:
    find = _metadata()["tool"]["setuptools"]["packages"]["find"]

    assert find["where"] == ["."]
    assert find["include"] == ["academic_pipeline*", "app_bundle*"]
    assert find["exclude"] == ["app_bundle.tests*"]
    assert find["namespaces"] is False


def test_pipfile_remains_runtime_dependency_authority() -> None:
    project = _metadata()["project"]

    assert (SOFTWARE_ROOT / "Pipfile").is_file()
    assert (SOFTWARE_ROOT / "Pipfile.lock").is_file()
    assert project["dependencies"] == []


@pytest.mark.parametrize(
    "module_name",
    (
        "academic_pipeline",
        "app_bundle",
        "app_bundle.scripts.pipeline",
    ),
)
def test_declared_source_packages_are_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
    assert module.__file__ is not None


def test_console_target_is_callable() -> None:
    module_name, function_name = CONSOLE_TARGET.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, function_name)

    assert callable(target)


@pytest.mark.parametrize(
    "arguments",
    (
        ("--help",),
        ("--list-institutions",),
    ),
)
def test_console_target_matches_official_module(
    arguments: tuple[str, ...],
) -> None:
    target = _run_python(
        "-c",
        (
            "import sys; "
            "from academic_pipeline.cli import main; "
            f"raise SystemExit(main({list(arguments)!r}))"
        ),
    )
    module = _run_python(
        "-m",
        "academic_pipeline",
        *arguments,
    )

    assert target.returncode == 0, target.stderr
    assert module.returncode == 0, module.stderr
    assert target.stdout == module.stdout
    assert target.stderr == module.stderr


def test_legacy_entrypoint_still_matches_console_target() -> None:
    target = _run_python(
        "-c",
        (
            "from academic_pipeline.cli import main; "
            "raise SystemExit(main(['--list-institutions']))"
        ),
    )
    legacy = _run_python(
        str(PIPELINE_DIR / "academic_pipeline_rc10.py"),
        "--list-institutions",
    )

    assert target.returncode == 0, target.stderr
    assert legacy.returncode == 0, legacy.stderr
    assert target.stdout == legacy.stdout
    assert target.stderr == legacy.stderr

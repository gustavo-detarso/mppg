from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOFTWARE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = SOFTWARE_ROOT / "app_bundle" / "scripts" / "pipeline"

PACKAGE_MODULES = (
    "app_bundle.scripts.pipeline.document_model",
    "app_bundle.scripts.pipeline.citation_renderer",
    "app_bundle.scripts.pipeline.document_validator",
    "app_bundle.scripts.pipeline.quality_report",
)

LEGACY_MODULES = (
    "document_model",
    "citation_renderer",
    "document_validator",
    "quality_report",
)


def _run_python(code: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["OPENAI_API_KEY"] = ""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_document_core_modules_import_by_package(module_name: str) -> None:
    proc = _run_python(
        (
            "import importlib; "
            f"m = importlib.import_module({module_name!r}); "
            "print(m.__name__); print(m.__file__)"
        ),
        cwd=SOFTWARE_ROOT,
    )

    assert proc.returncode == 0, proc.stderr
    assert module_name in proc.stdout
    assert "app_bundle/scripts/pipeline" in proc.stdout


@pytest.mark.parametrize("module_name", LEGACY_MODULES)
def test_document_core_modules_keep_legacy_bare_import(module_name: str) -> None:
    proc = _run_python(
        (
            "import importlib; "
            f"m = importlib.import_module({module_name!r}); "
            "print(m.__name__); print(m.__file__)"
        ),
        cwd=PIPELINE_DIR,
    )

    assert proc.returncode == 0, proc.stderr
    assert module_name in proc.stdout
    assert "app_bundle/scripts/pipeline" in proc.stdout


def test_package_imports_share_document_core_dependencies() -> None:
    model = importlib.import_module(
        "app_bundle.scripts.pipeline.document_model"
    )
    renderer = importlib.import_module(
        "app_bundle.scripts.pipeline.citation_renderer"
    )
    validator = importlib.import_module(
        "app_bundle.scripts.pipeline.document_validator"
    )
    quality = importlib.import_module(
        "app_bundle.scripts.pipeline.quality_report"
    )

    assert renderer.Citation is model.Citation
    assert renderer.Inline is model.Inline
    assert renderer.TextSpan is model.TextSpan

    assert validator.AcademicDocument is model.AcademicDocument
    assert validator.Citation is model.Citation
    assert validator.TextSpan is model.TextSpan
    assert validator.Block is model.Block
    assert (
        validator.extract_latex_cited_keys
        is renderer.extract_latex_cited_keys
    )

    assert quality.AcademicDocument is model.AcademicDocument
    assert quality.Block is model.Block
    assert quality.Citation is model.Citation
    assert quality.TextSpan is model.TextSpan


def test_official_entrypoint_remains_operational_after_core_migration() -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""
    proc = subprocess.run(
        [sys.executable, "-m", "academic_pipeline", "--list-institutions"],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()

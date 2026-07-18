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
    "app_bundle.scripts.pipeline.render_org_latex",
    "app_bundle.scripts.pipeline.render_docx",
    "app_bundle.scripts.pipeline.render_prisma_docx",
    "app_bundle.scripts.pipeline.render_prisma_flow",
    "app_bundle.scripts.pipeline.render_prisma_org",
    "app_bundle.scripts.pipeline.render_prisma_xlsx",
)

LEGACY_MODULES = (
    "render_org_latex",
    "render_docx",
    "render_prisma_docx",
    "render_prisma_flow",
    "render_prisma_org",
    "render_prisma_xlsx",
)


def _run_python(
    code: str,
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
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
def test_render_modules_import_by_package(module_name: str) -> None:
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
def test_render_modules_keep_legacy_bare_import(module_name: str) -> None:
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


def test_package_renderers_share_package_dependencies() -> None:
    model = importlib.import_module(
        "app_bundle.scripts.pipeline.document_model"
    )
    citations = importlib.import_module(
        "app_bundle.scripts.pipeline.citation_renderer"
    )
    org = importlib.import_module(
        "app_bundle.scripts.pipeline.render_org_latex"
    )
    docx = importlib.import_module(
        "app_bundle.scripts.pipeline.render_docx"
    )
    prisma_model = importlib.import_module(
        "app_bundle.scripts.pipeline.prisma_model"
    )
    prisma_docx = importlib.import_module(
        "app_bundle.scripts.pipeline.render_prisma_docx"
    )
    prisma_flow = importlib.import_module(
        "app_bundle.scripts.pipeline.render_prisma_flow"
    )
    prisma_org = importlib.import_module(
        "app_bundle.scripts.pipeline.render_prisma_org"
    )
    prisma_xlsx = importlib.import_module(
        "app_bundle.scripts.pipeline.render_prisma_xlsx"
    )

    assert org.AcademicDocument is model.AcademicDocument
    assert org.Block is model.Block
    assert org.render_latex_inlines is citations.render_latex_inlines

    assert docx.AcademicDocument is model.AcademicDocument
    assert docx.Citation is model.Citation
    assert docx.TextSpan is model.TextSpan
    assert docx.Block is model.Block
    assert (
        docx.is_ai_generated_reference_section_title
        is org.is_ai_generated_reference_section_title
    )

    assert prisma_docx.PrismaReport is prisma_model.PrismaReport
    assert prisma_docx.StudyRecord is prisma_model.StudyRecord
    assert prisma_flow.PrismaFlow is prisma_model.PrismaFlow
    assert prisma_flow.PrismaReport is prisma_model.PrismaReport
    assert prisma_org.PrismaReport is prisma_model.PrismaReport
    assert prisma_org.StudyRecord is prisma_model.StudyRecord
    assert (
        prisma_org.render_prisma_flow_latex
        is prisma_flow.render_prisma_flow_latex
    )
    assert (
        prisma_org.bibliography_style_from_cfg
        is org.bibliography_style_from_cfg
    )
    assert prisma_xlsx.PrismaReport is prisma_model.PrismaReport
    assert prisma_xlsx.StudyRecord is prisma_model.StudyRecord


def test_legacy_renderers_bind_legacy_dependencies() -> None:
    proc = _run_python(
        (
            "import render_docx, render_org_latex, render_prisma_org; "
            "print(render_docx.AcademicDocument.__module__); "
            "print(render_org_latex.AcademicDocument.__module__); "
            "print(render_prisma_org.PrismaReport.__module__); "
            "print(render_prisma_org.render_prisma_flow_latex.__module__)"
        ),
        cwd=PIPELINE_DIR,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "document_model",
        "document_model",
        "prisma_model",
        "render_prisma_flow",
    ]


def test_official_and_legacy_entrypoints_remain_equivalent() -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""

    official = subprocess.run(
        [sys.executable, "-m", "academic_pipeline", "--list-institutions"],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    legacy = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "academic_pipeline_rc10.py"),
            "--list-institutions",
        ],
        cwd=SOFTWARE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert official.returncode == 0, official.stderr
    assert legacy.returncode == 0, legacy.stderr
    assert official.stdout == legacy.stdout

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOFTWARE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = SOFTWARE_ROOT / "app_bundle" / "scripts" / "pipeline"

PACKAGE_MODULES = (
    "app_bundle.scripts.pipeline.corpus_manager",
    "app_bundle.scripts.pipeline.prompt_manager",
    "app_bundle.scripts.pipeline.bibliography_manager",
    "app_bundle.scripts.pipeline.prisma_builder",
    "app_bundle.scripts.pipeline.prisma_validator",
    "app_bundle.scripts.pipeline.prisma_pipeline",
)

LEGACY_MODULES = (
    "corpus_manager",
    "prompt_manager",
    "bibliography_manager",
    "prisma_builder",
    "prisma_validator",
    "prisma_pipeline",
)

MIGRATED_FILES = tuple(
    PIPELINE_DIR / f"{module_name}.py"
    for module_name in LEGACY_MODULES
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
def test_prisma_core_modules_import_by_package(module_name: str) -> None:
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
def test_prisma_core_modules_keep_legacy_bare_import(module_name: str) -> None:
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


def test_package_prisma_core_shares_package_dependencies() -> None:
    utils = importlib.import_module(
        "app_bundle.scripts.pipeline.utils"
    )
    corpus = importlib.import_module(
        "app_bundle.scripts.pipeline.corpus_manager"
    )
    bibliography = importlib.import_module(
        "app_bundle.scripts.pipeline.bibliography_manager"
    )
    prisma_model = importlib.import_module(
        "app_bundle.scripts.pipeline.prisma_model"
    )
    builder = importlib.import_module(
        "app_bundle.scripts.pipeline.prisma_builder"
    )
    validator = importlib.import_module(
        "app_bundle.scripts.pipeline.prisma_validator"
    )
    pipeline = importlib.import_module(
        "app_bundle.scripts.pipeline.prisma_pipeline"
    )
    prisma_org = importlib.import_module(
        "app_bundle.scripts.pipeline.render_prisma_org"
    )

    assert corpus.shorten_text is utils.shorten_text
    assert bibliography.SourceDoc is corpus.SourceDoc

    assert builder.BibBuildResult is bibliography.BibBuildResult
    assert builder.SourceDoc is corpus.SourceDoc
    assert builder.PrismaReport is prisma_model.PrismaReport

    assert validator.PrismaReport is prisma_model.PrismaReport

    assert pipeline.BibBuildResult is bibliography.BibBuildResult
    assert pipeline.SourceDoc is corpus.SourceDoc
    assert pipeline.build_prisma_report is builder.build_prisma_report
    assert (
        pipeline.validate_prisma_report
        is validator.validate_prisma_report
    )
    assert pipeline.render_prisma_org is prisma_org.render_prisma_org


def test_legacy_prisma_core_binds_legacy_dependencies() -> None:
    proc = _run_python(
        (
            "import bibliography_manager, prisma_builder, "
            "prisma_pipeline, prisma_validator; "
            "print(bibliography_manager.SourceDoc.__module__); "
            "print(prisma_builder.PrismaReport.__module__); "
            "print(prisma_pipeline.build_prisma_report.__module__); "
            "print(prisma_pipeline.validate_prisma_report.__module__); "
            "print(prisma_validator.PrismaReport.__module__)"
        ),
        cwd=PIPELINE_DIR,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "corpus_manager",
        "prisma_model",
        "prisma_builder",
        "prisma_validator",
        "prisma_model",
    ]


def test_migrated_prisma_core_has_no_top_level_bare_local_imports() -> None:
    local_modules = {
        path.stem
        for path in PIPELINE_DIR.glob("*.py")
        if path.name != "__init__.py"
    }
    failures: list[str] = []

    for path in MIGRATED_FILES:
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", 1)[0]
                if node.level == 0 and root in local_modules:
                    failures.append(
                        f"{path.name}:{node.lineno}: from {module} import ..."
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in local_modules:
                        failures.append(
                            f"{path.name}:{node.lineno}: import {alias.name}"
                        )

    assert failures == []


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

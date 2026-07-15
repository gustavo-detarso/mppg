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
    "app_bundle.scripts.pipeline.diagnostics",
    "app_bundle.scripts.pipeline.document_builder",
    "app_bundle.scripts.pipeline.institution_compliance",
    "app_bundle.scripts.pipeline.institution_explainer",
    "app_bundle.scripts.pipeline.mindmap_manager",
    "app_bundle.scripts.pipeline.paper_abstracts",
    "app_bundle.scripts.pipeline.project_tools",
    "app_bundle.scripts.pipeline.prompt_lock",
)

LEGACY_MODULES = (
    "diagnostics",
    "document_builder",
    "institution_compliance",
    "institution_explainer",
    "mindmap_manager",
    "paper_abstracts",
    "project_tools",
    "prompt_lock",
)

MIGRATED_FILES = tuple(
    PIPELINE_DIR / f"{module_name}.py"
    for module_name in LEGACY_MODULES
)

ALLOWED_REMAINING_TOP_LEVEL_BARE_IMPORTS = {
    "academic_pipeline_gui",
    "academic_pipeline_rc10",
    "academic_pipeline_toml_generator_interativo",
    "academic_pipeline_tui",
}


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


def _local_module_names() -> set[str]:
    return {
        path.stem
        for path in PIPELINE_DIR.glob("*.py")
        if path.name != "__init__.py"
    }


def _top_level_bare_local_imports(path: Path) -> list[str]:
    local_modules = _local_module_names()
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    rows: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if node.level == 0 and root in local_modules:
                rows.append(
                    f"{path.name}:{node.lineno}: from {module} import ..."
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in local_modules:
                    rows.append(
                        f"{path.name}:{node.lineno}: import {alias.name}"
                    )

    return rows


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_support_modules_import_by_package(module_name: str) -> None:
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
def test_support_modules_keep_legacy_bare_import(module_name: str) -> None:
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


def test_package_support_modules_share_package_dependencies() -> None:
    utils = importlib.import_module(
        "app_bundle.scripts.pipeline.utils"
    )
    model = importlib.import_module(
        "app_bundle.scripts.pipeline.document_model"
    )
    corpus = importlib.import_module(
        "app_bundle.scripts.pipeline.corpus_manager"
    )
    prompts = importlib.import_module(
        "app_bundle.scripts.pipeline.prompt_manager"
    )
    bibliography = importlib.import_module(
        "app_bundle.scripts.pipeline.bibliography_manager"
    )
    profiles = importlib.import_module(
        "app_bundle.scripts.pipeline.institution_profiles"
    )
    translation = importlib.import_module(
        "app_bundle.scripts.pipeline.document_translation"
    )

    diagnostics = importlib.import_module(
        "app_bundle.scripts.pipeline.diagnostics"
    )
    builder = importlib.import_module(
        "app_bundle.scripts.pipeline.document_builder"
    )
    compliance = importlib.import_module(
        "app_bundle.scripts.pipeline.institution_compliance"
    )
    explainer = importlib.import_module(
        "app_bundle.scripts.pipeline.institution_explainer"
    )
    mindmap = importlib.import_module(
        "app_bundle.scripts.pipeline.mindmap_manager"
    )
    abstracts = importlib.import_module(
        "app_bundle.scripts.pipeline.paper_abstracts"
    )
    project_tools = importlib.import_module(
        "app_bundle.scripts.pipeline.project_tools"
    )
    prompt_lock = importlib.import_module(
        "app_bundle.scripts.pipeline.prompt_lock"
    )

    assert diagnostics.resolve_path is utils.resolve_path
    assert diagnostics.validate_prompt_paths is prompts.validate_prompt_paths

    assert builder.AcademicDocument is model.AcademicDocument
    assert builder.SourceDoc is corpus.SourceDoc
    assert builder.load_prompt_bundle is prompts.load_prompt_bundle

    assert compliance.split_bib_entries is bibliography.split_bib_entries
    assert compliance.now_iso is diagnostics.now_iso
    assert compliance.find_app_bundle is profiles.find_app_bundle

    assert (
        explainer.available_institution_profiles
        is profiles.available_institution_profiles
    )
    assert explainer.find_app_bundle is profiles.find_app_bundle

    assert mindmap.AcademicDocument is model.AcademicDocument
    assert mindmap.load_prompt_bundle is prompts.load_prompt_bundle

    assert abstracts.normalize_language is translation.normalize_language
    assert (
        abstracts.requested_translation_languages
        is translation.requested_translation_languages
    )

    assert project_tools.split_bib_entries is bibliography.split_bib_entries
    assert project_tools.write_json is utils.write_json

    assert prompt_lock.now_iso is diagnostics.now_iso
    assert prompt_lock.prompt_report_for_cfg is prompts.prompt_report_for_cfg
    assert prompt_lock.write_json is utils.write_json


def test_migrated_support_modules_have_no_top_level_bare_local_imports() -> None:
    failures = [
        row
        for path in MIGRATED_FILES
        for row in _top_level_bare_local_imports(path)
    ]

    assert failures == []


def test_remaining_top_level_bare_imports_are_confined_to_entrypoints() -> None:
    failures: dict[str, list[str]] = {}

    for path in sorted(PIPELINE_DIR.glob("*.py")):
        rows = _top_level_bare_local_imports(path)
        if rows:
            failures[path.stem] = rows

    assert failures
    assert "academic_pipeline_rc10" in failures
    assert set(failures) <= ALLOWED_REMAINING_TOP_LEVEL_BARE_IMPORTS


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

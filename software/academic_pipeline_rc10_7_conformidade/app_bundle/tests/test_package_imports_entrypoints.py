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

ENTRYPOINTS = (
    "academic_pipeline_rc10",
    "academic_pipeline_toml_generator_interativo",
    "academic_pipeline_tui",
    "academic_pipeline_gui",
)

HELP_MARKERS = (
    ("academic_pipeline_rc10", "document_model canônico"),
    (
        "academic_pipeline_toml_generator_interativo",
        "Gerador interativo completo",
    ),
    ("academic_pipeline_tui", "Central operacional visual"),
    ("academic_pipeline_gui", "Interface gráfica FGV"),
)


def _run_python(
    *args: str,
    cwd: Path,
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


def _local_module_names() -> set[str]:
    return {
        path.stem
        for path in PIPELINE_DIR.glob("*.py")
        if path.name != "__init__.py"
    }


def _is_package_guard(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "__package__"


def _format_local_import(path: Path, node: ast.AST, module: str) -> str:
    if isinstance(node, ast.ImportFrom):
        return f"{path.name}:{node.lineno}: from {module} import ..."
    return f"{path.name}:{node.lineno}: import {module}"


def _unguarded_bare_local_imports(path: Path) -> list[str]:
    local_modules = _local_module_names()
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    rows: list[str] = []

    def visit(node: ast.AST, *, legacy_guarded: bool = False) -> None:
        if isinstance(node, ast.If) and _is_package_guard(node.test):
            for child in node.body:
                visit(child, legacy_guarded=legacy_guarded)
            for child in node.orelse:
                visit(child, legacy_guarded=True)
            return

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if (
                node.level == 0
                and root in local_modules
                and not legacy_guarded
            ):
                rows.append(_format_local_import(path, node, module))
            return

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in local_modules and not legacy_guarded:
                    rows.append(
                        _format_local_import(path, node, alias.name)
                    )
            return

        for child in ast.iter_child_nodes(node):
            visit(child, legacy_guarded=legacy_guarded)

    visit(tree)
    return rows


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
                rows.append(_format_local_import(path, node, module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in local_modules:
                    rows.append(
                        _format_local_import(path, node, alias.name)
                    )

    return rows


@pytest.mark.parametrize("module_name", ENTRYPOINTS)
def test_historical_entrypoints_import_by_package(module_name: str) -> None:
    proc = _run_python(
        "-c",
        (
            "import importlib; "
            f"m = importlib.import_module("
            f"'app_bundle.scripts.pipeline.{module_name}'); "
            "print(m.__name__); print(m.__file__)"
        ),
        cwd=SOFTWARE_ROOT,
    )

    assert proc.returncode == 0, proc.stderr
    assert f"app_bundle.scripts.pipeline.{module_name}" in proc.stdout
    assert "app_bundle/scripts/pipeline" in proc.stdout


@pytest.mark.parametrize("module_name", ENTRYPOINTS)
def test_historical_entrypoints_keep_legacy_bare_import(
    module_name: str,
) -> None:
    proc = _run_python(
        "-c",
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


@pytest.mark.parametrize(("module_name", "marker"), HELP_MARKERS)
def test_historical_entrypoints_run_as_package_modules(
    module_name: str,
    marker: str,
) -> None:
    proc = _run_python(
        "-m",
        f"app_bundle.scripts.pipeline.{module_name}",
        "--help",
        cwd=SOFTWARE_ROOT,
    )

    assert proc.returncode == 0, proc.stderr
    assert marker in proc.stdout


@pytest.mark.parametrize(("module_name", "marker"), HELP_MARKERS)
def test_historical_entrypoints_keep_direct_script_help(
    module_name: str,
    marker: str,
) -> None:
    proc = _run_python(
        str(PIPELINE_DIR / f"{module_name}.py"),
        "--help",
        cwd=SOFTWARE_ROOT,
    )

    assert proc.returncode == 0, proc.stderr
    assert marker in proc.stdout


def test_entrypoints_have_no_unguarded_bare_local_imports() -> None:
    failures = {
        module_name: rows
        for module_name in ENTRYPOINTS
        if (
            rows := _unguarded_bare_local_imports(
                PIPELINE_DIR / f"{module_name}.py"
            )
        )
    }

    assert failures == {}


def test_pipeline_has_no_top_level_bare_local_imports() -> None:
    failures = {
        path.name: rows
        for path in sorted(PIPELINE_DIR.glob("*.py"))
        if (rows := _top_level_bare_local_imports(path))
    }

    assert failures == {}


def test_rc10_preserves_two_main_definitions_and_wrapper_marker() -> None:
    path = PIPELINE_DIR / "academic_pipeline_rc10.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    mains = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]

    assert len(mains) == 2
    assert "_original_main_before_prisma_artigo_generico_wrapper" in source


def test_rc10_official_package_and_legacy_commands_match() -> None:
    official = _run_python(
        "-m",
        "academic_pipeline",
        "--list-institutions",
        cwd=SOFTWARE_ROOT,
    )
    package_module = _run_python(
        "-m",
        "app_bundle.scripts.pipeline.academic_pipeline_rc10",
        "--list-institutions",
        cwd=SOFTWARE_ROOT,
    )
    legacy = _run_python(
        str(PIPELINE_DIR / "academic_pipeline_rc10.py"),
        "--list-institutions",
        cwd=SOFTWARE_ROOT,
    )

    assert official.returncode == 0, official.stderr
    assert package_module.returncode == 0, package_module.stderr
    assert legacy.returncode == 0, legacy.stderr
    assert official.stdout == package_module.stdout == legacy.stdout

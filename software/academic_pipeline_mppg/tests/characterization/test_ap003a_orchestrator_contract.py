from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PACKAGE_MAIN = ROOT / "academic_pipeline" / "__main__.py"
PYPROJECT = ROOT / "pyproject.toml"
INVENTORY = ROOT / "docs/refactor/academic-pipeline/AP-003/ap003a_orchestrator_inventory.json"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots" / "ap003a"
WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_functions(tree: ast.Module):
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.update(
        {
            "COLUMNS": "120",
            "LINES": "40",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    return env


def _normalize(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    worktree = ROOT.parents[1]
    for source, replacement in (
        (str(ROOT), "<SOFTWARE_ROOT>"),
        (str(worktree), "<WORKTREE_ROOT>"),
        (str(Path(sys.executable)), "<PYTHON>"),
        (str(Path(sys.executable).resolve()), "<PYTHON>"),
    ):
        value = value.replace(source, replacement)
    return "\n".join(line.rstrip() for line in value.split("\n")).strip() + "\n"


def _capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    captured = (
        f"# command: {' '.join(command)}\n"
        f"# returncode: {completed.returncode}\n"
        "# stdout\n"
        f"{completed.stdout}"
        "# stderr\n"
        f"{completed.stderr}"
    )
    return _normalize(captured)


def test_ap003a_phase_gate_is_released_by_ap003f() -> None:
    functions = _top_level_functions(_tree(ORCHESTRATOR))
    mains = [
        node for node in functions
        if node.name == "main"
    ]
    cores = [
        node for node in functions
        if node.name == "_ap003f_pipeline_core"
    ]

    assert len(mains) == 1
    assert len(cores) == 1



def test_ap003a_historical_wrapper_state_is_documented() -> None:
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    wrapper = data["orchestrator"]["wrapper_symbol"]
    functions = _top_level_functions(_tree(ORCHESTRATOR))
    wrappers = [node for node in functions if node.name == WRAPPER_NAME]
    assert wrapper["name"] == WRAPPER_NAME
    assert wrapper["canonical_top_level_definition_count"] == len(wrappers)
    assert isinstance(wrapper["productive_occurrences"], list)


def test_ap003a_package_and_console_entrypoints_remain_declared() -> None:
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    scripts = project.get("scripts", {})
    assert PACKAGE_MAIN.is_file()
    assert project["name"] == "academic-pipeline-mppg"
    assert project["version"] == "0.1.0"
    assert "academic-pipeline" in scripts
    assert str(scripts["academic-pipeline"]).startswith("academic_pipeline")


@pytest.mark.parametrize(
    ("command", "snapshot_name"),
    [
        ([sys.executable, str(ORCHESTRATOR), "--help"], "direct_script_help.txt"),
        ([sys.executable, "-m", "academic_pipeline", "--help"], "package_module_help.txt"),
    ],
    ids=["historical-direct-script", "official-package-module"],
)
def test_ap003a_cli_help_contract(command: list[str], snapshot_name: str) -> None:
    expected = (SNAPSHOT_DIR / snapshot_name).read_text(encoding="utf-8")
    actual = _capture(command)
    assert actual == expected
    assert "# returncode: 0" in actual

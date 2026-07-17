from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PACKAGE_MAIN = ROOT / "academic_pipeline/__main__.py"
PARSER = ROOT / "academic_pipeline/cli_parser.py"
DISPATCH = ROOT / "academic_pipeline/command_dispatch.py"
DOCUMENT = ROOT / "academic_pipeline/document_orchestration.py"
PRISMA = ROOT / "academic_pipeline/prisma_generic_orchestration.py"
MANIFEST = ROOT / "docs/refactor/academic-pipeline/AP-003/ap003g_manifest.json"

DIRECT_SNAPSHOT = ROOT / "tests/characterization/snapshots/ap003a/direct_script_help.txt"
PACKAGE_SNAPSHOT = ROOT / "tests/characterization/snapshots/ap003a/package_module_help.txt"

CORE_NAME = "_ap003f_pipeline_core"
WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"
EXPECTED_HEAD = "7174664e22a941f4a6643d289106f37fa37289b5"
EXPECTED_HASHES = {'orchestrator': 'f385b32fed0445dde90a596440903a7c174e42eac2e1675251ddbd0ce516288f', 'package_main': '31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4', 'parser': 'f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8', 'dispatch': '42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3', 'document': 'c28a6201fcbd40339240fc3eac897c6924b9989810d8107038f445dde78e2c06', 'prisma': 'ff037dff7e83f4a48f607ad65fb31d075cf9a38f96493e472ea46727832b135b'}
EXPECTED_AP003G_PRODUCTION_HASHES = {'orchestrator': '8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977', 'package_main': '31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4', 'parser': 'f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8', 'dispatch': '42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3', 'document': '3f2a3c95e08ccc3c19e3019a225c36fcf532cf4468f75b13c56b7c43bbc88a8e', 'prisma': 'f250487a7787c967a0bad0ac38d5dbe210ff63981078d3c65e1d77655ff5f072'}


def _tree(path: Path) -> ast.Module:
    return ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )


def _calls(node: ast.AST) -> list[str]:
    result = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        result.append(ast.unparse(child.func))
    return result


def _snapshot_stdout(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    payload = content.split("# stdout\n", 1)[1]
    stdout, separator, _stderr = payload.partition("# stderr\n")
    assert separator
    return stdout


def _compact(text: str) -> str:
    return "".join(text.split())


def _run_help(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_final_orchestrator_shape() -> None:
    tree = _tree(ORCHESTRATOR)

    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    ]
    cores = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == CORE_NAME
    ]
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and "".join(ast.unparse(node.test).split())
        == "__name__=='__main__'"
    ]

    assert len(mains) == 1
    assert len(cores) == 1
    assert len(guards) == 1


def test_historical_alias_is_absent_from_production_ast() -> None:
    for path in (ORCHESTRATOR, PRISMA):
        tree = _tree(path)

        names = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == WRAPPER_NAME
        ]

        assert names == []


def test_public_main_core_and_prisma_delegations() -> None:
    orchestrator_tree = _tree(ORCHESTRATOR)
    prisma_tree = _tree(PRISMA)

    public_main = next(
        node
        for node in orchestrator_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )
    core = next(
        node
        for node in orchestrator_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == CORE_NAME
    )

    assert any(
        "_ap003e_entrypoint" in call
        for call in _calls(public_main)
    )
    assert "parse_cli_args" in _calls(core)

    prisma_loaded = [
        node.id
        for node in ast.walk(prisma_tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
    ]
    assert CORE_NAME in prisma_loaded


def test_direct_guard_and_package_entrypoint_call_main() -> None:
    orchestrator_tree = _tree(ORCHESTRATOR)
    package_tree = _tree(PACKAGE_MAIN)

    guard = next(
        node
        for node in orchestrator_tree.body
        if isinstance(node, ast.If)
        and "".join(ast.unparse(node.test).split())
        == "__name__=='__main__'"
    )

    assert "main" in _calls(guard)
    assert "main" in _calls(package_tree)


def test_structural_modules_compile_and_match_frozen_hashes() -> None:
    paths = {
        "orchestrator": ORCHESTRATOR,
        "package_main": PACKAGE_MAIN,
        "parser": PARSER,
        "dispatch": DISPATCH,
        "document": DOCUMENT,
        "prisma": PRISMA,
    }

    with tempfile.TemporaryDirectory(
        prefix="ap003g-contract-pyc-"
    ) as temporary:
        temporary_path = Path(temporary)

        for name, path in paths.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                EXPECTED_HASHES[name]
            )

            py_compile.compile(
                str(path),
                cfile=str(temporary_path / f"{name}.pyc"),
                doraise=True,
            )


def test_three_help_surfaces_remain_equivalent() -> None:
    console = shutil.which("academic-pipeline")
    assert console is not None

    direct = _run_help(
        [
            sys.executable,
            str(ORCHESTRATOR.relative_to(ROOT)),
            "--help",
        ]
    )
    package = _run_help(
        [
            sys.executable,
            "-m",
            "academic_pipeline",
            "--help",
        ]
    )
    command = _run_help([console, "--help"])

    assert _compact(direct) == _compact(
        _snapshot_stdout(DIRECT_SNAPSHOT)
    )
    assert _compact(package) == _compact(
        _snapshot_stdout(PACKAGE_SNAPSHOT)
    )
    assert _compact(command) == _compact(package)


def test_ap003g_manifest_freezes_final_contract() -> None:
    data = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    assert data["phase"] == "AP-003G"
    assert data["status"] in {"validating", "closed"}
    assert data["git"]["head"] == EXPECTED_HEAD
    assert data["production_hashes"] == EXPECTED_AP003G_PRODUCTION_HASHES

    result = data["final_architecture"]
    assert result["top_level_main_count"] == 1
    assert result["internal_core_count"] == 1
    assert result["historical_alias_count"] == 0

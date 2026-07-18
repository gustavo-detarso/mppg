from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PACKAGE_MAIN = ROOT / "academic_pipeline/__main__.py"
PARSER = ROOT / "academic_pipeline/cli_parser.py"
DISPATCH = ROOT / "academic_pipeline/command_dispatch.py"
DOCUMENT = ROOT / "academic_pipeline/document_orchestration.py"
PRISMA = ROOT / "academic_pipeline/prisma_generic_orchestration.py"
DIRECT_SNAPSHOT = (
    ROOT / "tests/characterization/snapshots/ap003a/direct_script_help.txt"
)
PACKAGE_SNAPSHOT = (
    ROOT / "tests/characterization/snapshots/ap003a/package_module_help.txt"
)

WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"
CORE_NAME = "_ap003f_pipeline_core"

EXPECTED_PACKAGE_MAIN_SHA256 = "31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4"
EXPECTED_PARSER_SHA256 = "f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8"
EXPECTED_DISPATCH_SHA256 = "42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3"
EXPECTED_DOCUMENT_SHA256 = 'c28a6201fcbd40339240fc3eac897c6924b9989810d8107038f445dde78e2c06'
EXPECTED_CORE_DUMP_SHA256 = "05ab0bb4c403f3a7322fe674157c54623535e71411d37e7470fd8af2715aba3e"


def _snapshot_stdout(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    payload = content.split("# stdout\n", 1)[1]
    stdout, separator, _stderr = payload.partition("# stderr\n")
    assert separator
    return stdout


def _run_help(*args: str) -> str:
    completed = subprocess.run(
        [sys.executable, *args, "--help"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _compact(text: str) -> str:
    return "".join(text.split())


def _normalized_function_dump(function: ast.FunctionDef) -> str:
    copied = ast.FunctionDef(
        name="<normalized>",
        args=function.args,
        body=function.body,
        decorator_list=function.decorator_list,
        returns=function.returns,
        type_comment=function.type_comment,
    )
    ast.fix_missing_locations(copied)
    return ast.dump(
        copied,
        annotate_fields=True,
        include_attributes=False,
    )


def test_orchestrator_has_one_public_main_and_one_core() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))

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

    assert len(mains) == 1
    assert len(cores) == 1


def test_historical_alias_assignment_was_removed() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))

    assignments = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == WRAPPER_NAME
            for target in node.targets
        ):
            assignments.append(node)

    assert assignments == []


def test_prisma_fallback_uses_new_core_name() -> None:
    source = PRISMA.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PRISMA))

    loaded_names = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
    ]

    assert WRAPPER_NAME not in loaded_names
    assert CORE_NAME in loaded_names


def test_core_body_matches_pre_unification_first_main() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))

    core = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == CORE_NAME
    )

    actual = hashlib.sha256(
        _normalized_function_dump(core).encode("utf-8")
    ).hexdigest()

    assert actual == EXPECTED_CORE_DUMP_SHA256


def test_public_main_and_direct_guard_keep_entry_contract() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))

    public_main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    main_calls = [
        ast.unparse(node.func)
        for node in ast.walk(public_main)
        if isinstance(node, ast.Call)
    ]
    assert any(
        "_ap003e_entrypoint" in call
        for call in main_calls
    )

    guards = []
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        if "".join(ast.unparse(node.test).split()) == (
            "__name__=='__main__'"
        ):
            guards.append(node)

    assert len(guards) == 1

    guard_calls = [
        ast.unparse(node.func)
        for node in ast.walk(guards[0])
        if isinstance(node, ast.Call)
    ]
    assert "main" in guard_calls


def test_previous_phase_files_except_prisma_are_byte_identical() -> None:
    assert hashlib.sha256(PACKAGE_MAIN.read_bytes()).hexdigest() == (
        EXPECTED_PACKAGE_MAIN_SHA256
    )
    assert hashlib.sha256(PARSER.read_bytes()).hexdigest() == (
        EXPECTED_PARSER_SHA256
    )
    assert hashlib.sha256(DISPATCH.read_bytes()).hexdigest() == (
        EXPECTED_DISPATCH_SHA256
    )
    assert hashlib.sha256(DOCUMENT.read_bytes()).hexdigest() == (
        EXPECTED_DOCUMENT_SHA256
    )


def test_help_surfaces_remain_semantically_identical() -> None:
    direct = _run_help(
        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    )
    package = _run_help("-m", "academic_pipeline")

    assert _compact(direct) == _compact(
        _snapshot_stdout(DIRECT_SNAPSHOT)
    )
    assert _compact(package) == _compact(
        _snapshot_stdout(PACKAGE_SNAPSHOT)
    )

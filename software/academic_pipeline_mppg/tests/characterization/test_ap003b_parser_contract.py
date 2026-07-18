from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from academic_pipeline.cli_parser import build_parser, parse_args


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
DIRECT_SNAPSHOT = ROOT / "tests/characterization/snapshots/ap003a/direct_script_help.txt"
PACKAGE_SNAPSHOT = ROOT / "tests/characterization/snapshots/ap003a/package_module_help.txt"
WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"


def _call_name(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


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


def test_build_parser_preserves_argument_surface() -> None:
    parser = build_parser(pipeline_version="test-version")
    user_actions = [action for action in parser._actions if action.dest != "help"]
    assert len(user_actions) == 62
    assert parser.description == (
        "academic_pipeline test-version — document_model canônico"
    )


def test_parse_args_preserves_representative_defaults_and_values() -> None:
    args = parse_args(
        [
            "--config",
            "projeto.toml",
            "--doctor",
            "--prisma-curadoria-max-incluir",
            "7",
            "--project-type",
            "paper_prisma",
        ],
        pipeline_version="test-version",
    )
    assert args.config == "projeto.toml"
    assert args.doctor is True
    assert args.prisma_curadoria_max_incluir == 7
    assert args.project_type == "paper_prisma"
    assert args.no_clear is False
    assert args.output_dir == ""


def test_ap003f_core_delegates_parser_without_embedded_argparse_surface() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))
    cores = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_ap003f_pipeline_core"
    ]
    assert len(cores) == 1

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(cores[0])
        if isinstance(node, ast.Call)
    ]

    assert sum(
        "parse_cli_args" in call
        for call in calls
    ) == 1
    assert not any(
        call == "argparse.ArgumentParser"
        or call.endswith(".add_argument")
        or call.endswith(".parse_args")
        for call in calls
    )



def test_ap003f_replaces_two_mains_and_alias_with_public_main_and_core() -> None:
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
        and node.name == "_ap003f_pipeline_core"
    ]
    aliases = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id
            == "_original_main_before_prisma_artigo_generico_wrapper"
            for target in node.targets
        )
    ]

    assert len(mains) == 1
    assert len(cores) == 1
    assert aliases == []



def _snapshot_stdout(path: Path) -> str:
    # Extrai somente stdout do formato de snapshot criado na AP-003A.
    content = path.read_text(encoding="utf-8")
    stdout_marker = "# stdout\n"
    stderr_marker = "# stderr\n"

    if stdout_marker not in content or stderr_marker not in content:
        raise AssertionError(
            f"Snapshot fora do formato AP-003A: {path}"
        )

    payload = content.split(stdout_marker, 1)[1]
    stdout, separator, _stderr = payload.partition(stderr_marker)

    if not separator:
        raise AssertionError(
            f"Seção stderr ausente no snapshot AP-003A: {path}"
        )

    return stdout


def _normalize_help(text: str) -> str:
    # O argparse pode quebrar linhas inclusive dentro de nomes longos,
    # como "--explain-profile". Para o snapshot semântico, comparamos
    # todos os caracteres não brancos, preservando conteúdo e ordem.
    return "".join(text.split())


def test_direct_script_help_snapshot_is_unchanged() -> None:
    actual = _run_help("app_bundle/scripts/pipeline/academic_pipeline_rc10.py")
    expected = _snapshot_stdout(DIRECT_SNAPSHOT)
    assert _normalize_help(actual) == _normalize_help(expected)


def test_package_module_help_snapshot_is_unchanged() -> None:
    actual = _run_help("-m", "academic_pipeline")
    expected = _snapshot_stdout(PACKAGE_SNAPSHOT)
    assert _normalize_help(actual) == _normalize_help(expected)

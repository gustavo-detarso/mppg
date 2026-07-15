from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import academic_pipeline.command_dispatch as command_dispatch


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PARSER = ROOT / "academic_pipeline/cli_parser.py"
DIRECT_SNAPSHOT = ROOT / "tests/characterization/snapshots/ap003a/direct_script_help.txt"
PACKAGE_SNAPSHOT = ROOT / "tests/characterization/snapshots/ap003a/package_module_help.txt"
WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"
EXPECTED_PARSER_SHA256 = "f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8"
EXPECTED_STAGES = ['dispatch_stage_001', 'dispatch_stage_002', 'dispatch_stage_003', 'dispatch_stage_004', 'dispatch_stage_005', 'dispatch_stage_006', 'dispatch_stage_007', 'dispatch_stage_008', 'dispatch_stage_009', 'dispatch_stage_010', 'dispatch_stage_011', 'dispatch_stage_012', 'dispatch_stage_013', 'dispatch_stage_014', 'dispatch_stage_015', 'dispatch_stage_016', 'dispatch_stage_017', 'dispatch_stage_018', 'dispatch_stage_019']
STAGE_METADATA = [{'name': 'dispatch_stage_001', 'line': 914, 'end_line': 920, 'args_attributes': ['gui']}, {'name': 'dispatch_stage_002', 'line': 922, 'end_line': 928, 'args_attributes': ['no_clear', 'tui']}, {'name': 'dispatch_stage_003', 'line': 930, 'end_line': 937, 'args_attributes': ['list_toml_profiles']}, {'name': 'dispatch_stage_004', 'line': 939, 'end_line': 946, 'args_attributes': ['init_toml', 'no_clear', 'toml_profile']}, {'name': 'dispatch_stage_005', 'line': 948, 'end_line': 950, 'args_attributes': ['list_institutions']}, {'name': 'dispatch_stage_006', 'line': 952, 'end_line': 967, 'args_attributes': ['config', 'list_layouts']}, {'name': 'dispatch_stage_007', 'line': 969, 'end_line': 971, 'args_attributes': ['explain_profile']}, {'name': 'dispatch_stage_008', 'line': 973, 'end_line': 978, 'args_attributes': ['config', 'show_prompts']}, {'name': 'dispatch_stage_009', 'line': 980, 'end_line': 990, 'args_attributes': ['base_dir', 'init_project', 'institution', 'overwrite_project', 'project_type']}, {'name': 'dispatch_stage_010', 'line': 992, 'end_line': 1009, 'args_attributes': ['input_dir', 'input_zip', 'make_doi_manifest', 'output']}, {'name': 'dispatch_stage_011', 'line': 1011, 'end_line': 1017, 'args_attributes': ['inspect_bib']}, {'name': 'dispatch_stage_012', 'line': 1045, 'end_line': 1046, 'args_attributes': ['somente_mapa_mental', 'somente_renderizar']}, {'name': 'dispatch_stage_013', 'line': 1047, 'end_line': 1048, 'args_attributes': ['forcar_regeneracao_mapa_mental', 'reusar_mapa_mental']}, {'name': 'dispatch_stage_014', 'line': 1050, 'end_line': 1060, 'args_attributes': ['write_prompt_lock']}, {'name': 'dispatch_stage_015', 'line': 1062, 'end_line': 1074, 'args_attributes': ['bib', 'check_institution_compliance', 'docx', 'org', 'pdf']}, {'name': 'dispatch_stage_016', 'line': 1076, 'end_line': 1082, 'args_attributes': ['doctor']}, {'name': 'dispatch_stage_017', 'line': 1084, 'end_line': 1091, 'args_attributes': ['check_config']}, {'name': 'dispatch_stage_018', 'line': 1093, 'end_line': 1094, 'args_attributes': ['recompile']}, {'name': 'dispatch_stage_019', 'line': 1099, 'end_line': 1139, 'args_attributes': ['prisma_importar_triagem']}]


def _call_name(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


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


def test_dispatch_module_exports_all_extracted_stages() -> None:
    assert command_dispatch.__all__[0] == "DispatchResult"
    assert command_dispatch.__all__[1:] == EXPECTED_STAGES
    for name in EXPECTED_STAGES:
        assert callable(getattr(command_dispatch, name))


def test_extracted_stages_have_result_contract() -> None:
    source = Path(command_dispatch.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in EXPECTED_STAGES:
        function = functions[name]
        assert [arg.arg for arg in function.args.args] == ["args", "runtime"]
        assert isinstance(function.body[-1], ast.Return)


def test_ap003f_core_delegates_each_dispatch_stage_once() -> None:
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

    for index, _stage in enumerate(EXPECTED_STAGES, start=1):
        assert calls.count(
            f"_ap003c_dispatch_{index:03d}"
        ) == 1



def test_ap003f_dispatch_contract_accepts_one_main_and_one_core() -> None:
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



def test_ap003b_parser_is_byte_identical() -> None:
    assert hashlib.sha256(PARSER.read_bytes()).hexdigest() == EXPECTED_PARSER_SHA256


def test_help_surfaces_are_semantically_identical() -> None:
    direct = _run_help("app_bundle/scripts/pipeline/academic_pipeline_rc10.py")
    package = _run_help("-m", "academic_pipeline")
    assert _compact(direct) == _compact(_snapshot_stdout(DIRECT_SNAPSHOT))
    assert _compact(package) == _compact(_snapshot_stdout(PACKAGE_SNAPSHOT))

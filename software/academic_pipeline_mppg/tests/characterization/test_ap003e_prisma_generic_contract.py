from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

import academic_pipeline.prisma_generic_orchestration as prisma_generic


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PARSER = ROOT / "academic_pipeline/cli_parser.py"
DISPATCH = ROOT / "academic_pipeline/command_dispatch.py"
DOCUMENT = ROOT / "academic_pipeline/document_orchestration.py"
DIRECT_SNAPSHOT = (
    ROOT / "tests/characterization/snapshots/ap003a/direct_script_help.txt"
)
PACKAGE_SNAPSHOT = (
    ROOT / "tests/characterization/snapshots/ap003a/package_module_help.txt"
)
WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"
EXPECTED_HELPERS = [{'occurrence': 'stage#1', 'name': 'stage', 'alias': '_ap003e_impl_stage_1', 'implementation': 'stage_impl_001'}, {'occurrence': '_json_or_none#1', 'name': '_json_or_none', 'alias': '_ap003e_impl__json_or_none_1', 'implementation': '_json_or_none_impl_001'}, {'occurrence': 'make_client#1', 'name': 'make_client', 'alias': '_ap003e_impl_make_client_1', 'implementation': 'make_client_impl_001'}, {'occurrence': '_section#1', 'name': '_section', 'alias': '_ap003e_impl__section_1', 'implementation': '_section_impl_001'}, {'occurrence': 'research_output_paths#1', 'name': 'research_output_paths', 'alias': '_ap003e_impl_research_output_paths_1', 'implementation': 'research_output_paths_impl_001'}, {'occurrence': 'render_external_prisma_outputs#1', 'name': 'render_external_prisma_outputs', 'alias': '_ap003e_impl_render_external_prisma_outputs_1', 'implementation': 'render_external_prisma_outputs_impl_001'}, {'occurrence': '_prisma_curadoria_default_config#1', 'name': '_prisma_curadoria_default_config', 'alias': '_ap003e_impl__prisma_curadoria_default_config_1', 'implementation': '_prisma_curadoria_default_config_impl_001'}, {'occurrence': '_prisma_curadoria_default_out_dir#1', 'name': '_prisma_curadoria_default_out_dir', 'alias': '_ap003e_impl__prisma_curadoria_default_out_dir_1', 'implementation': '_prisma_curadoria_default_out_dir_impl_001'}, {'occurrence': '_prisma_curadoria_default_prompt#1', 'name': '_prisma_curadoria_default_prompt', 'alias': '_ap003e_impl__prisma_curadoria_default_prompt_1', 'implementation': '_prisma_curadoria_default_prompt_impl_001'}, {'occurrence': '_prisma_curadoria_script_path#1', 'name': '_prisma_curadoria_script_path', 'alias': '_ap003e_impl__prisma_curadoria_script_path_1', 'implementation': '_prisma_curadoria_script_path_impl_001'}, {'occurrence': '_prisma_curadoria_arg#1', 'name': '_prisma_curadoria_arg', 'alias': '_ap003e_impl__prisma_curadoria_arg_1', 'implementation': '_prisma_curadoria_arg_impl_001'}, {'occurrence': '_prisma_curadoria_config_from_args#1', 'name': '_prisma_curadoria_config_from_args', 'alias': '_ap003e_impl__prisma_curadoria_config_from_args_1', 'implementation': '_prisma_curadoria_config_from_args_impl_001'}, {'occurrence': '_prisma_curadoria_out_from_args#1', 'name': '_prisma_curadoria_out_from_args', 'alias': '_ap003e_impl__prisma_curadoria_out_from_args_1', 'implementation': '_prisma_curadoria_out_from_args_impl_001'}, {'occurrence': '_prisma_curadoria_prompt_from_args#1', 'name': '_prisma_curadoria_prompt_from_args', 'alias': '_ap003e_impl__prisma_curadoria_prompt_from_args_1', 'implementation': '_prisma_curadoria_prompt_from_args_impl_001'}, {'occurrence': '_prisma_curadoria_input_from_args#1', 'name': '_prisma_curadoria_input_from_args', 'alias': '_ap003e_impl__prisma_curadoria_input_from_args_1', 'implementation': '_prisma_curadoria_input_from_args_impl_001'}, {'occurrence': '_prisma_curadoria_run_command#1', 'name': '_prisma_curadoria_run_command', 'alias': '_ap003e_impl__prisma_curadoria_run_command_1', 'implementation': '_prisma_curadoria_run_command_impl_001'}, {'occurrence': '_prisma_curadoria_build_cmd#1', 'name': '_prisma_curadoria_build_cmd', 'alias': '_ap003e_impl__prisma_curadoria_build_cmd_1', 'implementation': '_prisma_curadoria_build_cmd_impl_001'}, {'occurrence': '_prisma_curadoria_run_ia#1', 'name': '_prisma_curadoria_run_ia', 'alias': '_ap003e_impl__prisma_curadoria_run_ia_1', 'implementation': '_prisma_curadoria_run_ia_impl_001'}, {'occurrence': '_prisma_curadoria_reexportar_xlsx#1', 'name': '_prisma_curadoria_reexportar_xlsx', 'alias': '_ap003e_impl__prisma_curadoria_reexportar_xlsx_1', 'implementation': '_prisma_curadoria_reexportar_xlsx_impl_001'}, {'occurrence': '_prisma_curadoria_pipeline_supports_flag#1', 'name': '_prisma_curadoria_pipeline_supports_flag', 'alias': '_ap003e_impl__prisma_curadoria_pipeline_supports_flag_1', 'implementation': '_prisma_curadoria_pipeline_supports_flag_impl_001'}, {'occurrence': '_prisma_curadoria_importar_no_pipeline#1', 'name': '_prisma_curadoria_importar_no_pipeline', 'alias': '_ap003e_impl__prisma_curadoria_importar_no_pipeline_1', 'implementation': '_prisma_curadoria_importar_no_pipeline_impl_001'}, {'occurrence': '_prisma_curadoria_fluxo_completo#1', 'name': '_prisma_curadoria_fluxo_completo', 'alias': '_ap003e_impl__prisma_curadoria_fluxo_completo_1', 'implementation': '_prisma_curadoria_fluxo_completo_impl_001'}, {'occurrence': '_prisma_curadoria_mostrar_caminhos#1', 'name': '_prisma_curadoria_mostrar_caminhos', 'alias': '_ap003e_impl__prisma_curadoria_mostrar_caminhos_1', 'implementation': '_prisma_curadoria_mostrar_caminhos_impl_001'}, {'occurrence': '_prisma_curadoria_menu#1', 'name': '_prisma_curadoria_menu', 'alias': '_ap003e_impl__prisma_curadoria_menu_1', 'implementation': '_prisma_curadoria_menu_impl_001'}, {'occurrence': '_prisma_curadoria_dispatch#1', 'name': '_prisma_curadoria_dispatch', 'alias': '_ap003e_impl__prisma_curadoria_dispatch_1', 'implementation': '_prisma_curadoria_dispatch_impl_001'}, {'occurrence': '_prisma_artigo_generico_get_arg#1', 'name': '_prisma_artigo_generico_get_arg', 'alias': '_ap003e_impl__prisma_artigo_generico_get_arg_1', 'implementation': '_prisma_artigo_generico_get_arg_impl_001'}, {'occurrence': '_prisma_artigo_generico_strip#1', 'name': '_prisma_artigo_generico_strip', 'alias': '_ap003e_impl__prisma_artigo_generico_strip_1', 'implementation': '_prisma_artigo_generico_strip_impl_001'}, {'occurrence': '_prisma_artigo_generico_out_dir#1', 'name': '_prisma_artigo_generico_out_dir', 'alias': '_ap003e_impl__prisma_artigo_generico_out_dir_1', 'implementation': '_prisma_artigo_generico_out_dir_impl_001'}, {'occurrence': '_prisma_artigo_generico_run_export#1', 'name': '_prisma_artigo_generico_run_export', 'alias': '_ap003e_impl__prisma_artigo_generico_run_export_1', 'implementation': '_prisma_artigo_generico_run_export_impl_001'}, {'occurrence': '_prisma_artigo_generico_run_freeze#1', 'name': '_prisma_artigo_generico_run_freeze', 'alias': '_ap003e_impl__prisma_artigo_generico_run_freeze_1', 'implementation': '_prisma_artigo_generico_run_freeze_impl_001'}]
EXPECTED_STAGES = ['run_prisma_stage_001', 'run_prisma_stage_002', 'run_prisma_stage_003', 'run_prisma_stage_004', 'run_prisma_stage_005', 'run_prisma_stage_006', 'run_prisma_stage_007', 'run_prisma_stage_008']
EXPECTED_PARSER_SHA256 = "f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8"
EXPECTED_DISPATCH_SHA256 = "42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3"
EXPECTED_DOCUMENT_SHA256 = 'c28a6201fcbd40339240fc3eac897c6924b9989810d8107038f445dde78e2c06'


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


def test_prisma_module_exports_selected_contract() -> None:
    exported = prisma_generic.__all__

    assert exported[0] == "PrismaStageResult"
    assert "run_prisma_generic_entrypoint" in exported

    expected = [
        item["implementation"]
        for item in EXPECTED_HELPERS
    ] + EXPECTED_STAGES

    for name in expected:
        assert name in exported
        assert callable(getattr(prisma_generic, name))


def test_historical_helpers_have_expected_thin_wrappers() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    expected_by_name = Counter(
        item["name"] for item in EXPECTED_HELPERS
    )

    for helper_name, expected_count in expected_by_name.items():
        matches = [
            node for node in functions
            if node.name == helper_name
        ]
        wrappers = []

        for node in matches:
            calls = [
                ast.unparse(child.func)
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
            ]
            if any(
                call.startswith("_ap003e_impl_")
                for call in calls
            ):
                wrappers.append(node)

        assert len(wrappers) == expected_count


def test_ap003f_core_delegates_each_prisma_stage_once() -> None:
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
            f"_ap003e_stage_{index:03d}"
        ) == 1



def test_public_main_is_a_thin_prisma_entrypoint_wrapper() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))
    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    ]
    assert len(mains) == 1

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(mains[0])
        if isinstance(node, ast.Call)
    ]
    assert calls.count("_ap003e_entrypoint") == 1



def test_ap003f_prisma_contract_accepts_unified_main_and_removed_alias() -> None:
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



def test_previous_phase_modules_are_byte_identical() -> None:
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

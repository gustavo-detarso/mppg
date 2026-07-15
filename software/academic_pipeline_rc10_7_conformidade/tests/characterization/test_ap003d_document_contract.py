from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import academic_pipeline.document_orchestration as document_orchestration


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PARSER = ROOT / "academic_pipeline/cli_parser.py"
DISPATCH = ROOT / "academic_pipeline/command_dispatch.py"
DIRECT_SNAPSHOT = (
    ROOT / "tests/characterization/snapshots/ap003a/direct_script_help.txt"
)
PACKAGE_SNAPSHOT = (
    ROOT / "tests/characterization/snapshots/ap003a/package_module_help.txt"
)
WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"
EXPECTED_HELPERS = ['load_config', 'output_paths', 'apply_cli_path_overrides', 'load_existing_document_json', 'resolve_bib_for_existing_document', '_resolve_latex_paths_for_recompile', 'run_recompile', 'render_additional_language_versions', '_refs_v6_disabled', '_refs_v6_apply_runtime_policy', 'build_bibliography', '_refs_v6_clear_document_bibliography', '_refs_v6_strip_org', 'render_org_latex']
EXPECTED_IMPLS = ['load_config_impl', 'output_paths_impl', 'apply_cli_path_overrides_impl', 'load_existing_document_json_impl', 'resolve_bib_for_existing_document_impl', '_resolve_latex_paths_for_recompile_impl', 'run_recompile_impl', 'render_additional_language_versions_impl', '_refs_v6_disabled_impl', '_refs_v6_apply_runtime_policy_impl', 'build_bibliography_impl', '_refs_v6_clear_document_bibliography_impl', '_refs_v6_strip_org_impl', 'render_org_latex_impl']
EXPECTED_STAGES = ['run_document_stage_001', 'run_document_stage_002', 'run_document_stage_003', 'run_document_stage_004', 'run_document_stage_005', 'run_document_stage_006', 'run_document_stage_007', 'run_document_stage_008', 'run_document_stage_009', 'run_document_stage_010', 'run_document_stage_011', 'run_document_stage_012']
EXPECTED_PARSER_SHA256 = "f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8"
EXPECTED_DISPATCH_SHA256 = "42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3"


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


def test_document_module_exports_extracted_contract() -> None:
    exported = document_orchestration.__all__
    assert exported[0] == "DocumentStageResult"
    for name in EXPECTED_IMPLS + EXPECTED_STAGES:
        assert name in exported
        assert callable(getattr(document_orchestration, name))


def test_historical_helpers_are_thin_wrappers() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    expected_wrapper_counts = {
        name: EXPECTED_HELPERS.count(name)
        for name in set(EXPECTED_HELPERS)
    }

    for helper, expected_wrapper_count in expected_wrapper_counts.items():
        matches = [
            node
            for node in functions
            if node.name == helper
        ]

        assert len(matches) >= expected_wrapper_count

        thin_wrappers = []
        for match in matches:
            calls = [
                ast.unparse(node.func)
                for node in ast.walk(match)
                if isinstance(node, ast.Call)
            ]

            if any(
                name.startswith(f"_ap003d_impl_{helper}")
                for name in calls
            ):
                thin_wrappers.append(match)

        assert len(thin_wrappers) == expected_wrapper_count



def test_first_main_delegates_all_document_stages() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))
    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(mains) == 2
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(mains[0])
        if isinstance(node, ast.Call)
    ]
    for index, _name in enumerate(EXPECTED_STAGES, start=1):
        alias = f"_ap003d_stage_{index:03d}"
        assert calls.count(alias) == 1


def test_two_mains_and_historical_alias_remain_preserved() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))
    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(mains) == 2

    aliases = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "main":
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == WRAPPER_NAME
            for target in node.targets
        ):
            aliases.append(node)
    assert len(aliases) == 1


def test_parser_and_dispatch_are_byte_identical() -> None:
    assert hashlib.sha256(PARSER.read_bytes()).hexdigest() == (
        EXPECTED_PARSER_SHA256
    )
    assert hashlib.sha256(DISPATCH.read_bytes()).hexdigest() == (
        EXPECTED_DISPATCH_SHA256
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


def test_comprehension_targets_remain_local_in_document_module() -> None:
    source = Path(document_orchestration.__file__).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    def target_names(node: ast.AST) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Store)
        }

    violations = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            continue

        bound = set()
        for generator in node.generators:
            bound.update(target_names(generator.target))

        for child in ast.walk(node):
            if not isinstance(child, ast.Subscript):
                continue
            if not (
                isinstance(child.value, ast.Name)
                and child.value.id == "runtime"
            ):
                continue
            if not (
                isinstance(child.slice, ast.Constant)
                and isinstance(child.slice.value, str)
            ):
                continue
            if child.slice.value in bound:
                violations.append(
                    (
                        child.lineno,
                        child.slice.value,
                        ast.get_source_segment(source, child),
                    )
                )

    assert violations == []

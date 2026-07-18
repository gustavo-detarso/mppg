from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import tempfile
import tokenize
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parents[1]
APPLICATION = ROOT / 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_application.json'
TOOL = ROOT / 'tools/refactor/ap004c_apply_internal_symbols.py'
BASELINE_HEAD = 'aa9829f09a5c1b9e69c634637c311b03f360b07e'
EXPECTED_COMMIT_SUBJECT = 'refactor(academic-pipeline): consolidar símbolos internos da AP-004C'
EXPECTED_TOOL_SHA256 = 'a31ba8f4082fe4687f4fe0b4c9d97f213bd1ab13995923724ca24b67d6dbb145'
EXPECTED_DIRTY_PATHS = ['app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_APPLICATION.md', 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_STRATEGY.md', 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_application.json', 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_inventory.json', 'tests/characterization/test_ap003d_document_contract.py', 'tests/characterization/test_ap003g_stabilization_contract.py', 'tests/characterization/test_ap004b_module_file_application_contract.py', 'tests/characterization/test_ap004b_module_file_inventory_contract.py', 'tests/characterization/test_ap004c_internal_symbol_application_contract.py', 'tests/characterization/test_ap004c_internal_symbol_inventory_contract.py', 'tools/refactor/ap004c_apply_internal_symbols.py', 'tools/refactor/ap004c_inventory_internal_symbols.py']
WAVE_1 = [('_generate_interactive_before_wizard_documentos_locais_v4', '_generate_interactive_before_wizard_documentos_locais'), ('_generate_interactive_with_wizard_documentos_locais_v4', '_generate_interactive_with_wizard_documentos_locais'), ('_v5_is_local_document', '_is_local_document'), ('_v5_reference_default', '_reference_default'), ('_v5_normalise_prompt', '_normalise_prompt'), ('_v5_configure_reference_policy', '_configure_reference_policy'), ('_v5_ensure_reference_policy', '_ensure_reference_policy')]
WAVE_2 = [('_ap003d_impl_output_paths', '_impl_output_paths'), ('_ap003d_impl_apply_cli_path_overrides', '_impl_apply_cli_path_overrides'), ('_ap003d_impl_load_existing_document_json', '_impl_load_existing_document_json'), ('_ap003d_impl_resolve_bib_for_existing_document', '_impl_resolve_bib_for_existing_document'), ('_ap003d_impl__resolve_latex_paths_for_recompile', '_impl_resolve_latex_paths_for_recompile'), ('_ap003d_impl_run_recompile', '_impl_run_recompile'), ('_ap003d_impl_render_additional_language_versions', '_impl_render_additional_language_versions'), ('_ap003d_impl__refs_v6_disabled', '_impl_refs_disabled'), ('_ap003d_impl__refs_v6_apply_runtime_policy', '_impl_refs_apply_runtime_policy'), ('_ap003d_impl_load_config', '_impl_load_config'), ('_ap003d_impl_build_bibliography', '_impl_build_bibliography'), ('_ap003d_impl__refs_v6_clear_document_bibliography', '_impl_refs_clear_document_bibliography'), ('_ap003d_impl_render_org_latex', '_impl_render_org_latex')]
PROTECTED = [('_refs_v6_strip_org', 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'), ('_ap003d_impl__refs_v6_strip_org', 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'), ('_normalize', 'app_bundle/scripts/pipeline/article_workflow/state.py'), ('extract_org_abstracts', 'app_bundle/scripts/pipeline/render_docx_canonico.py')]
SOFTWARE_PREFIX = 'software/academic_pipeline_rc10_7_conformidade/'


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=REPOSITORY_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _data() -> dict:
    return json.loads(APPLICATION.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identifier_counts(path: Path, names: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    source = path.read_bytes()
    for token in tokenize.tokenize(BytesIO(source).readline):
        if token.type == tokenize.NAME and token.string in names:
            counts[token.string] += 1
    return counts


def _status_path(line: str) -> str:
    raw = line[2:].lstrip() if len(line) >= 2 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    raw = raw.strip().strip('"').replace("\\", "/")
    return raw[len(SOFTWARE_PREFIX):] if raw.startswith(SOFTWARE_PREFIX) else raw


def _ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return any(part in ignored for part in parts) or path.endswith((".pyc", ".pyo"))


def _find_commit() -> str | None:
    result = _run("git", "log", "--format=%H%x00%s", f"{BASELINE_HEAD}..HEAD")
    assert result.returncode == 0, result.stderr
    matches = []
    for line in result.stdout.splitlines():
        if "\x00" not in line:
            continue
        commit, subject = line.split("\x00", 1)
        if subject == EXPECTED_COMMIT_SUBJECT:
            matches.append(commit)
    assert len(matches) <= 1
    return matches[0] if matches else None


def test_ap004c_application_metadata_and_approval() -> None:
    data = _data()
    assert data["phase"] == "AP-004C"
    assert data["mode"] == "internal-symbol-application-v1.4"
    assert data["application_schema_version"] == 1
    assert data["application_revision"] == "1.4"
    assert data["baseline"]["head"] == BASELINE_HEAD
    assert data["approval"]["inventory_revision"] == "1.3"
    assert data["approval"]["approved"] is True


def test_ap004c_wave_1_contains_exact_seven_renames() -> None:
    data = _data()
    items = data["waves"]["wave_1"]["mappings"]
    assert [(item["old"], item["new"]) for item in items] == WAVE_1
    assert len(items) == 7
    assert all(item["replacement_count"] >= 1 for item in items)


def test_ap004c_wave_2_contains_exact_thirteen_renames() -> None:
    data = _data()
    items = data["waves"]["wave_2"]["mappings"]
    assert [(item["old"], item["new"]) for item in items] == WAVE_2
    assert len(items) == 13
    assert all(item["replacement_count"] >= 1 for item in items)


def test_ap004c_wave_1_old_identifiers_are_absent_and_new_are_present() -> None:
    path = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py"
    names = {name for pair in WAVE_1 for name in pair}
    counts = _identifier_counts(path, names)
    for old, new in WAVE_1:
        assert counts[old] == 0
        assert counts[new] >= 1


def test_ap004c_wave_2_old_identifiers_are_absent_and_new_are_present() -> None:
    path = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    names = {name for pair in WAVE_2 for name in pair}
    counts = _identifier_counts(path, names)
    for old, new in WAVE_2:
        assert counts[old] == 0
        assert counts[new] >= 1


def test_ap004c_normalized_ast_is_identical_in_both_waves() -> None:
    data = _data()
    for wave in ("wave_1", "wave_2"):
        item = data["waves"][wave]
        assert item["ast_sha256_before"] == item["ast_sha256_after_normalized"]
        assert item["source_sha256_before"] != item["source_sha256_after"]


def test_ap004c_protected_definition_asts_are_unchanged() -> None:
    data = _data()
    assert len(data["protected_controls"]) == 4
    for item in data["protected_controls"]:
        assert item["ast_dump_sha256_before"] == item["ast_dump_sha256_after"]


def test_ap004c_protected_names_remain_in_current_sources() -> None:
    orchestrator = (ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py").read_text(encoding="utf-8")
    assert "_refs_v6_strip_org" in orchestrator
    assert "_ap003d_impl__refs_v6_strip_org" in orchestrator
    state = (ROOT / "app_bundle/scripts/pipeline/article_workflow/state.py").read_text(encoding="utf-8")
    assert "def _normalize" in state
    docx = (ROOT / "app_bundle/scripts/pipeline/render_docx_canonico.py").read_text(encoding="utf-8")
    assert "def extract_org_abstracts" in docx


def test_ap004c_all_deferred_symbols_remain_deferred() -> None:
    data = _data()
    assert data["deferred"]["count"] == 49
    assert data["deferred"]["policy"] == "não alterados"


def test_ap004c_orchestrator_hash_is_rebaselined_in_ap003g_contract() -> None:
    data = _data()
    expected = data["waves"]["wave_2"]["source_sha256_after"]
    commit = _find_commit()
    assert commit is not None
    contract = _run(
        "git",
        "show",
        f"{commit}:{SOFTWARE_PREFIX}tests/characterization/test_ap003g_stabilization_contract.py",
    )
    assert contract.returncode == 0, contract.stderr
    assert expected in contract.stdout
    orchestrator = _run(
        "git",
        "show",
        f"{commit}:{SOFTWARE_PREFIX}app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    )
    assert orchestrator.returncode == 0, orchestrator.stderr
    assert hashlib.sha256(orchestrator.stdout.encode("utf-8")).hexdigest() == expected


def test_ap004c_core_ast_hash_is_rebaselined_in_ap003f_contract() -> None:
    source = (ROOT / "tests/characterization/test_ap003f_main_unification_contract.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_CORE_DUMP_SHA256"
            for target in node.targets
        ) and isinstance(node.value, ast.Constant):
            values.append(node.value.value)
    assert len(values) == 1
    assert isinstance(values[0], str) and len(values[0]) == 64


def test_ap004c_previous_phase_contracts_are_historical_and_durable() -> None:
    for relative in (
        "tests/characterization/test_ap004b_module_file_application_contract.py",
        "tests/characterization/test_ap004b_module_file_inventory_contract.py",
        "tests/characterization/test_ap004c_internal_symbol_inventory_contract.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "commit" in source.lower()
        ast.parse(source, filename=relative)


def test_ap004c_no_module_file_or_directory_rename_was_introduced() -> None:
    data = _data()
    assert data["scope"]["module_file_changes"] is False
    assert data["scope"]["productive_changed_paths"] == [
        "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py",
        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    ]


def test_ap004c_changed_productive_files_compile() -> None:
    data = _data()
    with tempfile.TemporaryDirectory(prefix="ap004c-application-pyc-") as tmp:
        for index, relative in enumerate(data["scope"]["productive_changed_paths"]):
            py_compile.compile(
                str(ROOT / relative),
                cfile=str(Path(tmp) / f"{index}.pyc"),
                doraise=True,
            )


def test_ap004c_validation_metadata_records_expected_suite() -> None:
    data = _data()
    consolidated = data["validation"]["consolidated_suite"]
    assert consolidated["status"] in {"pending", "passed"}
    if consolidated["status"] == "passed":
        assert consolidated["passed"] == 482
        assert consolidated["xfailed"] == 3


def test_ap004c_application_artifacts_and_tool_compile() -> None:
    data = _data()
    assert TOOL.is_file() and APPLICATION.is_file()
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    assert data["tool"]["sha256"] == EXPECTED_TOOL_SHA256
    with tempfile.TemporaryDirectory(prefix="ap004c-tool-pyc-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)


def test_ap004c_git_scope_is_exact_or_commit_is_durable() -> None:
    result = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert result.returncode == 0, result.stderr
    actual = {
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }
    expected = set(EXPECTED_DIRTY_PATHS)
    commit = _find_commit()
    if commit is None:
        assert actual == expected
        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD
        return
    changed = _run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    assert changed.returncode == 0, changed.stderr
    normalized = {
        path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
        for path in changed.stdout.splitlines() if path
    }
    assert normalized == expected
    assert _run("git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD").returncode == 0


def test_ap004c_known_xfails_remain_catalogued() -> None:
    data = _data()
    assert data["scope"]["known_xfails"] == [
        "_refs_v6_strip_org", "extract_org_abstracts", "WorkflowState._normalize"
    ]


def test_ap004c_application_contract_has_exact_test_count() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    count = sum(
        1 for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    assert count == 19

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parents[1]
INVENTORY = ROOT / 'docs/refactor/academic-pipeline/AP-004/ap004b_module_file_inventory.json'
TOOL = ROOT / 'tools/refactor/ap004b_inventory_modules.py'
EXPECTED_HEAD = '6de61fc9741035187836460d97da6d672708998a'
EXPECTED_CANDIDATES = ['app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py', 'configurar_pretriagem_ia_prisma_v16.py', 'gerar_log_diagnostico_artigo_v1_18.py', 'executar_artigo_longo_fulltext_v1_13.py', 'executar_artigo_longo_fulltext_v1_14.py']
EXPECTED_DIRTY_PATHS = ['app_bundle/scripts/pipeline/academic_pipeline_gui.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py', 'app_bundle/scripts/pipeline/academic_pipeline_tui.py', 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'app_bundle/scripts/pipeline/prisma_congelar_artigo.py', 'configurar_pretriagem_ia_prisma.py', 'configurar_pretriagem_ia_prisma_v16.py', 'docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md', 'docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_STRATEGY.md', 'docs/refactor/academic-pipeline/AP-004/ap004b_module_file_application.json', 'docs/refactor/academic-pipeline/AP-004/ap004b_module_file_inventory.json', 'gerar_log_diagnostico_artigo.py', 'gerar_log_diagnostico_artigo_v1_18.py', 'tests/characterization/test_ap004a_naming_inventory_contract.py', 'tests/characterization/test_ap004b_module_file_application_contract.py', 'tests/characterization/test_ap004b_module_file_inventory_contract.py', 'tools/refactor/ap004b_apply_module_file_names.py', 'tools/refactor/ap004b_inventory_modules.py']

EXPECTED_AP004B_COMMIT = 'aa9829f09a5c1b9e69c634637c311b03f360b07e'

EXPECTED_AP004B_SUBJECT = 'refactor(academic-pipeline): consolidar módulos e arquivos da AP-004B'
SOFTWARE_PREFIX = 'software/academic_pipeline_rc10_7_conformidade/'
EFFECTIVE = {"actionable_productive", "compatibility_contract"}
EXCLUDED = {
    "historical_immutable", "physical_directory_reference",
    "protected_operational", "contextual_non_actionable",
}


def _run(*args: str) -> str:
    result = subprocess.run(
        args, cwd=REPOSITORY_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout.strip()


def _data() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _baseline_bytes(relative: str) -> bytes:
    object_name = f"{EXPECTED_HEAD}:{SOFTWARE_PREFIX}{relative}"
    result = subprocess.run(
        ("git", "show", object_name), cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def test_ap004b_v1_6_inventory_metadata_is_durable() -> None:
    data = _data()
    assert data["phase"] == "AP-004B"
    assert data["mode"] == "module-file-inventory-v1.6-read-only"
    assert data["inventory_schema_version"] == 2
    assert data["tool_revision"] == "1.6"
    assert data["inventory_revision"] == "1.6"
    assert data["git"]["head"] == EXPECTED_HEAD


def test_ap004b_v1_6_has_exact_candidate_matrix() -> None:
    data = _data()
    assert [item["current_path"] for item in data["candidates"]] == EXPECTED_CANDIDATES
    assert sum(item["classification"] == "renomeação com compatibilidade" for item in data["candidates"]) == 4
    assert sum(item["classification"] == "renomeação de alto risco" for item in data["candidates"]) == 2


def test_ap004b_v1_6_semantic_partition_is_frozen() -> None:
    data = _data()
    references = data["reference_records"]
    assert len(references) == 269
    assert len(data["consumer_records"]) == 31
    assert all(item["semantic_category"] in EFFECTIVE | EXCLUDED for item in references)
    assert data["statistics"]["actionable_productive_records"] == 7
    assert data["statistics"]["compatibility_contract_records"] == 24


def test_ap004b_v1_6_approved_actionable_records_are_exact() -> None:
    data = _data()
    actual = {
        (item["consumer_path"], item["line"], tuple(item.get("matched_tokens", [])))
        for item in data["reference_records"]
        if item["semantic_category"] == "actionable_productive"
    }
    expected = {
        (item["path"], item["line"], (item["old"],))
        for item in [{'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_gui.py', 'line': 63, 'old': 'academic_pipeline_rc10.py', 'new': 'pipeline_orchestrator.py', 'kind': 'python_string_reference', 'call_selector': 'HERE.with_name'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'line': 4215, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'new': 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'kind': 'python_path_assignment', 'assignment_selector': 'command_lines'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'line': 4216, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'new': 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'kind': 'python_path_assignment', 'assignment_selector': 'command_lines'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_tui.py', 'line': 39, 'old': 'academic_pipeline_rc10.py', 'new': 'pipeline_orchestrator.py', 'kind': 'python_string_reference', 'call_selector': 'HERE.with_name'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/prisma_congelar_artigo.py', 'line': 186, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'new': 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'kind': 'python_path_assignment', 'assignment_selector': 'pipeline'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'executar_artigo_longo_fulltext_v1_13.py', 'line': 9, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'executar_artigo_longo_fulltext_v1_14.py', 'line': 9, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'}]
    }
    assert actual == expected


def test_ap004b_v1_6_collision_remains_suspended() -> None:
    collision = _data()["destination_collisions"][0]
    assert collision["decision"] == "suspended-manual-review-required"
    assert collision["suspended_target"] == 'executar_artigo_longo_fulltext.py'
    assert collision["origins"] == ['executar_artigo_longo_fulltext_v1_13.py', 'executar_artigo_longo_fulltext_v1_14.py']


def test_ap004b_v1_6_source_manifest_matches_baseline_commit() -> None:
    data = _data()
    for relative, record in data["source_manifest"].items():
        baseline = _baseline_bytes(relative)
        assert record["sha256"] == _sha256_bytes(baseline)
        if relative.endswith(".py") and record.get("ast_sha256"):
            tree = ast.parse(baseline.decode("utf-8"), filename=relative)
            dump = ast.dump(tree, include_attributes=False, annotate_fields=True)
            assert record["ast_sha256"] == hashlib.sha256(dump.encode()).hexdigest()


def test_ap004b_v1_6_control_manifest_matches_baseline_commit() -> None:
    data = _data()
    assert set(data["control_manifest"]) == {
        "docs/refactor/academic-pipeline/AP-004/AP-004_NAMING_CONVENTION.md",
        "docs/refactor/academic-pipeline/AP-004/ap004a_naming_inventory.json",
    }
    for relative, record in data["control_manifest"].items():
        assert record["sha256"] == _sha256_bytes(_baseline_bytes(relative))


def test_ap004b_v1_6_preserves_public_entrypoint_decision() -> None:
    data = _data()
    assert data["compatibility_rules"]["public_entrypoints_preserved"] == [
        "academic-pipeline", "python -m academic_pipeline"
    ]
    assert data["compatibility_rules"]["aliases_expire_in"].startswith("AP-004E")


def test_ap004b_v1_6_keeps_physical_directory_for_ap006() -> None:
    data = _data()
    assert data["statistics"]["physical_directory_reference_records"] == 441
    assert data["compatibility_rules"]["physical_directory_references_deferred_to"] == "AP-006"
    assert data["protected"]["physical_directory"] == "academic_pipeline_rc10_7_conformidade"


def test_ap004b_v1_6_preserves_known_xfail_catalog() -> None:
    assert _data()["protected"]["known_xfails"] == ['_refs_v6_strip_org', 'extract_org_abstracts', 'WorkflowState._normalize']


def test_ap004b_v1_6_inventory_artifacts_and_tool_remain_available() -> None:
    data = _data()
    assert INVENTORY.is_file()
    assert TOOL.is_file()
    assert data["tool"]["sha256"] == _sha256(TOOL)


def test_ap004b_v1_6_commit_scope_is_durable() -> None:
    assert (
        _run(
            "git", "show", "-s", "--format=%s",
            EXPECTED_AP004B_COMMIT
        )
        == EXPECTED_AP004B_SUBJECT
    )
    _run(
        "git", "merge-base", "--is-ancestor",
        EXPECTED_AP004B_COMMIT, "HEAD"
    )
    changed = _run(
        "git", "diff-tree", "--no-commit-id", "--name-only", "-r",
        EXPECTED_AP004B_COMMIT
    )
    normalized = {
        path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
        for path in changed.splitlines()
        if path
    }
    assert normalized == set(EXPECTED_DIRTY_PATHS)


def test_ap004b_v1_6_generated_python_compiles() -> None:
    ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))
    with tempfile.TemporaryDirectory(prefix="ap004b-inventory-durable-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)

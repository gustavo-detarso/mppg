from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parents[1]
APPLICATION = ROOT / 'docs/refactor/academic-pipeline/AP-004/ap004b_module_file_application.json'
TOOL = ROOT / 'tools/refactor/ap004b_apply_module_file_names.py'
EXPECTED_HEAD = '6de61fc9741035187836460d97da6d672708998a'
EXPECTED_TOOL_SHA256 = 'a91408393d26fef76d54f7acf3f5a8f9464c2bc505a52633e0c7a3a9d120071a'
EXPECTED_DIRTY_PATHS = ['app_bundle/scripts/pipeline/academic_pipeline_gui.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py', 'app_bundle/scripts/pipeline/academic_pipeline_tui.py', 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'app_bundle/scripts/pipeline/prisma_congelar_artigo.py', 'configurar_pretriagem_ia_prisma.py', 'configurar_pretriagem_ia_prisma_v16.py', 'docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md', 'docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_STRATEGY.md', 'docs/refactor/academic-pipeline/AP-004/ap004b_module_file_application.json', 'docs/refactor/academic-pipeline/AP-004/ap004b_module_file_inventory.json', 'gerar_log_diagnostico_artigo.py', 'gerar_log_diagnostico_artigo_v1_18.py', 'tests/characterization/test_ap004a_naming_inventory_contract.py', 'tests/characterization/test_ap004b_module_file_application_contract.py', 'tests/characterization/test_ap004b_module_file_inventory_contract.py', 'tools/refactor/ap004b_apply_module_file_names.py', 'tools/refactor/ap004b_inventory_modules.py']

EXPECTED_AP004B_COMMIT = 'aa9829f09a5c1b9e69c634637c311b03f360b07e'

EXPECTED_AP004B_SUBJECT = 'refactor(academic-pipeline): consolidar módulos e arquivos da AP-004B'
SOFTWARE_PREFIX = 'software/academic_pipeline_rc10_7_conformidade/'
CANDIDATES = [{'key': 'pipeline_orchestrator', 'historical': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'canonical': 'app_bundle/scripts/pipeline/pipeline_orchestrator.py'}, {'key': 'toml_generator', 'historical': 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py', 'canonical': 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py'}, {'key': 'prisma_ai_prescreen_configurator', 'historical': 'configurar_pretriagem_ia_prisma_v16.py', 'canonical': 'configurar_pretriagem_ia_prisma.py'}, {'key': 'article_diagnostic_log', 'historical': 'gerar_log_diagnostico_artigo_v1_18.py', 'canonical': 'gerar_log_diagnostico_artigo.py'}]
REPLACEMENTS = [{'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_gui.py', 'line': 63, 'old': 'academic_pipeline_rc10.py', 'new': 'pipeline_orchestrator.py', 'kind': 'python_string_reference', 'call_selector': 'HERE.with_name'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'line': 4215, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'new': 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'kind': 'python_path_assignment', 'assignment_selector': 'command_lines'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'line': 4216, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'new': 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'kind': 'python_path_assignment', 'assignment_selector': 'command_lines'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_tui.py', 'line': 39, 'old': 'academic_pipeline_rc10.py', 'new': 'pipeline_orchestrator.py', 'kind': 'python_string_reference', 'call_selector': 'HERE.with_name'}, {'candidate_key': 'pipeline_orchestrator', 'path': 'app_bundle/scripts/pipeline/prisma_congelar_artigo.py', 'line': 186, 'old': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'new': 'app_bundle/scripts/pipeline/pipeline_orchestrator.py', 'kind': 'python_path_assignment', 'assignment_selector': 'pipeline'}]
FULLTEXT_PATHS = ['executar_artigo_longo_fulltext_v1_13.py', 'executar_artigo_longo_fulltext_v1_14.py']
FORBIDDEN_FULLTEXT_CANONICAL = 'executar_artigo_longo_fulltext.py'


def _git_blob(commit: str, relative: str) -> bytes:
    repo_path = SOFTWARE_PREFIX + relative
    result = subprocess.run(
        ("git", "show", f"{commit}:{repo_path}"),
        cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout


def _data() -> dict:
    return json.loads(APPLICATION.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*args: str, cwd: Path = REPOSITORY_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )


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


def test_ap004b_application_metadata_and_approval() -> None:
    data = _data()
    assert data["phase"] == "AP-004B"
    assert data["mode"] == "module-file-application-v1.4"
    assert data["application_schema_version"] == 1
    assert data["tool"]["version"] == 1
    assert data["tool"]["revision"] == "1.4"
    assert data["baseline"]["head"] == EXPECTED_HEAD
    assert data["approval"]["inventory_revision"] == "1.6"
    assert data["approval"]["approved"] is True


def test_ap004b_module_paths_follow_approved_migration_policies() -> None:
    data = _data()
    assert len(data["module_migrations"]) == 4
    for item in data["module_migrations"]:
        historical = ROOT / item["historical_path"]
        canonical = ROOT / item["canonical_path"]
        assert historical.is_file()
        assert canonical.is_file()
        if item["key"] == "pipeline_orchestrator":
            assert item["migration_policy"] == "canonical-alias-over-frozen-historical"
            frozen = _git_blob(
                EXPECTED_AP004B_COMMIT, item["historical_path"]
            )
            assert hashlib.sha256(frozen).hexdigest() == item["source_sha256_before"]
            assert item["historical_sha256_after"] == item["source_sha256_before"]
            assert _sha256(canonical) == item["canonical_sha256_after"]
            alias_source = canonical.read_text(encoding="utf-8")
            assert "Alias canônico AP-004B" in alias_source
            assert "academic_pipeline_rc10.py" in alias_source
            current = historical.read_text(encoding="utf-8")
            assert "_refs_v6_strip_org" in current
            assert "_ap003d_impl__refs_v6_strip_org" in current
        else:
            assert item["migration_policy"] == "canonical-copy-with-historical-wrapper"
            assert _sha256(canonical) == item["source_sha256_before"]
            assert item["canonical_sha256_after"] == item["source_sha256_before"]
            tree = ast.parse(
                canonical.read_text(encoding="utf-8"),
                filename=str(canonical),
            )
            dump = ast.dump(
                tree, include_attributes=False, annotate_fields=True
            )
            assert hashlib.sha256(dump.encode()).hexdigest() == item["source_ast_sha256_before"]


def test_ap004b_three_non_orchestrator_historical_paths_are_loader_wrappers() -> None:
    data = _data()
    migrated = [item for item in data["module_migrations"] if item["key"] != "pipeline_orchestrator"]
    assert len(migrated) == 3
    for item in migrated:
        wrapper = ROOT / item["historical_path"]
        source = wrapper.read_text(encoding="utf-8")
        assert item["wrapper_sha256_after"] == _sha256(wrapper)
        assert "Wrapper transitório AP-004B" in source
        assert item["canonical_filename"] in source
        assert ".read_bytes()" in source
        assert "compile(" in source
        assert "exec(" in source
        ast.parse(source, filename=str(wrapper))


def test_ap004b_loader_aliases_preserve_namespace_strategy() -> None:
    data = _data()
    for item in data["module_migrations"]:
        relative = item["canonical_path"] if item["key"] == "pipeline_orchestrator" else item["historical_path"]
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        calls = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "globals" in calls
        assert "compile" in calls
        assert "exec" in calls
        prefix = "_ap004b_alias_" if item["key"] == "pipeline_orchestrator" else "_ap004b_compat_"
        assert any(name.startswith(prefix) for name in names)


def test_ap004b_consumer_replacements_remain_historical_snapshot() -> None:
    data = _data()
    assert len(data["consumer_replacements"]) == 5

    approved_e3_replacements = {
        (
            "app_bundle/scripts/pipeline/prisma_congelar_artigo.py",
            "app_bundle/scripts/pipeline/pipeline_orchestrator.py",
        ): 'command = [sys.executable, "-m", "academic_pipeline"',
    }

    for item in data["consumer_replacements"]:
        source = (ROOT / item["path"]).read_text(encoding="utf-8")
        if item["new"] in source:
            continue

        key = (item["path"], item["new"])
        assert key in approved_e3_replacements
        assert approved_e3_replacements[key] in source


def test_ap004b_only_five_approved_runtime_occurrences_were_migrated() -> None:
    data = _data()
    actual = {(item["path"], item["line"], item["old"], item["new"]) for item in data["consumer_replacements"]}
    expected = {(item["path"], item["line"], item["old"], item["new"]) for item in REPLACEMENTS}
    assert actual == expected
    assert data["scope"]["selected_actionable_records"] == 5
    assert data["scope"]["deferred_actionable_records"] == 2


def test_ap004b_fulltext_versions_are_untouched_and_target_absent() -> None:
    data = _data()
    for relative in FULLTEXT_PATHS:
        assert _sha256(ROOT / relative) == data["deferred_fulltext"][relative]["sha256_before"]
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "app_bundle/scripts/pipeline/academic_pipeline_rc10.py" in source
    assert not (ROOT / FORBIDDEN_FULLTEXT_CANONICAL).exists()


def test_ap004b_public_entrypoint_control_files_are_unchanged() -> None:
    data = _data()
    for relative, expected in data["unchanged_control_files"].items():
        historical = _git_blob(EXPECTED_AP004B_COMMIT, relative)
        assert hashlib.sha256(historical).hexdigest() == expected["sha256_before"]


def test_ap004b_legacy_module_and_compatibility_contracts_remain() -> None:
    data = _data()
    assert _sha256(ROOT / "academic_pipeline/legacy.py") == data["unchanged_control_files"]["academic_pipeline/legacy.py"]["sha256_before"]
    assert data["scope"]["compatibility_contract_records_preserved"] == 24
    assert (ROOT / 'tests/characterization/test_ap004a_naming_inventory_contract.py').is_file()
    assert (ROOT / 'tests/characterization/test_ap004b_module_file_inventory_contract.py').is_file()


def test_ap004b_historical_and_canonical_orchestrator_help_match() -> None:
    historical = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    canonical = ROOT / "app_bundle/scripts/pipeline/pipeline_orchestrator.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    old = _run(sys.executable, str(historical), "--help", cwd=ROOT)
    new = _run(sys.executable, str(canonical), "--help", cwd=ROOT)
    assert old.returncode == new.returncode == 0
    def normalize_program(text: str) -> str:
        normalized = text.replace(
            "academic_pipeline_rc10.py", "<PROGRAM>"
        ).replace("pipeline_orchestrator.py", "<PROGRAM>")
        # argparse calcula recuos e quebras conforme o comprimento de argv[0].
        # A equivalência relevante é o conteúdo e a ordem, não o espaçamento.
        return " ".join(normalized.split())
    assert normalize_program(old.stdout) == normalize_program(new.stdout)
    assert normalize_program(old.stderr) == normalize_program(new.stderr)


def test_ap004b_all_changed_productive_python_compiles() -> None:
    data = _data()
    with tempfile.TemporaryDirectory(prefix="ap004b-application-pyc-") as tmp:
        for index, relative in enumerate(data["scope"]["productive_changed_paths"]):
            py_compile.compile(
                str(ROOT / relative),
                cfile=str(Path(tmp) / f"{index}.pyc"),
                doraise=True,
            )


def test_ap004b_known_xfails_and_frozen_orchestrator_remain_unchanged() -> None:
    data = _data()
    assert data["protected"]["known_xfails"] == [
        "_refs_v6_strip_org", "extract_org_abstracts",
        "WorkflowState._normalize",
    ]
    orchestrator = next(
        item for item in data["module_migrations"]
        if item["key"] == "pipeline_orchestrator"
    )
    historical = _git_blob(
        EXPECTED_AP004B_COMMIT, orchestrator["historical_path"]
    )
    assert hashlib.sha256(historical).hexdigest() == orchestrator["source_sha256_before"]
    current = (ROOT / orchestrator["historical_path"]).read_text(encoding="utf-8")
    assert "_refs_v6_strip_org" in current
    assert "_ap003d_impl__refs_v6_strip_org" in current


def test_ap004b_commit_scope_is_durable() -> None:
    subject = _run(
        "git", "show", "-s", "--format=%s", EXPECTED_AP004B_COMMIT
    )
    assert subject.returncode == 0, subject.stderr
    assert subject.stdout.strip() == EXPECTED_AP004B_SUBJECT
    ancestor = _run(
        "git", "merge-base", "--is-ancestor",
        EXPECTED_AP004B_COMMIT, "HEAD"
    )
    assert ancestor.returncode == 0
    changed = _run(
        "git", "diff-tree", "--no-commit-id", "--name-only", "-r",
        EXPECTED_AP004B_COMMIT
    )
    assert changed.returncode == 0, changed.stderr
    normalized = {
        path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
        for path in changed.stdout.splitlines()
        if path
    }
    assert normalized == set(EXPECTED_DIRTY_PATHS)


def test_ap004b_application_artifacts_are_coherent() -> None:
    data = _data()
    assert TOOL.is_file()
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    assert data["tool"]["sha256"] == EXPECTED_TOOL_SHA256
    assert (ROOT / 'docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md').is_file()
    assert data["scope"]["forbidden_fulltext_target"] == FORBIDDEN_FULLTEXT_CANONICAL


def test_ap004b_inventory_contract_is_durable_and_compiles() -> None:
    path = ROOT / 'tests/characterization/test_ap004b_module_file_inventory_contract.py'
    source = path.read_text(encoding="utf-8")
    assert "source_manifest_matches_baseline_commit" in source
    assert "commit_scope_is_durable" in source
    ast.parse(source, filename=str(path))
    with tempfile.TemporaryDirectory(prefix="ap004b-inventory-contract-") as tmp:
        py_compile.compile(str(path), cfile=str(Path(tmp) / "inventory.pyc"), doraise=True)


def test_ap004b_ap004a_contract_remains_durable_and_compiles() -> None:
    path = ROOT / 'tests/characterization/test_ap004a_naming_inventory_contract.py'
    source = path.read_text(encoding="utf-8")
    assert "EXPECTED_AP004A_SUBJECT" in source
    assert "_find_ap004a_commit" in source
    ast.parse(source, filename=str(path))
    with tempfile.TemporaryDirectory(prefix="ap004a-contract-") as tmp:
        py_compile.compile(str(path), cfile=str(Path(tmp) / "ap004a.pyc"), doraise=True)


def test_ap004b_application_contract_and_tool_compile() -> None:
    with tempfile.TemporaryDirectory(prefix="ap004b-application-contract-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)

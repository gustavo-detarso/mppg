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
INVENTORY = ROOT / 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_inventory.json'
AP004C_APPLICATION = ROOT / 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_application.json'
STRATEGY = ROOT / 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_STRATEGY.md'
TOOL = ROOT / 'tools/refactor/ap004c_inventory_internal_symbols.py'
BASELINE_HEAD = 'aa9829f09a5c1b9e69c634637c311b03f360b07e'
EXPECTED_SUBJECT = 'chore(academic-pipeline): consolidar inventário de símbolos internos da AP-004C'
EXPECTED_AP004C_APPLICATION_SUBJECT = 'refactor(academic-pipeline): consolidar símbolos internos da AP-004C'
EXPECTED_AP004C_APPLICATION_OUTPUTS = ['app_bundle/scripts/pipeline/academic_pipeline_rc10.py', 'app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py', 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_APPLICATION.md', 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_STRATEGY.md', 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_application.json', 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_inventory.json', 'tests/characterization/test_ap003d_document_contract.py', 'tests/characterization/test_ap003g_stabilization_contract.py', 'tests/characterization/test_ap004b_module_file_application_contract.py', 'tests/characterization/test_ap004b_module_file_inventory_contract.py', 'tests/characterization/test_ap004c_internal_symbol_application_contract.py', 'tests/characterization/test_ap004c_internal_symbol_inventory_contract.py', 'tools/refactor/ap004c_apply_internal_symbols.py', 'tools/refactor/ap004c_inventory_internal_symbols.py']
EXPECTED_TOOL_SHA256 = '5220836c17d3e0b4d4b31a695ffa13e977d8a12606388eb66c15c838624881ea'
EXPECTED_OUTPUTS = ['docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-004/AP-004C_INTERNAL_SYMBOL_STRATEGY.md', 'docs/refactor/academic-pipeline/AP-004/ap004c_internal_symbol_inventory.json', 'tools/refactor/ap004c_inventory_internal_symbols.py', 'tests/characterization/test_ap004c_internal_symbol_inventory_contract.py', 'tests/characterization/test_ap004b_module_file_application_contract.py', 'tests/characterization/test_ap004b_module_file_inventory_contract.py']
EXPECTED_SAFE_ALIASES = [('_ap003d_impl_output_paths', '_impl_output_paths'), ('_ap003d_impl_apply_cli_path_overrides', '_impl_apply_cli_path_overrides'), ('_ap003d_impl_load_existing_document_json', '_impl_load_existing_document_json'), ('_ap003d_impl_resolve_bib_for_existing_document', '_impl_resolve_bib_for_existing_document'), ('_ap003d_impl__resolve_latex_paths_for_recompile', '_impl_resolve_latex_paths_for_recompile'), ('_ap003d_impl_run_recompile', '_impl_run_recompile'), ('_ap003d_impl_render_additional_language_versions', '_impl_render_additional_language_versions'), ('_ap003d_impl__refs_v6_disabled', '_impl_refs_disabled'), ('_ap003d_impl__refs_v6_apply_runtime_policy', '_impl_refs_apply_runtime_policy'), ('_ap003d_impl_load_config', '_impl_load_config'), ('_ap003d_impl_build_bibliography', '_impl_build_bibliography'), ('_ap003d_impl__refs_v6_clear_document_bibliography', '_impl_refs_clear_document_bibliography'), ('_ap003d_impl_render_org_latex', '_impl_render_org_latex')]
PROTECTED_CONTROLS = [{'qualified_name': '_refs_v6_strip_org', 'current_name': '_refs_v6_strip_org', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'}, {'qualified_name': 'extract_org_abstracts', 'current_name': 'extract_org_abstracts', 'path': 'app_bundle/scripts/pipeline/render_docx_canonico.py'}, {'qualified_name': 'WorkflowState._normalize', 'current_name': '_normalize', 'path': 'app_bundle/scripts/pipeline/article_workflow/state.py'}, {'qualified_name': '_ap003d_impl__refs_v6_strip_org', 'current_name': '_ap003d_impl__refs_v6_strip_org', 'path': 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'}]
SOFTWARE_PREFIX = 'software/academic_pipeline_rc10_7_conformidade/'


def _git_blob(commit: str, relative: str) -> bytes:
    repo_path = SOFTWARE_PREFIX + relative
    result = subprocess.run(
        ("git", "show", f"{commit}:{repo_path}"),
        cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=REPOSITORY_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _data() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    result = _run(
        "git", "log", "--format=%H%x00%s", f"{BASELINE_HEAD}..HEAD"
    )
    assert result.returncode == 0, result.stderr
    matches = []
    for line in result.stdout.splitlines():
        if "\x00" not in line:
            continue
        commit, subject = line.split("\x00", 1)
        if subject == EXPECTED_SUBJECT:
            matches.append(commit)
    assert len(matches) <= 1
    return matches[0] if matches else None


def test_ap004c_metadata_is_bound_to_published_ap004b() -> None:
    data = _data()
    assert data["phase"] == "AP-004C"
    assert data["mode"] == "internal-symbol-inventory-v1.3-read-only"
    assert data["inventory_schema_version"] == 1
    assert data["inventory_revision"] == "1.3"
    assert data["git"]["head"] == BASELINE_HEAD
    assert data["prior_phases"]["ap004a_revision"] == "4.2"
    assert data["prior_phases"]["ap004b_inventory_revision"] == "1.6"
    assert data["prior_phases"]["ap004b_application_mode"] == "module-file-application-v1.4"


def test_ap004c_candidates_are_exact_ap004a_internal_scope_plus_protections() -> None:
    data = _data()
    source = json.loads((ROOT / 'docs/refactor/academic-pipeline/AP-004/ap004a_naming_inventory.json').read_text(encoding="utf-8"))
    expected = {
        item["id"] for item in source["actionable_candidates"]
        if item["category"] in {"função", "classe", "constante", "alias"}
        and "AP-004C" in item["target_phase"]
    }
    actual_nonprotected = {
        item["id"] for item in data["candidates"]
        if item["disposition"] != "protected_xfail_out_of_scope"
    }
    assert expected <= actual_nonprotected
    assert all(item["category"] in {"função", "classe", "constante", "alias"} for item in data["candidates"])


def test_ap004c_candidate_definitions_and_hashes_are_current() -> None:
    data = _data()
    for item in data["candidates"]:
        path = ROOT / item["path"]
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        definition = item["definition"]
        assert definition["line"] is not None
        assert definition["definition_kind"] in {"function", "class", "assignment", "annotated_assignment", "import_alias", "import"}
        assert len(definition["ast_sha256"]) == 64
        assert len(definition["source_sha256"]) == 64
        assert isinstance(tree, ast.Module)


def test_ap004c_required_safe_orchestrator_aliases_are_preserved() -> None:
    data = _data()
    by_name = {item["current_name"]: item for item in data["candidates"]}
    for current, suggested in EXPECTED_SAFE_ALIASES:
        item = by_name[current]
        assert item["path"] == 'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'
        assert item["suggested_name"] == suggested
        assert item["classification"] == "renomeação segura"
        assert item["disposition"] in {
            "ready_contract_bound_ast_rename",
            "contract_update_required",
        }


def test_ap004c_xfail_controls_are_absolute_protections() -> None:
    data = _data()
    actual = {
        (item["current_name"], item["path"]): item
        for item in data["candidates"]
    }
    for control in PROTECTED_CONTROLS:
        item = actual[(control["current_name"], control["path"])]
        assert item["suggested_name"] is None
        assert item["disposition"] == "protected_xfail_out_of_scope"
    assert data["statistics"]["protected_count"] >= len(PROTECTED_CONTROLS)


def test_ap004c_core_and_opaque_stages_are_not_auto_renamed() -> None:
    data = _data()
    core = [item for item in data["candidates"] if item["current_name"] == "_ap003f_pipeline_core"]
    assert len(core) == 1
    assert core[0]["disposition"] == "deferred_structural_symbol"
    for item in data["candidates"]:
        name = item["current_name"]
        if name.startswith("_ap003c_dispatch_") or name.startswith("_ap003d_stage_") or name.startswith("_ap003e_stage_"):
            assert item["disposition"] == "deferred_structural_symbol"


def test_ap004c_dispositions_partition_every_candidate() -> None:
    data = _data()
    counts = data["statistics"]["by_disposition"]
    assert sum(counts.values()) == len(data["candidates"])
    assert set(counts) == {
        "ready_local_ast_rename", "ready_contract_bound_ast_rename",
        "contract_update_required", "compatibility_required",
        "deferred_structural_symbol", "manual_semantic_name_required",
        "blocked_destination_collision", "protected_xfail_out_of_scope",
    }
    assert data["statistics"]["ready_wave_1_count"] >= 0
    assert data["statistics"]["ready_wave_2_count"] >= len(EXPECTED_SAFE_ALIASES)


def test_ap004c_reference_records_are_exact_and_partitioned() -> None:
    data = _data()
    candidate_ids = {item["id"] for item in data["candidates"]}
    categories = set(data["statistics"]["by_reference_category"])
    for record in data["python_references"]:
        assert record["candidate_id"] in candidate_ids
        assert record["semantic_category"] in categories
        assert record["symbol"]
        assert record["path"].endswith(".py")
    assert sum(data["statistics"]["by_reference_category"].values()) == len(data["python_references"])


def test_ap004c_has_no_module_or_file_renames() -> None:
    data = _data()
    assert data["scope"]["productive_change"] is False
    assert data["scope"]["forbidden_module_file_changes"] is True
    assert all(item["category"] != "arquivo/módulo" for item in data["candidates"])
    assert not any(path.endswith("pipeline_orchestrator.py") for path in EXPECTED_OUTPUTS)


def test_ap004c_destination_collisions_are_explicit_blocks() -> None:
    data = _data()
    collision_ids = {candidate_id for collision in data["destination_collisions"] for candidate_id in collision["candidate_ids"]}
    for item in data["candidates"]:
        if item["id"] in collision_ids:
            assert item["disposition"] == "blocked_destination_collision"
        suggestion = item.get("suggested_name")
        if suggestion:
            assert suggestion.isidentifier()


def test_ap004c_source_manifest_matches_current_baseline() -> None:
    data = _data()
    application = json.loads(
        AP004C_APPLICATION.read_text(encoding="utf-8")
    )
    baseline = application["inventory_baseline"]
    assert baseline["source_manifest"] == data["source_manifest"]
    preparatory = set(baseline["preparatory_dirty_paths"])
    for record in data["source_manifest"]:
        if record["path"] in preparatory:
            continue
        historical = _git_blob(BASELINE_HEAD, record["path"])
        assert hashlib.sha256(historical).hexdigest() == record["sha256"]
        if record["path"].endswith(".py"):
            tree = ast.parse(
                historical.decode("utf-8"), filename=record["path"]
            )
            actual = hashlib.sha256(
                ast.dump(
                    tree, include_attributes=False, annotate_fields=True
                ).encode()
            ).hexdigest()
            assert actual == record["ast_sha256"]


def test_ap004c_preserves_ap003_and_ap004b_control_files() -> None:
    data = _data()
    manifest = {record["path"]: record for record in data["source_manifest"]}
    required = {
        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "tests/characterization/test_ap003g_stabilization_contract.py",
        "docs/refactor/academic-pipeline/AP-003/ap003g_manifest.json",
        "docs/refactor/academic-pipeline/AP-004/ap004b_module_file_application.json",
        "tests/characterization/test_ap004b_module_file_application_contract.py",
    }
    assert required <= set(manifest)
    historical = _git_blob(
        BASELINE_HEAD,
        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    )
    assert hashlib.sha256(historical).hexdigest() == (
        "8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977"
    )
    current = (
        ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    ).read_text(encoding="utf-8")
    assert "_refs_v6_strip_org" in current
    assert "_ap003d_impl__refs_v6_strip_org" in current
    for relative in (
        "tests/characterization/test_ap004b_module_file_application_contract.py",
        "tests/characterization/test_ap004b_module_file_inventory_contract.py",
    ):
        contract = (ROOT / relative).read_text(encoding="utf-8")
        assert "EXPECTED_AP004B_COMMIT" in contract
        assert "EXPECTED_AP004B_SUBJECT" in contract
        assert "commit_scope_is_durable" in contract


def test_ap004c_strategy_keeps_application_blocked_and_ordered() -> None:
    data = _data()
    text = STRATEGY.read_text(encoding="utf-8")
    assert data["next_gate"]["blocked"] is True
    assert "Onda 1" in text and "Onda 2" in text
    assert "_refs_v6_strip_org" in text
    assert "nenhum commit sem aprovação expressa" in text.lower()


def test_ap004c_current_status_is_output_scope_or_clean_and_commit_is_durable() -> None:
    result = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert result.returncode == 0, result.stderr
    actual = {
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }
    expected = set(EXPECTED_AP004C_APPLICATION_OUTPUTS)
    result = _run(
        "git", "log", "--format=%H%x00%s", f"{BASELINE_HEAD}..HEAD"
    )
    assert result.returncode == 0, result.stderr
    matches = []
    for line in result.stdout.splitlines():
        if "\x00" not in line:
            continue
        commit, subject = line.split("\x00", 1)
        if subject == EXPECTED_AP004C_APPLICATION_SUBJECT:
            matches.append(commit)
    if not matches:
        assert actual == expected
        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD
        return
    assert len(matches) == 1
    changed = _run(
        "git", "diff-tree", "--no-commit-id", "--name-only", "-r", matches[0]
    )
    assert changed.returncode == 0, changed.stderr
    normalized = {
        path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
        for path in changed.stdout.splitlines() if path
    }
    assert normalized == expected
    assert _run(
        "git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD"
    ).returncode == 0


def test_ap004c_generated_artifacts_and_tool_compile() -> None:
    data = _data()
    assert data["scope"]["allowed_outputs"] == EXPECTED_OUTPUTS
    assert TOOL.is_file() and INVENTORY.is_file() and STRATEGY.is_file()
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    assert data["tool"]["sha256"] == EXPECTED_TOOL_SHA256
    with tempfile.TemporaryDirectory(prefix="ap004c-contract-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)

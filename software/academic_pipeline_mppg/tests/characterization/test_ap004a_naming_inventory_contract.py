from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / 'docs/refactor/academic-pipeline/AP-004/ap004a_naming_inventory.json'
CONVENTION = ROOT / 'docs/refactor/academic-pipeline/AP-004/AP-004_NAMING_CONVENTION.md'
TOOL = ROOT / 'tools/refactor/ap004a_inventory_names.py'
EXPECTED_HEAD = '59ec50368de7302a9f25fe45809649e4baf2c144'
EXPECTED_AP003G_COMMIT = '59ec50368de7302a9f25fe45809649e4baf2c144'
EXPECTED_AP004A_SUBJECT = 'chore(academic-pipeline): consolidar inventário de nomes da AP-004A'
EXPECTED_TOOL_SHA256 = '9dc6e5a28de82a9ff4cb2019370132c032ce933dcd9062b3722136a92a8ac426'
EXPECTED_OUTPUTS = ['docs/refactor/academic-pipeline/AP-004/AP-004A_NAMING_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-004/AP-004_NAMING_CONVENTION.md', 'docs/refactor/academic-pipeline/AP-004/ap004a_naming_inventory.json', 'tools/refactor/ap004a_inventory_names.py', 'tests/characterization/test_ap004a_naming_inventory_contract.py']
CLASSIFICATIONS = ['renomeação segura', 'renomeação com compatibilidade', 'renomeação de alto risco', 'nome que deve permanecer']
CATEGORIES = ['arquivo/módulo', 'função', 'classe', 'constante', 'alias']
STRUCTURAL_KINDS = {
    "release_candidate", "version_marker", "refactor_phase", "explicit_version_word"
}


def _run(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def _data() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    raw = raw.strip().strip('"').replace("\\", "/")
    prefix = "software/academic_pipeline_rc10_7_conformidade/"
    return raw[len(prefix):] if raw.startswith(prefix) else raw


def _ephemeral(path: str) -> bool:
    parts = Path(path).parts
    return "__pycache__" in parts or ".pytest_cache" in parts or path.endswith((".pyc", ".pyo"))


def _software_relative(path: str) -> str:
    normalized = path.strip().strip('"').replace("\\", "/")
    prefix = "software/academic_pipeline_rc10_7_conformidade/"
    return normalized[len(prefix):] if normalized.startswith(prefix) else normalized


def _commit_paths(commit: str) -> set[str]:
    output = _run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    return {_software_relative(line) for line in output.splitlines() if line.strip()}


def _find_ap004a_commit() -> str:
    output = _run("git", "log", "--format=%H%x09%s", f"{EXPECTED_HEAD}..HEAD")
    matches = []
    for line in output.splitlines():
        if "\t" not in line:
            continue
        commit, subject = line.split("\t", 1)
        if subject == EXPECTED_AP004A_SUBJECT:
            matches.append(commit)
    assert len(matches) == 1, matches
    return matches[0]


def test_ap004a_v4_2_is_bound_to_inventory_baseline_and_ap003g() -> None:
    data = _data()
    assert data["phase"] == "AP-004A"
    assert data["mode"] == "inventory-and-convention-v4.2-read-only"
    assert data["inventory_schema_version"] == 4
    assert data["inventory_revision"] == "4.2"
    assert data["tool"]["version"] == 4
    assert data["tool"]["revision"] == "4.2"
    assert data["git"]["head"] == EXPECTED_HEAD
    current_head = _run("git", "rev-parse", "HEAD")
    if current_head != EXPECTED_HEAD:
        closure = _find_ap004a_commit()
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", closure, current_head),
            cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert _run("git", "rev-parse", f"{closure}^") == EXPECTED_HEAD
    assert data["ap003g_closure"]["commit"] == EXPECTED_AP003G_COMMIT
    assert data["ap003g_closure"]["published"] is True


def test_ap004a_v4_2_separates_raw_context_and_actionable_candidates() -> None:
    data = _data()
    assert data["raw_occurrences"]
    assert data["actionable_candidates"]
    assert data["candidates"] == data["actionable_candidates"]
    assert isinstance(data["contextual_review_occurrences"], list)
    assert data["historical_references"] is not None
    assert all(item["category"] in CATEGORIES for item in data["actionable_candidates"])
    assert all(item["category"] not in {"import", "entrypoint", "teste", "documentação"} for item in data["actionable_candidates"])


def test_ap004a_v4_2_actionable_candidates_require_structural_markers_or_explicit_preservation() -> None:
    data = _data()
    for item in data["actionable_candidates"]:
        marker_kinds = {marker["kind"] for marker in item["markers"]}
        if item["classification"] == "nome que deve permanecer":
            continue
        assert marker_kinds & STRUCTURAL_KINDS, item
    assert any(item.get("related_surfaces") for item in data["actionable_candidates"])
    assert data["entrypoints"]


def test_ap004a_v4_2_excludes_dunder_stdlib_and_semantic_verbs() -> None:
    data = _data()
    candidates = data["actionable_candidates"]
    names = {item["current_name"] for item in candidates}
    assert not any(re.fullmatch(r"__[^_].*__", name) for name in names)
    assert "copy" not in names
    assert "backup" not in names
    assert "copy_one" not in names
    assert "copy_if_exists" not in names
    assert "make_backup_and_copy" not in names
    assert any(item.get("module") == "copy" for item in data["import_inventory"])
    protected_prefixes = ("aplicar_", "atualizar_", "migrar_", "migrador_", "migration_", "patch_", "corrigir_")
    assert all(
        not (Path(item["path"]).parent == Path(".") and Path(item["path"]).name.startswith(protected_prefixes))
        for item in candidates
    )


def test_ap004a_v4_2_candidates_are_unique_and_suggestions_are_safe() -> None:
    data = _data()
    candidates = data["actionable_candidates"]
    ids = [item["id"] for item in candidates]
    assert len(ids) == len(set(ids))
    assert {item["classification"] for item in candidates} <= set(CLASSIFICATIONS)
    active_destinations = []
    unsafe = re.compile(r"(?:^|_)(?:original|pre|stage_?\d+|dispatch_?\d+)(?:_|$)|_\d+$", re.I)
    for item in candidates:
        assert item["classification_reason"]
        assert item["target_phase"]
        assert item["status"] == "candidate-only-no-change"
        suggestion = item.get("suggested_name")
        if not suggestion:
            continue
        assert not unsafe.search(Path(suggestion).stem), item
        if item["category"] == "arquivo/módulo":
            active_destinations.append(("file", suggestion))
        else:
            active_destinations.append((item["path"], suggestion))
    assert len(active_destinations) == len(set(active_destinations))
    assert all(collision["status"] == "suggestion-suspended" for collision in data["destination_collisions"])


def test_ap004a_v4_2_protects_operational_history_legacy_and_xfails() -> None:
    data = _data()
    protected = data["protected_names"]
    assert protected["physical_directory"] == "academic_pipeline_rc10_7_conformidade"
    assert protected["known_xfails"] == [
        "_refs_v6_strip_org", "extract_org_abstracts", "WorkflowState._normalize"
    ]
    assert protected["public_entrypoints"] == ["academic-pipeline", "python -m academic_pipeline"]
    assert all(
        not (
            item["path"].startswith("app_bundle/projetos/")
            and any(
                part == "execucoes_anteriores" or part.startswith("output")
                for part in Path(item["path"]).parts
            )
        )
        for item in data["actionable_candidates"]
    )
    legacy = [item for item in data["actionable_candidates"] if "legacy" in item["current_name"].lower()]
    assert legacy
    assert all(item["classification"] == "nome que deve permanecer" for item in legacy)
    assert all(item.get("suggested_name") is None for item in legacy)
    legacy_runtime_error = [
        item for item in legacy if item["current_name"] == "LegacyRuntimeError"
    ]
    assert legacy_runtime_error
    assert all(not item["markers"] for item in legacy_runtime_error)


def test_ap004a_v4_2_preserves_ap003_architecture_and_consolidates_entrypoints() -> None:
    data = _data()
    architecture = data["ap003_architecture"]
    assert architecture["status"] == "passed"
    assert architecture["public_main"]["name"] == "main"
    assert architecture["internal_core"]["name"] == "_ap003f_pipeline_core"
    assert architecture["historical_alias_assignments"] == 0
    assert architecture["direct_guard_calls"].count("main") == 1
    assert architecture["prisma_core_reference_count"] >= 1
    orchestrator = [
        item for item in data["actionable_candidates"]
        if item["path"] == "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    ]
    modules = [item for item in orchestrator if item["category"] == "arquivo/módulo"]
    assert len(modules) == 1
    assert modules[0]["related_surfaces"]
    assert any(item["current_name"] == "_ap003f_pipeline_core" for item in orchestrator)
    assert all(item["category"] != "entrypoint" for item in data["actionable_candidates"])


def test_ap004a_v4_2_xfails_are_bound_to_exact_production_symbols() -> None:
    data = _data()
    candidates = data["actionable_candidates"]
    expected = {
        "_refs_v6_strip_org": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "extract_org_abstracts": "app_bundle/scripts/pipeline/render_docx_canonico.py",
        "WorkflowState._normalize": "app_bundle/scripts/pipeline/article_workflow/state.py",
    }
    for name, path in expected.items():
        matches = [item for item in candidates if item["current_name"] == name]
        assert len(matches) == 1, (name, matches)
        item = matches[0]
        assert item["path"] == path
        assert item["classification"] == "nome que deve permanecer"
        assert item["target_phase"] == "fora da AP-004"
        assert item.get("suggested_name") is None
        assert not item["path"].startswith("tests/")
    aliases = [
        item for item in candidates
        if item["current_name"] == "_ap003d_impl__refs_v6_strip_org"
    ]
    assert len(aliases) == 1
    assert aliases[0]["path"] == "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    assert aliases[0]["classification"] == "nome que deve permanecer"
    bindings = [
        item for item in data["import_inventory"]
        if item["path"] == "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
        and item.get("form") == "from"
        and item.get("module") == "academic_pipeline.document_orchestration"
        and item.get("name") == "_refs_v6_strip_org_impl"
        and item.get("asname") == "_ap003d_impl__refs_v6_strip_org"
    ]
    assert len(bindings) == 1


def test_ap004a_v4_2_core_symbol_is_scoped_to_symbol_normalization() -> None:
    data = _data()
    candidates = data["actionable_candidates"]
    core = [
        item for item in candidates
        if item["current_name"] == "_ap003f_pipeline_core"
        and item["path"] == "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    ]
    assert len(core) == 1
    item = core[0]
    assert item["category"] == "função"
    assert item["classification"] == "renomeação de alto risco"
    assert item["suggested_name"] == "_run_pipeline"
    assert item["target_phase"] == "AP-004C/AP-004D"
    assert "AP-004B" not in item["target_phase"]
    module = [
        candidate for candidate in candidates
        if candidate["category"] == "arquivo/módulo"
        and candidate["path"] == "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    ]
    assert len(module) == 1
    assert module[0]["target_phase"] == "AP-004B/AP-004E"


def test_ap004a_v4_2_commit_scope_and_generated_python_are_durable() -> None:
    current_head = _run("git", "rev-parse", "HEAD")
    if current_head == EXPECTED_HEAD:
        status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
        actual = {
            path for line in status.splitlines() if line.strip()
            for path in [_status_path(line)] if not _ephemeral(path)
        }
        assert actual == set(EXPECTED_OUTPUTS)
    else:
        closure = _find_ap004a_commit()
        assert _commit_paths(closure) == set(EXPECTED_OUTPUTS)
        for relative in EXPECTED_OUTPUTS:
            _run("git", "ls-files", "--error-unmatch", relative)
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))
    assert CONVENTION.is_file()
    with tempfile.TemporaryDirectory(prefix="ap004a-contract-pyc-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)

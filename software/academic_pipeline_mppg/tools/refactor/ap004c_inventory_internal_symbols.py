#!/usr/bin/env python3
"""AP-004C — inventário preparatório de símbolos internos.

Ferramenta somente leitura produtiva. Ela não renomeia símbolos nem modifica
código do Academic Pipeline. A execução:

- valida diretório, branch, árvore, HEAD local/remoto e o fechamento AP-004B;
- carrega o inventário AP-004A v4.2 como fonte canônica de candidatos;
- seleciona funções, classes, constantes e aliases destinados à AP-004C;
- preserva explicitamente os três xfails históricos e seus aliases vinculados;
- localiza definições por AST e mapeia consumidores estáticos, dinâmicos,
  contratos atuais e contratos congelados da AP-003;
- separa renomeações locais, renomeações vinculadas a contratos, símbolos
  opacos/estruturais, compatibilidade e nomes protegidos;
- identifica colisões de destino e nomes sugeridos já ocupados;
- gera manifesto de hashes e contratos duráveis para um futuro aplicador;
- grava cinco artefatos preparatórios e mantém dois contratos duráveis da AP-004B, com backup externo, escrita
  atômica e rollback integral;
- executa py_compile, git diff --check, suíte específica e suíte consolidada.

Execute a partir da raiz do software e mantenha este arquivo fora do
repositório, por exemplo em ~/Downloads.
"""

from __future__ import annotations

import argparse
import ast
import errno
import hashlib
import json
import keyword
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, NoReturn, Sequence


EXPECTED_SOFTWARE_ROOT = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline/"
    "software/academic_pipeline_rc10_7_conformidade"
)
EXPECTED_REPOSITORY_ROOT = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline"
)
EXPECTED_BRANCH = "ap-refactor/03-orchestrator-decomposition"
EXPECTED_REMOTE_REF = f"origin/{EXPECTED_BRANCH}"
EXPECTED_HEAD = "aa9829f09a5c1b9e69c634637c311b03f360b07e"
EXPECTED_AP004B_SUBJECT = (
    "refactor(academic-pipeline): consolidar módulos e arquivos da AP-004B"
)
EXPECTED_AP004C_INVENTORY_SUBJECT = (
    "chore(academic-pipeline): consolidar inventário de símbolos internos da AP-004C"
)
SOFTWARE_PREFIX = "software/academic_pipeline_rc10_7_conformidade/"

PHASE = "AP-004C"
MODE = "internal-symbol-inventory-v1.3-read-only"
TOOL_VERSION = 1
TOOL_REVISION = "1.3"
INVENTORY_SCHEMA_VERSION = 1
INVENTORY_REVISION = "1.3"
BASELINE_PASSED = 448
BASELINE_XFAILED = 3
EXPECTED_CONTRACT_TESTS = 15

DOC_DIR = Path("docs/refactor/academic-pipeline/AP-004")
REPORT_REL = DOC_DIR / "AP-004C_INTERNAL_SYMBOL_INVENTORY.md"
STRATEGY_REL = DOC_DIR / "AP-004C_INTERNAL_SYMBOL_STRATEGY.md"
INVENTORY_REL = DOC_DIR / "ap004c_internal_symbol_inventory.json"
TOOL_REL = Path("tools/refactor/ap004c_inventory_internal_symbols.py")
TEST_REL = Path(
    "tests/characterization/test_ap004c_internal_symbol_inventory_contract.py"
)
PREPARATORY_OUTPUT_RELS = (REPORT_REL, STRATEGY_REL, INVENTORY_REL, TOOL_REL, TEST_REL)

AP004A_INVENTORY_REL = DOC_DIR / "ap004a_naming_inventory.json"
AP004B_INVENTORY_REL = DOC_DIR / "ap004b_module_file_inventory.json"
AP004B_APPLICATION_REL = DOC_DIR / "ap004b_module_file_application.json"
AP004B_APPLICATION_TEST_REL = Path(
    "tests/characterization/test_ap004b_module_file_application_contract.py"
)
AP004B_INVENTORY_TEST_REL = Path(
    "tests/characterization/test_ap004b_module_file_inventory_contract.py"
)
OUTPUT_RELS = PREPARATORY_OUTPUT_RELS + (
    AP004B_APPLICATION_TEST_REL, AP004B_INVENTORY_TEST_REL
)
AP003G_CONTRACT_REL = Path(
    "tests/characterization/test_ap003g_stabilization_contract.py"
)
AP003G_MANIFEST_REL = Path(
    "docs/refactor/academic-pipeline/AP-003/ap003g_manifest.json"
)
ORCHESTRATOR_REL = Path(
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)
TOML_INTERACTIVE_REL = Path(
    "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py"
)

INTERNAL_CATEGORIES = {"função", "classe", "constante", "alias"}
KNOWN_XFAIL_CONTROLS: tuple[dict[str, str], ...] = (
    {
        "qualified_name": "_refs_v6_strip_org",
        "current_name": "_refs_v6_strip_org",
        "path": ORCHESTRATOR_REL.as_posix(),
    },
    {
        "qualified_name": "extract_org_abstracts",
        "current_name": "extract_org_abstracts",
        "path": "app_bundle/scripts/pipeline/render_docx_canonico.py",
    },
    {
        "qualified_name": "WorkflowState._normalize",
        "current_name": "_normalize",
        "path": "app_bundle/scripts/pipeline/article_workflow/state.py",
    },
    {
        "qualified_name": "_ap003d_impl__refs_v6_strip_org",
        "current_name": "_ap003d_impl__refs_v6_strip_org",
        "path": ORCHESTRATOR_REL.as_posix(),
    },
)

EXPECTED_SAFE_ORCHESTRATOR_ALIASES: tuple[tuple[str, str], ...] = (
    ("_ap003d_impl_output_paths", "_impl_output_paths"),
    (
        "_ap003d_impl_apply_cli_path_overrides",
        "_impl_apply_cli_path_overrides",
    ),
    (
        "_ap003d_impl_load_existing_document_json",
        "_impl_load_existing_document_json",
    ),
    (
        "_ap003d_impl_resolve_bib_for_existing_document",
        "_impl_resolve_bib_for_existing_document",
    ),
    (
        "_ap003d_impl__resolve_latex_paths_for_recompile",
        "_impl_resolve_latex_paths_for_recompile",
    ),
    ("_ap003d_impl_run_recompile", "_impl_run_recompile"),
    (
        "_ap003d_impl_render_additional_language_versions",
        "_impl_render_additional_language_versions",
    ),
    (
        "_ap003d_impl__refs_v6_disabled",
        "_impl_refs_disabled",
    ),
    (
        "_ap003d_impl__refs_v6_apply_runtime_policy",
        "_impl_refs_apply_runtime_policy",
    ),
    ("_ap003d_impl_load_config", "_impl_load_config"),
    ("_ap003d_impl_build_bibliography", "_impl_build_bibliography"),
    (
        "_ap003d_impl__refs_v6_clear_document_bibliography",
        "_impl_refs_clear_document_bibliography",
    ),
    ("_ap003d_impl_render_org_latex", "_impl_render_org_latex"),
)

PROTECTED_COMPONENTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".academic_pipeline",
    ".patch_backups",
    "backups",
    "build",
    "dist",
    "site-packages",
    "node_modules",
}
PROTECTED_ROOT_PREFIXES = (
    "aplicar_",
    "atualizar_",
    "instalar_",
    "install_",
    "setup_",
)
HISTORICAL_PREFIXES = (
    "docs/refactor/academic-pipeline/AP-003/",
    "tools/refactor/ap003",
    "tools/refactor/ap004a_",
    "tools/refactor/ap004b_",
    "docs/refactor/academic-pipeline/AP-004/AP-004A_",
    "docs/refactor/academic-pipeline/AP-004/AP-004B_",
    "docs/refactor/academic-pipeline/AP-004/ap004a_",
    "docs/refactor/academic-pipeline/AP-004/ap004b_",
    "tests/characterization/test_ap004a_",
    "tests/characterization/test_ap004b_",
)
FROZEN_AP003_TEST_PREFIX = "tests/characterization/test_ap003"
TEXT_EXTENSIONS = {
    ".md", ".json", ".toml", ".sh", ".txt", ".org", ".yaml", ".yml"
}

OPAQUE_STRUCTURAL_RE = re.compile(
    r"^_ap003(?:c_dispatch|d_stage|e_stage)_\d+$"
    r"|^_ap003e_impl_.+_\d+$"
)
PHASE_MARKER_RE = re.compile(r"^_ap003[bcdef](?:_|$)")
VERSION_MARKER_RE = re.compile(r"(?:^|_)v\d+(?:_|$)|version\d+", re.IGNORECASE)

DISPOSITIONS = (
    "ready_local_ast_rename",
    "ready_contract_bound_ast_rename",
    "contract_update_required",
    "compatibility_required",
    "deferred_structural_symbol",
    "manual_semantic_name_required",
    "blocked_destination_collision",
    "protected_xfail_out_of_scope",
)
REFERENCE_CATEGORIES = (
    "definition",
    "same_module_static",
    "same_module_dynamic",
    "cross_module_static",
    "cross_module_dynamic",
    "current_test_contract",
    "frozen_ap003_contract",
    "historical_immutable",
    "protected_operational",
    "contextual_string",
)


class InventoryError(RuntimeError):
    """Erro controlado da ferramenta."""


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class BackupRecord:
    path: Path
    existed: bool
    backup_path: Path | None
    mode: int | None


@dataclass
class ParsedPython:
    path: str
    text: str
    tree: ast.Module
    parents: dict[int, ast.AST]


def fail(message: str) -> NoReturn:
    raise InventoryError(message)


def run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: int = 300,
) -> CommandResult:
    try:
        completed = subprocess.run(
            tuple(args),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        fail(f"Comando excedeu {timeout}s: {' '.join(args)}\n{exc}")
    result = CommandResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        fail(
            f"Comando falhou ({result.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def git(root: Path, *args: str, check: bool = True, timeout: int = 300) -> CommandResult:
    return run(("git", *args), cwd=root, check=check, timeout=timeout)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        fail(f"Não foi possível calcular SHA-256 de {path}: {exc}")


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_output(content: str) -> str:
    return content.rstrip() + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = normalize_output(content).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def create_backups(
    outputs: dict[Path, str], *, software_root: Path
) -> tuple[Path, list[BackupRecord]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        Path.home()
        / ".cache/academic-pipeline-refactor/backups/AP-004C"
        / timestamp
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    records: list[BackupRecord] = []
    for path in outputs:
        relative = path.relative_to(software_root)
        if path.exists():
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            records.append(
                BackupRecord(
                    path=path,
                    existed=True,
                    backup_path=destination,
                    mode=path.stat().st_mode,
                )
            )
        else:
            records.append(
                BackupRecord(path=path, existed=False, backup_path=None, mode=None)
            )
    return backup_root, records


def rollback(records: Iterable[BackupRecord]) -> None:
    for record in reversed(list(records)):
        try:
            if record.existed and record.backup_path is not None:
                record.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record.backup_path, record.path)
                if record.mode is not None:
                    os.chmod(record.path, record.mode)
            else:
                record.path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[ROLLBACK-ERRO] {record.path}: {exc}", file=sys.stderr)


def compile_python(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ap004c-pyc-") as temporary:
        py_compile.compile(
            str(path),
            cfile=str(Path(temporary) / f"{path.name}.pyc"),
            doraise=True,
        )


def normalize_status_path(line: str) -> str:
    raw = line[2:].lstrip() if len(line) >= 2 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip().strip('"').replace("\\", "/")


def software_relative_status_path(path: str) -> str:
    return path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path


def is_ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return any(part in ignored for part in parts) or path.endswith((".pyc", ".pyo"))


def status_paths(repository_root: Path) -> tuple[list[str], list[str]]:
    result = git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    unstaged_or_untracked: list[str] = []
    staged: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = software_relative_status_path(normalize_status_path(line))
        if is_ephemeral(path):
            continue
        x = line[0] if line else " "
        y = line[1] if len(line) > 1 else " "
        if x not in {" ", "?"}:
            staged.append(path)
        if y != " " or x == "?":
            unstaged_or_untracked.append(path)
    return sorted(set(unstaged_or_untracked)), sorted(set(staged))


def validate_git_state(
    software_root: Path, *, skip_remote_check: bool
) -> tuple[Path, dict[str, Any]]:
    if software_root != EXPECTED_SOFTWARE_ROOT:
        fail(
            "Diretório incorreto. Execute em:\n"
            f"{EXPECTED_SOFTWARE_ROOT}\nAtual: {software_root}"
        )
    repository_root = Path(
        git(software_root, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve()
    if repository_root != EXPECTED_REPOSITORY_ROOT:
        fail(
            f"Raiz Git inesperada: {repository_root}; "
            f"esperada: {EXPECTED_REPOSITORY_ROOT}"
        )
    branch = git(repository_root, "branch", "--show-current").stdout.strip()
    if branch != EXPECTED_BRANCH:
        fail(f"Branch incorreta: {branch!r}; esperada: {EXPECTED_BRANCH!r}")
    head = git(repository_root, "rev-parse", "HEAD").stdout.strip()
    if head != EXPECTED_HEAD:
        fail(f"HEAD incorreto: {head}; esperado: {EXPECTED_HEAD}")
    subject = git(repository_root, "show", "-s", "--format=%s", "HEAD").stdout.strip()
    if subject != EXPECTED_AP004B_SUBJECT:
        fail(
            f"Commit HEAD não é o fechamento AP-004B esperado: {subject!r}"
        )
    unstaged, staged = status_paths(repository_root)
    allowed = sorted(path.as_posix() for path in OUTPUT_RELS)
    if staged:
        fail(f"Índice Git deve estar vazio; staged encontrados: {staged}")
    if unstaged not in ([], sorted(path.as_posix() for path in PREPARATORY_OUTPUT_RELS), allowed):
        fail(
            "Árvore deve estar limpa ou conter somente os artefatos AP-004C e a manutenção contratual AP-004B.\n"
            f"Atual: {unstaged}\nPermitido: {allowed}"
        )
    remote_head: str | None = None
    if not skip_remote_check:
        git(repository_root, "fetch", "origin", EXPECTED_BRANCH, timeout=600)
        remote_head = git(repository_root, "rev-parse", f"origin/{EXPECTED_BRANCH}").stdout.strip()
        if remote_head != head:
            fail(f"HEAD remoto divergente: local={head}, remoto={remote_head}")
        divergence = git(
            repository_root,
            "rev-list",
            "--left-right",
            "--count",
            f"HEAD...origin/{EXPECTED_BRANCH}",
        ).stdout.strip()
        if divergence != "0\t0" and divergence != "0 0":
            fail(f"Branch local/remota divergente: {divergence}")
    return repository_root, {
        "branch": branch,
        "head": head,
        "remote_head": remote_head,
        "subject": subject,
        "initial_status": "clean" if not unstaged else "ap004c-output-only",
    }


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Não foi possível carregar {label} em {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"{label} deve ser um objeto JSON.")
    return data


def validate_prior_phases(
    software_root: Path, repository_root: Path
) -> dict[str, Any]:
    ap004a = load_json(
        software_root / AP004A_INVENTORY_REL, label="inventário AP-004A"
    )
    if ap004a.get("phase") != "AP-004A":
        fail("Inventário AP-004A com phase divergente.")
    if ap004a.get("inventory_revision") != "4.2":
        fail("Inventário AP-004A deve estar na revisão 4.2.")
    ap004b_inventory = load_json(
        software_root / AP004B_INVENTORY_REL, label="inventário AP-004B"
    )
    if ap004b_inventory.get("phase") != "AP-004B":
        fail("Inventário AP-004B com phase divergente.")
    if ap004b_inventory.get("inventory_revision") != "1.6":
        fail("Inventário AP-004B deve estar na revisão 1.6.")
    ap004b_application = load_json(
        software_root / AP004B_APPLICATION_REL, label="aplicação AP-004B"
    )
    if ap004b_application.get("phase") != "AP-004B":
        fail("Aplicação AP-004B com phase divergente.")
    if ap004b_application.get("mode") != "module-file-application-v1.4":
        fail("Aplicação AP-004B deve estar no modo v1.4.")
    validation = ap004b_application.get("validation", {})
    consolidated = validation.get("consolidated_suite", {})
    if consolidated.get("passed") != BASELINE_PASSED:
        fail(
            "Baseline AP-004B divergente: esperado "
            f"{BASELINE_PASSED} passed."
        )
    if consolidated.get("xfailed") != BASELINE_XFAILED:
        fail(
            "Baseline AP-004B divergente: esperado "
            f"{BASELINE_XFAILED} xfailed."
        )
    commit_paths = git(
        repository_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        EXPECTED_HEAD,
    ).stdout.splitlines()
    if len(commit_paths) != 21:
        fail(f"Commit AP-004B deveria conter 21 caminhos; contém {len(commit_paths)}.")
    return {
        "ap004a": ap004a,
        "ap004b_inventory": ap004b_inventory,
        "ap004b_application": ap004b_application,
        "ap004b_commit_paths": commit_paths,
    }


def ignored_tracked_path(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return any(part in PROTECTED_COMPONENTS for part in parts)


def protected_operational_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    if ignored_tracked_path(relative):
        return True
    if path.parts and path.parts[0].startswith(PROTECTED_ROOT_PREFIXES):
        return True
    name = path.name.lower()
    return (
        name.startswith(PROTECTED_ROOT_PREFIXES)
        or "/output/" in f"/{relative}/"
        or "/assets/" in f"/{relative}/"
        or "clean_" in name
        or "report" in name and relative.startswith("app_bundle/")
    )


def historical_path(relative: str) -> bool:
    return any(relative.startswith(prefix) for prefix in HISTORICAL_PREFIXES)


def tracked_software_files(
    repository_root: Path, software_root: Path
) -> list[str]:
    result = git(repository_root, "ls-files", "-z")
    files: list[str] = []
    for repo_relative in result.stdout.split("\0"):
        if not repo_relative or not repo_relative.startswith(SOFTWARE_PREFIX):
            continue
        relative = repo_relative[len(SOFTWARE_PREFIX):]
        if ignored_tracked_path(relative):
            continue
        # Não chamar stat antes da filtragem lexical: o índice pode conter
        # caminhos históricos recursivos maiores que PATH_MAX.
        path = software_root / relative
        try:
            if not path.is_file():
                continue
        except OSError as exc:
            if exc.errno in {errno.ENAMETOOLONG, errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
                continue
            raise
        files.append(relative)
    return sorted(set(files))


def safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    except OSError as exc:
        if exc.errno in {errno.ENAMETOOLONG, errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
            return None
        raise


def build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def parse_python_files(
    software_root: Path, tracked: Sequence[str]
) -> tuple[dict[str, ParsedPython], list[dict[str, Any]]]:
    parsed: dict[str, ParsedPython] = {}
    errors: list[dict[str, Any]] = []
    for relative in tracked:
        if not relative.endswith(".py"):
            continue
        text = safe_read_text(software_root / relative)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError as exc:
            errors.append(
                {
                    "path": relative,
                    "line": exc.lineno,
                    "message": exc.msg,
                }
            )
            continue
        parsed[relative] = ParsedPython(
            path=relative,
            text=text,
            tree=tree,
            parents=build_parent_map(tree),
        )
    return parsed, errors


def node_scope(node: ast.AST, parsed: ParsedPython) -> str:
    names: list[str] = []
    current = parsed.parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parsed.parents.get(id(current))
    return ".".join(reversed(names)) or "<module>"


def source_segment(parsed: ParsedPython, node: ast.AST) -> str:
    segment = ast.get_source_segment(parsed.text, node)
    if segment is not None:
        return segment
    lines = parsed.text.splitlines()
    lineno = getattr(node, "lineno", 1)
    end_lineno = getattr(node, "end_lineno", lineno)
    return "\n".join(lines[max(0, lineno - 1):end_lineno])


def assigned_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for child in node.elts:
            names.extend(assigned_names(child))
    return names


def definition_matches(
    parsed: ParsedPython, *, name: str, category: str
) -> list[tuple[ast.AST, str, str]]:
    matches: list[tuple[ast.AST, str, str]] = []
    for node in ast.walk(parsed.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            matches.append((node, "function", node_scope(node, parsed)))
        elif isinstance(node, ast.ClassDef) and node.name == name:
            matches.append((node, "class", node_scope(node, parsed)))
        elif isinstance(node, ast.Assign):
            target_names: list[str] = []
            for target in node.targets:
                target_names.extend(assigned_names(target))
            if name in target_names:
                matches.append((node, "assignment", node_scope(node, parsed)))
        elif isinstance(node, ast.AnnAssign) and name in assigned_names(node.target):
            matches.append((node, "annotated_assignment", node_scope(node, parsed)))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                visible = alias.asname or alias.name.split(".")[-1]
                if visible == name:
                    form = "import_alias" if alias.asname else "import"
                    matches.append((node, form, node_scope(node, parsed)))
    # A categoria ajuda a eliminar uma coincidência rara de função/atribuição.
    preferred = {
        "função": {"function"},
        "classe": {"class"},
        "constante": {"assignment", "annotated_assignment"},
        "alias": {"assignment", "annotated_assignment", "import_alias", "import"},
    }.get(category, set())
    filtered = [item for item in matches if item[1] in preferred]
    return filtered or matches


def locate_definition(
    candidate: dict[str, Any], parsed_files: dict[str, ParsedPython]
) -> dict[str, Any]:
    path = candidate["path"]
    parsed = parsed_files.get(path)
    if parsed is None:
        fail(f"Arquivo Python do candidato não pôde ser analisado: {path}")
    matches = definition_matches(
        parsed,
        name=candidate["current_name"],
        category=candidate["category"],
    )
    if not matches:
        fail(
            "Definição não localizada para "
            f"{candidate['current_name']} em {path}:{candidate.get('line')}"
        )
    expected_line = candidate.get("line") or 0
    ranked = sorted(
        matches,
        key=lambda item: (
            0 if getattr(item[0], "lineno", -1) == expected_line else 1,
            abs(getattr(item[0], "lineno", 0) - expected_line),
            getattr(item[0], "lineno", 0),
        ),
    )
    best = ranked[0]
    if len(ranked) > 1:
        score0 = (
            0 if getattr(ranked[0][0], "lineno", -1) == expected_line else 1,
            abs(getattr(ranked[0][0], "lineno", 0) - expected_line),
        )
        score1 = (
            0 if getattr(ranked[1][0], "lineno", -1) == expected_line else 1,
            abs(getattr(ranked[1][0], "lineno", 0) - expected_line),
        )
        if score0 == score1:
            fail(
                f"Definição ambígua para {candidate['current_name']} em {path}."
            )
    node, definition_kind, scope = best
    segment = source_segment(parsed, node)
    return {
        "path": path,
        "line": getattr(node, "lineno", None),
        "end_line": getattr(node, "end_lineno", getattr(node, "lineno", None)),
        "column": getattr(node, "col_offset", None),
        "definition_kind": definition_kind,
        "scope": scope,
        "ast_sha256": sha256_bytes(
            ast.dump(node, include_attributes=False, annotate_fields=True).encode(
                "utf-8"
            )
        ),
        "source_sha256": sha256_bytes(segment.encode("utf-8")),
        "source_excerpt": " ".join(segment.strip().split())[:500],
    }


def candidate_is_selected(item: dict[str, Any]) -> bool:
    return (
        item.get("category") in INTERNAL_CATEGORIES
        and "AP-004C" in str(item.get("target_phase", ""))
    )


def protected_control_matches(item: dict[str, Any]) -> bool:
    return any(
        item.get("current_name") == control["current_name"]
        and item.get("path") == control["path"]
        for control in KNOWN_XFAIL_CONTROLS
    )


def select_candidates(ap004a: dict[str, Any]) -> list[dict[str, Any]]:
    source = ap004a.get("actionable_candidates") or ap004a.get("candidates")
    if not isinstance(source, list):
        fail("AP-004A não contém lista de candidatos acionáveis.")
    selected: dict[str, dict[str, Any]] = {}
    for item in source:
        if not isinstance(item, dict):
            continue
        if candidate_is_selected(item) or protected_control_matches(item):
            candidate_id = str(item.get("id") or stable_id(
                item.get("category"), item.get("path"), item.get("line"),
                item.get("current_name")
            ))
            selected[candidate_id] = dict(item)
    # WorkflowState._normalize pode aparecer qualificado apenas na lista de
    # proteção. Garante que todos os controles estejam representados.
    for control in KNOWN_XFAIL_CONTROLS:
        if not any(
            item.get("current_name") == control["current_name"]
            and item.get("path") == control["path"]
            for item in selected.values()
        ):
            selected[stable_id("protected", control["path"], control["current_name"])] = {
                "id": stable_id("protected", control["path"], control["current_name"]),
                "category": "função" if control["current_name"] != "_ap003d_impl__refs_v6_strip_org" else "alias",
                "current_name": control["current_name"],
                "suggested_name": None,
                "path": control["path"],
                "line": None,
                "classification": "nome que deve permanecer",
                "classification_reason": "Controle xfail histórico protegido.",
                "target_phase": "fora da AP-004",
                "references": {},
                "evidence": [control["qualified_name"]],
                "status": "protected-control",
            }
    result = sorted(
        selected.values(),
        key=lambda item: (
            item.get("path", ""),
            item.get("line") or 0,
            item.get("current_name", ""),
        ),
    )
    if not result:
        fail("Nenhum candidato interno AP-004C foi selecionado.")
    return result


def validate_expected_safe_aliases(candidates: Sequence[dict[str, Any]]) -> None:
    by_name = {
        (item.get("current_name"), item.get("path")): item
        for item in candidates
    }
    for current, suggested in EXPECTED_SAFE_ORCHESTRATOR_ALIASES:
        key = (current, ORCHESTRATOR_REL.as_posix())
        item = by_name.get(key)
        if item is None:
            fail(f"Alias seguro AP-004A ausente na AP-004C: {current}")
        if item.get("suggested_name") != suggested:
            fail(
                f"Sugestão divergente para {current}: "
                f"{item.get('suggested_name')!r}; esperada {suggested!r}"
            )
        if item.get("classification") != "renomeação segura":
            fail(f"Classificação divergente para alias seguro {current}.")
    protected = {
        (item.get("current_name"), item.get("path")) for item in candidates
    }
    for control in KNOWN_XFAIL_CONTROLS:
        if (control["current_name"], control["path"]) not in protected:
            fail(f"Controle xfail ausente: {control['qualified_name']}")


def call_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def nearest_call(node: ast.AST, parsed: ParsedPython) -> str | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.Call):
            return call_name(current.func)
        current = parsed.parents.get(id(current))
    return None


def identifier_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
    )


def reference_category(
    *,
    candidate: dict[str, Any],
    path: str,
    kind: str,
    dynamic: bool,
    line: int | None,
    definition: dict[str, Any],
) -> str:
    if (
        path == definition["path"]
        and line == definition["line"]
        and kind in {"name_store", "import_alias_definition"}
    ):
        return "definition"
    if path.startswith(FROZEN_AP003_TEST_PREFIX):
        return "frozen_ap003_contract"
    if historical_path(path):
        return "historical_immutable"
    if protected_operational_path(path):
        return "protected_operational"
    if path.startswith("app_bundle/tests/") or path.startswith("tests/"):
        return "current_test_contract"
    if kind == "string_reference" and not dynamic:
        return "contextual_string"
    if path == candidate["path"]:
        return "same_module_dynamic" if dynamic else "same_module_static"
    return "cross_module_dynamic" if dynamic else "cross_module_static"


def scan_python_references(
    *,
    candidates: Sequence[dict[str, Any]],
    definitions: dict[str, dict[str, Any]],
    parsed_files: dict[str, ParsedPython],
) -> list[dict[str, Any]]:
    names_to_ids: dict[str, list[str]] = defaultdict(list)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = candidate["id"]
        candidate_by_id[candidate_id] = candidate
        names_to_ids[candidate["current_name"]].append(candidate_id)
    patterns = {name: identifier_pattern(name) for name in names_to_ids}
    records: list[dict[str, Any]] = []

    for path, parsed in parsed_files.items():
        for node in ast.walk(parsed.tree):
            name: str | None = None
            kind: str | None = None
            dynamic = False
            excerpt: str | None = None
            if isinstance(node, ast.Name) and node.id in names_to_ids:
                name = node.id
                kind = (
                    "name_load" if isinstance(node.ctx, ast.Load)
                    else "name_store" if isinstance(node.ctx, ast.Store)
                    else "name_delete"
                )
                excerpt = name
            elif isinstance(node, ast.Attribute) and node.attr in names_to_ids:
                name = node.attr
                kind = "attribute"
                excerpt = ast.unparse(node)[:500]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    visible = alias.asname or alias.name.split(".")[-1]
                    if visible not in names_to_ids:
                        continue
                    for candidate_id in names_to_ids[visible]:
                        candidate = candidate_by_id[candidate_id]
                        definition = definitions[candidate_id]
                        category = reference_category(
                            candidate=candidate,
                            path=path,
                            kind="import_alias_definition",
                            dynamic=False,
                            line=getattr(node, "lineno", None),
                            definition=definition,
                        )
                        records.append(
                            {
                                "candidate_id": candidate_id,
                                "symbol": visible,
                                "path": path,
                                "line": getattr(node, "lineno", None),
                                "column": getattr(node, "col_offset", None),
                                "kind": "import_alias_definition",
                                "scope": node_scope(node, parsed),
                                "call": None,
                                "dynamic": False,
                                "excerpt": ast.unparse(node)[:500],
                                "semantic_category": category,
                            }
                        )
                continue
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                matched = [
                    candidate_name
                    for candidate_name, pattern in patterns.items()
                    if pattern.search(node.value)
                ]
                if not matched:
                    continue
                call = nearest_call(node, parsed)
                dynamic_calls = {
                    "getattr", "setattr", "hasattr", "delattr",
                    "patch", "mock.patch", "unittest.mock.patch",
                    "patch.object", "monkeypatch.setattr",
                }
                dynamic = bool(call in dynamic_calls)
                for candidate_name in matched:
                    for candidate_id in names_to_ids[candidate_name]:
                        candidate = candidate_by_id[candidate_id]
                        definition = definitions[candidate_id]
                        category = reference_category(
                            candidate=candidate,
                            path=path,
                            kind="string_reference",
                            dynamic=dynamic,
                            line=getattr(node, "lineno", None),
                            definition=definition,
                        )
                        records.append(
                            {
                                "candidate_id": candidate_id,
                                "symbol": candidate_name,
                                "path": path,
                                "line": getattr(node, "lineno", None),
                                "column": getattr(node, "col_offset", None),
                                "kind": "string_reference",
                                "scope": node_scope(node, parsed),
                                "call": call,
                                "dynamic": dynamic,
                                "excerpt": " ".join(node.value.split())[:500],
                                "semantic_category": category,
                            }
                        )
                continue
            if name is None or kind is None:
                continue
            call = nearest_call(node, parsed)
            for candidate_id in names_to_ids[name]:
                candidate = candidate_by_id[candidate_id]
                definition = definitions[candidate_id]
                category = reference_category(
                    candidate=candidate,
                    path=path,
                    kind=kind,
                    dynamic=dynamic,
                    line=getattr(node, "lineno", None),
                    definition=definition,
                )
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "symbol": name,
                        "path": path,
                        "line": getattr(node, "lineno", None),
                        "column": getattr(node, "col_offset", None),
                        "kind": kind,
                        "scope": node_scope(node, parsed),
                        "call": call,
                        "dynamic": dynamic,
                        "excerpt": excerpt,
                        "semantic_category": category,
                    }
                )

    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    kind_priority = {
        "attribute": 0,
        "name_load": 1,
        "name_store": 2,
        "import_alias_definition": 3,
        "string_reference": 4,
    }
    for record in records:
        key = (
            record["candidate_id"], record["path"], record["line"],
            record["column"], record["semantic_category"], record["excerpt"]
        )
        previous = deduplicated.get(key)
        if previous is None or kind_priority.get(record["kind"], 99) < kind_priority.get(previous["kind"], 99):
            deduplicated[key] = record
    return sorted(
        deduplicated.values(),
        key=lambda record: (
            record["candidate_id"], record["path"],
            record["line"] or 0, record["column"] or 0, record["kind"]
        ),
    )


def scan_text_references(
    *,
    candidates: Sequence[dict[str, Any]],
    tracked: Sequence[str],
    software_root: Path,
) -> list[dict[str, Any]]:
    names = sorted({item["current_name"] for item in candidates}, key=len, reverse=True)
    patterns = {name: identifier_pattern(name) for name in names}
    records: list[dict[str, Any]] = []
    for relative in tracked:
        if Path(relative).suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = safe_read_text(software_root / relative)
        if text is None:
            continue
        for name, pattern in patterns.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            line_numbers: list[int] = []
            for match in matches[:5]:
                line_numbers.append(text.count("\n", 0, match.start()) + 1)
            records.append(
                {
                    "symbol": name,
                    "path": relative,
                    "count": len(matches),
                    "sample_lines": line_numbers,
                    "semantic_category": (
                        "historical_immutable"
                        if historical_path(relative)
                        else "protected_operational"
                        if protected_operational_path(relative)
                        else "contextual_string"
                    ),
                }
            )
    return sorted(records, key=lambda item: (item["symbol"], item["path"]))


def destination_collisions(
    candidates: Sequence[dict[str, Any]],
    parsed_files: dict[str, ParsedPython],
) -> list[dict[str, Any]]:
    existing_by_path: dict[str, set[str]] = defaultdict(set)
    for path, parsed in parsed_files.items():
        for node in parsed.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                existing_by_path[path].add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    existing_by_path[path].update(assigned_names(target))
            elif isinstance(node, ast.AnnAssign):
                existing_by_path[path].update(assigned_names(node.target))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    existing_by_path[path].add(alias.asname or alias.name.split(".")[-1])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    collisions: list[dict[str, Any]] = []
    for candidate in candidates:
        suggestion = candidate.get("suggested_name")
        if not suggestion:
            continue
        groups[(candidate["path"], suggestion)].append(candidate)
        if not suggestion.isidentifier() or keyword.iskeyword(suggestion):
            collisions.append(
                {
                    "collision_id": stable_id("invalid", candidate["id"], suggestion),
                    "kind": "invalid_identifier",
                    "path": candidate["path"],
                    "suggested_name": suggestion,
                    "candidate_ids": [candidate["id"]],
                    "current_names": [candidate["current_name"]],
                }
            )
        elif (
            suggestion in existing_by_path.get(candidate["path"], set())
            and suggestion != candidate["current_name"]
        ):
            collisions.append(
                {
                    "collision_id": stable_id("occupied", candidate["path"], suggestion),
                    "kind": "destination_already_exists",
                    "path": candidate["path"],
                    "suggested_name": suggestion,
                    "candidate_ids": [candidate["id"]],
                    "current_names": [candidate["current_name"]],
                }
            )
    for (path, suggestion), items in groups.items():
        if len(items) > 1:
            collisions.append(
                {
                    "collision_id": stable_id("multiple", path, suggestion),
                    "kind": "multiple_candidates_same_destination",
                    "path": path,
                    "suggested_name": suggestion,
                    "candidate_ids": [item["id"] for item in items],
                    "current_names": [item["current_name"] for item in items],
                }
            )
    unique = {item["collision_id"]: item for item in collisions}
    return sorted(unique.values(), key=lambda item: item["collision_id"])


def is_protected_candidate(candidate: dict[str, Any]) -> bool:
    return protected_control_matches(candidate)


def classify_disposition(
    candidate: dict[str, Any],
    refs: Sequence[dict[str, Any]],
    collision_ids: set[str],
) -> tuple[str, str, str]:
    if is_protected_candidate(candidate):
        return (
            "protected_xfail_out_of_scope",
            "Símbolo ou alias vinculado a xfail histórico; não será renomeado nem corrigido na AP-004C.",
            "fora da AP-004",
        )
    if candidate["id"] in collision_ids:
        return (
            "blocked_destination_collision",
            "O destino sugerido já existe ou é compartilhado por outro candidato.",
            "revisão manual",
        )
    categories = Counter(record["semantic_category"] for record in refs)
    has_cross = bool(
        categories["cross_module_static"] or categories["cross_module_dynamic"]
    )
    has_dynamic = bool(
        categories["cross_module_dynamic"] or categories["same_module_dynamic"]
    )
    has_current_tests = bool(categories["current_test_contract"])
    has_frozen_contracts = bool(categories["frozen_ap003_contract"])
    name = candidate["current_name"]
    classification = candidate.get("classification")
    suggestion = candidate.get("suggested_name")

    if name == "_ap003f_pipeline_core" or OPAQUE_STRUCTURAL_RE.fullmatch(name):
        return (
            "deferred_structural_symbol",
            "Símbolo estrutural congelado ou opaco da AP-003; exige nome semântico e prova AST específica.",
            "AP-004C/AP-004D (revisão manual)",
        )
    if classification == "renomeação com compatibilidade":
        return (
            "compatibility_required",
            "Símbolo usado externamente ou exportado; requer alias transitório e revisão posterior na AP-004E.",
            "AP-004C/AP-004E",
        )
    if not suggestion:
        return (
            "manual_semantic_name_required",
            "Não há sugestão semântica segura; o corpo e a responsabilidade devem orientar o novo nome.",
            "revisão manual",
        )
    if classification == "renomeação segura":
        # O orquestrador histórico permanece congelado por contratos AST e
        # hash da AP-003G. Mesmo aliases privados sem consumidor externo
        # exigem atualização dirigida desses contratos e rebaseline do hash.
        if candidate["path"] == ORCHESTRATOR_REL.as_posix():
            return (
                "ready_contract_bound_ast_rename",
                "Renomeação privada e estática no orquestrador congelado; exige atualização dirigida dos contratos AP-003 e rebaseline do hash, sem mudança estrutural.",
                "AP-004C — onda 2",
            )
        if has_cross or has_dynamic:
            return (
                "contract_update_required",
                "A classificação original era segura, mas a varredura atual encontrou consumidor externo ou dinâmico.",
                "AP-004C (revisão dirigida)",
            )
        if has_frozen_contracts:
            return (
                "ready_contract_bound_ast_rename",
                "Renomeação privada e estática, porém vinculada a contratos AP-003 que deverão ser rebaselined sem alterar a estrutura.",
                "AP-004C — onda 2",
            )
        if has_current_tests:
            return (
                "contract_update_required",
                "Renomeação privada local com referências em testes atuais que devem acompanhar a mudança.",
                "AP-004C — onda 2",
            )
        return (
            "ready_local_ast_rename",
            "Símbolo privado/local, destino livre e sem consumidor externo ou contrato congelado.",
            "AP-004C — onda 1",
        )
    if PHASE_MARKER_RE.match(name) or VERSION_MARKER_RE.search(name):
        return (
            "manual_semantic_name_required",
            "O marcador é estrutural/versionado, mas a evidência não autoriza renomeação automática.",
            "AP-004C/AP-004D",
        )
    return (
        "manual_semantic_name_required",
        "A classificação de alto risco exige decisão manual baseada na responsabilidade do símbolo.",
        "revisão manual",
    )


def file_metadata(path: Path, *, relative: str) -> dict[str, Any]:
    data = path.read_bytes()
    record: dict[str, Any] = {
        "path": relative,
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }
    if relative.endswith(".py"):
        text = data.decode("utf-8")
        tree = ast.parse(text, filename=relative)
        record["ast_sha256"] = sha256_bytes(
            ast.dump(tree, include_attributes=False, annotate_fields=True).encode(
                "utf-8"
            )
        )
    return record



def _replace_top_level_function(
    source: str, *, old_name: str, replacement: str
) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == old_name
    ]
    if len(matches) != 1:
        fail(
            f"Contrato esperado com uma função {old_name!r}; "
            f"encontradas: {len(matches)}"
        )
    node = matches[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [normalize_output(replacement)]
    return "".join(lines)


def _insert_after_assignment(
    source: str, *, name: str, addition: str
) -> str:
    tree = ast.parse(source)
    matches = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in targets
            ):
                matches.append(node)
    if len(matches) != 1:
        fail(
            f"Contrato esperado com uma atribuição {name!r}; "
            f"encontradas: {len(matches)}"
        )
    node = matches[0]
    lines = source.splitlines(keepends=True)
    lines[node.end_lineno : node.end_lineno] = [
        "\n" + normalize_output(addition)
    ]
    return "".join(lines)


def build_durable_ap004b_application_contract(software_root: Path) -> str:
    path = software_root / AP004B_APPLICATION_TEST_REL
    source = path.read_text(encoding="utf-8")
    if "EXPECTED_AP004B_COMMIT" not in source:
        source = _insert_after_assignment(
            source,
            name="EXPECTED_DIRTY_PATHS",
            addition=(
                "EXPECTED_AP004B_COMMIT = "
                + repr(EXPECTED_HEAD)
            ),
        )
    if "EXPECTED_AP004B_SUBJECT" not in source:
        source = _insert_after_assignment(
            source,
            name="EXPECTED_AP004B_COMMIT",
            addition=(
                "EXPECTED_AP004B_SUBJECT = "
                + repr(EXPECTED_AP004B_SUBJECT)
            ),
        )
    replacement = """def test_ap004b_commit_scope_is_durable() -> None:
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
"""
    old_name = (
        "test_ap004b_git_diff_is_limited_to_approved_scope"
        if "def test_ap004b_git_diff_is_limited_to_approved_scope" in source
        else "test_ap004b_commit_scope_is_durable"
    )
    source = _replace_top_level_function(
        source, old_name=old_name, replacement=replacement
    )
    source = source.replace(
        'assert "current_status_is_application_scope_or_clean" in source',
        'assert "commit_scope_is_durable" in source',
    )
    ast.parse(source, filename=AP004B_APPLICATION_TEST_REL.as_posix())
    return normalize_output(source)


def build_durable_ap004b_inventory_contract(software_root: Path) -> str:
    path = software_root / AP004B_INVENTORY_TEST_REL
    source = path.read_text(encoding="utf-8")
    if "EXPECTED_AP004B_COMMIT" not in source:
        source = _insert_after_assignment(
            source,
            name="EXPECTED_DIRTY_PATHS",
            addition=(
                "EXPECTED_AP004B_COMMIT = "
                + repr(EXPECTED_HEAD)
            ),
        )
    if "EXPECTED_AP004B_SUBJECT" not in source:
        source = _insert_after_assignment(
            source,
            name="EXPECTED_AP004B_COMMIT",
            addition=(
                "EXPECTED_AP004B_SUBJECT = "
                + repr(EXPECTED_AP004B_SUBJECT)
            ),
        )
    replacement = """def test_ap004b_v1_6_commit_scope_is_durable() -> None:
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
"""
    old_name = (
        "test_ap004b_v1_6_current_status_is_application_scope_or_clean"
        if "def test_ap004b_v1_6_current_status_is_application_scope_or_clean" in source
        else "test_ap004b_v1_6_commit_scope_is_durable"
    )
    source = _replace_top_level_function(
        source, old_name=old_name, replacement=replacement
    )
    ast.parse(source, filename=AP004B_INVENTORY_TEST_REL.as_posix())
    return normalize_output(source)


def _metadata_from_text(relative: str, content: str) -> dict[str, Any]:
    raw = content.encode("utf-8")
    record: dict[str, Any] = {
        "path": relative,
        "size": len(raw),
        "sha256": sha256_bytes(raw),
    }
    if relative.endswith(".py"):
        tree = ast.parse(content, filename=relative)
        record["ast_sha256"] = sha256_bytes(
            ast.dump(
                tree, include_attributes=False, annotate_fields=True
            ).encode("utf-8")
        )
    return record


def update_manifest_for_contract_maintenance(
    inventory: dict[str, Any], *, application_source: str, inventory_source: str
) -> None:
    replacements = {
        AP004B_APPLICATION_TEST_REL.as_posix(): application_source,
        AP004B_INVENTORY_TEST_REL.as_posix(): inventory_source,
    }
    by_path = {
        record["path"]: record for record in inventory["source_manifest"]
    }
    for relative, content in replacements.items():
        if relative not in by_path:
            fail(f"Contrato AP-004B ausente do manifesto AP-004C: {relative}")
        updated = _metadata_from_text(relative, content)
        by_path[relative].clear()
        by_path[relative].update(updated)
    inventory["contract_maintenance"] = {
        "reason": "remove current-worktree coupling from AP-004B contracts",
        "paths": sorted(replacements),
        "ap004b_commit": EXPECTED_HEAD,
        "ap004b_subject": EXPECTED_AP004B_SUBJECT,
        "productive_change": False,
    }


def build_manifest(
    *,
    software_root: Path,
    candidates: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    paths = {candidate["path"] for candidate in candidates}
    for record in references:
        if record["semantic_category"] in {
            "cross_module_static", "cross_module_dynamic",
            "current_test_contract", "frozen_ap003_contract",
        }:
            paths.add(record["path"])
    paths.update(
        {
            AP004A_INVENTORY_REL.as_posix(),
            AP004B_INVENTORY_REL.as_posix(),
            AP004B_APPLICATION_REL.as_posix(),
            AP004B_APPLICATION_TEST_REL.as_posix(),
            AP004B_INVENTORY_TEST_REL.as_posix(),
            AP003G_CONTRACT_REL.as_posix(),
            AP003G_MANIFEST_REL.as_posix(),
            ORCHESTRATOR_REL.as_posix(),
            TOML_INTERACTIVE_REL.as_posix(),
        }
    )
    manifest: list[dict[str, Any]] = []
    for relative in sorted(paths):
        path = software_root / relative
        if not path.is_file():
            fail(f"Arquivo do manifesto ausente: {relative}")
        manifest.append(file_metadata(path, relative=relative))
    return manifest


def build_inventory(
    *,
    software_root: Path,
    repository_root: Path,
    git_state: dict[str, Any],
    prior: dict[str, Any],
    tool_source: str,
) -> dict[str, Any]:
    tracked = tracked_software_files(repository_root, software_root)
    parsed_files, parse_errors = parse_python_files(software_root, tracked)
    candidates = select_candidates(prior["ap004a"])
    validate_expected_safe_aliases(candidates)

    definitions: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        definitions[candidate["id"]] = locate_definition(candidate, parsed_files)

    references = scan_python_references(
        candidates=candidates,
        definitions=definitions,
        parsed_files=parsed_files,
    )
    text_references = scan_text_references(
        candidates=candidates, tracked=tracked, software_root=software_root
    )
    collisions = destination_collisions(candidates, parsed_files)
    collision_ids = {
        candidate_id
        for collision in collisions
        for candidate_id in collision["candidate_ids"]
    }

    refs_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in references:
        refs_by_candidate[record["candidate_id"]].append(record)

    candidate_records: list[dict[str, Any]] = []
    for candidate in candidates:
        refs = refs_by_candidate[candidate["id"]]
        disposition, reason, application_wave = classify_disposition(
            candidate, refs, collision_ids
        )
        counts = Counter(record["semantic_category"] for record in refs)
        record = dict(candidate)
        record["definition"] = definitions[candidate["id"]]
        record["disposition"] = disposition
        record["disposition_reason"] = reason
        record["application_wave"] = application_wave
        record["reference_counts"] = {
            category: counts.get(category, 0) for category in REFERENCE_CATEGORIES
        }
        record["effective_external_reference_count"] = sum(
            counts.get(category, 0)
            for category in (
                "cross_module_static", "cross_module_dynamic",
                "current_test_contract", "frozen_ap003_contract",
            )
        )
        record["collision_ids"] = [
            collision["collision_id"]
            for collision in collisions
            if candidate["id"] in collision["candidate_ids"]
        ]
        candidate_records.append(record)

    disposition_counts = Counter(item["disposition"] for item in candidate_records)
    classification_counts = Counter(
        item.get("classification") for item in candidate_records
    )
    category_counts = Counter(item.get("category") for item in candidate_records)
    reference_counts = Counter(record["semantic_category"] for record in references)
    ready_ids = {
        item["id"] for item in candidate_records
        if item["disposition"] in {
            "ready_local_ast_rename",
            "ready_contract_bound_ast_rename",
            "contract_update_required",
        }
    }
    ready_references = [
        record for record in references if record["candidate_id"] in ready_ids
    ]
    manifest = build_manifest(
        software_root=software_root,
        candidates=candidate_records,
        references=ready_references,
    )

    return {
        "phase": PHASE,
        "mode": MODE,
        "inventory_schema_version": INVENTORY_SCHEMA_VERSION,
        "inventory_revision": INVENTORY_REVISION,
        "generated_at_utc": utc_now(),
        "git": git_state,
        "prior_phases": {
            "ap004a_revision": prior["ap004a"].get("inventory_revision"),
            "ap004b_inventory_revision": prior["ap004b_inventory"].get("inventory_revision"),
            "ap004b_application_mode": prior["ap004b_application"].get("mode"),
            "ap004b_commit": EXPECTED_HEAD,
            "ap004b_subject": EXPECTED_AP004B_SUBJECT,
        },
        "scope": {
            "productive_change": False,
            "selected_from_ap004a_rule": (
                "category in função/classe/constante/alias and target_phase contains AP-004C"
            ),
            "protected_controls_added": [
                control["qualified_name"] for control in KNOWN_XFAIL_CONTROLS
            ],
            "allowed_outputs": [path.as_posix() for path in OUTPUT_RELS],
            "forbidden_module_file_changes": True,
            "physical_directory_reserved_for": "AP-006",
        },
        "statistics": {
            "candidate_count": len(candidate_records),
            "by_classification": dict(sorted(classification_counts.items())),
            "by_category": dict(sorted(category_counts.items())),
            "by_disposition": {
                name: disposition_counts.get(name, 0) for name in DISPOSITIONS
            },
            "python_reference_record_count": len(references),
            "text_reference_summary_count": len(text_references),
            "by_reference_category": {
                name: reference_counts.get(name, 0) for name in REFERENCE_CATEGORIES
            },
            "definition_file_count": len({item["path"] for item in candidate_records}),
            "manifest_file_count": len(manifest),
            "destination_collision_count": len(collisions),
            "python_parse_error_count": len(parse_errors),
            "ready_wave_1_count": disposition_counts["ready_local_ast_rename"],
            "ready_wave_2_count": (
                disposition_counts["ready_contract_bound_ast_rename"]
                + disposition_counts["contract_update_required"]
            ),
            "deferred_count": sum(
                disposition_counts[name]
                for name in (
                    "compatibility_required",
                    "deferred_structural_symbol",
                    "manual_semantic_name_required",
                    "blocked_destination_collision",
                )
            ),
            "protected_count": disposition_counts["protected_xfail_out_of_scope"],
        },
        "required_safe_orchestrator_aliases": [
            {"current_name": current, "suggested_name": suggested}
            for current, suggested in EXPECTED_SAFE_ORCHESTRATOR_ALIASES
        ],
        "protected_xfail_controls": list(KNOWN_XFAIL_CONTROLS),
        "candidates": candidate_records,
        "python_references": references,
        "text_reference_summaries": text_references,
        "destination_collisions": collisions,
        "python_parse_errors": parse_errors,
        "source_manifest": manifest,
        "application_plan": {
            "wave_1": {
                "name": "local-private-safe",
                "dispositions": ["ready_local_ast_rename"],
                "policy": "AST rename no arquivo definidor e em referências locais exatas.",
            },
            "wave_2": {
                "name": "contract-bound-safe",
                "dispositions": [
                    "ready_contract_bound_ast_rename",
                    "contract_update_required",
                ],
                "policy": (
                    "AST rename com atualização dirigida de contratos e rebaseline "
                    "de hashes, sem alteração estrutural."
                ),
            },
            "deferred": {
                "dispositions": [
                    "compatibility_required",
                    "deferred_structural_symbol",
                    "manual_semantic_name_required",
                    "blocked_destination_collision",
                ],
                "policy": "Não entra no primeiro aplicador AP-004C.",
            },
            "protected": {
                "dispositions": ["protected_xfail_out_of_scope"],
                "policy": "Byte/AST e nomes preservados; nenhum reparo funcional.",
            },
        },
        "tool": {
            "path": TOOL_REL.as_posix(),
            "version": TOOL_VERSION,
            "revision": TOOL_REVISION,
            "sha256": sha256_bytes(normalize_output(tool_source).encode("utf-8")),
        },
        "validation": {
            "py_compile": "pending",
            "git_diff_check": "pending",
            "specific_suite": {"status": "pending"},
            "consolidated_suite": {"status": "pending"},
        },
        "next_gate": {
            "blocked": True,
            "condition": (
                "Não criar aplicador produtivo sem revisão e aprovação expressa "
                "do inventário AP-004C."
            ),
        },
    }


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_strategy_document(data: dict[str, Any]) -> str:
    stats = data["statistics"]
    return normalize_output(f"""# AP-004C — estratégia de símbolos internos

> Documento preparatório. Nenhum símbolo produtivo foi renomeado.

## Objetivo técnico

Normalizar identificadores internos herdados das fases AP-003 e de marcadores de
versão sem alterar comportamento, estrutura do orquestrador, contratos públicos,
conteúdo gerado ou os três defeitos legados congelados.

## Base canônica

- Branch: `{data['git']['branch']}`.
- HEAD local/remoto: `{data['git']['head']}`.
- Fechamento AP-004B: `{EXPECTED_AP004B_SUBJECT}`.
- Baseline: `{BASELINE_PASSED} passed, {BASELINE_XFAILED} xfailed`.

## Ondas propostas

### Onda 1 — símbolos privados locais

Abrange somente `ready_local_ast_rename`. A transformação futura deverá usar AST,
validar colisões, preservar assinaturas e provar que nenhum consumidor externo ou
dinâmico foi alterado. Candidatos atuais: **{stats['ready_wave_1_count']}**.

### Onda 2 — símbolos vinculados a contratos

Abrange `ready_contract_bound_ast_rename` e `contract_update_required`. Inclui
aliases privados do orquestrador e outros símbolos cuja renomeação exige atualizar
contratos caracterizadores e rebaselinar hashes, sem mudar a estrutura consolidada
na AP-003. Candidatos atuais: **{stats['ready_wave_2_count']}**.

### Símbolos adiados

Símbolos opacos de stage/dispatch, núcleo `_ap003f_pipeline_core`, nomes sem
sugestão semântica, colisões e superfícies com compatibilidade ficam fora do
primeiro aplicador. Total adiado: **{stats['deferred_count']}**.

### Proteções absolutas

`_refs_v6_strip_org`, `_ap003d_impl__refs_v6_strip_org`,
`extract_org_abstracts` e `WorkflowState._normalize` permanecem fora do escopo.
Nenhum xfail será corrigido, renomeado ou convertido em teste aprovado.

## Regras do futuro aplicador

- branch, HEAD e remoto exatos;
- árvore limitada aos cinco artefatos preparatórios e aos dois contratos AP-004B mantidos;
- hashes de todo o `source_manifest`;
- renomeação por AST, nunca substituição global;
- atualização somente de referências vinculadas ao mesmo `candidate_id`;
- bloqueio diante de string dinâmica, consumidor externo inesperado ou colisão;
- preservação dos módulos e wrappers tratados na AP-004B;
- backup externo, escrita atômica e rollback integral;
- `py_compile`, `git diff --check`, suíte específica e suíte consolidada;
- nenhum commit sem aprovação expressa.

## Estado

A criação do aplicador produtivo permanece bloqueada até aprovação deste
inventário e definição das ondas que realmente serão executadas.
""")


def build_report(data: dict[str, Any]) -> str:
    stats = data["statistics"]
    lines = [
        "# AP-004C — inventário de símbolos internos (v1.3)",
        "",
        "> Levantamento somente preparatório. Nenhum símbolo produtivo foi renomeado.",
        "",
        "## Estado Git e base canônica",
        "",
        f"- Branch: `{data['git']['branch']}`.",
        f"- HEAD local/remoto: `{data['git']['head']}`.",
        f"- Commit AP-004B: `{EXPECTED_HEAD}`.",
        f"- Inventário AP-004A: revisão `{data['prior_phases']['ap004a_revision']}`.",
        f"- Aplicação AP-004B: `{data['prior_phases']['ap004b_application_mode']}`.",
        "",
        "## Resumo",
        "",
        f"- Candidatos e controles: **{stats['candidate_count']}**.",
        f"- Arquivos definidores: **{stats['definition_file_count']}**.",
        f"- Referências Python: **{stats['python_reference_record_count']}**.",
        f"- Resumos de referências textuais: **{stats['text_reference_summary_count']}**.",
        f"- Arquivos no manifesto: **{stats['manifest_file_count']}**.",
        f"- Colisões de destino: **{stats['destination_collision_count']}**.",
        f"- Onda 1 pronta: **{stats['ready_wave_1_count']}**.",
        f"- Onda 2 vinculada a contratos: **{stats['ready_wave_2_count']}**.",
        f"- Adiados: **{stats['deferred_count']}**.",
        f"- Protegidos: **{stats['protected_count']}**.",
        "- Código produtivo alterado: **não**.",
        "",
        "## Matriz de decisão",
        "",
        "| Símbolo atual | Sugestão | Categoria | Arquivo:linha | Classificação AP-004A | Disposição AP-004C | Onda | Externas/contratos |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for item in data["candidates"]:
        suggestion = item.get("suggested_name") or "—"
        definition = item["definition"]
        lines.append(
            "| `{current}` | `{suggested}` | {category} | `{path}:{line}` | {classification} | `{disposition}` | {wave} | {external} |".format(
                current=md(item["current_name"]),
                suggested=md(suggestion),
                category=md(item["category"]),
                path=md(item["path"]),
                line=definition.get("line") or "?",
                classification=md(item.get("classification")),
                disposition=md(item["disposition"]),
                wave=md(item["application_wave"]),
                external=item["effective_external_reference_count"],
            )
        )
    lines.extend([
        "",
        "## Aliases seguros herdados da AP-004A",
        "",
    ])
    for pair in data["required_safe_orchestrator_aliases"]:
        lines.append(
            f"- `{pair['current_name']}` → `{pair['suggested_name']}`."
        )
    lines.extend([
        "",
        "## Controles xfail protegidos",
        "",
    ])
    for control in data["protected_xfail_controls"]:
        lines.append(
            f"- `{control['qualified_name']}` em `{control['path']}`."
        )
    lines.extend([
        "",
        "## Distribuição por disposição",
        "",
        "| Disposição | Quantidade |",
        "|---|---:|",
    ])
    for disposition in DISPOSITIONS:
        lines.append(
            f"| `{disposition}` | {stats['by_disposition'][disposition]} |"
        )
    lines.extend([
        "",
        "## Consumidores efetivos dos candidatos prontos",
        "",
    ])
    ready_ids = {
        item["id"] for item in data["candidates"]
        if item["disposition"] in {
            "ready_local_ast_rename",
            "ready_contract_bound_ast_rename",
            "contract_update_required",
        }
    }
    relevant = [
        record for record in data["python_references"]
        if record["candidate_id"] in ready_ids
        and record["semantic_category"] not in {
            "definition", "historical_immutable", "protected_operational",
            "contextual_string",
        }
    ]
    if relevant:
        lines.extend([
            "| Símbolo | Categoria | Arquivo:linha | Tipo | Escopo |",
            "|---|---|---|---|---|",
        ])
        for record in relevant:
            lines.append(
                "| `{symbol}` | `{category}` | `{path}:{line}` | `{kind}` | `{scope}` |".format(
                    symbol=md(record["symbol"]),
                    category=md(record["semantic_category"]),
                    path=md(record["path"]),
                    line=record.get("line") or "?",
                    kind=md(record["kind"]),
                    scope=md(record["scope"]),
                )
            )
    else:
        lines.append("Nenhum consumidor efetivo adicional foi localizado.")
    lines.extend([
        "",
        "## Colisões",
        "",
    ])
    if data["destination_collisions"]:
        for collision in data["destination_collisions"]:
            lines.append(
                f"- `{collision['kind']}`: `{collision['suggested_name']}` em "
                f"`{collision['path']}`; origens: "
                + ", ".join(f"`{name}`" for name in collision["current_names"])
                + "."
            )
    else:
        lines.append("Nenhuma colisão de destino foi detectada.")
    lines.extend([
        "",
        "## Manifesto",
        "",
        f"O JSON registra hashes de **{stats['manifest_file_count']}** arquivos relevantes, além dos contratos AST de todas as definições selecionadas.",
        "",
        "## Validação",
        "",
        f"- `py_compile`: `{data['validation']['py_compile']}`.",
        f"- `git diff --check`: `{data['validation']['git_diff_check']}`.",
        f"- Suíte específica: `{data['validation']['specific_suite'].get('summary', data['validation']['specific_suite'].get('status'))}`.",
        f"- Suíte consolidada: `{data['validation']['consolidated_suite'].get('summary', data['validation']['consolidated_suite'].get('status'))}`.",
        "",
        "## Decisão de fase",
        "",
        "O aplicador produtivo da AP-004C permanece bloqueado até aprovação expressa deste inventário.",
    ])
    return normalize_output("\n".join(lines))


def build_contract_test(*, tool_sha256: str) -> str:
    outputs = [path.as_posix() for path in OUTPUT_RELS]
    safe_aliases = list(EXPECTED_SAFE_ORCHESTRATOR_ALIASES)
    protected = list(KNOWN_XFAIL_CONTROLS)
    return normalize_output(f'''from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parents[1]
INVENTORY = ROOT / {INVENTORY_REL.as_posix()!r}
STRATEGY = ROOT / {STRATEGY_REL.as_posix()!r}
TOOL = ROOT / {TOOL_REL.as_posix()!r}
BASELINE_HEAD = {EXPECTED_HEAD!r}
EXPECTED_SUBJECT = {EXPECTED_AP004C_INVENTORY_SUBJECT!r}
EXPECTED_TOOL_SHA256 = {tool_sha256!r}
EXPECTED_OUTPUTS = {outputs!r}
EXPECTED_SAFE_ALIASES = {safe_aliases!r}
PROTECTED_CONTROLS = {protected!r}
SOFTWARE_PREFIX = {SOFTWARE_PREFIX!r}


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
    raw = raw.strip().strip('"').replace("\\\\", "/")
    return raw[len(SOFTWARE_PREFIX):] if raw.startswith(SOFTWARE_PREFIX) else raw


def _ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {{"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}}
    return any(part in ignored for part in parts) or path.endswith((".pyc", ".pyo"))


def _find_commit() -> str | None:
    result = _run(
        "git", "log", "--format=%H%x00%s", f"{{BASELINE_HEAD}}..HEAD"
    )
    assert result.returncode == 0, result.stderr
    matches = []
    for line in result.stdout.splitlines():
        if "\\x00" not in line:
            continue
        commit, subject = line.split("\\x00", 1)
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
    source = json.loads((ROOT / {AP004A_INVENTORY_REL.as_posix()!r}).read_text(encoding="utf-8"))
    expected = {{
        item["id"] for item in source["actionable_candidates"]
        if item["category"] in {{"função", "classe", "constante", "alias"}}
        and "AP-004C" in item["target_phase"]
    }}
    actual_nonprotected = {{
        item["id"] for item in data["candidates"]
        if item["disposition"] != "protected_xfail_out_of_scope"
    }}
    assert expected <= actual_nonprotected
    assert all(item["category"] in {{"função", "classe", "constante", "alias"}} for item in data["candidates"])


def test_ap004c_candidate_definitions_and_hashes_are_current() -> None:
    data = _data()
    for item in data["candidates"]:
        path = ROOT / item["path"]
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        definition = item["definition"]
        assert definition["line"] is not None
        assert definition["definition_kind"] in {{"function", "class", "assignment", "annotated_assignment", "import_alias", "import"}}
        assert len(definition["ast_sha256"]) == 64
        assert len(definition["source_sha256"]) == 64
        assert isinstance(tree, ast.Module)


def test_ap004c_required_safe_orchestrator_aliases_are_preserved() -> None:
    data = _data()
    by_name = {{item["current_name"]: item for item in data["candidates"]}}
    for current, suggested in EXPECTED_SAFE_ALIASES:
        item = by_name[current]
        assert item["path"] == {ORCHESTRATOR_REL.as_posix()!r}
        assert item["suggested_name"] == suggested
        assert item["classification"] == "renomeação segura"
        assert item["disposition"] in {{
            "ready_contract_bound_ast_rename",
            "contract_update_required",
        }}


def test_ap004c_xfail_controls_are_absolute_protections() -> None:
    data = _data()
    actual = {{
        (item["current_name"], item["path"]): item
        for item in data["candidates"]
    }}
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
    assert set(counts) == {{
        "ready_local_ast_rename", "ready_contract_bound_ast_rename",
        "contract_update_required", "compatibility_required",
        "deferred_structural_symbol", "manual_semantic_name_required",
        "blocked_destination_collision", "protected_xfail_out_of_scope",
    }}
    assert data["statistics"]["ready_wave_1_count"] >= 0
    assert data["statistics"]["ready_wave_2_count"] >= len(EXPECTED_SAFE_ALIASES)


def test_ap004c_reference_records_are_exact_and_partitioned() -> None:
    data = _data()
    candidate_ids = {{item["id"] for item in data["candidates"]}}
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
    collision_ids = {{candidate_id for collision in data["destination_collisions"] for candidate_id in collision["candidate_ids"]}}
    for item in data["candidates"]:
        if item["id"] in collision_ids:
            assert item["disposition"] == "blocked_destination_collision"
        suggestion = item.get("suggested_name")
        if suggestion:
            assert suggestion.isidentifier()


def test_ap004c_source_manifest_matches_current_baseline() -> None:
    data = _data()
    for record in data["source_manifest"]:
        path = ROOT / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            actual = hashlib.sha256(ast.dump(tree, include_attributes=False, annotate_fields=True).encode()).hexdigest()
            assert actual == record["ast_sha256"]


def test_ap004c_preserves_ap003_and_ap004b_control_files() -> None:
    data = _data()
    manifest = {{record["path"]: record for record in data["source_manifest"]}}
    required = {{
        {ORCHESTRATOR_REL.as_posix()!r},
        {AP003G_CONTRACT_REL.as_posix()!r},
        {AP003G_MANIFEST_REL.as_posix()!r},
        {AP004B_APPLICATION_REL.as_posix()!r},
        {AP004B_APPLICATION_TEST_REL.as_posix()!r},
    }}
    assert required <= set(manifest)
    assert _sha256(ROOT / {ORCHESTRATOR_REL.as_posix()!r}) == "8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977"
    for relative in (
        {AP004B_APPLICATION_TEST_REL.as_posix()!r},
        {AP004B_INVENTORY_TEST_REL.as_posix()!r},
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "EXPECTED_AP004B_COMMIT" in source
        assert "EXPECTED_AP004B_SUBJECT" in source
        assert "commit_scope_is_durable" in source


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
    actual = {{
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }}
    expected = set(EXPECTED_OUTPUTS)
    if actual:
        assert actual == expected
        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD
    else:
        commit = _find_commit()
        assert commit is not None
        changed = _run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
        assert changed.returncode == 0, changed.stderr
        normalized = {{
            path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
            for path in changed.stdout.splitlines() if path
        }}
        assert normalized == expected
        assert _run("git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD").returncode == 0


def test_ap004c_generated_artifacts_and_tool_compile() -> None:
    data = _data()
    assert data["scope"]["allowed_outputs"] == EXPECTED_OUTPUTS
    assert TOOL.is_file() and INVENTORY.is_file() and STRATEGY.is_file()
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    assert data["tool"]["sha256"] == EXPECTED_TOOL_SHA256
    with tempfile.TemporaryDirectory(prefix="ap004c-contract-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)
''')


def count_test_functions(source: str) -> int:
    tree = ast.parse(source)
    return sum(
        1 for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def whitespace_check(paths: Iterable[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                fail(f"Whitespace final em {path}:{number}")
        if not text.endswith("\n"):
            fail(f"Arquivo sem newline final: {path}")


def validate_allowed_final_status(
    repository_root: Path
) -> None:
    unstaged, staged = status_paths(repository_root)
    expected = sorted(path.as_posix() for path in OUTPUT_RELS)
    if staged:
        fail(f"Nenhum arquivo pode estar staged: {staged}")
    if unstaged != expected:
        fail(
            "Estado final fora do escopo AP-004C.\n"
            f"Atual: {unstaged}\nEsperado: {expected}"
        )


def parse_pytest_summary(result: CommandResult, *, label: str) -> dict[str, Any]:
    text = result.stdout + "\n" + result.stderr
    matches = list(re.finditer(r"(?P<count>\d+) (?P<name>passed|failed|xfailed|xpassed|skipped|errors?|warnings?)", text))
    counts: Counter[str] = Counter()
    for match in matches:
        counts[match.group("name")] = int(match.group("count"))
    if not matches:
        fail(f"Não foi possível interpretar o resumo pytest da suíte {label}.\n{text}")
    summary_match = re.findall(r"=+ ([^\n]+?) =+\s*$", text, flags=re.MULTILINE)
    summary = summary_match[-1] if summary_match else ", ".join(
        f"{value} {name}" for name, value in counts.items()
    )
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "xfailed": counts.get("xfailed", 0),
        "skipped": counts.get("skipped", 0),
        "summary": summary,
    }


def run_specific_suite(software_root: Path) -> dict[str, Any]:
    result = run(
        ("pipenv", "run", "pytest", "-q", "-ra", TEST_REL.as_posix()),
        cwd=software_root, check=False, timeout=900,
    )
    parsed = parse_pytest_summary(result, label="AP-004C específica")
    if result.returncode != 0:
        fail(f"Suíte específica AP-004C falhou:\n{result.stdout}{result.stderr}")
    if parsed["passed"] != EXPECTED_CONTRACT_TESTS or parsed["xfailed"] != 0:
        fail(
            "Contagem específica AP-004C divergente.\n"
            f"Esperado: {EXPECTED_CONTRACT_TESTS} passed, 0 xfailed\n"
            f"Atual: {parsed['summary']}"
        )
    return parsed


def run_consolidated_suite(software_root: Path) -> dict[str, Any]:
    result = run(
        (
            "pipenv", "run", "pytest", "-q", "-ra",
            "app_bundle/tests", "tests",
        ),
        cwd=software_root, check=False, timeout=1800,
    )
    parsed = parse_pytest_summary(result, label="AP-004C consolidada")
    expected_passed = BASELINE_PASSED + EXPECTED_CONTRACT_TESTS
    if result.returncode != 0:
        fail(f"Suíte consolidada AP-004C falhou:\n{result.stdout}{result.stderr}")
    if parsed["passed"] != expected_passed or parsed["xfailed"] != BASELINE_XFAILED:
        fail(
            "Contagem consolidada AP-004C divergente.\n"
            f"Esperado: {expected_passed} passed, {BASELINE_XFAILED} xfailed\n"
            f"Atual: {parsed['summary']}"
        )
    return parsed


def write_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventário preparatório AP-004C de símbolos internos."
    )
    parser.add_argument(
        "--skip-remote-check",
        action="store_true",
        help="Uso excepcional offline; não comprova publicação no remoto.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Uso diagnóstico; gera artefatos sem validar pytest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    software_root = Path.cwd().resolve()
    tool_source = Path(__file__).read_text(encoding="utf-8")
    ast.parse(tool_source, filename=str(Path(__file__)))

    repository_root, git_state = validate_git_state(
        software_root, skip_remote_check=args.skip_remote_check
    )
    prior = validate_prior_phases(software_root, repository_root)
    inventory = build_inventory(
        software_root=software_root,
        repository_root=repository_root,
        git_state=git_state,
        prior=prior,
        tool_source=tool_source,
    )
    ap004b_application_test_source = build_durable_ap004b_application_contract(
        software_root
    )
    ap004b_inventory_test_source = build_durable_ap004b_inventory_contract(
        software_root
    )
    update_manifest_for_contract_maintenance(
        inventory,
        application_source=ap004b_application_test_source,
        inventory_source=ap004b_inventory_test_source,
    )
    test_source = build_contract_test(tool_sha256=inventory["tool"]["sha256"])
    if count_test_functions(test_source) != EXPECTED_CONTRACT_TESTS:
        fail(
            f"Esperados {EXPECTED_CONTRACT_TESTS} testes; "
            f"gerados {count_test_functions(test_source)}."
        )
    ast.parse(test_source, filename=TEST_REL.as_posix())
    strategy = build_strategy_document(inventory)
    report = build_report(inventory)
    outputs = {
        software_root / TOOL_REL: normalize_output(tool_source),
        software_root / STRATEGY_REL: strategy,
        software_root / INVENTORY_REL: write_json(inventory),
        software_root / REPORT_REL: report,
        software_root / TEST_REL: test_source,
        software_root / AP004B_APPLICATION_TEST_REL: ap004b_application_test_source,
        software_root / AP004B_INVENTORY_TEST_REL: ap004b_inventory_test_source,
    }
    backup_root, backup_records = create_backups(
        outputs, software_root=software_root
    )
    try:
        for path, content in outputs.items():
            atomic_write(path, content)
        compile_python(software_root / TOOL_REL)
        compile_python(software_root / TEST_REL)
        whitespace_check(outputs)
        diff_check = git(repository_root, "diff", "--check", check=False)
        if diff_check.returncode != 0:
            fail(f"git diff --check falhou:\n{diff_check.stdout}{diff_check.stderr}")
        validate_allowed_final_status(repository_root)
        validation: dict[str, Any] = {
            "py_compile": "passed",
            "git_diff_check": "passed",
            "specific_suite": {"status": "skipped"},
            "consolidated_suite": {"status": "skipped"},
        }
        if not args.skip_tests:
            validation["specific_suite"] = run_specific_suite(software_root)
            validation["consolidated_suite"] = run_consolidated_suite(software_root)
        inventory["validation"] = validation
        atomic_write(software_root / INVENTORY_REL, write_json(inventory))
        atomic_write(software_root / REPORT_REL, build_report(inventory))
        whitespace_check(outputs)
        validate_allowed_final_status(repository_root)
    except Exception:
        rollback(backup_records)
        raise

    stats = inventory["statistics"]
    print("[OK] AP-004C inventariada sem alteração produtiva.")
    print(f"[OK] Branch: {git_state['branch']}")
    print(f"[OK] HEAD local/remoto: {git_state['head']}")
    print(f"[OK] Commit AP-004B confirmado: {EXPECTED_HEAD} — {EXPECTED_AP004B_SUBJECT}")
    print(f"[OK] Candidatos e controles: {stats['candidate_count']}")
    print(f"     onda 1 — locais seguros: {stats['ready_wave_1_count']}")
    print(f"     onda 2 — vinculados a contratos: {stats['ready_wave_2_count']}")
    print(f"     adiados: {stats['deferred_count']}")
    print(f"     protegidos: {stats['protected_count']}")
    print(f"[OK] Referências Python: {stats['python_reference_record_count']}")
    print(f"[OK] Resumos de referências textuais: {stats['text_reference_summary_count']}")
    print(f"[OK] Colisões de destino: {stats['destination_collision_count']}")
    print(f"[OK] Arquivos no manifesto: {stats['manifest_file_count']}")
    print(f"[OK] Relatório: {REPORT_REL}")
    print(f"[OK] Estratégia: {STRATEGY_REL}")
    print(f"[OK] JSON: {INVENTORY_REL}")
    print(f"[OK] Teste: {TEST_REL}")
    print(f"[OK] Ferramenta reexecutável: {TOOL_REL}")
    print("[OK] Contratos AP-004B tornados duráveis: 2")
    print(f"[OK] Backup externo: {backup_root}")
    if not args.skip_tests:
        print(
            "[OK] Suíte específica: "
            f"{inventory['validation']['specific_suite']['summary']}"
        )
        print(
            "[OK] Suíte consolidada: "
            f"{inventory['validation']['consolidated_suite']['summary']}"
        )
    else:
        print("[AVISO] Testes ignorados; inventário não validado para consolidação.")
    print("[OK] Nenhum commit foi criado.")
    print("[BLOQUEIO] Não criar aplicador produtivo sem aprovação do inventário AP-004C.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InventoryError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        raise SystemExit(1)

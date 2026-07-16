#!/usr/bin/env python3
"""Preparatory, read-mostly inventory for AP-004D version-marker consolidation.

The tool validates the AP-004C Git baseline before creating only the five
preparatory AP-004D artifacts. It never edits productive source files, never
creates commits, and never pushes.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tokenize
import uuid
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = "ap004d-version-marker-inventory/2"
EXPECTED_BRANCH = "ap-refactor/03-orchestrator-decomposition"
REMOTE_NAME = "origin"
REMOTE_REF = f"{REMOTE_NAME}/{EXPECTED_BRANCH}"
EXPECTED_AP004C_SUBJECT = (
    "refactor(academic-pipeline): consolidar símbolos internos da AP-004C"
)
SOFTWARE_RELATIVE = PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)

OUTPUT_INVENTORY_MD = PurePosixPath(
    "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_INVENTORY.md"
)
OUTPUT_STRATEGY_MD = PurePosixPath(
    "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_STRATEGY.md"
)
OUTPUT_INVENTORY_JSON = PurePosixPath(
    "docs/refactor/academic-pipeline/AP-004/ap004d_version_marker_inventory.json"
)
OUTPUT_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap004d_version_marker_inventory_contract.py"
)
OUTPUT_TOOL = PurePosixPath("tools/refactor/ap004d_inventory_version_markers.py")
OUTPUT_PATHS = (
    OUTPUT_INVENTORY_MD,
    OUTPUT_STRATEGY_MD,
    OUTPUT_INVENTORY_JSON,
    OUTPUT_TEST,
    OUTPUT_TOOL,
)
OUTPUT_PATH_SET = {str(path) for path in OUTPUT_PATHS}

PROTECTED_SYMBOLS = {
    "_refs_v6_strip_org",
    "_ap003d_impl__refs_v6_strip_org",
    "extract_org_abstracts",
}
PROTECTED_QUALIFIED = {
    "WorkflowState._normalize",
}
STRUCTURAL_AP003_SYMBOLS = {
    "_ap003f_pipeline_core",
}
FROZEN_FULLTEXT_FILES = {
    "executar_artigo_longo_fulltext_v1_13.py",
    "executar_artigo_longo_fulltext_v1_14.py",
}
HISTORICAL_ORCHESTRATOR_FILES = {
    "academic_pipeline_rc10.py",
}

CLASSIFICATIONS = {
    "marcador_interno_removivel",
    "marcador_privado_renomeavel_ast",
    "marcador_preso_contrato_historico",
    "marcador_necessario_compatibilidade",
    "marcador_comentario_historico",
    "marcador_string_operacional",
    "marcador_caminho_fisico_fora_escopo",
    "marcador_ambiguo_decisao_manual",
    "marcador_protegido_xfail",
    "colisao_destino",
    "ocorrencia_apenas_documental",
    "ocorrencia_snapshot_fixture_manifesto",
}
DECISIONS = {"candidato", "preservar", "adiar", "revisao_manual"}
RISKS = {"baixo", "medio", "alto", "proibitivo"}
WAVES = {"onda_0_preservacao", "onda_1_ast", "onda_2_textual", "onda_3_manual"}

TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".org",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
    ".service",
    ".desktop",
    ".xml",
    ".jinja",
    ".j2",
    ".tex",
}
TEXT_BASENAMES = {
    "Pipfile",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "MANIFEST.in",
    "Makefile",
}
LOCKFILE_BASENAMES = {
    "Pipfile.lock",
    "poetry.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}
MAX_TEXT_BYTES = 2_500_000
MAX_RECORDS = 30_000
MAX_CONSUMERS_PER_RECORD = 250

SCAN_ROOTS = (
    SOFTWARE_RELATIVE,
    PurePosixPath("docs/refactor/academic-pipeline"),
    PurePosixPath("tools/refactor"),
)
PROJECT_INSTANCES_PREFIX = SOFTWARE_RELATIVE / PurePosixPath("app_bundle/projetos")
PROJECT_EXECUTABLE_METADATA_SUFFIXES = {
    ".cfg",
    ".conf",
    ".ini",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
EXCLUDED_PATH_PREFIXES: tuple[PurePosixPath, ...] = ()
EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "archive",
    "archives",
    "backup",
    "backups",
    "build",
    "cache",
    "coverage",
    "dist",
    "export",
    "exports",
    "generated",
    "htmlcov",
    "log",
    "logs",
    "node_modules",
    "output",
    "outputs",
    "site-packages",
    "temp",
    "tmp",
    "venv",
}
EXCLUDED_DATA_SUFFIXES = {
    ".csv",
    ".db",
    ".doc",
    ".docx",
    ".feather",
    ".ods",
    ".parquet",
    ".pdf",
    ".sqlite",
    ".sqlite3",
    ".tsv",
    ".xls",
    ".xlsm",
    ".xlsx",
}

MARKER_RE = re.compile(
    r"(?P<phase>\bAP[-_]?\d{3,4}[A-Z]?\b)"
    r"|(?P<ap_symbol>(?<![A-Za-z0-9])_?ap\d{3,4}[a-z]?(?=\b|_))"
    r"|(?P<rc>(?<![A-Za-z0-9])rc\d+(?:[._-]\d+)*)"
    r"|(?P<vmarker>(?<![A-Za-z0-9])v\d+(?:[._-]\d+)*)"
    r"|(?P<version_word>(?<![A-Za-z0-9])(?:ver(?:sion|sao)?)[_-]?\d+(?:[._-]\d+)*)",
    re.IGNORECASE,
)
SEMVER_RE = re.compile(r"(?P<semver>\b\d+\.\d+(?:\.\d+){0,2}\b)")
SEMVER_CONTEXT_RE = re.compile(
    r"(?:__version__|api[_-]?version|schema[_-]?version|version|vers[aã]o|"
    r"requires[-_]?python|python[_-]?requires|python[_-]?version|release|"
    r"compat(?:ibility|ibilidade)?|minimum|m[ií]nim[oa]|maximum|m[aá]xim[oa])",
    re.IGNORECASE,
)
IDENTIFIER_MARKER_RE = re.compile(
    r"(?:^|_)(?:v\d+(?:_\d+)*|rc\d+(?:_\d+)*|ap\d{3,4}[a-z]?)(?:_|$)",
    re.IGNORECASE,
)
VERSION_SEGMENT_RE = re.compile(
    r"(?:(?<=_)|^)(?:v|rc)\d+(?:_\d+)*(?=_|$)",
    re.IGNORECASE,
)
HISTORICAL_WORDS_RE = re.compile(
    r"\b(?:histor(?:ic|ico|ica)|legacy|legado|compat(?:ibility|ibilidade)?|"
    r"wrapper|deprecated|obsoleto|preservad[oa]|congelad[oa]|snapshot|fixture|"
    r"manifesto|fase|phase|baseline)\b",
    re.IGNORECASE,
)
REMOVABLE_COMMENT_RE = re.compile(
    r"\b(?:patch|hotfix|marcador\s+interno|internal\s+marker|vers[aã]o\s+interna|"
    r"tag\s+tempor[aá]ria|temporary\s+tag)\b",
    re.IGNORECASE,
)
OPERATIONAL_TARGET_RE = re.compile(
    r"(?:version|versao|vers[aã]o|patch|tag|selector|seletor|path|caminho|file|"
    r"arquivo|module|modulo|m[oó]dulo|command|comando|entry|entrada|template|"
    r"schema|manifest|profile|perfil)",
    re.IGNORECASE,
)
OPERATIONAL_CALL_RE = re.compile(
    r"(?:open|Path|import_module|find_spec|getenv|setdefault|run|Popen|check_call|"
    r"check_output|read_text|write_text|loads|load|compile|render|template)",
    re.IGNORECASE,
)


class InventoryError(RuntimeError):
    """Controlled failure that must not leave partial AP-004D artifacts."""


@dataclasses.dataclass(frozen=True)
class GitBaseline:
    repo_root: Path
    branch: str
    head: str
    subject: str
    remote_head: str
    divergence_left: int
    divergence_right: int
    commit_time: str
    preexisting_status: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TrackedSelection:
    total_count: int
    included: tuple[PurePosixPath, ...]
    excluded_counts: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True)
class IdentifierOccurrence:
    path: str
    line: int
    column: int
    name: str
    kind: str
    scope: str
    is_definition: bool
    qualified_name: str


@dataclasses.dataclass(frozen=True)
class StringRole:
    role: str
    context: str


@dataclasses.dataclass
class ScanState:
    identifier_occurrences: list[IdentifierOccurrence] = dataclasses.field(default_factory=list)
    records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    skipped_files: list[dict[str, str]] = dataclasses.field(default_factory=list)
    parse_errors: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    definition_names_by_scope: dict[tuple[str, str], set[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )


class AstIdentifierVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope_stack: list[str] = ["<module>"]
        self.occurrences: list[IdentifierOccurrence] = []
        self.definition_names_by_scope: dict[tuple[str, str], set[str]] = defaultdict(set)

    @property
    def scope(self) -> str:
        return ".".join(self.scope_stack)

    def _qualified(self, name: str) -> str:
        visible = [item for item in self.scope_stack if item != "<module>"]
        return ".".join([*visible, name]) if visible else name

    def _add(
        self,
        *,
        name: str,
        node: ast.AST,
        kind: str,
        is_definition: bool,
        scope_override: str | None = None,
    ) -> None:
        scope = scope_override or self.scope
        occurrence = IdentifierOccurrence(
            path=self.path,
            line=int(getattr(node, "lineno", 1)),
            column=int(getattr(node, "col_offset", 0)),
            name=name,
            kind=kind,
            scope=scope,
            is_definition=is_definition,
            qualified_name=self._qualified(name),
        )
        self.occurrences.append(occurrence)
        if is_definition:
            self.definition_names_by_scope[(self.path, scope)].add(name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        parent_scope = self.scope
        self._add(
            name=node.name,
            node=node,
            kind="function_definition",
            is_definition=True,
            scope_override=parent_scope,
        )
        self.scope_stack.append(node.name)
        self._visit_arguments(node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns:
            self.visit(node.returns)
        for statement in node.body:
            self.visit(statement)
        self.scope_stack.pop()
        return None

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        parent_scope = self.scope
        self._add(
            name=node.name,
            node=node,
            kind="class_definition",
            is_definition=True,
            scope_override=parent_scope,
        )
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.scope_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope_stack.pop()
        return None

    def _visit_arguments(self, args: ast.arguments) -> None:
        all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)
        for argument in all_args:
            self._add(
                name=argument.arg,
                node=argument,
                kind="argument_definition",
                is_definition=True,
            )
            if argument.annotation:
                self.visit(argument.annotation)
        for default in [*args.defaults, *args.kw_defaults]:
            if default is not None:
                self.visit(default)

    def visit_Name(self, node: ast.Name) -> Any:
        is_definition = isinstance(node.ctx, (ast.Store, ast.Param))
        self._add(
            name=node.id,
            node=node,
            kind=f"name_{node.ctx.__class__.__name__.lower()}",
            is_definition=is_definition,
        )
        return None

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        self.visit(node.value)
        is_definition = isinstance(node.ctx, ast.Store)
        self._add(
            name=node.attr,
            node=node,
            kind=f"attribute_{node.ctx.__class__.__name__.lower()}",
            is_definition=is_definition,
        )
        return None

    def visit_alias(self, node: ast.alias) -> Any:
        bound_name = node.asname or node.name.split(".")[0]
        self._add(
            name=bound_name,
            node=node,
            kind="import_binding",
            is_definition=True,
        )
        return None

    def visit_Global(self, node: ast.Global) -> Any:
        for name in node.names:
            self._add(
                name=name,
                node=node,
                kind="global_declaration",
                is_definition=False,
            )
        return None

    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:
        for name in node.names:
            self._add(
                name=name,
                node=node,
                kind="nonlocal_declaration",
                is_definition=False,
            )
        return None


class StringRoleCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.roles: dict[tuple[int, int], StringRole] = {}
        self.parent_by_id: dict[int, ast.AST] = {}

    def collect(self, tree: ast.AST) -> dict[tuple[int, int], StringRole]:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parent_by_id[id(child)] = parent
        docstring_nodes: set[int] = set()
        for owner in ast.walk(tree):
            if isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if owner.body and isinstance(owner.body[0], ast.Expr):
                    value = owner.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        docstring_nodes.add(id(value))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            key = (int(getattr(node, "lineno", 1)), int(getattr(node, "col_offset", 0)))
            if id(node) in docstring_nodes:
                self.roles[key] = StringRole("docstring", "documentation")
                continue
            parent = self.parent_by_id.get(id(node))
            role, context = self._classify_parent(node, parent)
            self.roles[key] = StringRole(role, context)
        return self.roles

    def _classify_parent(
        self, node: ast.Constant, parent: ast.AST | None
    ) -> tuple[str, str]:
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            names = sorted(_assignment_target_names(parent))
            context = "assignment:" + ",".join(names)
            if any(OPERATIONAL_TARGET_RE.search(name) for name in names):
                return "operational_assignment", context
            return "assignment", context
        if isinstance(parent, ast.Call):
            function_name = _call_name(parent.func)
            if OPERATIONAL_CALL_RE.search(function_name):
                return "operational_call_argument", f"call:{function_name}"
            return "call_argument", f"call:{function_name}"
        if isinstance(parent, (ast.Compare, ast.MatchValue, ast.MatchSingleton)):
            return "operational_comparison", parent.__class__.__name__
        if isinstance(parent, ast.Dict):
            return "metadata_literal", "dict"
        if isinstance(parent, (ast.Subscript, ast.Slice)):
            return "selector_literal", parent.__class__.__name__
        if isinstance(parent, ast.Expr):
            return "standalone_literal", "expression"
        return "string_literal", parent.__class__.__name__ if parent else "unknown"


def _assignment_target_names(node: ast.AST) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    elif isinstance(node, ast.NamedExpr):
        targets.append(node.target)
    names: set[str] = set()
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)
    return names


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts)) or "<call>"


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    if check and completed.returncode != 0:
        rendered = " ".join(args)
        raise InventoryError(
            f"Comando falhou ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _discover_repo_root(explicit: str | None) -> Path:
    start = Path(explicit).expanduser().resolve() if explicit else Path.cwd().resolve()
    completed = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise InventoryError(
            "Não foi possível localizar a raiz Git. Execute dentro do repositório "
            "ou informe --repo-root."
        )
    return Path(completed.stdout.strip()).resolve()


def _parse_status(repo_root: Path) -> tuple[str, ...]:
    completed = _run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        cwd=repo_root,
    )
    entries: list[str] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line:
            continue
        path_part = raw_line[3:]
        if " -> " in path_part:
            old_path, new_path = path_part.split(" -> ", 1)
            entries.extend([old_path.strip('"'), new_path.strip('"')])
        else:
            entries.append(path_part.strip('"'))
    return tuple(sorted(set(entries)))


def _validate_git_baseline(repo_root: Path, *, fetch: bool) -> GitBaseline:
    software_root = repo_root / SOFTWARE_RELATIVE
    if not software_root.is_dir():
        raise InventoryError(f"Raiz canônica do software ausente: {software_root}")

    if fetch:
        _run(["git", "fetch", REMOTE_NAME], cwd=repo_root)

    branch = _run(["git", "branch", "--show-current"], cwd=repo_root).stdout.strip()
    if branch != EXPECTED_BRANCH:
        raise InventoryError(
            f"Branch incorreta: {branch!r}. Esperada: {EXPECTED_BRANCH!r}."
        )

    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    subject = _run(
        ["git", "show", "-s", "--format=%s", "HEAD"], cwd=repo_root
    ).stdout.strip()
    if subject != EXPECTED_AP004C_SUBJECT:
        raise InventoryError(
            "A AP-004D está bloqueada porque o HEAD não é o commit isolado aprovado "
            f"da AP-004C. Assunto encontrado: {subject!r}."
        )

    remote_head = _run(
        ["git", "rev-parse", REMOTE_REF], cwd=repo_root
    ).stdout.strip()
    divergence_text = _run(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{REMOTE_REF}"],
        cwd=repo_root,
    ).stdout.strip()
    try:
        divergence_left, divergence_right = map(int, divergence_text.split())
    except ValueError as exc:
        raise InventoryError(f"Divergência Git inválida: {divergence_text!r}") from exc

    if head != remote_head or (divergence_left, divergence_right) != (0, 0):
        raise InventoryError(
            "A AP-004D está bloqueada: HEAD local e branch remota não coincidem. "
            f"local={head}, remoto={remote_head}, "
            f"divergência={divergence_left} {divergence_right}."
        )

    git_dir = Path(
        _run(["git", "rev-parse", "--git-dir"], cwd=repo_root).stdout.strip()
    )
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    in_progress_markers = [
        git_dir / "MERGE_HEAD",
        git_dir / "CHERRY_PICK_HEAD",
        git_dir / "REVERT_HEAD",
        git_dir / "rebase-merge",
        git_dir / "rebase-apply",
    ]
    active = [str(path) for path in in_progress_markers if path.exists()]
    if active:
        raise InventoryError(
            "Operação Git em andamento; finalize-a antes da AP-004D: " + ", ".join(active)
        )

    status = _parse_status(repo_root)
    unexpected = [path for path in status if path not in OUTPUT_PATH_SET]
    if unexpected:
        raise InventoryError(
            "Árvore de trabalho não está limpa. Apenas artefatos preparatórios AP-004D "
            "preexistentes são tolerados em reexecução. Caminhos inesperados:\n- "
            + "\n- ".join(unexpected)
        )

    commit_time = _run(
        ["git", "show", "-s", "--format=%cI", "HEAD"], cwd=repo_root
    ).stdout.strip()
    return GitBaseline(
        repo_root=repo_root,
        branch=branch,
        head=head,
        subject=subject,
        remote_head=remote_head,
        divergence_left=divergence_left,
        divergence_right=divergence_right,
        commit_time=commit_time,
        preexisting_status=status,
    )


def _is_under(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents


def _scope_exclusion_reason(path: PurePosixPath) -> str | None:
    if str(path) in OUTPUT_PATH_SET:
        return "generated_ap004d_output"
    if not any(_is_under(path, root) for root in SCAN_ROOTS):
        return "outside_academic_pipeline_scope"
    if any(_is_under(path, prefix) for prefix in EXCLUDED_PATH_PREFIXES):
        return "excluded_path_prefix"
    lowered_parts = {part.lower() for part in path.parts}
    excluded_parts = lowered_parts & EXCLUDED_DIRECTORY_NAMES
    if excluded_parts:
        return "excluded_directory:" + ",".join(sorted(excluded_parts))
    if _is_under(path, PROJECT_INSTANCES_PREFIX):
        is_metadata = (
            path.suffix.lower() in PROJECT_EXECUTABLE_METADATA_SUFFIXES
            or path.name in TEXT_BASENAMES
        )
        if not is_metadata:
            return "generated_project_content"
    if path.suffix.lower() in EXCLUDED_DATA_SUFFIXES:
        return "data_or_binary_artifact"
    return None


def _git_tracked_files(repo_root: Path) -> TrackedSelection:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise InventoryError(
            "git ls-files falhou: " + completed.stderr.decode("utf-8", errors="replace")
        )
    included: list[PurePosixPath] = []
    excluded = Counter()
    total_count = 0
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        total_count += 1
        decoded = raw.decode("utf-8", errors="surrogateescape")
        path = PurePosixPath(decoded)
        reason = _scope_exclusion_reason(path)
        if reason is not None:
            excluded[reason] += 1
            continue
        included.append(path)
    return TrackedSelection(
        total_count=total_count,
        included=tuple(sorted(included, key=str)),
        excluded_counts=tuple(sorted(excluded.items())),
    )


def _is_text_candidate(relative: PurePosixPath) -> bool:
    return relative.name in TEXT_BASENAMES or relative.suffix.lower() in TEXT_EXTENSIONS


def _is_docs_path(path: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(path).parts}
    return bool(parts & {"docs", "doc", "documentation"}) or path.lower().endswith(
        (".md", ".rst", ".org")
    )


def _is_snapshot_fixture_manifest(path: str) -> bool:
    lowered = path.lower()
    parts = {part.lower() for part in PurePosixPath(path).parts}
    return bool(
        parts & {"snapshot", "snapshots", "fixture", "fixtures", "manifests", "manifest"}
        or "snapshot" in lowered
        or "fixture" in lowered
        or "manifest" in lowered
    )


def _is_test_path(path: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(path).parts}
    return "tests" in parts or Path(path).name.startswith("test_")


def _is_frozen_file(path: str) -> bool:
    return PurePosixPath(path).name in FROZEN_FULLTEXT_FILES


def _is_historical_orchestrator(path: str) -> bool:
    return PurePosixPath(path).name in HISTORICAL_ORCHESTRATOR_FILES


def _semver_context_allowed(
    *, text: str, path: str, occurrence_type: str, role: str
) -> bool:
    if not SEMVER_RE.search(text):
        return False
    if SEMVER_CONTEXT_RE.search(text) or SEMVER_CONTEXT_RE.search(role):
        return True
    basename = PurePosixPath(path).name
    if basename in {"pyproject.toml", "setup.py", "setup.cfg", "Pipfile"}:
        return occurrence_type != "python_comment"
    return False


def _iter_marker_matches(
    *, text: str, path: str, occurrence_type: str, role: str
) -> Iterator[re.Match[str]]:
    matches = list(MARKER_RE.finditer(text))
    if _semver_context_allowed(
        text=text,
        path=path,
        occurrence_type=occurrence_type,
        role=role,
    ):
        matches.extend(SEMVER_RE.finditer(text))
    matches.sort(key=lambda item: (item.start(), -(item.end() - item.start())))
    accepted: list[re.Match[str]] = []
    for match in matches:
        if any(match.start() < prior.end() and prior.start() < match.end() for prior in accepted):
            continue
        accepted.append(match)
    yield from accepted


def _marker_kind(text: str) -> str:
    match = MARKER_RE.search(text) or SEMVER_RE.search(text)
    return match.lastgroup if match else "unknown"


def _propose_identifier(name: str) -> str | None:
    if re.search(r"(?:^|_)ap\d{3,4}[a-z]?(?:_|$)", name, re.IGNORECASE):
        return None
    leading = "_" if name.startswith("_") else ""
    updated = VERSION_SEGMENT_RE.sub("", name)
    updated = re.sub(r"_{2,}", "_", updated).strip("_")
    if leading and updated:
        updated = "_" + updated
    if not updated or updated == name or not updated.isidentifier():
        return None
    return updated


def _record_id(parts: Iterable[Any]) -> str:
    canonical = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _make_record(
    *,
    current: str,
    proposed: str | None,
    path: str,
    line: int,
    column: int,
    occurrence_type: str,
    ast_scope: str | None,
    consumers: Sequence[str],
    contracts: Sequence[str],
    risk: str,
    decision: str,
    reason: str,
    wave: str,
    compatibility_required: bool,
    classification: str,
    marker_kind: str,
    context: str,
    collision: bool = False,
    consumer_count: int | None = None,
) -> dict[str, Any]:
    if classification not in CLASSIFICATIONS:
        raise InventoryError(f"Classificação inválida: {classification}")
    if decision not in DECISIONS:
        raise InventoryError(f"Decisão inválida: {decision}")
    if risk not in RISKS:
        raise InventoryError(f"Risco inválido: {risk}")
    if wave not in WAVES:
        raise InventoryError(f"Onda inválida: {wave}")
    record = {
        "id": _record_id(
            [
                current,
                proposed or "",
                path,
                line,
                column,
                occurrence_type,
                ast_scope or "",
                classification,
                context,
            ]
        ),
        "current": current,
        "proposed": proposed,
        "path": path,
        "line": line,
        "column": column,
        "occurrence_type": occurrence_type,
        "ast_scope": ast_scope,
        "consumers": list(consumers)[:MAX_CONSUMERS_PER_RECORD],
        "consumer_count": consumer_count if consumer_count is not None else len(consumers),
        "contracts": sorted(set(contracts)),
        "risk": risk,
        "decision": decision,
        "reason": reason,
        "wave": wave,
        "compatibility_required": compatibility_required,
        "classification": classification,
        "marker_kind": marker_kind,
        "context": context,
        "collision": collision,
    }
    return record


def _scan_python_file(
    *,
    absolute: Path,
    relative: str,
    state: ScanState,
) -> None:
    try:
        source = absolute.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            source = absolute.read_text(encoding="latin-1")
        except Exception as exc:  # noqa: BLE001
            state.skipped_files.append({"path": relative, "reason": f"decode_error:{exc}"})
            return
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        state.parse_errors.append(
            {
                "path": relative,
                "line": exc.lineno,
                "column": exc.offset,
                "message": exc.msg,
            }
        )
        _scan_text_lines(source=source, relative=relative, state=state, kind="python_unparsed")
        return

    visitor = AstIdentifierVisitor(relative)
    visitor.visit(tree)
    state.identifier_occurrences.extend(visitor.occurrences)
    for key, names in visitor.definition_names_by_scope.items():
        state.definition_names_by_scope[key].update(names)

    roles = StringRoleCollector().collect(tree)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                _scan_token_text(
                    text=token.string,
                    relative=relative,
                    line=token.start[0],
                    column=token.start[1],
                    occurrence_type="python_comment",
                    ast_scope=None,
                    role="comment",
                    state=state,
                )
            elif token.type == tokenize.STRING:
                role = roles.get(token.start, StringRole("string_literal", "unknown"))
                _scan_token_text(
                    text=token.string,
                    relative=relative,
                    line=token.start[0],
                    column=token.start[1],
                    occurrence_type="python_string",
                    ast_scope=None,
                    role=f"{role.role}:{role.context}",
                    state=state,
                )
    except (tokenize.TokenError, IndentationError) as exc:
        state.parse_errors.append(
            {"path": relative, "line": 0, "column": 0, "message": f"tokenize:{exc}"}
        )


def _scan_text_lines(
    *,
    source: str,
    relative: str,
    state: ScanState,
    kind: str = "text",
) -> None:
    for line_number, line_text in enumerate(source.splitlines(), start=1):
        if not MARKER_RE.search(line_text) and not _semver_context_allowed(
            text=line_text,
            path=relative,
            occurrence_type=kind,
            role="text_line",
        ):
            continue
        _scan_token_text(
            text=line_text,
            relative=relative,
            line=line_number,
            column=0,
            occurrence_type=kind,
            ast_scope=None,
            role="text_line",
            state=state,
        )


def _scan_token_text(
    *,
    text: str,
    relative: str,
    line: int,
    column: int,
    occurrence_type: str,
    ast_scope: str | None,
    role: str,
    state: ScanState,
) -> None:
    for match in _iter_marker_matches(
        text=text,
        path=relative,
        occurrence_type=occurrence_type,
        role=role,
    ):
        marker = match.group(0)
        classification = _classify_text_occurrence(
            marker=marker,
            full_text=text,
            path=relative,
            occurrence_type=occurrence_type,
            role=role,
        )
        proposed = None
        if classification == "marcador_interno_removivel":
            proposed = _remove_marker_from_text(text, marker)
        decision, risk, wave, compatibility, reason, contracts = _text_policy(
            classification=classification,
            marker=marker,
            path=relative,
            role=role,
        )
        state.records.append(
            _make_record(
                current=marker,
                proposed=proposed,
                path=relative,
                line=line,
                column=column + match.start(),
                occurrence_type=occurrence_type,
                ast_scope=ast_scope,
                consumers=[],
                contracts=contracts,
                risk=risk,
                decision=decision,
                reason=reason,
                wave=wave,
                compatibility_required=compatibility,
                classification=classification,
                marker_kind=match.lastgroup or "unknown",
                context=_compact_context(text),
            )
        )
        if len(state.records) > MAX_RECORDS:
            raise InventoryError(
                f"Inventário excedeu {MAX_RECORDS} registros. Refine as exclusões antes de prosseguir."
            )


def _remove_marker_from_text(text: str, marker: str) -> str:
    updated = text.replace(marker, "", 1)
    updated = re.sub(r"[ \t]{2,}", " ", updated)
    updated = re.sub(r"\s+([,;:.])", r"\1", updated)
    return updated.strip()


def _compact_context(text: str, limit: int = 220) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _classify_text_occurrence(
    *,
    marker: str,
    full_text: str,
    path: str,
    occurrence_type: str,
    role: str,
) -> str:
    lowered_marker = marker.lower()
    lowered_text = full_text.lower()
    basename = PurePosixPath(path).name

    if _is_frozen_file(path):
        return "marcador_preso_contrato_historico"
    if _is_snapshot_fixture_manifest(path):
        return "ocorrencia_snapshot_fixture_manifesto"
    if any(symbol.lower() in lowered_text for symbol in PROTECTED_SYMBOLS) or any(
        symbol.lower() in lowered_text for symbol in PROTECTED_QUALIFIED
    ):
        return "marcador_protegido_xfail"
    if basename in {"pyproject.toml", "setup.py", "setup.cfg", "Pipfile"} and re.search(
        r"\bversion\b|\brequires[-_]?python\b|\bpython_version\b", full_text, re.IGNORECASE
    ):
        return "marcador_necessario_compatibilidade"
    if lowered_marker.startswith("ap") or lowered_marker.startswith("_ap"):
        return "marcador_preso_contrato_historico"
    if occurrence_type == "python_comment":
        if REMOVABLE_COMMENT_RE.search(full_text) and not HISTORICAL_WORDS_RE.search(full_text):
            return "marcador_interno_removivel"
        if HISTORICAL_WORDS_RE.search(full_text):
            return "marcador_comentario_historico"
        return "marcador_ambiguo_decisao_manual"
    if role.startswith("docstring") or _is_docs_path(path):
        return "ocorrencia_apenas_documental"
    if occurrence_type == "python_string":
        if role.startswith(
            (
                "operational_",
                "metadata_literal",
                "selector_literal",
            )
        ):
            return "marcador_string_operacional"
        if HISTORICAL_WORDS_RE.search(full_text):
            return "marcador_preso_contrato_historico"
        return "marcador_ambiguo_decisao_manual"
    if _is_test_path(path):
        return "ocorrencia_snapshot_fixture_manifesto"
    if PurePosixPath(path).suffix.lower() in {".toml", ".json", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".service", ".desktop", ".sh"}:
        return "marcador_string_operacional"
    return "marcador_ambiguo_decisao_manual"


def _text_policy(
    *, classification: str, marker: str, path: str, role: str
) -> tuple[str, str, str, bool, str, list[str]]:
    if classification == "marcador_interno_removivel":
        return (
            "candidato",
            "baixo",
            "onda_2_textual",
            False,
            "Comentário indica marcador interno/temporário e não apresenta sinal de contrato histórico.",
            [],
        )
    if classification == "marcador_protegido_xfail":
        return (
            "preservar",
            "proibitivo",
            "onda_0_preservacao",
            True,
            "Ocorrência vinculada a símbolo protegido por defeito histórico congelado (xfail).",
            ["AP-004D: três xfail históricos congelados"],
        )
    if classification == "marcador_preso_contrato_historico":
        return (
            "preservar",
            "alto",
            "onda_0_preservacao",
            True,
            "Marcador identifica fase, artefato congelado ou contrato histórico que não deve ser reescrito.",
            ["histórico de refatoração", "compatibilidade histórica"],
        )
    if classification == "marcador_necessario_compatibilidade":
        return (
            "preservar",
            "alto",
            "onda_0_preservacao",
            True,
            "Marcador pertence a metadado público ou requisito de compatibilidade.",
            ["metadado público", "compatibilidade de ambiente"],
        )
    if classification == "marcador_comentario_historico":
        return (
            "preservar",
            "baixo",
            "onda_0_preservacao",
            False,
            "Comentário registra histórico; removê-lo reescreveria contexto sem benefício funcional demonstrado.",
            ["registro histórico"],
        )
    if classification == "marcador_string_operacional":
        return (
            "revisao_manual",
            "alto",
            "onda_3_manual",
            True,
            f"String participa de contexto operacional ({role}); alteração exige prova de consumidores e contrato.",
            ["string operacional"],
        )
    if classification == "marcador_caminho_fisico_fora_escopo":
        return (
            "preservar",
            "proibitivo",
            "onda_0_preservacao",
            True,
            "Nome físico versionado está explicitamente fora do escopo da AP-004D.",
            ["limites AP-004D"],
        )
    if classification == "ocorrencia_apenas_documental":
        return (
            "adiar",
            "baixo",
            "onda_3_manual",
            False,
            "Ocorrência é documental; não há justificativa automática para reescrever documentação histórica.",
            ["documentação"],
        )
    if classification == "ocorrencia_snapshot_fixture_manifesto":
        return (
            "preservar",
            "alto",
            "onda_0_preservacao",
            True,
            "Snapshot, fixture ou manifesto deve permanecer estável até decisão explícita de atualização contratual.",
            ["snapshot/fixture/manifesto"],
        )
    return (
        "revisao_manual",
        "medio",
        "onda_3_manual",
        False,
        f"Sem evidência suficiente para classificar automaticamente o marcador {marker!r} em {path}.",
        [],
    )


def _scan_paths(tracked: Sequence[PurePosixPath], state: ScanState) -> None:
    seen: set[tuple[str, str]] = set()
    for path in tracked:
        cumulative: list[str] = []
        for component in path.parts:
            cumulative.append(component)
            if not MARKER_RE.search(component):
                continue
            logical_path = "/".join(cumulative)
            key = (logical_path, component)
            if key in seen:
                continue
            seen.add(key)
            if component in FROZEN_FULLTEXT_FILES:
                classification = "marcador_preso_contrato_historico"
            elif component == SOFTWARE_RELATIVE.name:
                classification = "marcador_caminho_fisico_fora_escopo"
            elif component in HISTORICAL_ORCHESTRATOR_FILES:
                classification = "marcador_necessario_compatibilidade"
            elif component.lower().startswith("ap-004") or component.lower().startswith("ap004"):
                classification = "marcador_preso_contrato_historico"
            elif component.endswith((".py", ".sh")):
                classification = "marcador_ambiguo_decisao_manual"
            else:
                classification = "marcador_caminho_fisico_fora_escopo"
            decision, risk, wave, compatibility, reason, contracts = _text_policy(
                classification=classification,
                marker=component,
                path=logical_path,
                role="physical_path",
            )
            state.records.append(
                _make_record(
                    current=component,
                    proposed=None,
                    path=logical_path,
                    line=0,
                    column=0,
                    occurrence_type="physical_path_component",
                    ast_scope=None,
                    consumers=[],
                    contracts=contracts,
                    risk=risk,
                    decision=decision,
                    reason=reason,
                    wave=wave,
                    compatibility_required=compatibility,
                    classification=classification,
                    marker_kind=_marker_kind(component),
                    context=logical_path,
                )
            )


def _classify_identifiers(state: ScanState) -> None:
    by_name: dict[str, list[IdentifierOccurrence]] = defaultdict(list)
    for occurrence in state.identifier_occurrences:
        is_explicit_protected = (
            occurrence.name in PROTECTED_SYMBOLS
            or occurrence.qualified_name in PROTECTED_QUALIFIED
        )
        if IDENTIFIER_MARKER_RE.search(occurrence.name) or is_explicit_protected:
            by_name[occurrence.name].append(occurrence)

    for name in sorted(by_name):
        occurrences = sorted(
            by_name[name], key=lambda item: (item.path, item.line, item.column, item.kind)
        )
        definitions = [item for item in occurrences if item.is_definition]
        anchors = definitions or [occurrences[0]]
        consumers = [
            f"{item.path}:{item.line}:{item.kind}:{item.scope}" for item in occurrences
        ]
        proposed = _propose_identifier(name)

        for anchor in anchors:
            qualified = anchor.qualified_name
            xfail_protected = (
                name in PROTECTED_SYMBOLS or qualified in PROTECTED_QUALIFIED
            )
            structural_ap003 = (
                name in STRUCTURAL_AP003_SYMBOLS
                or bool(re.search(r"(?:^|_)ap003[a-z]?(?:_|$)", name, re.IGNORECASE))
            )
            protected = xfail_protected or structural_ap003
            if not IDENTIFIER_MARKER_RE.search(name) and not xfail_protected:
                continue
            collision = False
            if proposed:
                same_scope_names = state.definition_names_by_scope.get(
                    (anchor.path, anchor.scope), set()
                )
                collision = proposed in same_scope_names and proposed != name

            contracts: list[str] = []
            if protected:
                classification = (
                    "marcador_protegido_xfail"
                    if xfail_protected
                    else "marcador_preso_contrato_historico"
                )
                decision = "preservar"
                risk = "proibitivo" if classification == "marcador_protegido_xfail" else "alto"
                wave = "onda_0_preservacao"
                compatibility = True
                reason = (
                    "Símbolo protegido pelos xfail históricos congelados."
                    if classification == "marcador_protegido_xfail"
                    else "Símbolo AP-003 integra a estrutura consolidada e não pode ser reaberto na AP-004D."
                )
                contracts.append(
                    "AP-004D: xfail congelado"
                    if classification == "marcador_protegido_xfail"
                    else "AP-003 estrutural consolidada"
                )
            elif _is_frozen_file(anchor.path):
                classification = "marcador_preso_contrato_historico"
                decision = "preservar"
                risk = "proibitivo"
                wave = "onda_0_preservacao"
                compatibility = True
                reason = "Símbolo está em arquivo fulltext explicitamente congelado."
                contracts.append("fulltext v1_13/v1_14 congelado")
            elif collision:
                classification = "colisao_destino"
                decision = "revisao_manual"
                risk = "alto"
                wave = "onda_3_manual"
                compatibility = False
                reason = f"Destino proposto {proposed!r} já existe no mesmo escopo AST."
                contracts.append("unicidade de símbolo no escopo")
            elif not definitions:
                classification = "marcador_ambiguo_decisao_manual"
                decision = "revisao_manual"
                risk = "medio"
                wave = "onda_3_manual"
                compatibility = False
                reason = "Não foi encontrada definição AST local; o símbolo pode vir de importação dinâmica ou contrato externo."
            elif name.startswith("_") and proposed:
                classification = "marcador_privado_renomeavel_ast"
                decision = "candidato"
                risk = "medio" if len(occurrences) > 10 else "baixo"
                wave = "onda_1_ast"
                compatibility = False
                reason = "Símbolo privado possui marcador removível, definição AST identificada e nenhum destino colidente no escopo."
            else:
                classification = "marcador_necessario_compatibilidade"
                decision = "preservar" if not name.startswith("_") else "revisao_manual"
                risk = "alto" if not name.startswith("_") else "medio"
                wave = "onda_0_preservacao" if decision == "preservar" else "onda_3_manual"
                compatibility = not name.startswith("_")
                reason = (
                    "Símbolo não privado pode integrar superfície pública ou compatibilidade; não é candidato automático."
                    if not name.startswith("_")
                    else "Não foi possível produzir destino AST válido sem ambiguidade."
                )
                if compatibility:
                    contracts.append("superfície potencialmente pública")

            state.records.append(
                _make_record(
                    current=name,
                    proposed=proposed,
                    path=anchor.path,
                    line=anchor.line,
                    column=anchor.column,
                    occurrence_type="python_identifier",
                    ast_scope=anchor.scope,
                    consumers=consumers,
                    consumer_count=len(occurrences),
                    contracts=contracts,
                    risk=risk,
                    decision=decision,
                    reason=reason,
                    wave=wave,
                    compatibility_required=compatibility,
                    classification=classification,
                    marker_kind=_marker_kind(name),
                    context=f"{anchor.kind}; qualified={qualified}",
                    collision=collision,
                )
            )


def _scan_repository(repo_root: Path, tracked: Sequence[PurePosixPath]) -> ScanState:
    state = ScanState()
    _scan_paths(tracked, state)

    for relative_path in tracked:
        relative = str(relative_path)
        absolute = repo_root / relative_path
        if not absolute.is_file():
            state.skipped_files.append({"path": relative, "reason": "not_regular_file"})
            continue
        if relative_path.name in LOCKFILE_BASENAMES:
            state.skipped_files.append({"path": relative, "reason": "third_party_lockfile"})
            continue
        if not _is_text_candidate(relative_path):
            continue
        size = absolute.stat().st_size
        if size > MAX_TEXT_BYTES:
            state.skipped_files.append(
                {"path": relative, "reason": f"text_file_too_large:{size}"}
            )
            continue
        if relative_path.suffix.lower() in {".py", ".pyi"}:
            _scan_python_file(absolute=absolute, relative=relative, state=state)
        else:
            try:
                source = absolute.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    source = absolute.read_text(encoding="latin-1")
                except Exception as exc:  # noqa: BLE001
                    state.skipped_files.append(
                        {"path": relative, "reason": f"decode_error:{exc}"}
                    )
                    continue
            _scan_text_lines(source=source, relative=relative, state=state)

    _classify_identifiers(state)
    state.records.sort(
        key=lambda item: (
            item["decision"] != "candidato",
            item["classification"],
            item["path"],
            item["line"],
            item["column"],
            item["current"],
        )
    )
    _deduplicate_records(state)
    return state


def _deduplicate_records(state: ScanState) -> None:
    unique: dict[str, dict[str, Any]] = {}
    for record in state.records:
        existing = unique.get(record["id"])
        if existing is None:
            unique[record["id"]] = record
            continue
        merged_consumers = sorted(set(existing["consumers"]) | set(record["consumers"]))
        existing["consumers"] = merged_consumers[:MAX_CONSUMERS_PER_RECORD]
        existing["consumer_count"] = max(
            existing["consumer_count"], record["consumer_count"], len(merged_consumers)
        )
    state.records = list(unique.values())
    state.records.sort(
        key=lambda item: (
            item["decision"] != "candidato",
            item["classification"],
            item["path"],
            item["line"],
            item["column"],
            item["current"],
        )
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _logical_inventory_digest(payload: Mapping[str, Any]) -> str:
    copy_payload = dict(payload)
    copy_payload.pop("inventory_sha256", None)
    encoded = json.dumps(
        copy_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _build_inventory_payload(
    *,
    baseline: GitBaseline,
    state: ScanState,
    tool_sha256: str,
    tracked_total_count: int,
    tracked_scoped_count: int,
    excluded_counts: Mapping[str, int],
) -> dict[str, Any]:
    classification_counts = Counter(record["classification"] for record in state.records)
    decision_counts = Counter(record["decision"] for record in state.records)
    wave_counts = Counter(record["wave"] for record in state.records)
    risk_counts = Counter(record["risk"] for record in state.records)
    candidates = [record for record in state.records if record["decision"] == "candidato"]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "AP-004D",
        "purpose": "inventário preparatório de marcadores internos de versão",
        "generated_from_commit_time": baseline.commit_time,
        "git": {
            "branch": baseline.branch,
            "head": baseline.head,
            "head_subject": baseline.subject,
            "remote_ref": REMOTE_REF,
            "remote_head": baseline.remote_head,
            "divergence": [baseline.divergence_left, baseline.divergence_right],
            "unexpected_worktree_status": [],
        },
        "tool": {
            "path": str(OUTPUT_TOOL),
            "sha256": tool_sha256,
            "writes_productive_files": False,
            "creates_commit": False,
            "pushes": False,
        },
        "scope": {
            "repo_root_redacted": "<git-root>",
            "software_root": str(SOFTWARE_RELATIVE),
            "tracked_files_total": tracked_total_count,
            "tracked_files_considered": tracked_scoped_count,
            "excluded_tracked_files_by_reason": dict(sorted(excluded_counts.items())),
            "scan_roots": [str(path) for path in SCAN_ROOTS],
            "excluded_path_prefixes": [str(path) for path in EXCLUDED_PATH_PREFIXES],
            "project_instances_prefix": str(PROJECT_INSTANCES_PREFIX),
            "project_executable_metadata_suffixes": sorted(PROJECT_EXECUTABLE_METADATA_SUFFIXES),
            "excluded_directory_names": sorted(EXCLUDED_DIRECTORY_NAMES),
            "excluded_data_suffixes": sorted(EXCLUDED_DATA_SUFFIXES),
            "bare_semver_requires_version_context": True,
            "excluded_output_paths": sorted(OUTPUT_PATH_SET),
            "protected_symbols": sorted(PROTECTED_SYMBOLS),
            "protected_qualified_symbols": sorted(PROTECTED_QUALIFIED),
            "structural_ap003_symbols": sorted(STRUCTURAL_AP003_SYMBOLS),
            "frozen_fulltext_files": sorted(FROZEN_FULLTEXT_FILES),
        },
        "classification_catalog": sorted(CLASSIFICATIONS),
        "summary": {
            "record_count": len(state.records),
            "candidate_count": len(candidates),
            "classification_counts": dict(sorted(classification_counts.items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "wave_counts": dict(sorted(wave_counts.items())),
            "risk_counts": dict(sorted(risk_counts.items())),
            "skipped_file_count": len(state.skipped_files),
            "parse_error_count": len(state.parse_errors),
            "collision_count": sum(bool(record["collision"]) for record in state.records),
        },
        "records": state.records,
        "skipped_files": sorted(state.skipped_files, key=lambda item: (item["path"], item["reason"])),
        "parse_errors": sorted(
            state.parse_errors,
            key=lambda item: (item["path"], item.get("line") or 0, item.get("column") or 0),
        ),
        "application_gate": {
            "productive_applicator_allowed": False,
            "reason": "Exige aprovação expressa do inventário AP-004D.",
        },
    }
    payload["inventory_sha256"] = _logical_inventory_digest(payload)
    return payload


def _escape_md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _render_inventory_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    records = payload["records"]
    candidates = [record for record in records if record["decision"] == "candidato"]
    lines = [
        "# AP-004D — Inventário de marcadores de versão",
        "",
        "> **Estado:** preparação somente leitura concluída. O aplicador produtivo permanece bloqueado até aprovação expressa.",
        "",
        "## Baseline verificada pelo inventariador",
        "",
        f"- Branch: `{payload['git']['branch']}`",
        f"- HEAD: `{payload['git']['head']}`",
        f"- Assunto: `{payload['git']['head_subject']}`",
        f"- Remoto: `{payload['git']['remote_ref']}` = `{payload['git']['remote_head']}`",
        f"- Divergência: `{payload['git']['divergence'][0]} {payload['git']['divergence'][1]}`",
        f"- Digest lógico do inventário: `{payload['inventory_sha256']}`",
        "",
        "## Resumo",
        "",
        f"- Registros: **{summary['record_count']}**",
        f"- Candidatos: **{summary['candidate_count']}**",
        f"- Colisões de destino: **{summary['collision_count']}**",
        f"- Arquivos ignorados com justificativa: **{summary['skipped_file_count']}**",
        f"- Erros de análise sintática/tokenização: **{summary['parse_error_count']}**",
        "",
        "### Contagem por classificação",
        "",
        "| Classificação | Quantidade |",
        "|---|---:|",
    ]
    for key, count in summary["classification_counts"].items():
        lines.append(f"| `{_escape_md(key)}` | {count} |")

    lines.extend(
        [
            "",
            "## Candidatos à AP-004D",
            "",
            "Nenhum item desta seção autoriza alteração produtiva. Cada candidato requer aprovação expressa.",
            "",
            "| ID | Atual | Proposto | Arquivo:linha | Tipo | Escopo AST | Consumidores | Risco | Onda | Motivo |",
            "|---|---|---|---|---|---|---:|---|---|---|",
        ]
    )
    if candidates:
        for record in candidates:
            lines.append(
                "| {id} | `{current}` | `{proposed}` | `{path}:{line}` | `{otype}` | `{scope}` | {consumers} | `{risk}` | `{wave}` | {reason} |".format(
                    id=record["id"],
                    current=_escape_md(record["current"]),
                    proposed=_escape_md(record["proposed"]),
                    path=_escape_md(record["path"]),
                    line=record["line"],
                    otype=_escape_md(record["occurrence_type"]),
                    scope=_escape_md(record["ast_scope"] or "—"),
                    consumers=record["consumer_count"],
                    risk=record["risk"],
                    wave=record["wave"],
                    reason=_escape_md(record["reason"]),
                )
            )
    else:
        lines.append("| — | — | — | — | — | — | 0 | — | — | Nenhum candidato automático identificado. |")

    lines.extend(
        [
            "",
            "## Inventário completo",
            "",
            "| ID | Classificação | Decisão | Atual | Proposto | Arquivo:linha:coluna | Tipo | Escopo AST | Compatibilidade | Contratos | Contexto |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for record in records:
        lines.append(
            "| {id} | `{classification}` | `{decision}` | `{current}` | `{proposed}` | `{path}:{line}:{column}` | `{otype}` | `{scope}` | `{compat}` | {contracts} | {context} |".format(
                id=record["id"],
                classification=record["classification"],
                decision=record["decision"],
                current=_escape_md(record["current"]),
                proposed=_escape_md(record["proposed"] or "—"),
                path=_escape_md(record["path"]),
                line=record["line"],
                column=record["column"],
                otype=_escape_md(record["occurrence_type"]),
                scope=_escape_md(record["ast_scope"] or "—"),
                compat="sim" if record["compatibility_required"] else "não",
                contracts=_escape_md(", ".join(record["contracts"]) or "—"),
                context=_escape_md(record["context"]),
            )
        )

    if payload["parse_errors"]:
        lines.extend(["", "## Erros de análise", ""])
        for error in payload["parse_errors"]:
            lines.append(
                f"- `{error['path']}:{error.get('line') or 0}:{error.get('column') or 0}` — {_escape_md(error['message'])}"
            )

    if payload["skipped_files"]:
        lines.extend(["", "## Arquivos ignorados", ""])
        for item in payload["skipped_files"]:
            lines.append(f"- `{item['path']}` — `{item['reason']}`")

    lines.extend(
        [
            "",
            "## Bloqueio de aplicação",
            "",
            "O aplicador produtivo **não deve ser criado nem executado** antes da aprovação expressa deste inventário. A eventual aplicação deverá usar transformação AST para símbolos, escrita atômica, backup externo e rollback integral.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_strategy_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    candidate_records = [
        record for record in payload["records"] if record["decision"] == "candidato"
    ]
    ast_candidates = [
        record
        for record in candidate_records
        if record["classification"] == "marcador_privado_renomeavel_ast"
    ]
    textual_candidates = [
        record
        for record in candidate_records
        if record["classification"] == "marcador_interno_removivel"
    ]
    return textwrap.dedent(
        f"""\
        # AP-004D — Estratégia de consolidação dos marcadores de versão

        > **Gate:** aplicador produtivo bloqueado até aprovação expressa do inventário `{payload['inventory_sha256']}`.

        ## Objetivo

        Consolidar somente marcadores internos de versão sem contrato público, sem função de compatibilidade, sem vínculo com artefatos históricos congelados e sem alteração comportamental.

        ## Baseline

        - Branch: `{payload['git']['branch']}`
        - Commit AP-004C: `{payload['git']['head']}`
        - Remoto sincronizado: `{payload['git']['remote_head']}`
        - Divergência: `{payload['git']['divergence'][0]} {payload['git']['divergence'][1]}`
        - Registros inventariados: **{summary['record_count']}**
        - Candidatos AST: **{len(ast_candidates)}**
        - Candidatos textuais: **{len(textual_candidates)}**
        - Colisões: **{summary['collision_count']}**

        ## Ondas propostas

        ### Onda 0 — preservação explícita

        Não alterar símbolos AP-003, os quatro símbolos protegidos, os três defeitos congelados, os dois arquivos fulltext, caminhos físicos fora do escopo, superfícies públicas, wrappers, snapshots, fixtures, manifestos e metadados de compatibilidade.

        ### Onda 1 — símbolos privados por AST

        Considerar exclusivamente registros classificados como `marcador_privado_renomeavel_ast`, sem colisão e com definição/consumidores identificados. A transformação deverá resolver todas as referências da onda antes da primeira escrita; substituição textual é proibida.

        ### Onda 2 — comentários internos removíveis

        Considerar apenas registros classificados como `marcador_interno_removivel`. Cada alteração deve preservar o significado do comentário e não pode alcançar strings, documentos gerados ou registros históricos.

        ### Onda 3 — revisão manual

        Adiar strings operacionais, identificadores sem definição local, ocorrências documentais, caminhos versionados e qualquer item ambíguo. Esses itens exigem contrato explícito ou prova adicional de ausência de consumidores.

        ## Contrato do futuro aplicador

        O eventual aplicador deverá ser idempotente, pré-validar todas as ondas de uma vez, usar backup externo, escrita atômica e rollback integral, informar caminhos alterados, executar `py_compile`, `git diff --check`, testes específicos e suíte consolidada, e nunca criar commit ou publicar automaticamente.

        ## Critério de autorização

        A criação do aplicador produtivo só poderá começar após aprovação expressa dos candidatos listados no inventário. Aprovação genérica da fase não substitui a aprovação da lista de candidatos e dos destinos propostos.
        """
    )


def _render_contract_test(inventory_digest: str) -> str:
    classifications_literal = repr(sorted(CLASSIFICATIONS))
    return textwrap.dedent(
        f'''\
        """Contrato de caracterização do inventário preparatório AP-004D."""

        from __future__ import annotations

        import hashlib
        import json
        from pathlib import Path

        EXPECTED_SCHEMA = {SCHEMA_VERSION!r}
        EXPECTED_DIGEST = {inventory_digest!r}
        EXPECTED_CLASSIFICATIONS = set({classifications_literal})
        REQUIRED_RECORD_KEYS = {{
            "id",
            "current",
            "proposed",
            "path",
            "line",
            "column",
            "occurrence_type",
            "ast_scope",
            "consumers",
            "consumer_count",
            "contracts",
            "risk",
            "decision",
            "reason",
            "wave",
            "compatibility_required",
            "classification",
            "marker_kind",
            "context",
            "collision",
        }}


        def _find_repo_root() -> Path:
            current = Path(__file__).resolve()
            for parent in [current.parent, *current.parents]:
                if (parent / ".git").exists():
                    return parent
            raise AssertionError("Raiz Git não encontrada a partir do teste AP-004D")


        def _load_inventory() -> tuple[Path, dict]:
            repo_root = _find_repo_root()
            path = repo_root / "docs/refactor/academic-pipeline/AP-004/ap004d_version_marker_inventory.json"
            return path, json.loads(path.read_text(encoding="utf-8"))


        def _logical_digest(payload: dict) -> str:
            normalized = dict(payload)
            normalized.pop("inventory_sha256", None)
            encoded = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()


        def test_ap004d_inventory_schema_and_digest() -> None:
            path, payload = _load_inventory()
            assert path.is_file()
            assert payload["schema_version"] == EXPECTED_SCHEMA
            assert payload["inventory_sha256"] == EXPECTED_DIGEST
            assert _logical_digest(payload) == EXPECTED_DIGEST
            assert payload["application_gate"]["productive_applicator_allowed"] is False


        def test_ap004d_inventory_records_are_complete_and_unique() -> None:
            _, payload = _load_inventory()
            records = payload["records"]
            assert payload["summary"]["record_count"] == len(records)
            assert len({{record["id"] for record in records}}) == len(records)
            assert all(REQUIRED_RECORD_KEYS <= set(record) for record in records)
            assert all(record["classification"] in EXPECTED_CLASSIFICATIONS for record in records)
            assert all(not (record["decision"] == "candidato" and record["collision"]) for record in records)


        def test_ap004d_protected_and_frozen_contracts_are_not_candidates() -> None:
            _, payload = _load_inventory()
            protected = set(payload["scope"]["protected_symbols"])
            frozen = set(payload["scope"]["frozen_fulltext_files"])
            record_names = {{record["current"] for record in payload["records"]}}
            assert protected <= record_names
            assert any(
                record["current"] == "_normalize"
                and "WorkflowState._normalize" in record["context"]
                for record in payload["records"]
            )
            for record in payload["records"]:
                if record["current"] in protected or Path(record["path"]).name in frozen:
                    assert record["decision"] != "candidato"
                if record["classification"] == "marcador_protegido_xfail":
                    assert record["decision"] == "preservar"
                    assert record["wave"] == "onda_0_preservacao"


        def test_ap004d_markdown_artifacts_reference_same_digest() -> None:
            repo_root = _find_repo_root()
            for relative in (
                "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_INVENTORY.md",
                "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_STRATEGY.md",
            ):
                text = (repo_root / relative).read_text(encoding="utf-8")
                assert EXPECTED_DIGEST in text
        '''
    )


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InventoryError("Schema do inventário inválido.")
    records = payload.get("records")
    if not isinstance(records, list):
        raise InventoryError("Campo records inválido.")
    if payload["summary"]["record_count"] != len(records):
        raise InventoryError("Contagem de registros inconsistente.")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise InventoryError("IDs duplicados no inventário.")
    for record in records:
        if record["classification"] not in CLASSIFICATIONS:
            raise InventoryError(f"Classificação desconhecida: {record['classification']}")
        if record["decision"] == "candidato" and record["collision"]:
            raise InventoryError(f"Candidato com colisão indevida: {record['id']}")
    if _logical_inventory_digest(payload) != payload["inventory_sha256"]:
        raise InventoryError("Digest lógico do inventário inconsistente.")


def _validate_generated_python(source: str, label: str) -> None:
    try:
        compile(source, label, "exec")
    except SyntaxError as exc:
        raise InventoryError(f"Python gerado inválido em {label}: {exc}") from exc


def _ensure_output_safety(repo_root: Path) -> None:
    resolved_root = repo_root.resolve()
    for relative in OUTPUT_PATHS:
        target = repo_root / relative
        resolved_parent = target.parent.resolve(strict=False)
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError as exc:
            raise InventoryError(f"Destino escapa da raiz Git: {target}") from exc
        current = target.parent
        while current != repo_root and not current.exists():
            current = current.parent
        if current.is_symlink():
            raise InventoryError(f"Ancestral de destino não pode ser symlink: {current}")
        if target.exists() and target.is_symlink():
            raise InventoryError(f"Destino não pode ser symlink: {target}")


def _backup_root(repo_root: Path, head: str) -> Path:
    base = Path.home() / ".cache" / "mppg-refactor" / "ap004d" / "backups"
    candidate = base / f"{head[:12]}-{uuid.uuid4().hex}"
    resolved_repo = repo_root.resolve()
    resolved_candidate_parent = candidate.parent.resolve()
    try:
        resolved_candidate_parent.relative_to(resolved_repo)
    except ValueError:
        pass
    else:
        raise InventoryError("Diretório de backup deve ficar fora do repositório.")
    return candidate


def _create_backup(
    repo_root: Path, backup_dir: Path
) -> dict[str, dict[str, Any]]:
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, dict[str, Any]] = {}
    for relative in OUTPUT_PATHS:
        target = repo_root / relative
        key = str(relative)
        if target.exists():
            backup_target = backup_dir / relative
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_target)
            manifest[key] = {
                "existed": True,
                "sha256": _sha256_bytes(target.read_bytes()),
                "mode": target.stat().st_mode & 0o777,
            }
        else:
            manifest[key] = {"existed": False, "sha256": None, "mode": None}
    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _rollback(
    repo_root: Path,
    backup_dir: Path,
    manifest: Mapping[str, Mapping[str, Any]],
) -> None:
    errors: list[str] = []
    for relative in reversed(OUTPUT_PATHS):
        key = str(relative)
        target = repo_root / relative
        metadata = manifest[key]
        try:
            if metadata["existed"]:
                backup_target = backup_dir / relative
                _atomic_write(
                    target,
                    backup_target.read_bytes(),
                    int(metadata["mode"] or 0o644),
                )
            else:
                target.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{target}: {exc}")
    if errors:
        raise InventoryError("Rollback incompleto:\n- " + "\n- ".join(errors))


def _check_no_trailing_whitespace(contents: Mapping[PurePosixPath, bytes]) -> None:
    problems: list[str] = []
    for relative, data in contents.items():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InventoryError(f"Saída não UTF-8: {relative}: {exc}") from exc
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.endswith((" ", "\t")):
                problems.append(f"{relative}:{line_no}: whitespace final")
    if problems:
        raise InventoryError("Falha equivalente a diff --check:\n- " + "\n- ".join(problems))


def _post_write_validation(
    *,
    repo_root: Path,
    contents: Mapping[PurePosixPath, bytes],
    skip_tests: bool,
) -> None:
    for relative, expected in contents.items():
        target = repo_root / relative
        if not target.is_file():
            raise InventoryError(f"Saída ausente após escrita: {relative}")
        actual = target.read_bytes()
        if actual != expected:
            raise InventoryError(f"Conteúdo divergente após escrita: {relative}")

    _check_no_trailing_whitespace(contents)
    _run(["git", "diff", "--check", "--"], cwd=repo_root)

    status = _parse_status(repo_root)
    unexpected = [path for path in status if path not in OUTPUT_PATH_SET]
    if unexpected:
        raise InventoryError(
            "A escrita preparatória alterou caminhos inesperados:\n- " + "\n- ".join(unexpected)
        )

    if not skip_tests:
        software_root = repo_root / SOFTWARE_RELATIVE
        _run(
            [
                "pipenv",
                "run",
                "pytest",
                "-q",
                str(PurePosixPath("tests/characterization/test_ap004d_version_marker_inventory_contract.py")),
            ],
            cwd=software_root,
        )


def _write_transaction(
    *,
    baseline: GitBaseline,
    contents: Mapping[PurePosixPath, bytes],
    skip_tests: bool,
) -> Path:
    repo_root = baseline.repo_root
    _ensure_output_safety(repo_root)
    _check_no_trailing_whitespace(contents)
    for relative, data in contents.items():
        if relative.suffix == ".py":
            _validate_generated_python(data.decode("utf-8"), str(relative))

    backup_dir = _backup_root(repo_root, baseline.head)
    manifest = _create_backup(repo_root, backup_dir)
    try:
        for relative in OUTPUT_PATHS:
            mode = 0o755 if relative == OUTPUT_TOOL else 0o644
            _atomic_write(repo_root / relative, contents[relative], mode)
        _post_write_validation(
            repo_root=repo_root,
            contents=contents,
            skip_tests=skip_tests,
        )
    except Exception as original_exc:
        try:
            _rollback(repo_root, backup_dir, manifest)
        except Exception as rollback_exc:
            raise InventoryError(
                f"Falha na preparação: {original_exc}\nFalha adicional no rollback: {rollback_exc}"
            ) from original_exc
        raise InventoryError(
            f"Falha na preparação; rollback integral aplicado. Motivo: {original_exc}"
        ) from original_exc
    return backup_dir


def _read_self_source() -> bytes:
    source_path = Path(__file__).resolve()
    data = source_path.read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError(f"O próprio inventariador não está em UTF-8: {exc}") from exc
    return data


def _build_contents(
    *, payload: Mapping[str, Any], self_source: bytes
) -> dict[PurePosixPath, bytes]:
    inventory_json = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    inventory_md = _render_inventory_markdown(payload).encode("utf-8")
    strategy_md = _render_strategy_markdown(payload).encode("utf-8")
    contract_test = _render_contract_test(payload["inventory_sha256"]).encode("utf-8")
    return {
        OUTPUT_INVENTORY_MD: inventory_md,
        OUTPUT_STRATEGY_MD: strategy_md,
        OUTPUT_INVENTORY_JSON: inventory_json,
        OUTPUT_TEST: contract_test,
        OUTPUT_TOOL: self_source,
    }


def _print_success(
    *, baseline: GitBaseline, payload: Mapping[str, Any], backup_dir: Path
) -> None:
    summary = payload["summary"]
    print("[OK] Baseline AP-004C confirmada.")
    print(f"[OK] Branch: {baseline.branch}")
    print(f"[OK] HEAD local/remoto: {baseline.head}")
    print("[OK] Divergência: 0 0")
    print(f"[OK] Registros inventariados: {summary['record_count']}")
    print(f"[OK] Candidatos: {summary['candidate_count']}")
    print(f"[OK] Colisões: {summary['collision_count']}")
    print(f"[OK] Digest lógico: {payload['inventory_sha256']}")
    print(f"[OK] Backup externo: {backup_dir}")
    print("[OK] Caminhos preparatórios escritos:")
    for relative in OUTPUT_PATHS:
        print(f"  - {relative}")
    print("[BLOQUEIO] Aplicador produtivo não criado. Aguarde aprovação expressa do inventário.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Valida a consolidação AP-004C e gera somente os cinco artefatos "
            "preparatórios do inventário AP-004D."
        )
    )
    parser.add_argument(
        "--repo-root",
        help="Raiz do repositório Git. Por padrão, detectada a partir do diretório atual.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Não executar git fetch origin (uso excepcional/offline; ainda exige ref remota local).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Não executar o teste de caracterização após a escrita (uso diagnóstico).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        repo_root = _discover_repo_root(args.repo_root)
        baseline = _validate_git_baseline(repo_root, fetch=not args.no_fetch)
        selection = _git_tracked_files(repo_root)
        self_source = _read_self_source()
        tool_sha256 = _sha256_bytes(self_source)
        state = _scan_repository(repo_root, selection.included)
        payload = _build_inventory_payload(
            baseline=baseline,
            state=state,
            tool_sha256=tool_sha256,
            tracked_total_count=selection.total_count,
            tracked_scoped_count=len(selection.included),
            excluded_counts=dict(selection.excluded_counts),
        )
        _validate_payload(payload)
        contents = _build_contents(payload=payload, self_source=self_source)

        # Complete prevalidation of all five outputs before the first repository write.
        for relative, data in contents.items():
            if relative.suffix == ".py":
                _validate_generated_python(data.decode("utf-8"), str(relative))
        _check_no_trailing_whitespace(contents)

        backup_dir = _write_transaction(
            baseline=baseline,
            contents=contents,
            skip_tests=args.skip_tests,
        )
        _print_success(baseline=baseline, payload=payload, backup_dir=backup_dir)
        return 0
    except InventoryError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        print(
            "[BLOQUEIO] Nenhum aplicador produtivo deve ser criado ou executado.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("[ERRO] Execução interrompida pelo usuário.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

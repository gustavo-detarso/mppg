#!/usr/bin/env python3
"""AP-004A — inventário e convenção canônica de nomes internos.

Este aplicador preparatório v4.2 (inventário canônico estrito) não altera código produtivo. Ele:
- valida branch, sincronização local/remota e encerramento AP-003G;
- aceita árvore limpa ou exclusivamente os cinco artefatos não rastreados da AP-004A;
- valida a arquitetura congelada da AP-003;
- separa ocorrências brutas, evidências contextuais e candidatos acionáveis;
- protege saídas operacionais, scripts históricos de manutenção, assets e compatibilidade real;
- exclui dunder, imports da biblioteca padrão e falsos gatilhos semânticos;
- consolida módulo e entrypoint numa única decisão acionável;
- detecta colisões e suspende sugestões semanticamente inseguras;
- grava somente ferramentas, documentação, JSON e teste contratual;
- executa py_compile, git diff --check, suíte específica e suíte consolidada;
- cria backup externo e realiza rollback integral em caso de falha.

Execute a partir da raiz do software, mantendo este arquivo fora do repositório
(por exemplo, em ~/Downloads).
"""

from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, NoReturn, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 não é suportado
    tomllib = None  # type: ignore[assignment]


EXPECTED_SOFTWARE_ROOT = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline/"
    "software/academic_pipeline_rc10_7_conformidade"
)
EXPECTED_REPOSITORY_ROOT = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline"
)
EXPECTED_BRANCH = "ap-refactor/03-orchestrator-decomposition"
EXPECTED_REMOTE_REF = f"origin/{EXPECTED_BRANCH}"
EXPECTED_REMOTE_BRANCH_REF = f"refs/heads/{EXPECTED_BRANCH}"
SOFTWARE_DIRNAME = "academic_pipeline_rc10_7_conformidade"

PHASE = "AP-004A"
MODE = "inventory-and-convention-v4.2-read-only"
BASELINE_PASSED = 408
BASELINE_XFAILED = 3
EXPECTED_CONTRACT_TESTS = 10

DOC_DIR = Path("docs/refactor/academic-pipeline/AP-004")
REPORT_REL = DOC_DIR / "AP-004A_NAMING_INVENTORY.md"
CONVENTION_REL = DOC_DIR / "AP-004_NAMING_CONVENTION.md"
INVENTORY_REL = DOC_DIR / "ap004a_naming_inventory.json"
TOOL_REL = Path("tools/refactor/ap004a_inventory_names.py")
TEST_REL = Path(
    "tests/characterization/test_ap004a_naming_inventory_contract.py"
)

OUTPUT_RELS = (
    REPORT_REL,
    CONVENTION_REL,
    INVENTORY_REL,
    TOOL_REL,
    TEST_REL,
)
ALLOWED_OUTPUT_PREFIXES = (
    "docs/refactor/academic-pipeline/AP-004/",
    "tests/characterization/test_ap004a_",
    "tools/refactor/ap004a_",
)

AP003G_ARTIFACTS = (
    Path(
        "docs/refactor/academic-pipeline/AP-003/"
        "AP-003G_STABILIZATION_PREPARATION.md"
    ),
    Path(
        "docs/refactor/academic-pipeline/AP-003/"
        "ap003g_stabilization_inventory.json"
    ),
    Path(
        "docs/refactor/academic-pipeline/AP-003/"
        "AP-003G_STABILIZATION_CLOSURE.md"
    ),
    Path(
        "docs/refactor/academic-pipeline/AP-003/"
        "ap003g_manifest.json"
    ),
    Path(
        "tests/characterization/"
        "test_ap003g_stabilization_contract.py"
    ),
)

ORCHESTRATOR_REL = Path(
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)
PACKAGE_MAIN_REL = Path("academic_pipeline/__main__.py")
PARSER_REL = Path("academic_pipeline/cli_parser.py")
DISPATCH_REL = Path("academic_pipeline/command_dispatch.py")
DOCUMENT_REL = Path("academic_pipeline/document_orchestration.py")
PRISMA_REL = Path("academic_pipeline/prisma_generic_orchestration.py")

AP003_PRODUCTION_MODULES = (
    ORCHESTRATOR_REL,
    PACKAGE_MAIN_REL,
    PARSER_REL,
    DISPATCH_REL,
    DOCUMENT_REL,
    PRISMA_REL,
)

AP003_CORE_NAME = "_ap003f_pipeline_core"
AP003_HISTORICAL_ALIAS = (
    "_original_main_before_prisma_artigo_generico_wrapper"
)
KNOWN_XFAIL_SYMBOLS = (
    "_refs_v6_strip_org",
    "extract_org_abstracts",
    "WorkflowState._normalize",
)

RENDER_DOCX_REL = Path("app_bundle/scripts/pipeline/render_docx_canonico.py")
ARTICLE_STATE_REL = Path("app_bundle/scripts/pipeline/article_workflow/state.py")
KNOWN_XFAIL_DEFINITIONS = (
    {
        "qualified_name": "_refs_v6_strip_org",
        "symbol": "_refs_v6_strip_org",
        "path": ORCHESTRATOR_REL.as_posix(),
    },
    {
        "qualified_name": "extract_org_abstracts",
        "symbol": "extract_org_abstracts",
        "path": RENDER_DOCX_REL.as_posix(),
    },
    {
        "qualified_name": "WorkflowState._normalize",
        "symbol": "_normalize",
        "path": ARTICLE_STATE_REL.as_posix(),
    },
)
KNOWN_XFAIL_LINKED_ALIASES = (
    {
        "name": "_ap003d_impl__refs_v6_strip_org",
        "path": ORCHESTRATOR_REL.as_posix(),
        "form": "from",
        "module": "academic_pipeline.document_orchestration",
        "imported_name": "_refs_v6_strip_org_impl",
    },
)

CLASSIFICATIONS = (
    "renomeação segura",
    "renomeação com compatibilidade",
    "renomeação de alto risco",
    "nome que deve permanecer",
)
CATEGORIES = (
    "arquivo/módulo",
    "função",
    "classe",
    "constante",
    "alias",
)
RAW_SURFACES = CATEGORIES + (
    "import",
    "entrypoint",
    "teste",
    "documentação",
    "operacional",
    "histórico",
)

TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".txt",
    ".org",
    ".toml",
    ".cfg",
    ".ini",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".service",
    ".desktop",
    ".env.example",
}
MAX_TEXT_BYTES = 2_500_000

IGNORED_PATH_PARTS = {
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
}

STRUCTURAL_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "release_candidate",
        re.compile(
            r"(?i)(?<![a-z0-9])rc[_-]?\d+(?:[._-]\d+)*(?![a-z0-9])"
        ),
    ),
    (
        "version_marker",
        re.compile(
            r"(?i)(?<![a-z0-9])v[_-]?\d+(?:[._-]\d+)*(?![a-z0-9])"
        ),
    ),
    (
        "refactor_phase",
        re.compile(
            r"(?i)(?<![a-z0-9])ap[_-]?\d{3}[a-z]?(?![a-z0-9])"
        ),
    ),
    (
        "explicit_version_word",
        re.compile(
            r"(?i)(?<![a-z0-9])"
            r"(?:version|versao|versão|ver|rev|revision|revisao|revisão|release)"
            r"[_-]?\d+(?:[._-]\d+)*(?![a-z0-9])"
        ),
    ),
)

CONTEXTUAL_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "contextual_temporal_label",
        re.compile(
            r"(?i)(?<![a-z0-9])"
            r"(?:novo|nova|new|final|legacy|legado|old|antigo|antiga|latest|"
            r"definitivo|definitiva|original|pre)"
            r"(?![a-z0-9])"
        ),
    ),
)

MARKER_PATTERNS = STRUCTURAL_MARKER_PATTERNS + CONTEXTUAL_MARKER_PATTERNS

PHASE_AUDIT_PATH_RE = re.compile(
    r"(?i)(?:^|/)(?:AP-\d{3}[A-Z]?|test_ap\d{3}[a-z]?_|ap\d{3}[a-z]?_)"
)
TEST_PATH_RE = re.compile(r"(?:^|/)(?:tests?|test_[^/]+)(?:/|$)", re.I)
DOC_PATH_RE = re.compile(r"(?:^|/)(?:docs?|documentation)(?:/|$)", re.I)
SUMMARY_PATTERN = re.compile(
    r"(?P<passed>\d+)\s+passed"
    r"(?:,\s+(?P<xfailed>\d+)\s+xfailed)?"
)

SPECIAL_SUGGESTIONS = {
    "academic_pipeline_rc10.py": "pipeline_orchestrator.py",
    "academic_pipeline_rc10": "pipeline_orchestrator",
    AP003_CORE_NAME: "_run_pipeline",
}


LEADING_IMPLEMENTATION_MARKER_RE = re.compile(
    r"^(?P<private>_?)(?:(?:rc|v)_?\d+(?:_\d+)*[a-z]?|ap_?\d{3}[a-z]?)_(?P<rest>.+)$",
    re.I,
)
IMPLEMENTATION_MARKER_TOKEN_RE = re.compile(
    r"^(?:(?:rc|v)\d+[a-z]?|ap\d{3}[a-z]?)$",
    re.I,
)
OPAQUE_STAGE_DISPATCH_RE = re.compile(
    r"^_?(?:ap_?\d{3}[a-z]?_)?(?:stage|dispatch)_\d+$",
    re.I,
)
LEGACY_SEMANTIC_RE = re.compile(
    r"(?:^|_)(?:legacy|legado)(?:_|$)|(?:^|\.)(?:legacy|legado)(?:\.|$)",
    re.I,
)
INSTALLER_NAME_RE = re.compile(r"^install(?:er|acao|ação)?[^/]*\.(?:sh|bash|zsh)$", re.I)
ROOT_MAINTENANCE_SCRIPT_RE = re.compile(
    r"^(?:aplicar|atualizar|migrar|migrador|migration|patch|corrigir)_.*\.py$",
    re.I,
)
OUTPUT_DIR_RE = re.compile(r"^output(?:_|$)", re.I)
PHASE_TOOL_NAME_RE = re.compile(r"^ap\d{3}[a-z]?_", re.I)
DUNDER_NAME_RE = re.compile(r"^__[^_].*__$")
UNSAFE_SUGGESTION_TOKEN_RE = re.compile(
    r"(?:^|_)(?:original|pre|stage_?\d+|dispatch_?\d+)(?:_|$)",
    re.I,
)
OPAQUE_NUMERIC_SUFFIX_RE = re.compile(r"_\d+$")


def protected_path_reason(path: str) -> str | None:
    """Return why a tracked path is outside the actionable AP-004 matrix."""
    pure = PurePosixPath(path)
    parts = pure.parts
    lower_parts = tuple(part.lower() for part in parts)

    if len(parts) >= 2 and lower_parts[:2] == ("app_bundle", "projetos"):
        if any(
            OUTPUT_DIR_RE.match(part) or part == "execucoes_anteriores"
            for part in lower_parts[2:]
        ):
            return "artefato operacional gerado ou execução histórica"

    if INSTALLER_NAME_RE.match(pure.name):
        return "instalador ou script operacional fora do escopo da AP-004"

    if len(parts) == 1 and ROOT_MAINTENANCE_SCRIPT_RE.match(pure.name):
        return "script histórico de aplicação, atualização ou migração na raiz"

    if "assets" in lower_parts:
        return "asset operacional fora do escopo da AP-004"

    if path.startswith("docs/refactor/") and PHASE_AUDIT_PATH_RE.search(path):
        return "documentação auditável de fase da refatoração"

    if path.startswith("tools/refactor/") and PHASE_TOOL_NAME_RE.match(pure.name):
        return "ferramenta auditável vinculada a uma fase da refatoração"

    return None


def is_test_or_documentation_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(path) or DOC_PATH_RE.search(path))


def legacy_is_semantic(name: str, path: str) -> bool:
    """Return whether ``legacy`` denotes compatibility, not a version target.

    Compatibility names may use snake_case, dotted module notation or
    CamelCase (for example ``LegacyRuntimeError``).  AP-004A must preserve
    every explicit legacy/legado symbol for the dedicated AP-004E review; it
    must never turn those names into automatic renaming candidates merely
    because their spelling is historical.
    """
    lowered_name = name.casefold()
    lowered_path = path.casefold()
    explicitly_legacy = (
        "legacy" in lowered_name
        or "legado" in lowered_name
        or "/legacy.py" in f"/{lowered_path.lstrip('/')}"
        or lowered_path == "academic_pipeline/legacy.py"
    )
    if not explicitly_legacy:
        return False

    # Conservative rule: every explicit legacy/legado spelling is preserved
    # for AP-004E.  This covers snake_case, dotted names and CamelCase without
    # turning contextual historical labels into automatic rename targets.
    return True


def known_xfail_protection(name: str, path: str) -> str | None:
    """Return an exact AP-004 scope guard for frozen historical defects.

    Protection is path-bound.  A same-named helper in tests or another module
    is evidence only and must never become a preserved production candidate.
    """
    for target in KNOWN_XFAIL_DEFINITIONS:
        if path == target["path"] and name in {
            target["qualified_name"], target["symbol"]
        }:
            return "xfail histórico congelado"
    for target in KNOWN_XFAIL_LINKED_ALIASES:
        if path == target["path"] and name == target["name"]:
            return "alias diretamente ligado a xfail histórico congelado"
    return None


class InventoryError(RuntimeError):
    """Falha controlada da AP-004A."""


def fail(message: str) -> NoReturn:
    raise InventoryError(message)


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class BackupRecord:
    path: Path
    existed: bool
    backup: Path | None


def run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> CommandResult:
    merged_env = os.environ.copy()
    merged_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        merged_env.update(env)

    try:
        completed = subprocess.run(
            [str(item) for item in args],
            cwd=cwd,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        fail(
            "Executável obrigatório não encontrado: "
            f"{args[0]!s}. Verifique o ambiente antes de executar a AP-004A."
        )
    except subprocess.TimeoutExpired:
        fail(
            "Comando excedeu o tempo limite de "
            f"{timeout}s: " + " ".join(str(item) for item in args)
        )
    result = CommandResult(
        args=tuple(str(item) for item in args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        fail(
            "Comando falhou: "
            + " ".join(result.args)
            + f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def git(
    root: Path,
    *args: str,
    check: bool = True,
    timeout: int = 180,
) -> CommandResult:
    return run(("git", *args), cwd=root, check=check, timeout=timeout)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_id(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_output(content: str) -> str:
    return "\n".join(line.rstrip() for line in content.splitlines()) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_backups(
    paths: Iterable[Path],
    *,
    software_root: Path,
) -> tuple[Path, list[BackupRecord]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        Path.home()
        / ".cache"
        / "academic-pipeline-refactor"
        / "backups"
        / PHASE
        / timestamp
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    records: list[BackupRecord] = []

    for path in paths:
        if path.exists():
            relative = path.relative_to(software_root)
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            records.append(BackupRecord(path, True, destination))
        else:
            records.append(BackupRecord(path, False, None))
    return backup_root, records


def rollback(records: Iterable[BackupRecord]) -> None:
    for record in records:
        if record.existed and record.backup is not None:
            record.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.backup, record.path)
        elif record.path.exists():
            record.path.unlink()


def compile_python(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ap004a-pyc-") as temporary:
        py_compile.compile(
            str(path),
            cfile=str(Path(temporary) / f"{path.name}c"),
            doraise=True,
        )


def normalize_status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip().strip('"').replace("\\", "/")


def repository_relative(
    path: Path,
    *,
    repository_root: Path,
) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def software_relative_status_path(
    line: str,
    *,
    software_root: Path,
    repository_root: Path,
) -> str:
    raw = normalize_status_path(line)
    prefix = repository_relative(
        software_root,
        repository_root=repository_root,
    ).rstrip("/") + "/"
    if raw.startswith(prefix):
        return raw[len(prefix):]
    return raw


def parse_remote_head(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        fail(
            "Não foi possível resolver univocamente a branch remota.\n"
            f"Saída de git ls-remote:\n{output}"
        )
    fields = lines[0].split()
    if len(fields) != 2:
        fail(f"Saída inválida de git ls-remote: {lines[0]}")
    return fields[0]



def validate_git_state(
    software_root: Path,
    *,
    skip_remote_check: bool,
) -> tuple[Path, dict[str, Any]]:
    if software_root.resolve() != EXPECTED_SOFTWARE_ROOT.resolve():
        fail(
            "Diretório incorreto. Execute em:\n"
            f"  {EXPECTED_SOFTWARE_ROOT}\n"
            f"Diretório atual:\n  {software_root}"
        )
    if software_root.name != SOFTWARE_DIRNAME:
        fail(
            f"Raiz de software inesperada: {software_root.name}; "
            f"esperada: {SOFTWARE_DIRNAME}."
        )

    repository_root = Path(
        git(software_root, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve()
    if repository_root != EXPECTED_REPOSITORY_ROOT.resolve():
        fail(
            f"Worktree Git inesperado: {repository_root}\n"
            f"Esperado: {EXPECTED_REPOSITORY_ROOT}"
        )

    branch = git(repository_root, "branch", "--show-current").stdout.strip()
    if branch != EXPECTED_BRANCH:
        fail(
            f"Branch incorreta: {branch or '<detached>'}\n"
            f"Esperada: {EXPECTED_BRANCH}"
        )

    status_lines = [
        line
        for line in git(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout.splitlines()
        if line.strip()
    ]
    expected_outputs = {item.as_posix() for item in OUTPUT_RELS}
    initial_paths: set[str] = set()
    initial_status: dict[str, str] = {}
    unexpected: list[str] = []
    for line in status_lines:
        relative = software_relative_status_path(
            line,
            software_root=software_root,
            repository_root=repository_root,
        )
        parts = PurePosixPath(relative).parts
        if any(part in IGNORED_PATH_PARTS for part in parts) or relative.endswith((".pyc", ".pyo")):
            continue
        code = line[:2]
        initial_paths.add(relative)
        initial_status[relative] = code
        if relative not in expected_outputs or code != "??":
            unexpected.append(f"{code} {relative}")

    if unexpected:
        fail(
            "A árvore contém alterações não relacionadas ou artefatos AP-004A "
            "já rastreados/modificados. A v4.2 aceita apenas árvore limpa ou os "
            "cinco artefatos não rastreados da AP-004A atual:\n"
            + "\n".join(f"  - {item}" for item in unexpected)
        )

    head = git(repository_root, "rev-parse", "HEAD").stdout.strip()
    tracking_head = git(
        repository_root,
        "rev-parse",
        EXPECTED_REMOTE_REF,
    ).stdout.strip()
    if head != tracking_head:
        fail(
            "HEAD local divergente da referência remota local.\n"
            f"HEAD:                  {head}\n"
            f"{EXPECTED_REMOTE_REF}: {tracking_head}"
        )

    upstream = git(
        repository_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    upstream_name = upstream.stdout.strip() if upstream.returncode == 0 else ""
    if upstream_name and upstream_name != EXPECTED_REMOTE_REF:
        fail(
            f"Upstream inesperado: {upstream_name}\n"
            f"Esperado: {EXPECTED_REMOTE_REF}"
        )

    remote_head = tracking_head
    remote_check = "skipped"
    if not skip_remote_check:
        remote = git(
            repository_root,
            "ls-remote",
            "--heads",
            "origin",
            EXPECTED_REMOTE_BRANCH_REF,
            timeout=60,
        )
        remote_head = parse_remote_head(remote.stdout)
        remote_check = "passed"
        if head != remote_head:
            fail(
                "HEAD local não corresponde ao estado publicado no remoto.\n"
                f"HEAD local:  {head}\n"
                f"HEAD remoto: {remote_head}"
            )

    tree_state = "clean" if not initial_paths else "ap004a-artifacts-only"
    return repository_root, {
        "branch": branch,
        "head": head,
        "upstream": upstream_name or EXPECTED_REMOTE_REF,
        "tracking_head": tracking_head,
        "remote_head": remote_head,
        "remote_check": remote_check,
        "tree": tree_state,
        "initial_allowed_paths": sorted(initial_paths),
        "initial_status": initial_status,
    }



def git_last_commit_for_path(
    repository_root: Path,
    repository_relative_path: str,
) -> str:
    return git(
        repository_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        repository_relative_path,
    ).stdout.strip()


def validate_ap003g_commit(
    *,
    software_root: Path,
    repository_root: Path,
    git_state: dict[str, Any],
) -> dict[str, Any]:
    commit_by_artifact: dict[str, str] = {}
    missing: list[str] = []

    for relative in AP003G_ARTIFACTS:
        path = software_root / relative
        if not path.is_file():
            missing.append(relative.as_posix())
            continue
        repository_path = repository_relative(
            path,
            repository_root=repository_root,
        )
        commit = git_last_commit_for_path(repository_root, repository_path)
        if not commit:
            fail(f"Artefato AP-003G não versionado: {relative}")
        commit_by_artifact[relative.as_posix()] = commit

    if missing:
        fail(
            "Artefatos obrigatórios da AP-003G ausentes:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )

    commits = set(commit_by_artifact.values())
    if len(commits) != 1:
        details = "\n".join(
            f"  - {path}: {commit}"
            for path, commit in sorted(commit_by_artifact.items())
        )
        fail(
            "Os artefatos finais da AP-003G não convergem para um commit "
            f"isolado:\n{details}"
        )

    commit = next(iter(commits))
    for descendant in (git_state["head"], git_state["remote_head"]):
        result = git(
            repository_root,
            "merge-base",
            "--is-ancestor",
            commit,
            descendant,
            check=False,
        )
        if result.returncode != 0:
            fail(
                f"O commit AP-003G {commit} não é ancestral de {descendant}."
            )

    metadata_text = git(
        repository_root,
        "show",
        "-s",
        "--format=%H%n%P%n%an%n%ae%n%aI%n%s%n%b",
        commit,
    ).stdout
    metadata_lines = metadata_text.splitlines()
    while len(metadata_lines) < 6:
        metadata_lines.append("")

    changed = [
        line.strip()
        for line in git(
            repository_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        ).stdout.splitlines()
        if line.strip()
    ]
    production_prefix = repository_relative(
        software_root,
        repository_root=repository_root,
    ).rstrip("/") + "/"
    forbidden_production = []
    for item in changed:
        relative = item[len(production_prefix):] if item.startswith(
            production_prefix
        ) else item
        if relative.startswith("academic_pipeline/"):
            forbidden_production.append(relative)
        if relative.startswith("app_bundle/scripts/"):
            forbidden_production.append(relative)

    if forbidden_production:
        fail(
            "O commit identificado como encerramento AP-003G alterou código "
            "produtivo, contrariando o estado canônico:\n"
            + "\n".join(f"  - {item}" for item in forbidden_production)
        )

    return {
        "commit": commit,
        "published": True,
        "subject": metadata_lines[5],
        "parents": metadata_lines[1].split(),
        "author_name": metadata_lines[2],
        "author_email": metadata_lines[3],
        "authored_at": metadata_lines[4],
        "body": "\n".join(metadata_lines[6:]).strip(),
        "changed_files": changed,
        "artifact_commits": commit_by_artifact,
        "productive_files_changed": [],
    }


def top_level_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def name_load_count(tree: ast.AST, name: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == name
    )


def direct_guard_calls(tree: ast.Module) -> list[str]:
    calls: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_guard = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )
        if not is_guard:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                try:
                    calls.append(ast.unparse(child.func))
                except Exception:
                    calls.append(child.func.__class__.__name__)
    return calls


def validate_ap003_architecture(software_root: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for relative in AP003_PRODUCTION_MODULES:
        path = software_root / relative
        if not path.is_file():
            fail(f"Módulo estrutural AP-003 ausente: {relative}")
        compile_python(path)
        hashes[relative.as_posix()] = sha256_path(path)

    orchestrator = software_root / ORCHESTRATOR_REL
    source = orchestrator.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(orchestrator))
    functions = top_level_functions(tree)
    mains = [node for node in functions if node.name == "main"]
    cores = [node for node in functions if node.name == AP003_CORE_NAME]
    alias_assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(child, ast.Name)
            and child.id == AP003_HISTORICAL_ALIAS
            for child in ast.walk(node)
        )
    ]
    guards = direct_guard_calls(tree)

    if len(mains) != 1:
        fail(f"Esperado um main() público; encontrados {len(mains)}.")
    if len(cores) != 1:
        fail(
            f"Esperado um núcleo {AP003_CORE_NAME}; encontrados {len(cores)}."
        )
    if alias_assignments or name_load_count(tree, AP003_HISTORICAL_ALIAS):
        fail(f"Alias histórico reapareceu: {AP003_HISTORICAL_ALIAS}.")
    if guards.count("main") != 1:
        fail(
            "A guarda direta do orquestrador não chama main() exatamente uma "
            f"vez: {guards}"
        )

    package_main_path = software_root / PACKAGE_MAIN_REL
    package_tree = ast.parse(
        package_main_path.read_text(encoding="utf-8"),
        filename=str(package_main_path),
    )
    package_calls = [
        ast.unparse(node.func)
        for node in ast.walk(package_tree)
        if isinstance(node, ast.Call)
    ]
    if not any(call == "main" or call.endswith(".main") for call in package_calls):
        fail("academic_pipeline/__main__.py não chama a entrada pública main().")

    prisma_path = software_root / PRISMA_REL
    prisma_source = prisma_path.read_text(encoding="utf-8")
    prisma_tree = ast.parse(prisma_source, filename=str(prisma_path))
    prisma_references = name_load_count(prisma_tree, AP003_CORE_NAME) + sum(
        1
        for node in ast.walk(prisma_tree)
        if isinstance(node, ast.Attribute) and node.attr == AP003_CORE_NAME
    )
    if prisma_references < 1:
        fail(
            f"O módulo PRISMA não referencia {AP003_CORE_NAME}."
        )

    return {
        "status": "passed",
        "production_hashes": hashes,
        "public_main": {
            "name": "main",
            "line": mains[0].lineno,
            "end_line": mains[0].end_lineno,
        },
        "internal_core": {
            "name": AP003_CORE_NAME,
            "line": cores[0].lineno,
            "end_line": cores[0].end_lineno,
        },
        "historical_alias_assignments": 0,
        "direct_guard_calls": guards,
        "package_main_calls": sorted(set(package_calls)),
        "prisma_core_reference_count": prisma_references,
    }


def tracked_files(
    *,
    software_root: Path,
    repository_root: Path,
) -> list[Path]:
    prefix = repository_relative(
        software_root,
        repository_root=repository_root,
    )
    result = git(
        repository_root,
        "ls-files",
        "-z",
        "--",
        prefix,
    )
    paths: list[Path] = []
    for raw in result.stdout.split("\0"):
        if not raw:
            continue
        path = repository_root / PurePosixPath(raw)
        try:
            relative = path.relative_to(software_root)
        except ValueError:
            continue
        if any(part in IGNORED_PATH_PARTS for part in relative.parts):
            continue
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def read_python_source(path: Path) -> tuple[str, str]:
    try:
        with tokenize.open(path) as handle:
            return handle.read(), handle.encoding
    except (SyntaxError, UnicodeDecodeError):
        data = path.read_bytes()
        return data.decode("utf-8", errors="replace"), "utf-8-replace"


def read_text_file(path: Path) -> tuple[str | None, dict[str, Any]]:
    stat = path.stat()
    metadata = {
        "bytes": stat.st_size,
        "sha256": sha256_path(path),
        "encoding": None,
        "skipped": None,
    }
    if stat.st_size > MAX_TEXT_BYTES:
        metadata["skipped"] = "size-limit"
        return None, metadata
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        metadata["skipped"] = "binary-null-byte"
        return None, metadata
    if path.suffix == ".py":
        text, encoding = read_python_source(path)
        metadata["encoding"] = encoding
        return text, metadata
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = data.decode(encoding)
            metadata["encoding"] = encoding
            return text, metadata
        except UnicodeDecodeError:
            continue
    text = data.decode("latin-1")
    metadata["encoding"] = "latin-1"
    return text, metadata


def _marker_hits(
    value: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for kind, pattern in patterns:
        for match in pattern.finditer(value):
            hits.append(
                {
                    "kind": kind,
                    "value": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return hits


def marker_hits(value: str) -> list[dict[str, Any]]:
    """Return structural and contextual occurrences for the raw inventory."""
    return _marker_hits(value, MARKER_PATTERNS)


def structural_marker_hits(value: str) -> list[dict[str, Any]]:
    """Return only unequivocal implementation/version markers."""
    return _marker_hits(value, STRUCTURAL_MARKER_PATTERNS)


def contextual_marker_hits(value: str) -> list[dict[str, Any]]:
    return _marker_hits(value, CONTEXTUAL_MARKER_PATTERNS)


def has_markers(value: str) -> bool:
    return bool(marker_hits(value))


def has_structural_markers(value: str) -> bool:
    return bool(structural_marker_hits(value))


def is_dunder_name(name: str) -> bool:
    return bool(DUNDER_NAME_RE.fullmatch(name))


def suggestion_is_semantically_safe(name: str | None) -> bool:
    if not name:
        return False
    stem = PurePosixPath(name).stem if "/" in name or name.endswith(".py") else name
    if UNSAFE_SUGGESTION_TOKEN_RE.search(stem):
        return False
    if OPAQUE_STAGE_DISPATCH_RE.fullmatch(stem):
        return False
    if OPAQUE_NUMERIC_SUFFIX_RE.search(stem):
        return False
    return True


def is_snake_case(name: str) -> bool:
    return bool(re.fullmatch(r"_?[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name))



def is_pascal_case(name: str) -> bool:
    # Classes privadas continuam canônicas: `_ClassName` não é API pública.
    return bool(re.fullmatch(r"_?[A-Z][A-Za-z0-9]*", name))



def is_constant_case(name: str) -> bool:
    return bool(re.fullmatch(r"_?[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*", name))


def module_name_for(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    else:
        targets.append(node.target)
    names: list[str] = []
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.append(child.id)
    return names


def import_records(
    tree: ast.AST,
    *,
    relative: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(
                    {
                        "path": relative.as_posix(),
                        "line": node.lineno,
                        "form": "import",
                        "module": alias.name,
                        "name": None,
                        "asname": alias.asname,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                records.append(
                    {
                        "path": relative.as_posix(),
                        "line": node.lineno,
                        "form": "from",
                        "level": node.level,
                        "module": node.module or "",
                        "name": alias.name,
                        "asname": alias.asname,
                    }
                )
    return records


def all_exports(tree: ast.Module) -> set[str]:
    exports: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        names = assigned_names(node)
        if "__all__" not in names:
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for item in value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    exports.add(item.value)
    return exports


def string_literals(tree: ast.AST) -> Counter[str]:
    values: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values[node.value] += 1
    return values


def is_alias_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    return isinstance(value, (ast.Name, ast.Attribute))


def ast_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def enclosing_class_name(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            return current.name
        current = parents.get(current)
    return None


def ast_scan(
    python_files: list[Path],
    *,
    software_root: Path,
) -> dict[str, Any]:
    modules: dict[str, dict[str, Any]] = {}
    imports: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    main_guards: list[dict[str, Any]] = []
    syntax_errors: list[dict[str, Any]] = []
    name_loads_by_file: dict[str, Counter[str]] = {}
    attributes_by_file: dict[str, Counter[str]] = {}
    strings_by_file: dict[str, Counter[str]] = {}

    for path in python_files:
        relative = path.relative_to(software_root)
        source, encoding = read_python_source(path)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(
                {
                    "path": relative.as_posix(),
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                }
            )
            continue

        parents = ast_parent_map(tree)
        module_name = module_name_for(relative)
        exports = all_exports(tree)
        module_imports = import_records(tree, relative=relative)
        imports.extend(module_imports)
        name_loads = Counter(
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        attributes = Counter(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        strings = string_literals(tree)
        name_loads_by_file[relative.as_posix()] = name_loads
        attributes_by_file[relative.as_posix()] = attributes
        strings_by_file[relative.as_posix()] = strings

        top_level_names: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top_level_names.append(node.name)
                definitions.append(
                    {
                        "kind": "function",
                        "name": node.name,
                        "path": relative.as_posix(),
                        "module": module_name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "top_level": True,
                        "owner": None,
                        "qualified_name": node.name,
                        "async": isinstance(node, ast.AsyncFunctionDef),
                        "exported": node.name in exports,
                        "decorators": [
                            ast.unparse(item) for item in node.decorator_list
                        ],
                    }
                )
            elif isinstance(node, ast.ClassDef):
                top_level_names.append(node.name)
                definitions.append(
                    {
                        "kind": "class",
                        "name": node.name,
                        "path": relative.as_posix(),
                        "module": module_name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "top_level": True,
                        "owner": None,
                        "qualified_name": node.name,
                        "exported": node.name in exports,
                        "decorators": [
                            ast.unparse(item) for item in node.decorator_list
                        ],
                    }
                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for name in assigned_names(node):
                    top_level_names.append(name)
                    if is_constant_case(name):
                        definitions.append(
                            {
                                "kind": "constant",
                                "name": name,
                                "path": relative.as_posix(),
                                "module": module_name,
                                "line": node.lineno,
                                "end_line": node.end_lineno,
                                "top_level": True,
                                "owner": None,
                                "qualified_name": name,
                                "exported": name in exports,
                            }
                        )
                    if is_alias_assignment(node):
                        aliases.append(
                            {
                                "kind": "assignment",
                                "name": name,
                                "path": relative.as_posix(),
                                "module": module_name,
                                "line": node.lineno,
                                "target": ast.unparse(node.value),
                            }
                        )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node in tree.body:
                    continue
                owner = enclosing_class_name(node, parents)
                definitions.append(
                    {
                        "kind": "function",
                        "name": node.name,
                        "path": relative.as_posix(),
                        "module": module_name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "top_level": False,
                        "owner": owner,
                        "qualified_name": f"{owner}.{node.name}" if owner else node.name,
                        "async": isinstance(node, ast.AsyncFunctionDef),
                        "exported": False,
                        "decorators": [
                            ast.unparse(item) for item in node.decorator_list
                        ],
                    }
                )
            elif isinstance(node, ast.ClassDef) and node not in tree.body:
                owner = enclosing_class_name(node, parents)
                definitions.append(
                    {
                        "kind": "class",
                        "name": node.name,
                        "path": relative.as_posix(),
                        "module": module_name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "top_level": False,
                        "owner": owner,
                        "qualified_name": f"{owner}.{node.name}" if owner else node.name,
                        "exported": False,
                        "decorators": [
                            ast.unparse(item) for item in node.decorator_list
                        ],
                    }
                )

        guards = direct_guard_calls(tree)
        if guards:
            main_guards.append(
                {
                    "path": relative.as_posix(),
                    "module": module_name,
                    "calls": guards,
                }
            )

        modules[module_name] = {
            "path": relative.as_posix(),
            "encoding": encoding,
            "sha256": sha256_path(path),
            "lines": len(source.splitlines()),
            "top_level_names": sorted(set(top_level_names)),
            "exports": sorted(exports),
            "import_count": len(module_imports),
            "main_guard_calls": guards,
        }

    return {
        "modules": modules,
        "imports": imports,
        "definitions": definitions,
        "aliases": aliases,
        "main_guards": main_guards,
        "syntax_errors": syntax_errors,
        "name_loads_by_file": name_loads_by_file,
        "attributes_by_file": attributes_by_file,
        "strings_by_file": strings_by_file,
    }


def detect_console_entrypoints(
    *,
    software_root: Path,
    tracked: list[Path],
    ast_data: dict[str, Any],
) -> list[dict[str, Any]]:
    entrypoints: list[dict[str, Any]] = []

    for item in ast_data["main_guards"]:
        entrypoints.append(
            {
                "kind": "python-main-guard",
                "name": item["module"],
                "target": item["calls"],
                "path": item["path"],
            }
        )

    package_main = software_root / PACKAGE_MAIN_REL
    if package_main in tracked:
        entrypoints.append(
            {
                "kind": "package-main-module",
                "name": "python -m academic_pipeline",
                "target": "academic_pipeline.__main__",
                "path": PACKAGE_MAIN_REL.as_posix(),
            }
        )

    pyproject = software_root / "pyproject.toml"
    if pyproject in tracked and tomllib is not None:
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except Exception as exc:
            entrypoints.append(
                {
                    "kind": "parse-error",
                    "name": "pyproject.toml",
                    "target": str(exc),
                    "path": "pyproject.toml",
                }
            )
        else:
            scripts = (data.get("project") or {}).get("scripts") or {}
            for name, target in sorted(scripts.items()):
                entrypoints.append(
                    {
                        "kind": "project-script",
                        "name": name,
                        "target": target,
                        "path": "pyproject.toml",
                    }
                )
            poetry_scripts = (
                ((data.get("tool") or {}).get("poetry") or {}).get("scripts")
                or {}
            )
            for name, target in sorted(poetry_scripts.items()):
                entrypoints.append(
                    {
                        "kind": "poetry-script",
                        "name": name,
                        "target": target,
                        "path": "pyproject.toml",
                    }
                )

    setup_cfg = software_root / "setup.cfg"
    if setup_cfg in tracked:
        parser = configparser.ConfigParser()
        try:
            parser.read(setup_cfg, encoding="utf-8")
            if parser.has_section("options.entry_points"):
                for group, value in parser.items("options.entry_points"):
                    for line in value.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entrypoints.append(
                            {
                                "kind": f"setup-cfg-{group}",
                                "name": line.split("=", 1)[0].strip(),
                                "target": line.split("=", 1)[-1].strip(),
                                "path": "setup.cfg",
                            }
                        )
        except Exception as exc:
            entrypoints.append(
                {
                    "kind": "parse-error",
                    "name": "setup.cfg",
                    "target": str(exc),
                    "path": "setup.cfg",
                }
            )

    setup_py = software_root / "setup.py"
    if setup_py in tracked:
        source, _ = read_python_source(setup_py)
        for match in re.finditer(
            r"(?m)^\s*['\"](?P<name>[^'\"]+)['\"]\s*=\s*"
            r"['\"](?P<target>[^'\"]+)['\"]\s*,?\s*$",
            source,
        ):
            if ":" not in match.group("target"):
                continue
            entrypoints.append(
                {
                    "kind": "setup-py-possible-console-script",
                    "name": match.group("name"),
                    "target": match.group("target"),
                    "path": "setup.py",
                }
            )

    unique: dict[str, dict[str, Any]] = {}
    for item in entrypoints:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        unique[key] = item
    return sorted(
        unique.values(),
        key=lambda item: (
            str(item.get("kind")),
            str(item.get("name")),
            str(item.get("path")),
        ),
    )


def style_issues_for_definition(item: dict[str, Any]) -> list[str]:
    name = item["name"]
    kind = item["kind"]
    issues: list[str] = []
    if kind == "function" and not is_snake_case(name):
        issues.append("function-not-snake-case")
    elif kind == "class" and not is_pascal_case(name):
        issues.append("class-not-pascal-case")
    elif kind == "constant" and not is_constant_case(name):
        issues.append("constant-not-upper-snake-case")
    return issues



def normalize_identifier_suggestion(name: str, *, kind: str) -> str | None:
    """Produce only conservative suggestions from structural markers.

    v4.2 never treats domain words such as ``copy``, ``backup``, ``final``,
    ``new`` or ``old`` as defects. Dunder methods are outside the naming
    analysis. Suggestions that still contain ``original``, ``pre``, opaque
    stage/dispatch numbers or a numeric tail are rejected.
    """
    if is_dunder_name(name):
        return None
    if name in SPECIAL_SUGGESTIONS:
        suggestion = SPECIAL_SUGGESTIONS[name]
        return suggestion if suggestion_is_semantically_safe(suggestion) else None
    if OPAQUE_STAGE_DISPATCH_RE.fullmatch(name):
        return None
    if not has_structural_markers(name):
        return None
    if kind == "constant" and is_constant_case(name) and not has_structural_markers(name):
        return None
    if kind == "class" and is_pascal_case(name) and not has_structural_markers(name):
        return None

    private = "_" if name.startswith("_") and not name.startswith("__") else ""
    raw = name[1:] if private else name
    tokens = [token for token in raw.split("_") if token]
    kept: list[str] = []
    removed_marker = False
    skip_numeric_tail = False
    for token in tokens:
        if IMPLEMENTATION_MARKER_TOKEN_RE.fullmatch(token):
            removed_marker = True
            skip_numeric_tail = token.lower().startswith(("v", "rc"))
            continue
        if skip_numeric_tail and token.isdigit():
            continue
        skip_numeric_tail = False
        kept.append(token)

    if not removed_marker or not kept:
        return None

    rest = "_".join(kept)
    if kind == "class":
        proposed = private + "".join(piece[:1].upper() + piece[1:] for piece in kept)
    elif kind == "constant":
        proposed = private + rest.upper()
    else:
        proposed = private + rest.lower()

    if proposed == name or not suggestion_is_semantically_safe(proposed):
        return None
    return proposed



def normalize_dotted_suggestion(name: str) -> str | None:
    parts = name.split(".")
    normalized_parts: list[str] = []
    changed = False
    for part in parts:
        if not part:
            normalized_parts.append(part)
            continue
        replacement = SPECIAL_SUGGESTIONS.get(part)
        if replacement is None and not LEGACY_SEMANTIC_RE.search(part):
            replacement = normalize_identifier_suggestion(part, kind="module")
        if replacement and replacement != part:
            normalized_parts.append(replacement)
            changed = True
        else:
            normalized_parts.append(part)
    suggestion = ".".join(normalized_parts)
    return (
        suggestion
        if changed and suggestion != name and suggestion_is_semantically_safe(suggestion)
        else None
    )




def path_suggestion(relative: Path) -> str | None:
    path_text = relative.as_posix()
    if relative.suffix != ".py" or protected_path_reason(path_text):
        return None
    if relative.name in SPECIAL_SUGGESTIONS:
        candidate = str(relative.with_name(SPECIAL_SUGGESTIONS[relative.name])).replace("\\", "/")
        return candidate if suggestion_is_semantically_safe(candidate) else None
    normalized = normalize_identifier_suggestion(relative.stem, kind="module")
    if not normalized or normalized in {"test", "tests", "module", "file"}:
        return None
    candidate = str(relative.with_name(normalized + relative.suffix)).replace("\\", "/")
    return candidate if suggestion_is_semantically_safe(candidate) else None



def reference_evidence(
    name: str,
    *,
    defining_path: str | None,
    ast_data: dict[str, Any],
    text_occurrences: dict[str, Counter[str]],
) -> dict[str, Any]:
    python_files: set[str] = set()
    python_load_count = 0
    attribute_count = 0
    string_literal_count = 0
    string_files: set[str] = set()

    for path, counts in ast_data["name_loads_by_file"].items():
        count = counts.get(name, 0)
        if count:
            python_files.add(path)
            python_load_count += count
    for path, counts in ast_data["attributes_by_file"].items():
        count = counts.get(name, 0)
        if count:
            python_files.add(path)
            attribute_count += count
    for path, counts in ast_data["strings_by_file"].items():
        count = sum(
            amount
            for value, amount in counts.items()
            if value == name or name in value
        )
        if count:
            string_files.add(path)
            string_literal_count += count

    text_files: set[str] = set()
    text_count = 0
    for path, counts in text_occurrences.items():
        count = counts.get(name, 0)
        if count:
            text_files.add(path)
            text_count += count

    external_python_files = sorted(
        path for path in python_files if path != defining_path
    )
    return {
        "python_load_count": python_load_count,
        "attribute_count": attribute_count,
        "python_files": sorted(python_files),
        "external_python_files": external_python_files,
        "string_literal_count": string_literal_count,
        "string_literal_files": sorted(string_files),
        "text_reference_count": text_count,
        "text_reference_files": sorted(text_files),
        "test_reference_files": sorted(
            path
            for path in python_files | text_files
            if TEST_PATH_RE.search(path)
        ),
        "documentation_reference_files": sorted(
            path
            for path in text_files
            if DOC_PATH_RE.search(path)
        ),
    }



def classify_candidate(
    *,
    category: str,
    name: str,
    path: str,
    markers: list[dict[str, Any]],
    style_issues: list[str],
    references: dict[str, Any],
    suggestion: str | None,
    top_level: bool | None = None,
    exported: bool = False,
    entrypoint: bool = False,
) -> tuple[str, str, str]:
    protected_reason = protected_path_reason(path)
    if path == "." or name == SOFTWARE_DIRNAME:
        return (
            "nome que deve permanecer",
            "O diretório físico está explicitamente fora do escopo da AP-004 e reservado para a AP-006.",
            "AP-006",
        )
    if protected_reason:
        return (
            "nome que deve permanecer",
            f"O caminho é protegido: {protected_reason}.",
            "fora da AP-004",
        )
    xfail_reason = known_xfail_protection(name, path)
    if xfail_reason:
        return (
            "nome que deve permanecer",
            f"{xfail_reason.capitalize()}; renomeá-lo ampliaria o escopo da AP-004.",
            "fora da AP-004",
        )
    if legacy_is_semantic(name, path):
        return (
            "nome que deve permanecer",
            "`legacy` identifica uma camada real de compatibilidade; sua necessidade será revista na AP-004E, sem remoção automática.",
            "AP-004E (revisão de compatibilidade)",
        )
    if OPAQUE_STAGE_DISPATCH_RE.fullmatch(name):
        return (
            "renomeação de alto risco",
            "O identificador é opaco. Remover apenas o marcador AP produziria outro nome não semântico; requer leitura do corpo e nome explícito.",
            "AP-004C/AP-004D (revisão manual)",
        )

    if name == AP003_CORE_NAME and path == ORCHESTRATOR_REL.as_posix():
        return (
            "renomeação de alto risco",
            "O núcleo interno foi congelado pela AP-003. Sua normalização é uma decisão de símbolo, dependente de contratos AST, sem pertencer à movimentação de módulos da AP-004B.",
            "AP-004C/AP-004D",
        )

    external = bool(references["external_python_files"])
    dynamic = references["string_literal_count"] > 0
    public = exported or (top_level is True and not name.startswith("_"))

    if entrypoint or category == "entrypoint":
        if has_markers(name) or has_markers(path):
            return (
                "renomeação com compatibilidade",
                "A superfície é pública/operacional e só pode migrar mantendo wrapper ou alias transitório.",
                "AP-004B/AP-004E",
            )
        return (
            "nome que deve permanecer",
            "O entrypoint é contrato público estável e não contém marcador inadequado.",
            "contrato público",
        )

    if category == "import":
        if "academic_pipeline_rc10" in name:
            return (
                "renomeação com compatibilidade",
                "O import consome o orquestrador canônico; deve acompanhar um módulo-ponte na AP-004B.",
                "AP-004B/AP-004E",
            )
        if legacy_is_semantic(name, path):
            return (
                "nome que deve permanecer",
                "O import explicita a camada de compatibilidade histórica.",
                "AP-004E (revisão de compatibilidade)",
            )
        return (
            "renomeação de alto risco",
            "Imports são consumidores, não definições; a mudança depende da decisão do módulo ou símbolo de origem.",
            "AP-004B/AP-004C (revisão manual)",
        )

    if not suggestion:
        return (
            "renomeação de alto risco",
            "Não existe sugestão semântica automática segura. A decisão exige nome explícito baseado na responsabilidade real.",
            "revisão manual",
        )

    if dynamic and (external or public or category == "arquivo/módulo"):
        return (
            "renomeação de alto risco",
            "Há referências dinâmicas e alcance externo; a migração exige prova específica e plano de compatibilidade.",
            "AP-004B/AP-004C/AP-004D",
        )
    if category == "arquivo/módulo":
        if external or references["text_reference_count"]:
            return (
                "renomeação com compatibilidade",
                "O módulo possui consumidores identificados e requer módulo-ponte ou wrapper.",
                "AP-004B/AP-004E",
            )
        return (
            "renomeação segura",
            "Arquivo Python produtivo interno, com sugestão conservadora e sem consumidores externos detectados.",
            "AP-004B",
        )
    if public or external or exported:
        return (
            "renomeação com compatibilidade",
            "O símbolo é público, exportado ou usado por outro módulo; a migração precisa preservar compatibilidade.",
            "AP-004C/AP-004E",
        )
    if dynamic:
        return (
            "renomeação de alto risco",
            "O nome aparece em string literal e não pode ser considerado seguro apenas pela AST.",
            "AP-004C/AP-004D",
        )
    if name.startswith("_") or top_level is False:
        return (
            "renomeação segura",
            "Símbolo privado/local, com sugestão conservadora e sem referências externas detectadas.",
            "AP-004C/AP-004D",
        )
    if style_issues:
        return (
            "renomeação segura",
            "Inconsistência estritamente estilística e sem consumidor externo detectado.",
            "AP-004C",
        )
    return (
        "renomeação de alto risco",
        "A evidência estática não basta para assegurar uma migração sem impacto.",
        "revisão manual",
    )




def candidate_record(
    *,
    category: str,
    name: str,
    path: str,
    line: int | None,
    markers: list[dict[str, Any]],
    style_issues: list[str],
    references: dict[str, Any],
    suggestion: str | None,
    top_level: bool | None = None,
    exported: bool = False,
    entrypoint: bool = False,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    classification, reason, target_phase = classify_candidate(
        category=category,
        name=name,
        path=path,
        markers=markers,
        style_issues=style_issues,
        references=references,
        suggestion=suggestion,
        top_level=top_level,
        exported=exported,
        entrypoint=entrypoint,
    )
    if classification == "nome que deve permanecer":
        suggestion = None
    return {
        "id": stable_id(category, path, line or 0, name),
        "category": category,
        "current_name": name,
        "suggested_name": suggestion,
        "path": path,
        "line": line,
        "markers": markers,
        "style_issues": style_issues,
        "classification": classification,
        "classification_reason": reason,
        "target_phase": target_phase,
        "top_level": top_level,
        "exported": exported,
        "entrypoint": entrypoint,
        "references": references,
        "evidence": evidence or [],
        "status": "candidate-only-no-change",
    }



def empty_references() -> dict[str, Any]:
    return {
        "python_load_count": 0,
        "attribute_count": 0,
        "python_files": [],
        "external_python_files": [],
        "string_literal_count": 0,
        "string_literal_files": [],
        "text_reference_count": 0,
        "text_reference_files": [],
        "test_reference_files": [],
        "documentation_reference_files": [],
    }


def occurrence_counter_for_names(
    text_files: dict[str, str],
    names: Iterable[str],
) -> dict[str, Counter[str]]:
    unique_names = sorted(
        {name for name in names if len(name) >= 3},
        key=len,
        reverse=True,
    )
    results: dict[str, Counter[str]] = {}
    for path, text in text_files.items():
        counter: Counter[str] = Counter()
        for name in unique_names:
            count = text.count(name)
            if count:
                counter[name] = count
        results[path] = counter
    return results


def line_hits_for_markers(text: str, *, limit: int = 20) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        markers = marker_hits(line)
        if not markers:
            continue
        hits.append(
            {
                "line": number,
                "markers": markers,
                "excerpt": line.strip()[:240],
            }
        )
        if len(hits) >= limit:
            break
    return hits



def build_inventory_data(
    *,
    software_root: Path,
    repository_root: Path,
    git_state: dict[str, Any],
    ap003g: dict[str, Any],
    architecture: dict[str, Any],
    tool_source: str,
) -> dict[str, Any]:
    tracked = tracked_files(
        software_root=software_root,
        repository_root=repository_root,
    )
    python_files = [path for path in tracked if path.suffix == ".py"]
    ast_data = ast_scan(python_files, software_root=software_root)
    if ast_data["syntax_errors"]:
        fail(
            "Há arquivos Python rastreados que não puderam ser analisados por AST:\n"
            + json.dumps(ast_data["syntax_errors"], ensure_ascii=False, indent=2)
        )

    text_files: dict[str, str] = {}
    file_metadata: dict[str, dict[str, Any]] = {}
    skipped_text: list[dict[str, Any]] = []
    for path in tracked:
        relative = path.relative_to(software_root)
        suffix = path.suffix.lower()
        is_known_text = (
            suffix in TEXT_SUFFIXES
            or path.name in {"Pipfile", "Dockerfile", "Makefile", "LICENSE"}
        )
        if not is_known_text:
            continue
        text_value, metadata = read_text_file(path)
        file_metadata[relative.as_posix()] = metadata
        if text_value is None:
            skipped_text.append(
                {
                    "path": relative.as_posix(),
                    "reason": metadata["skipped"],
                    "bytes": metadata["bytes"],
                }
            )
            continue
        text_files[relative.as_posix()] = text_value

    names_for_reference = {item["name"] for item in ast_data["definitions"]}
    names_for_reference |= {item["name"] for item in ast_data["aliases"]}
    names_for_reference.update(
        path.stem for path in tracked if path.suffix == ".py"
    )
    text_occurrences = occurrence_counter_for_names(
        text_files,
        names_for_reference,
    )

    entrypoints = detect_console_entrypoints(
        software_root=software_root,
        tracked=tracked,
        ast_data=ast_data,
    )
    entrypoint_targets = {
        str(item.get("target", "")) for item in entrypoints
    }

    raw_occurrences: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    protected_operational_names: list[dict[str, Any]] = []
    historical_references: list[dict[str, Any]] = []

    def add_raw(
        *,
        surface: str,
        name: str,
        path: str,
        line: int | None,
        markers: list[dict[str, Any]],
        style_issues: list[str] | None = None,
        evidence: list[str] | None = None,
        disposition: str,
    ) -> None:
        raw_occurrences.append(
            {
                "id": stable_id("raw", surface, path, line or 0, name),
                "surface": surface,
                "name": name,
                "path": path,
                "line": line,
                "markers": markers,
                "style_issues": style_issues or [],
                "evidence": evidence or [],
                "disposition": disposition,
            }
        )

    # Limite físico explicitamente preservado até a AP-006.
    add_raw(
        surface="arquivo/módulo",
        name=SOFTWARE_DIRNAME,
        path=".",
        line=None,
        markers=marker_hits(SOFTWARE_DIRNAME),
        evidence=["limite explícito da AP-004"],
        disposition="protected-physical-directory",
    )
    candidates.append(
        candidate_record(
            category="arquivo/módulo",
            name=SOFTWARE_DIRNAME,
            path=".",
            line=None,
            markers=marker_hits(SOFTWARE_DIRNAME),
            style_issues=[],
            references=empty_references(),
            suggestion=None,
            evidence=["limite explícito da AP-004"],
        )
    )

    protected_python_paths = {
        path.relative_to(software_root).as_posix()
        for path in python_files
        if protected_path_reason(
            path.relative_to(software_root).as_posix()
        )
    }

    # Arquivos e módulos. Somente marcadores estruturais criam candidato.
    for path in tracked:
        relative = path.relative_to(software_root)
        relative_text = relative.as_posix()
        markers = marker_hits(relative.name)
        structural_markers = structural_marker_hits(relative.name)
        module_style_issue = (
            path.suffix == ".py"
            and relative.stem not in {"__init__", "__main__"}
            and not is_snake_case(relative.stem)
        )
        protected_reason = protected_path_reason(relative_text)

        if not markers and not module_style_issue and not protected_reason:
            continue

        if protected_reason:
            historical_only = (
                protected_reason.startswith("documentação auditável")
                or protected_reason.startswith("ferramenta auditável")
            )
            record = {
                "id": stable_id("protected", relative_text),
                "name": relative.name,
                "path": relative_text,
                "reason": protected_reason,
                "markers": markers,
            }
            if historical_only:
                historical_references.append(
                    {
                        "id": stable_id("historical-path", relative_text),
                        "surface": (
                            "documentação"
                            if relative_text.startswith("docs/")
                            else "histórico"
                        ),
                        "path": relative_text,
                        "hit_count": 1,
                        "hits": [
                            {
                                "line": None,
                                "markers": markers,
                                "excerpt": relative.name,
                            }
                        ],
                        "role": "auditable-phase-history",
                    }
                )
            else:
                protected_operational_names.append(record)
            add_raw(
                surface="histórico" if historical_only else "operacional",
                name=relative.name,
                path=relative_text,
                line=None,
                markers=markers,
                style_issues=(
                    ["module-filename-not-snake-case"]
                    if module_style_issue
                    else []
                ),
                evidence=[protected_reason],
                disposition=(
                    "historical-evidence"
                    if historical_only
                    else "protected-operational-name"
                ),
            )
            continue

        if is_test_or_documentation_path(relative_text):
            add_raw(
                surface=(
                    "teste"
                    if TEST_PATH_RE.search(relative_text)
                    else "documentação"
                ),
                name=relative.name,
                path=relative_text,
                line=None,
                markers=markers,
                style_issues=(
                    ["module-filename-not-snake-case"]
                    if module_style_issue
                    else []
                ),
                evidence=["nome de arquivo em evidência histórica/contratual"],
                disposition="evidence-only",
            )
            continue

        if path.suffix != ".py":
            add_raw(
                surface="operacional",
                name=relative.name,
                path=relative_text,
                line=None,
                markers=markers,
                evidence=["arquivo não Python; nenhuma renomeação automática"],
                disposition="manual-non-python-review",
            )
            protected_operational_names.append(
                {
                    "id": stable_id("protected-non-python", relative_text),
                    "name": relative.name,
                    "path": relative_text,
                    "reason": (
                        "arquivo não Python fora da matriz produtiva automática"
                    ),
                    "markers": markers,
                }
            )
            continue

        if not structural_markers:
            add_raw(
                surface="arquivo/módulo",
                name=relative.name,
                path=relative_text,
                line=None,
                markers=markers,
                style_issues=(
                    ["module-filename-not-snake-case"]
                    if module_style_issue
                    else []
                ),
                evidence=[
                    "marcador contextual ou questão estilística sem gatilho "
                    "estrutural inequívoco"
                ],
                disposition="contextual-evidence-only",
            )
            continue

        references = reference_evidence(
            relative.stem,
            defining_path=relative_text,
            ast_data=ast_data,
            text_occurrences=text_occurrences,
        )
        module_name = module_name_for(relative)
        related_entrypoints = [
            item
            for item in entrypoints
            if (
                relative_text == str(item.get("path", ""))
                or relative.stem in str(item.get("target", ""))
                or module_name == str(item.get("name", ""))
            )
        ]
        add_raw(
            surface="arquivo/módulo",
            name=relative.name,
            path=relative_text,
            line=None,
            markers=markers,
            style_issues=(
                ["module-filename-not-snake-case"]
                if module_style_issue
                else []
            ),
            evidence=["tracked Python path with structural marker"],
            disposition="actionable-candidate",
        )
        record = candidate_record(
            category="arquivo/módulo",
            name=relative.name,
            path=relative_text,
            line=None,
            markers=structural_markers,
            style_issues=(
                ["module-filename-not-snake-case"]
                if module_style_issue
                else []
            ),
            references=references,
            suggestion=path_suggestion(relative),
            entrypoint=bool(related_entrypoints),
            evidence=["tracked Python path with structural marker"],
        )
        record["related_surfaces"] = related_entrypoints
        candidates.append(record)

    # Definições AST. Dunder e símbolos de arquivos protegidos são excluídos.
    definition_seen: set[tuple[str, int, str]] = set()
    for item in ast_data["definitions"]:
        if (
            item["path"] in protected_python_paths
            or is_dunder_name(item["name"])
        ):
            continue
        markers = marker_hits(item["name"])
        structural_markers = structural_marker_hits(item["name"])
        style_issues = style_issues_for_definition(item)
        if not markers and not style_issues:
            continue
        key = (item["path"], item["line"], item["name"])
        if key in definition_seen:
            continue
        definition_seen.add(key)
        surface = {
            "function": "função",
            "class": "classe",
            "constant": "constante",
        }[item["kind"]]

        if is_test_or_documentation_path(item["path"]):
            add_raw(
                surface=(
                    "teste"
                    if TEST_PATH_RE.search(item["path"])
                    else "documentação"
                ),
                name=item["name"],
                path=item["path"],
                line=item["line"],
                markers=markers,
                style_issues=style_issues,
                evidence=[f"AST {item['kind']} definition"],
                disposition="evidence-only",
            )
            continue

        if not structural_markers:
            add_raw(
                surface=surface,
                name=item["name"],
                path=item["path"],
                line=item["line"],
                markers=markers,
                style_issues=style_issues,
                evidence=[f"AST {item['kind']} definition"],
                disposition="contextual-evidence-only",
            )
            continue

        references = reference_evidence(
            item["name"],
            defining_path=item["path"],
            ast_data=ast_data,
            text_occurrences=text_occurrences,
        )
        target_notation = f"{item['module']}:{item['name']}"
        is_entrypoint = (
            target_notation in entrypoint_targets
            or any(
                item["name"] in target and item["module"] in target
                for target in entrypoint_targets
            )
        )
        add_raw(
            surface=surface,
            name=item["name"],
            path=item["path"],
            line=item["line"],
            markers=markers,
            style_issues=style_issues,
            evidence=[f"AST {item['kind']} definition with structural marker"],
            disposition="actionable-candidate",
        )
        record = candidate_record(
            category=surface,
            name=item["name"],
            path=item["path"],
            line=item["line"],
            markers=structural_markers,
            style_issues=style_issues,
            references=references,
            suggestion=normalize_identifier_suggestion(
                item["name"],
                kind=item["kind"],
            ),
            top_level=item["top_level"],
            exported=item.get("exported", False),
            entrypoint=is_entrypoint,
            evidence=[f"AST {item['kind']} definition with structural marker"],
        )
        record["related_surfaces"] = [
            ep
            for ep in entrypoints
            if (
                item["path"] == str(ep.get("path", ""))
                and item["name"] in str(ep.get("target", ""))
            )
        ]
        candidates.append(record)

    # Imports são consumidores; nunca duplicam a decisão acionável da origem.
    for item in ast_data["imports"]:
        path = item["path"]
        if path in protected_python_paths:
            continue
        import_name = ".".join(
            part
            for part in (item.get("module"), item.get("name"))
            if part
        )
        alias_name = item.get("asname")
        import_markers = marker_hits(import_name)
        alias_markers = marker_hits(alias_name or "")

        if import_markers:
            evidence_only = is_test_or_documentation_path(path)
            add_raw(
                surface=(
                    "teste"
                    if evidence_only and TEST_PATH_RE.search(path)
                    else "documentação"
                    if evidence_only
                    else "import"
                ),
                name=import_name,
                path=path,
                line=item["line"],
                markers=import_markers,
                evidence=[item["form"], "AST import consumer"],
                disposition=(
                    "evidence-only"
                    if evidence_only
                    else "consumer-evidence-only"
                ),
            )

        if alias_name and alias_markers and not is_dunder_name(alias_name):
            evidence_only = is_test_or_documentation_path(path)
            structural_alias_markers = structural_marker_hits(alias_name)
            add_raw(
                surface=(
                    "teste"
                    if evidence_only and TEST_PATH_RE.search(path)
                    else "documentação"
                    if evidence_only
                    else "alias"
                ),
                name=alias_name,
                path=path,
                line=item["line"],
                markers=alias_markers,
                evidence=[f"alias de import: {import_name}"],
                disposition=(
                    "evidence-only"
                    if evidence_only
                    else "actionable-candidate"
                    if structural_alias_markers
                    else "contextual-evidence-only"
                ),
            )
            if evidence_only or not structural_alias_markers:
                continue
            references = reference_evidence(
                alias_name,
                defining_path=path,
                ast_data=ast_data,
                text_occurrences=text_occurrences,
            )
            candidates.append(
                candidate_record(
                    category="alias",
                    name=alias_name,
                    path=path,
                    line=item["line"],
                    markers=structural_alias_markers,
                    style_issues=[],
                    references=references,
                    suggestion=normalize_identifier_suggestion(
                        alias_name,
                        kind="alias",
                    ),
                    evidence=[
                        f"alias de import com marcador estrutural: {import_name}"
                    ],
                )
            )

    # Aliases por atribuição. Marcador no alvo é apenas evidência de consumo.
    for item in ast_data["aliases"]:
        if (
            item["path"] in protected_python_paths
            or is_dunder_name(item["name"])
        ):
            continue
        markers = marker_hits(item["name"])
        target_markers = marker_hits(item["target"])
        if not markers and not target_markers:
            continue
        structural_markers = structural_marker_hits(item["name"])
        evidence_only = is_test_or_documentation_path(item["path"])
        add_raw(
            surface=(
                "teste"
                if evidence_only and TEST_PATH_RE.search(item["path"])
                else "documentação"
                if evidence_only
                else "alias"
            ),
            name=item["name"],
            path=item["path"],
            line=item["line"],
            markers=markers + target_markers,
            evidence=[f"AST assignment alias -> {item['target']}"],
            disposition=(
                "evidence-only"
                if evidence_only
                else "actionable-candidate"
                if structural_markers
                else "consumer-evidence-only"
                if target_markers
                else "contextual-evidence-only"
            ),
        )
        if evidence_only or not structural_markers:
            continue
        references = reference_evidence(
            item["name"],
            defining_path=item["path"],
            ast_data=ast_data,
            text_occurrences=text_occurrences,
        )
        candidates.append(
            candidate_record(
                category="alias",
                name=item["name"],
                path=item["path"],
                line=item["line"],
                markers=structural_markers,
                style_issues=[],
                references=references,
                suggestion=normalize_identifier_suggestion(
                    item["name"],
                    kind="alias",
                ),
                evidence=[f"AST assignment alias -> {item['target']}"],
            )
        )

    # Entrypoints são superfícies relacionadas, não candidatos independentes.
    for entrypoint in entrypoints:
        name = str(entrypoint.get("name", ""))
        path = str(entrypoint.get("path", ""))
        target = str(entrypoint.get("target", ""))
        add_raw(
            surface="entrypoint",
            name=name,
            path=path,
            line=None,
            markers=(
                marker_hits(name)
                + marker_hits(target)
                + marker_hits(path)
            ),
            evidence=[f"{entrypoint['kind']} -> {target}"],
            disposition="related-surface-only",
        )

    # Testes e documentação permanecem evidência ampla.
    for path, text_value in text_files.items():
        if not is_test_or_documentation_path(path):
            continue
        hits = line_hits_for_markers(text_value, limit=200)
        if not hits:
            continue
        category = "teste" if TEST_PATH_RE.search(path) else "documentação"
        historical_references.append(
            {
                "id": stable_id("historical", path),
                "surface": category,
                "path": path,
                "hit_count": len(hits),
                "hits": hits,
                "role": "consumer-or-historical-evidence",
            }
        )
        for hit in hits:
            add_raw(
                surface=category,
                name=", ".join(
                    sorted(
                        {marker["value"] for marker in hit["markers"]},
                        key=str.lower,
                    )
                ),
                path=path,
                line=hit["line"],
                markers=hit["markers"],
                evidence=[hit["excerpt"]],
                disposition="evidence-only",
            )

    # Decisões explícitas de preservação da camada legacy.
    legacy_path = "academic_pipeline/legacy.py"
    if "academic_pipeline.legacy" in ast_data["modules"]:
        candidates.append(
            candidate_record(
                category="arquivo/módulo",
                name="legacy.py",
                path=legacy_path,
                line=None,
                markers=marker_hits("legacy.py"),
                style_issues=[],
                references=reference_evidence(
                    "legacy",
                    defining_path=legacy_path,
                    ast_data=ast_data,
                    text_occurrences=text_occurrences,
                ),
                suggestion=None,
                evidence=["camada explícita de compatibilidade histórica"],
            )
        )
    for item in ast_data["definitions"]:
        if (
            item["path"] != legacy_path
            or "legacy" not in item["name"].lower()
        ):
            continue
        candidates.append(
            candidate_record(
                category={
                    "function": "função",
                    "class": "classe",
                    "constant": "constante",
                }[item["kind"]],
                name=item["name"],
                path=item["path"],
                line=item["line"],
                markers=marker_hits(item["name"]),
                style_issues=[],
                references=reference_evidence(
                    item["name"],
                    defining_path=item["path"],
                    ast_data=ast_data,
                    text_occurrences=text_occurrences,
                ),
                suggestion=None,
                top_level=item["top_level"],
                exported=item.get("exported", False),
                evidence=[
                    "símbolo da camada explícita de compatibilidade legacy"
                ],
            )
        )

    # Três xfails históricos, vinculados a definições produtivas exatas.
    for target in KNOWN_XFAIL_DEFINITIONS:
        matches = [
            item
            for item in ast_data["definitions"]
            if item["path"] == target["path"]
            and item.get("qualified_name", item["name"])
            == target["qualified_name"]
        ]
        if len(matches) != 1:
            fail(
                "Definição produtiva esperada para xfail não é única: "
                f"{target['qualified_name']} em {target['path']} "
                f"(encontradas: {len(matches)})."
            )
        item = matches[0]
        candidates.append(
            candidate_record(
                category={
                    "function": "função",
                    "class": "classe",
                    "constant": "constante",
                }[item["kind"]],
                name=target["qualified_name"],
                path=item["path"],
                line=item["line"],
                markers=marker_hits(target["qualified_name"]),
                style_issues=[],
                references=reference_evidence(
                    item["name"],
                    defining_path=item["path"],
                    ast_data=ast_data,
                    text_occurrences=text_occurrences,
                ),
                suggestion=None,
                top_level=item["top_level"],
                exported=item.get("exported", False),
                evidence=["xfail histórico congelado em definição produtiva exata"],
            )
        )

    # Alias produtivo diretamente acoplado ao primeiro xfail.
    # No orquestrador real ele é materializado por ``from ... import ... as ...``;
    # portanto, deve ser validado no inventário de imports, não no de atribuições.
    for target in KNOWN_XFAIL_LINKED_ALIASES:
        matches = [
            item
            for item in ast_data["imports"]
            if item["path"] == target["path"]
            and item.get("form") == target["form"]
            and item.get("module") == target["module"]
            and item.get("name") == target["imported_name"]
            and item.get("asname") == target["name"]
        ]
        if len(matches) != 1:
            fail(
                "Vínculo de import esperado para o alias do xfail não é único: "
                f"{target['module']}.{target['imported_name']} as "
                f"{target['name']} em {target['path']} "
                f"(encontrados: {len(matches)})."
            )
        item = matches[0]
        candidates.append(
            candidate_record(
                category="alias",
                name=target["name"],
                path=item["path"],
                line=item["line"],
                markers=marker_hits(target["name"]),
                style_issues=[],
                references=reference_evidence(
                    target["name"],
                    defining_path=item["path"],
                    ast_data=ast_data,
                    text_occurrences=text_occurrences,
                ),
                suggestion=None,
                evidence=[
                    "alias de import produtivo diretamente ligado a xfail histórico congelado",
                    f"{target['module']}.{target['imported_name']} as {target['name']}",
                ],
            )
        )

    def deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for item in items:
            key = item["id"]
            existing = unique.get(key)
            if existing is None:
                unique[key] = item
                continue
            for list_key in ("evidence", "markers", "style_issues"):
                if list_key not in item or list_key not in existing:
                    continue
                existing[list_key] += [
                    value
                    for value in item[list_key]
                    if value not in existing[list_key]
                ]
            if item.get("related_surfaces"):
                existing.setdefault("related_surfaces", [])
                for surface in item["related_surfaces"]:
                    if surface not in existing["related_surfaces"]:
                        existing["related_surfaces"].append(surface)
        return list(unique.values())

    candidates = deduplicate(candidates)
    raw_occurrences = deduplicate(raw_occurrences)
    protected_operational_names = deduplicate(protected_operational_names)
    historical_references = deduplicate(historical_references)

    # Colisões de destino suspendem a sugestão.
    destination_groups: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for candidate in candidates:
        suggestion = candidate.get("suggested_name")
        if not suggestion:
            continue
        if candidate["category"] == "arquivo/módulo":
            key = ("file", suggestion)
        else:
            key = (candidate["path"], suggestion)
        destination_groups[key].append(candidate)

    destination_collisions: list[dict[str, Any]] = []
    for (scope, destination), group in sorted(destination_groups.items()):
        sources = {
            (item["path"], item["line"], item["current_name"])
            for item in group
        }
        if len(sources) < 2:
            continue
        collision_id = stable_id("collision", scope, destination)
        destination_collisions.append(
            {
                "id": collision_id,
                "scope": scope,
                "proposed_destination": destination,
                "candidate_ids": sorted(item["id"] for item in group),
                "sources": [
                    {
                        "path": path,
                        "line": line,
                        "current_name": name,
                    }
                    for path, line, name in sorted(
                        sources,
                        key=lambda value: (
                            value[0],
                            value[1] or 0,
                            value[2],
                        ),
                    )
                ],
                "status": "suggestion-suspended",
            }
        )
        for item in group:
            item["suspended_suggestion"] = item["suggested_name"]
            item["suggested_name"] = None
            item["classification"] = "renomeação de alto risco"
            item["classification_reason"] = (
                "A sugestão foi suspensa porque múltiplas origens convergem "
                "para o mesmo destino. É necessário comparar conteúdo, "
                "consumidores e papel operacional."
            )
            item["target_phase"] = (
                "revisão manual antes da AP-004B/AP-004C"
            )
            item["collision_id"] = collision_id

    candidates.sort(
        key=lambda item: (
            CLASSIFICATIONS.index(item["classification"]),
            CATEGORIES.index(item["category"]),
            item["path"],
            item["line"] or 0,
            item["current_name"],
        )
    )
    raw_occurrences.sort(
        key=lambda item: (
            item["path"],
            item["line"] or 0,
            item["surface"],
            item["name"],
        )
    )
    protected_operational_names.sort(
        key=lambda item: (item["path"], item["name"])
    )
    historical_references.sort(key=lambda item: item["path"])

    structural_kinds = {
        kind for kind, _pattern in STRUCTURAL_MARKER_PATTERNS
    }
    contextual_review_occurrences = [
        item
        for item in raw_occurrences
        if any(
            marker["kind"] == "contextual_temporal_label"
            for marker in item["markers"]
        )
        and not any(
            marker["kind"] in structural_kinds
            for marker in item["markers"]
        )
    ]

    classification_counts = Counter(
        item["classification"] for item in candidates
    )
    category_counts = Counter(item["category"] for item in candidates)
    marker_counts = Counter(
        marker["kind"]
        for item in raw_occurrences
        for marker in item["markers"]
    )
    manual_review_required = sorted(
        item["id"]
        for item in candidates
        if item["classification"] == "renomeação de alto risco"
    )

    coverage = {
        surface: {
            "raw_occurrence_count": sum(
                1
                for item in raw_occurrences
                if item["surface"] == surface
            ),
            "actionable_candidate_count": sum(
                1
                for item in candidates
                if item["category"] == surface
            ),
            "status": "scanned",
        }
        for surface in RAW_SURFACES
    }

    return {
        "phase": PHASE,
        "mode": MODE,
        "inventory_schema_version": 4,
        "inventory_revision": "4.2",
        "generated_at_utc": utc_now(),
        "git": git_state,
        "ap003g_closure": ap003g,
        "ap003_architecture": architecture,
        "scope": {
            "repository_root": str(repository_root),
            "software_root": str(software_root),
            "tracked_files_scanned": len(tracked),
            "python_files_scanned": len(python_files),
            "text_files_scanned": len(text_files),
            "text_files_skipped": skipped_text,
            "source": "git ls-files at recorded HEAD",
            "productive_changes": [],
            "excluded_from_actionable_matrix": [
                "testes e documentação usados somente como evidência",
                "saídas de app_bundle/projetos/**/output* e execucoes_anteriores",
                "instaladores, scripts históricos de manutenção e assets",
                "símbolos internos de arquivos protegidos",
                "métodos especiais Python __dunder__",
                "imports consumidores como decisões duplicadas",
                "palavras contextuais final/novo/old/original/pre sem marcador estrutural",
                "arquivos não Python sem decisão manual explícita",
                "diretório físico academic_pipeline_rc10_7_conformidade",
                "correção dos três xfails históricos",
            ],
        },
        "canonical_convention": {
            "document": CONVENTION_REL.as_posix(),
            "semantic_names_only": True,
            "automatic_temporal_word_removal": False,
            "private_class_prefix_preserved": True,
            "upper_snake_constants_preserved": True,
            "legacy": (
                "preservado quando identifica compatibilidade real"
            ),
            "tests_and_docs": (
                "evidência, não candidatos por palavra isolada"
            ),
            "collisions": (
                "sugestão suspensa e classificação de alto risco"
            ),
            "actionable_triggers": [
                "rcNN",
                "vNN",
                "versionNN",
                "apNNN[x]",
            ],
            "dunder": (
                "excluído integralmente da análise de nomenclatura"
            ),
            "imports": (
                "inventariados como consumidores, sem decisão acionável duplicada"
            ),
            "entrypoints": (
                "relacionados ao candidato principal do módulo"
            ),
            "unsafe_suggestions": (
                "rejeitadas quando preservam original/pre/stage_N/dispatch_N "
                "ou cauda numérica opaca"
            ),
        },
        "coverage": coverage,
        "statistics": {
            "raw_occurrence_count": len(raw_occurrences),
            "actionable_candidate_count": len(candidates),
            "protected_operational_count": len(protected_operational_names),
            "historical_reference_count": len(historical_references),
            "contextual_review_occurrence_count": len(
                contextual_review_occurrences
            ),
            "destination_collision_count": len(destination_collisions),
            "manual_review_count": len(manual_review_required),
            "by_classification": {
                name: classification_counts.get(name, 0)
                for name in CLASSIFICATIONS
            },
            "by_category": {
                name: category_counts.get(name, 0)
                for name in CATEGORIES
            },
            "by_marker": dict(sorted(marker_counts.items())),
        },
        "entrypoints": entrypoints,
        "module_inventory": ast_data["modules"],
        "import_inventory": ast_data["imports"],
        "alias_inventory": ast_data["aliases"],
        "file_metadata": file_metadata,
        "raw_occurrences": raw_occurrences,
        "actionable_candidates": candidates,
        "candidates": candidates,
        "protected_operational_names": protected_operational_names,
        "historical_references": historical_references,
        "contextual_review_occurrences": contextual_review_occurrences,
        "destination_collisions": destination_collisions,
        "manual_review_required": manual_review_required,
        "protected_names": {
            "physical_directory": SOFTWARE_DIRNAME,
            "known_xfails": list(KNOWN_XFAIL_SYMBOLS),
            "historical_phase_records": (
                "AP-xxx em docs/testes/ferramentas"
            ),
            "public_entrypoints": [
                "academic-pipeline",
                "python -m academic_pipeline",
            ],
        },
        "tool": {
            "path": TOOL_REL.as_posix(),
            "sha256": sha256_bytes(tool_source.encode("utf-8")),
            "version": 4,
            "revision": "4.2",
        },
        "validation": {
            "py_compile": "pending",
            "git_diff_check": "pending",
            "specific_suite": {"status": "pending"},
            "consolidated_suite": {"status": "pending"},
        },
        "next_gate": {
            "next_subphase": "AP-004B",
            "blocked_until_user_approval": True,
            "required_inputs": [
                "aprovação da convenção canônica v4.2",
                "aprovação da matriz acionável restritiva",
                "seleção explícita dos candidatos da AP-004B",
            ],
        },
    }



def build_convention_document() -> str:
    return normalize_output(
        """
        # AP-004 — convenção canônica de nomes internos (v4.2)

        ## Finalidade

        Esta convenção disciplina a normalização de nomes sem alterar comportamento,
        superfícies de entrada, documentos gerados ou caminhos operacionais. O
        inventário v4.2 distingue marcadores estruturais inequívocos de palavras
        contextuais que podem representar conceitos legítimos do domínio.

        ## Gatilhos acionáveis automáticos

        Somente os seguintes marcadores podem criar automaticamente um candidato:

        - `rcNN` e equivalentes de release candidate;
        - `vNN`, `v1_18`, `v0_3_1` e equivalentes;
        - palavras explícitas de versão seguidas por número;
        - prefixos de refatoração `apNNN`, como `_ap003d_` e `_ap003f_`.

        Palavras como `final`, `novo`, `new`, `old`, `legacy`, `original` e `pre`
        são apenas evidência contextual. Elas não criam candidato por si mesmas.
        `copy` e `backup` são verbos legítimos e não são marcadores de nomenclatura.

        ## Regras canônicas

        1. **Módulos e arquivos Python produtivos** usam `snake_case` e nomes
           semânticos. A sugestão deve descrever responsabilidade real.
        2. **Funções e métodos** usam `snake_case` e preservam verbos de ação.
        3. **Métodos especiais `__dunder__`** ficam integralmente fora da análise.
        4. **Classes** usam `PascalCase`; classes privadas podem usar `_PascalCase`.
        5. **Constantes** usam `UPPER_SNAKE_CASE`.
        6. **`legacy`/`legado`** permanece quando identifica uma camada real de
           compatibilidade; sua necessidade será revista na AP-004E.
        7. **Aliases numéricos**, como `stage_001` e `dispatch_001`, são opacos e
           exigem nome semântico explícito.
        8. **Imports** são consumidores e não duplicam a decisão do módulo ou símbolo
           de origem.
        9. **Entrypoints** são superfícies relacionadas ao candidato principal do
           módulo; não aparecem como candidatos independentes.
        10. **Testes e documentação** registram consumidores, contratos e história.
        11. **Saídas operacionais, execuções anteriores, instaladores, assets e
            scripts históricos de aplicação/atualização/migração na raiz** ficam fora
            da AP-004.
        12. **Colisões de destino** suspendem a sugestão e elevam os envolvidos para
            alto risco.
        13. Sugestões que ainda contenham `original`, `pre`, `stage_N`, `dispatch_N`
            ou cauda numérica opaca são rejeitadas.

        ## Estruturas do inventário

        - `raw_occurrences`: levantamento amplo de ocorrências e evidências;
        - `actionable_candidates`: decisões possíveis sobre módulos e símbolos;
        - `contextual_review_occurrences`: palavras contextuais não acionáveis;
        - `protected_operational_names`: caminhos operacionais preservados;
        - `historical_references`: testes e documentação como evidência;
        - `destination_collisions`: destinos propostos por mais de uma origem;
        - `manual_review_required`: candidatos de alto risco.

        ## Critérios de classificação

        ### Renomeação segura

        Símbolo privado/local ou arquivo Python produtivo interno, com marcador
        estrutural inequívoco, sugestão semântica conservadora e sem consumidor
        externo, string dinâmica ou colisão.

        ### Renomeação com compatibilidade

        Módulo ou símbolo público, exportado, consumido externamente ou relacionado
        a entrypoint, cuja migração exige wrapper ou alias transitório documentado.

        ### Renomeação de alto risco

        Marcador estrutural presente, mas sem sugestão semântica segura, com alias
        opaco, referência dinâmica, colisão ou múltiplos consumidores não resolvidos.

        ### Nome que deve permanecer

        Diretório físico reservado à AP-006, xfail congelado, histórico auditável,
        script operacional protegido ou camada real de compatibilidade `legacy`.

        ## Limites

        O diretório `academic_pipeline_rc10_7_conformidade`, os três xfails, a
        semântica da CLI, instaladores, assets, caminhos de implantação e conteúdo
        documental gerado permanecem inalterados.

        Nenhuma renomeação será feita por substituição textual global. Aplicadores
        futuros deverão validar `HEAD`, hashes, AST, conjunto permitido de arquivos,
        escrita atômica e rollback integral.
        """
    )



def markdown_escape(value: object) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text



def build_report(
    inventory: dict[str, Any],
    *,
    validation_override: dict[str, Any] | None = None,
) -> str:
    validation = validation_override or inventory["validation"]
    git_state = inventory["git"]
    ap003g = inventory["ap003g_closure"]
    stats = inventory["statistics"]

    lines = [
        "# AP-004A — inventário e convenção canônica (v4.2)",
        "",
        "> Levantamento somente preparatório. Nenhum arquivo produtivo foi modificado.",
        "",
        "## Estado Git confirmado",
        "",
        f"- Branch: `{git_state['branch']}`.",
        f"- HEAD local: `{git_state['head']}`.",
        f"- Referência local `{EXPECTED_REMOTE_REF}`: `{git_state['tracking_head']}`.",
        f"- HEAD publicado: `{git_state['remote_head']}`.",
        f"- Verificação remota: `{git_state['remote_check']}`.",
        f"- Estado inicial aceito: `{git_state['tree']}`.",
        "",
        "## Encerramento AP-003G confirmado",
        "",
        f"- Commit: `{ap003g['commit']}`.",
        f"- Assunto: `{ap003g['subject']}`.",
        f"- Data: `{ap003g['authored_at']}`.",
        "- Commit ancestral do HEAD local e do HEAD publicado.",
        "- Alterações produtivas no commit: nenhuma.",
        "",
        "## Escopo técnico",
        "",
        f"- Arquivos rastreados: **{inventory['scope']['tracked_files_scanned']}**.",
        f"- Python analisados por AST: **{inventory['scope']['python_files_scanned']}**.",
        f"- Textos analisados: **{inventory['scope']['text_files_scanned']}**.",
        "- Testes e documentação: evidência, não candidatos por palavra isolada.",
        "- Saídas operacionais, scripts históricos de manutenção e assets: protegidos.",
        "- Métodos `__dunder__` e imports consumidores não duplicam decisões acionáveis.",
        "- Código produtivo alterado: **não**.",
        "",
        "## Totais v4.2",
        "",
        f"- Ocorrências brutas: **{stats['raw_occurrence_count']}**.",
        f"- Candidatos acionáveis: **{stats['actionable_candidate_count']}**.",
        f"- Nomes operacionais protegidos: **{stats['protected_operational_count']}**.",
        f"- Registros históricos/testes: **{stats['historical_reference_count']}**.",
        f"- Ocorrências contextuais não acionáveis: **{stats['contextual_review_occurrence_count']}**.",
        f"- Colisões de destino: **{stats['destination_collision_count']}**.",
        f"- Revisões manuais: **{stats['manual_review_count']}**.",
    ]
    for classification in CLASSIFICATIONS:
        lines.append(f"- {classification.capitalize()}: **{stats['by_classification'][classification]}**.")

    lines.extend(["", "## Candidatos acionáveis", "", "| ID | Categoria | Nome atual | Sugestão | Arquivo:linha | Classificação | Fase |", "|---|---|---|---|---|---|---|"])
    for item in inventory["actionable_candidates"]:
        location = item["path"] + (f":{item['line']}" if item["line"] else "")
        suggestion = f"`{markdown_escape(item['suggested_name'])}`" if item.get("suggested_name") else "—"
        if item.get("suspended_suggestion"):
            suggestion = f"suspensa: `{markdown_escape(item['suspended_suggestion'])}`"
        lines.append(
            "| `{id}` | {category} | `{current}` | {suggestion} | `{location}` | {classification} | {phase} |".format(
                id=item["id"], category=markdown_escape(item["category"]),
                current=markdown_escape(item["current_name"]), suggestion=suggestion,
                location=markdown_escape(location), classification=markdown_escape(item["classification"]),
                phase=markdown_escape(item["target_phase"]),
            )
        )

    lines.extend(["", "## Superfícies relacionadas", ""])
    related = [
        item for item in inventory["actionable_candidates"]
        if item.get("related_surfaces")
    ]
    if not related:
        lines.append("Nenhum candidato acionável está ligado a entrypoint.")
    else:
        for item in related:
            lines.append(f"- `{item['current_name']}`:")
            for surface in item["related_surfaces"]:
                lines.append(
                    "  - {kind}: `{name}` → `{target}` em `{path}`".format(
                        kind=markdown_escape(surface.get("kind", "")),
                        name=markdown_escape(surface.get("name", "")),
                        target=markdown_escape(surface.get("target", "")),
                        path=markdown_escape(surface.get("path", "")),
                    )
                )

    lines.extend(["", "## Colisões de destino", ""])
    if not inventory["destination_collisions"]:
        lines.append("Nenhuma colisão detectada.")
    else:
        for collision in inventory["destination_collisions"]:
            lines.append(f"- `{collision['proposed_destination']}` — sugestão suspensa para {len(collision['sources'])} origens.")
            for source in collision["sources"]:
                location = source["path"] + (f":{source['line']}" if source["line"] else "")
                lines.append(f"  - `{source['current_name']}` em `{location}`")

    lines.extend(["", "## Caminhos operacionais protegidos", ""])
    if not inventory["protected_operational_names"]:
        lines.append("Nenhum caminho protegido com marcador foi encontrado.")
    else:
        for item in inventory["protected_operational_names"]:
            lines.append(f"- `{item['path']}` — {item['reason']}.")

    lines.extend([
        "", "## Entry points identificados", "", "| Tipo | Nome | Destino | Arquivo |", "|---|---|---|---|",
    ])
    for item in inventory["entrypoints"]:
        lines.append("| {kind} | `{name}` | `{target}` | `{path}` |".format(
            kind=markdown_escape(item.get("kind", "")), name=markdown_escape(item.get("name", "")),
            target=markdown_escape(item.get("target", "")), path=markdown_escape(item.get("path", "")),
        ))

    lines.extend([
        "", "## Nomes protegidos", "",
        f"- Diretório físico: `{SOFTWARE_DIRNAME}` — permanece até AP-006.",
        f"- `{KNOWN_XFAIL_SYMBOLS[0]}` — xfail histórico congelado.",
        f"- `{KNOWN_XFAIL_SYMBOLS[1]}` — xfail histórico congelado.",
        f"- `{KNOWN_XFAIL_SYMBOLS[2]}` — xfail histórico congelado.",
        "- `legacy` permanece quando identifica compatibilidade real.",
        "- `academic-pipeline` e `python -m academic_pipeline` permanecem contratos públicos.",
        "", "## Validação", "",
        f"- `py_compile`: `{validation['py_compile']}`.",
        f"- `git diff --check`: `{validation['git_diff_check']}`.",
        "- Suíte específica: `" + str(validation["specific_suite"].get("summary", validation["specific_suite"].get("status"))) + "`.",
        "- Suíte consolidada: `" + str(validation["consolidated_suite"].get("summary", validation["consolidated_suite"].get("status"))) + "`.",
        "", "## Decisão de fase", "",
        "A AP-004B permanece bloqueada. A matriz v4.2 deve ser revisada e aprovada antes de qualquer renomeação ou commit.",
        "",
    ])
    return normalize_output("\n".join(lines))




def build_contract_test(
    *,
    git_head: str,
    ap003g_commit: str,
    tool_sha256: str,
) -> str:
    output_rels = [item.as_posix() for item in OUTPUT_RELS]
    return normalize_output(
        f'''from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / {INVENTORY_REL.as_posix()!r}
CONVENTION = ROOT / {CONVENTION_REL.as_posix()!r}
TOOL = ROOT / {TOOL_REL.as_posix()!r}
EXPECTED_HEAD = {git_head!r}
EXPECTED_AP003G_COMMIT = {ap003g_commit!r}
EXPECTED_TOOL_SHA256 = {tool_sha256!r}
EXPECTED_OUTPUTS = {output_rels!r}
CLASSIFICATIONS = {list(CLASSIFICATIONS)!r}
CATEGORIES = {list(CATEGORIES)!r}
STRUCTURAL_KINDS = {{
    "release_candidate", "version_marker", "refactor_phase", "explicit_version_word"
}}


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
    raw = raw.strip().strip('"').replace("\\\\", "/")
    prefix = "software/academic_pipeline_rc10_7_conformidade/"
    return raw[len(prefix):] if raw.startswith(prefix) else raw


def _ephemeral(path: str) -> bool:
    parts = Path(path).parts
    return "__pycache__" in parts or ".pytest_cache" in parts or path.endswith((".pyc", ".pyo"))


def test_ap004a_v4_2_is_bound_to_current_head_and_ap003g() -> None:
    data = _data()
    assert data["phase"] == "AP-004A"
    assert data["mode"] == "inventory-and-convention-v4.2-read-only"
    assert data["inventory_schema_version"] == 4
    assert data["inventory_revision"] == "4.2"
    assert data["tool"]["version"] == 4
    assert data["tool"]["revision"] == "4.2"
    assert data["git"]["head"] == EXPECTED_HEAD
    assert _run("git", "rev-parse", "HEAD") == EXPECTED_HEAD
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
    assert all(item["category"] not in {{"import", "entrypoint", "teste", "documentação"}} for item in data["actionable_candidates"])


def test_ap004a_v4_2_actionable_candidates_require_structural_markers_or_explicit_preservation() -> None:
    data = _data()
    for item in data["actionable_candidates"]:
        marker_kinds = {{marker["kind"] for marker in item["markers"]}}
        if item["classification"] == "nome que deve permanecer":
            continue
        assert marker_kinds & STRUCTURAL_KINDS, item
    assert any(item.get("related_surfaces") for item in data["actionable_candidates"])
    assert data["entrypoints"]


def test_ap004a_v4_2_excludes_dunder_stdlib_and_semantic_verbs() -> None:
    data = _data()
    candidates = data["actionable_candidates"]
    names = {{item["current_name"] for item in candidates}}
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
    assert {{item["classification"] for item in candidates}} <= set(CLASSIFICATIONS)
    active_destinations = []
    unsafe = re.compile(r"(?:^|_)(?:original|pre|stage_?\\d+|dispatch_?\\d+)(?:_|$)|_\\d+$", re.I)
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
    expected = {{
        "_refs_v6_strip_org": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "extract_org_abstracts": "app_bundle/scripts/pipeline/render_docx_canonico.py",
        "WorkflowState._normalize": "app_bundle/scripts/pipeline/article_workflow/state.py",
    }}
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


def test_ap004a_v4_2_changes_only_allowed_files_and_generated_python_compiles() -> None:
    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    actual = {{
        path for line in status.splitlines() if line.strip()
        for path in [_status_path(line)] if not _ephemeral(path)
    }}
    assert actual == set(EXPECTED_OUTPUTS)
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))
    assert CONVENTION.is_file()
    with tempfile.TemporaryDirectory(prefix="ap004a-contract-pyc-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)
'''
    )



def count_test_functions(source: str) -> int:
    tree = ast.parse(source)
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def whitespace_check(paths: Iterable[Path]) -> None:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.endswith((" ", "\t")):
                errors.append(f"{path}:{number}: trailing whitespace")
        if text and not text.endswith("\n"):
            errors.append(f"{path}: missing final newline")
    if errors:
        fail("Falhas de whitespace:\n" + "\n".join(errors))


def validate_allowed_final_status(
    *,
    repository_root: Path,
    software_root: Path,
) -> list[str]:
    status_lines = [
        line
        for line in git(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout.splitlines()
        if line.strip()
    ]
    actual: set[str] = set()
    for line in status_lines:
        path = software_relative_status_path(
            line,
            software_root=software_root,
            repository_root=repository_root,
        )
        parts = PurePosixPath(path).parts
        if any(part in IGNORED_PATH_PARTS for part in parts):
            continue
        if path.endswith((".pyc", ".pyo")):
            continue
        actual.add(path)
    expected = {item.as_posix() for item in OUTPUT_RELS}
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        details: list[str] = []
        if unexpected:
            details.append(
                "Arquivos inesperados:\n"
                + "\n".join(f"  - {item}" for item in unexpected)
            )
        if missing:
            details.append(
                "Arquivos ausentes:\n"
                + "\n".join(f"  - {item}" for item in missing)
            )
        fail(
            "O conjunto final de alterações diverge do permitido:\n"
            + "\n".join(details)
        )
    return status_lines


def parse_pytest_summary(result: CommandResult, *, label: str) -> dict[str, Any]:
    combined = "\n".join(
        value
        for value in (result.stdout.strip(), result.stderr.strip())
        if value
    )
    matches = list(SUMMARY_PATTERN.finditer(combined))
    if not matches:
        fail(f"Resumo pytest não reconhecido ({label}):\n{combined}")
    match = matches[-1]
    passed = int(match.group("passed"))
    xfailed = int(match.group("xfailed") or 0)
    summary = match.group(0)
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "passed": passed,
        "xfailed": xfailed,
        "summary": summary,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_specific_suite(software_root: Path) -> dict[str, Any]:
    result = run(
        (
            "pipenv",
            "run",
            "pytest",
            "-q",
            "-ra",
            TEST_REL.as_posix(),
        ),
        cwd=software_root,
        check=False,
        timeout=600,
    )
    parsed = parse_pytest_summary(result, label="AP-004A específica")
    if result.returncode != 0:
        fail(
            "Suíte específica AP-004A falhou:\n"
            f"{result.stdout}{result.stderr}"
        )
    if parsed["passed"] != EXPECTED_CONTRACT_TESTS or parsed["xfailed"] != 0:
        fail(
            "Contagem específica AP-004A divergente.\n"
            f"Esperado: {EXPECTED_CONTRACT_TESTS} passed, 0 xfailed\n"
            f"Atual:    {parsed['summary']}"
        )
    return parsed


def run_consolidated_suite(software_root: Path) -> dict[str, Any]:
    result = run(
        (
            "pipenv",
            "run",
            "pytest",
            "-q",
            "-ra",
            "app_bundle/tests",
            "tests",
        ),
        cwd=software_root,
        check=False,
        timeout=1800,
    )
    parsed = parse_pytest_summary(result, label="AP-004A consolidada")
    if result.returncode != 0:
        fail(
            "Suíte consolidada AP-004A falhou:\n"
            f"{result.stdout}{result.stderr}"
        )
    expected_passed = BASELINE_PASSED + EXPECTED_CONTRACT_TESTS
    if (
        parsed["passed"] != expected_passed
        or parsed["xfailed"] != BASELINE_XFAILED
    ):
        fail(
            "Contagem consolidada AP-004A divergente.\n"
            f"Esperado: {expected_passed} passed, "
            f"{BASELINE_XFAILED} xfailed\n"
            f"Atual:    {parsed['summary']}"
        )
    return parsed


def write_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventário AP-004A v4.2, somente leitura produtiva."
    )
    parser.add_argument(
        "--skip-remote-check",
        action="store_true",
        help=(
            "Uso excepcional/offline: compara apenas HEAD e origin/branch local. "
            "Não satisfaz a confirmação de publicação exigida para a fase."
        ),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help=(
            "Uso diagnóstico: gera inventário sem executar pytest. Não encerra "
            "nem valida a AP-004A."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    software_root = Path.cwd().resolve()
    tool_source = Path(__file__).read_text(encoding="utf-8")
    ast.parse(tool_source, filename=str(Path(__file__)))

    repository_root, git_state = validate_git_state(
        software_root,
        skip_remote_check=args.skip_remote_check,
    )
    if args.skip_remote_check:
        print(
            "[AVISO] Verificação remota ignorada. O resultado não confirma "
            "publicação no GitHub.",
            file=sys.stderr,
        )

    ap003g = validate_ap003g_commit(
        software_root=software_root,
        repository_root=repository_root,
        git_state=git_state,
    )
    architecture = validate_ap003_architecture(software_root)

    inventory = build_inventory_data(
        software_root=software_root,
        repository_root=repository_root,
        git_state=git_state,
        ap003g=ap003g,
        architecture=architecture,
        tool_source=tool_source,
    )
    convention = build_convention_document()
    test_source = build_contract_test(
        git_head=git_state["head"],
        ap003g_commit=ap003g["commit"],
        tool_sha256=inventory["tool"]["sha256"],
    )
    if count_test_functions(test_source) != EXPECTED_CONTRACT_TESTS:
        fail(
            f"Esperados {EXPECTED_CONTRACT_TESTS} testes contratuais; "
            f"gerados {count_test_functions(test_source)}."
        )
    ast.parse(test_source, filename=TEST_REL.as_posix())

    preliminary_report = build_report(inventory)
    outputs = {
        software_root / TOOL_REL: normalize_output(tool_source),
        software_root / CONVENTION_REL: convention,
        software_root / INVENTORY_REL: write_json(inventory),
        software_root / REPORT_REL: preliminary_report,
        software_root / TEST_REL: test_source,
    }

    backup_root, backup_records = create_backups(
        outputs,
        software_root=software_root,
    )

    try:
        for path, content in outputs.items():
            atomic_write(path, content)

        compile_python(software_root / TOOL_REL)
        compile_python(software_root / TEST_REL)
        whitespace_check(outputs)

        git_diff = git(repository_root, "diff", "--check", check=False)
        if git_diff.returncode != 0:
            fail(
                "git diff --check falhou:\n"
                f"{git_diff.stdout}{git_diff.stderr}"
            )

        validate_allowed_final_status(
            repository_root=repository_root,
            software_root=software_root,
        )

        validation: dict[str, Any] = {
            "py_compile": "passed",
            "git_diff_check": "passed",
            "specific_suite": {"status": "skipped"},
            "consolidated_suite": {"status": "skipped"},
        }

        if not args.skip_tests:
            validation["specific_suite"] = run_specific_suite(software_root)
            validation["consolidated_suite"] = run_consolidated_suite(
                software_root
            )

        inventory["validation"] = validation
        final_report = build_report(
            inventory,
            validation_override=validation,
        )
        atomic_write(software_root / INVENTORY_REL, write_json(inventory))
        atomic_write(software_root / REPORT_REL, final_report)
        whitespace_check(outputs)

        validate_allowed_final_status(
            repository_root=repository_root,
            software_root=software_root,
        )
        final_diff = git(repository_root, "diff", "--check", check=False)
        if final_diff.returncode != 0:
            fail(
                "git diff --check final falhou:\n"
                f"{final_diff.stdout}{final_diff.stderr}"
            )

        if not args.skip_tests:
            # Revalida o JSON e o relatório finais após a atualização dos resultados.
            final_specific = run_specific_suite(software_root)
            inventory["validation"]["specific_suite_final"] = final_specific
            atomic_write(
                software_root / INVENTORY_REL,
                write_json(inventory),
            )

        print("[OK] AP-004A v4.2 inventariada sem alteração produtiva.")
        print(f"[OK] Branch: {git_state['branch']}")
        print(f"[OK] HEAD local/remoto: {git_state['head']}")
        print(
            "[OK] Commit AP-003G publicado: "
            f"{ap003g['commit']} — {ap003g['subject']}"
        )
        print(
            "[OK] Ocorrências brutas: "
            f"{inventory['statistics']['raw_occurrence_count']}"
        )
        print(
            "[OK] Candidatos acionáveis: "
            f"{inventory['statistics']['actionable_candidate_count']}"
        )
        for classification in CLASSIFICATIONS:
            print(
                f"     {classification}: "
                f"{inventory['statistics']['by_classification'][classification]}"
            )
        print(f"[OK] Relatório: {REPORT_REL}")
        print(f"[OK] Convenção: {CONVENTION_REL}")
        print(f"[OK] JSON: {INVENTORY_REL}")
        print(f"[OK] Teste: {TEST_REL}")
        print(f"[OK] Ferramenta reexecutável: {TOOL_REL}")
        print(f"[OK] Backup externo: {backup_root}")
        if args.skip_tests:
            print(
                "[ATENÇÃO] Testes ignorados; a AP-004A não está validada para "
                "consolidação."
            )
        else:
            print(
                "[OK] Suíte específica: "
                f"{validation['specific_suite']['summary']}"
            )
            print(
                "[OK] Suíte consolidada: "
                f"{validation['consolidated_suite']['summary']}"
            )
        print("[OK] Nenhum commit foi criado.")
        print("[BLOQUEIO] Não avançar para AP-004B sem aprovação do inventário.")
        return 0
    except Exception:
        rollback(backup_records)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InventoryError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        raise SystemExit(1)

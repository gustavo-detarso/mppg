#!/usr/bin/env python3
"""AP-004C — aplicador produtivo de símbolos internos em duas ondas atômicas.

Vinculado ao inventário AP-004C v1.3 aprovado e ao commit publicado da
AP-004B. O aplicador:

1. valida branch, HEAD, remoto, árvore preparatória e manifests;
2. aplica sete renomeações locais no gerador TOML interativo;
3. valida e registra o checkpoint da onda 1;
4. aplica treze renomeações privadas no orquestrador;
5. atualiza somente contratos de caracterização e hashes necessários;
6. preserva os quatro controles de xfail e todos os símbolos adiados;
7. gera relatório, JSON, ferramenta reexecutável e contrato de aplicação;
8. executa py_compile, testes específicos e suíte consolidada;
9. usa backup externo, escrita atômica e rollback integral das duas ondas.

Nenhum commit é criado automaticamente.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import os
import py_compile
import re
import runpy
import shutil
import stat
import subprocess
import sys
import symtable
import tempfile
import tokenize
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
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
EXPECTED_AP004C_COMMIT_SUBJECT = (
    "refactor(academic-pipeline): consolidar símbolos internos da AP-004C"
)
SOFTWARE_PREFIX = "software/academic_pipeline_rc10_7_conformidade/"

PHASE = "AP-004C"
MODE = "internal-symbol-application-v1.4"
TOOL_VERSION = 1
TOOL_REVISION = "1.4"
APPLICATION_SCHEMA_VERSION = 1
BASELINE_PASSED = 463
BASELINE_XFAILED = 3
EXPECTED_APPLICATION_TESTS = 19
EXPECTED_CONSOLIDATED_PASSED = BASELINE_PASSED + EXPECTED_APPLICATION_TESTS

DOC_DIR = Path("docs/refactor/academic-pipeline/AP-004")
INVENTORY_REPORT_REL = DOC_DIR / "AP-004C_INTERNAL_SYMBOL_INVENTORY.md"
INVENTORY_STRATEGY_REL = DOC_DIR / "AP-004C_INTERNAL_SYMBOL_STRATEGY.md"
INVENTORY_JSON_REL = DOC_DIR / "ap004c_internal_symbol_inventory.json"
INVENTORY_TEST_REL = Path(
    "tests/characterization/test_ap004c_internal_symbol_inventory_contract.py"
)
INVENTORY_TOOL_REL = Path("tools/refactor/ap004c_inventory_internal_symbols.py")

AP004B_APPLICATION_TEST_REL = Path(
    "tests/characterization/test_ap004b_module_file_application_contract.py"
)
AP004B_INVENTORY_TEST_REL = Path(
    "tests/characterization/test_ap004b_module_file_inventory_contract.py"
)
AP003F_TEST_REL = Path(
    "tests/characterization/test_ap003f_main_unification_contract.py"
)
AP003G_TEST_REL = Path(
    "tests/characterization/test_ap003g_stabilization_contract.py"
)
AP003D_TEST_REL = Path(
    "tests/characterization/test_ap003d_document_contract.py"
)

APPLICATION_REPORT_REL = DOC_DIR / "AP-004C_INTERNAL_SYMBOL_APPLICATION.md"
APPLICATION_JSON_REL = DOC_DIR / "ap004c_internal_symbol_application.json"
APPLICATION_TEST_REL = Path(
    "tests/characterization/test_ap004c_internal_symbol_application_contract.py"
)
APPLICATION_TOOL_REL = Path("tools/refactor/ap004c_apply_internal_symbols.py")

TOML_GENERATOR_REL = Path(
    "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py"
)
ORCHESTRATOR_REL = Path(
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)

PREPARATORY_DIRTY_PATHS = {
    AP004B_APPLICATION_TEST_REL.as_posix(),
    AP004B_INVENTORY_TEST_REL.as_posix(),
    INVENTORY_REPORT_REL.as_posix(),
    INVENTORY_STRATEGY_REL.as_posix(),
    INVENTORY_JSON_REL.as_posix(),
    INVENTORY_TEST_REL.as_posix(),
    INVENTORY_TOOL_REL.as_posix(),
}

WAVE_1: tuple[tuple[str, str], ...] = (
    (
        "_generate_interactive_before_wizard_documentos_locais_v4",
        "_generate_interactive_before_wizard_documentos_locais",
    ),
    (
        "_generate_interactive_with_wizard_documentos_locais_v4",
        "_generate_interactive_with_wizard_documentos_locais",
    ),
    ("_v5_is_local_document", "_is_local_document"),
    ("_v5_reference_default", "_reference_default"),
    ("_v5_normalise_prompt", "_normalise_prompt"),
    ("_v5_configure_reference_policy", "_configure_reference_policy"),
    ("_v5_ensure_reference_policy", "_ensure_reference_policy"),
)

WAVE_2: tuple[tuple[str, str], ...] = (
    ("_ap003d_impl_output_paths", "_impl_output_paths"),
    ("_ap003d_impl_apply_cli_path_overrides", "_impl_apply_cli_path_overrides"),
    ("_ap003d_impl_load_existing_document_json", "_impl_load_existing_document_json"),
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
    ("_ap003d_impl__refs_v6_disabled", "_impl_refs_disabled"),
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

PROTECTED_CONTROLS = (
    ("_refs_v6_strip_org", ORCHESTRATOR_REL.as_posix()),
    ("_ap003d_impl__refs_v6_strip_org", ORCHESTRATOR_REL.as_posix()),
    ("_normalize", "app_bundle/scripts/pipeline/article_workflow/state.py"),
    ("extract_org_abstracts", "app_bundle/scripts/pipeline/render_docx_canonico.py"),
)

APPLICATION_ARTIFACT_PATHS = {
    APPLICATION_REPORT_REL.as_posix(),
    APPLICATION_JSON_REL.as_posix(),
    APPLICATION_TEST_REL.as_posix(),
    APPLICATION_TOOL_REL.as_posix(),
}

KNOWN_XFAILS = (
    "_refs_v6_strip_org",
    "extract_org_abstracts",
    "WorkflowState._normalize",
)

AP003G_HISTORICAL_ORCHESTRATOR_SHA256 = (
    "8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977"
)


class ApplicationError(RuntimeError):
    """Erro controlado do aplicador AP-004C."""


def fail(message: str) -> NoReturn:
    raise ApplicationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_output(text: str) -> str:
    return text.rstrip() + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: int = 300,
) -> CommandResult:
    completed = subprocess.run(
        tuple(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    result = CommandResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        fail(
            f"Comando falhou ({result.returncode}): {' '.join(result.args)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def git(repository_root: Path, *args: str, check: bool = True) -> CommandResult:
    return run(("git", *args), cwd=repository_root, check=check, timeout=900)


def resolve_roots() -> tuple[Path, Path]:
    software_root = Path.cwd().resolve()
    repository_root = Path(
        git(software_root, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve()
    if software_root != EXPECTED_SOFTWARE_ROOT:
        fail(
            "Raiz do software incorreta.\n"
            f"Esperada: {EXPECTED_SOFTWARE_ROOT}\nAtual: {software_root}"
        )
    if repository_root != EXPECTED_REPOSITORY_ROOT:
        fail(
            "Raiz Git incorreta.\n"
            f"Esperada: {EXPECTED_REPOSITORY_ROOT}\nAtual: {repository_root}"
        )
    return software_root, repository_root


def status_path(line: str) -> str:
    raw = line[2:].lstrip() if len(line) >= 2 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    raw = raw.strip().strip('"').replace("\\", "/")
    if raw.startswith(SOFTWARE_PREFIX):
        raw = raw[len(SOFTWARE_PREFIX):]
    return raw


def ephemeral(path: str) -> bool:
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
    unstaged: set[str] = set()
    staged: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = status_path(line)
        if ephemeral(path):
            continue
        x = line[0] if len(line) > 0 else " "
        y = line[1] if len(line) > 1 else " "
        if x not in {" ", "?", "!"}:
            staged.add(path)
        if y != " " or line.startswith("??"):
            unstaged.add(path)
    return sorted(unstaged), sorted(staged)


def validate_git_state(software_root: Path, repository_root: Path) -> dict[str, str]:
    branch = git(repository_root, "branch", "--show-current").stdout.strip()
    head = git(repository_root, "rev-parse", "HEAD").stdout.strip()
    subject = git(repository_root, "show", "-s", "--format=%s", "HEAD").stdout.strip()
    if branch != EXPECTED_BRANCH:
        fail(f"Branch incorreta: {branch!r}; esperada: {EXPECTED_BRANCH!r}")
    if head != EXPECTED_HEAD:
        fail(f"HEAD incorreto: {head}; esperado: {EXPECTED_HEAD}")
    if subject != EXPECTED_AP004B_SUBJECT:
        fail(f"Assunto do HEAD divergente: {subject!r}")

    unstaged, staged = status_paths(repository_root)
    if staged:
        fail(f"Nenhum arquivo pode estar staged: {staged}")
    expected = sorted(PREPARATORY_DIRTY_PATHS)
    if unstaged != expected:
        fail(
            "Árvore preparatória AP-004C divergente.\n"
            f"Atual: {unstaged}\nEsperado: {expected}"
        )

    git(repository_root, "fetch", "origin", EXPECTED_BRANCH)
    remote = git(repository_root, "rev-parse", EXPECTED_REMOTE_REF).stdout.strip()
    if remote != head:
        fail(f"HEAD remoto divergente: local={head}, remoto={remote}")
    divergence = git(
        repository_root,
        "rev-list",
        "--left-right",
        "--count",
        f"HEAD...{EXPECTED_REMOTE_REF}",
    ).stdout.strip()
    if divergence != "0\t0" and divergence != "0 0":
        fail(f"Branch local/remota divergente: {divergence}")

    return {"branch": branch, "head": head, "remote_head": remote, "subject": subject}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"JSON inválido ou indisponível: {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON raiz não é objeto: {path}")
    return value


def identifier_counts(source: str, names: Iterable[str]) -> Counter[str]:
    wanted = set(names)
    counts: Counter[str] = Counter()
    for token in tokenize.tokenize(BytesIO(source.encode("utf-8")).readline):
        if token.type == tokenize.NAME and token.string in wanted:
            counts[token.string] += 1
    return counts


def rewrite_identifier_tokens(source: str, mapping: dict[str, str]) -> tuple[str, Counter[str]]:
    rewritten, counts, _trace = rewrite_identifier_tokens_traced(source, mapping)
    return rewritten, counts


def rewrite_identifier_tokens_traced(
    source: str,
    mapping: dict[str, str],
) -> tuple[str, Counter[str], dict[int, str]]:
    """Renomeia tokens NAME e registra os índices efetivamente alterados.

    O trace permite restaurar somente os identificadores modificados pela
    aplicação, sem confundir homônimos preexistentes em outros escopos.
    """
    tokens: list[tokenize.TokenInfo] = []
    counts: Counter[str] = Counter()
    trace: dict[int, str] = {}
    for index, token in enumerate(
        tokenize.tokenize(BytesIO(source.encode("utf-8")).readline)
    ):
        if token.type == tokenize.NAME and token.string in mapping:
            old_name = token.string
            counts[old_name] += 1
            trace[index] = old_name
            token = tokenize.TokenInfo(
                type=token.type,
                string=mapping[old_name],
                start=token.start,
                end=token.end,
                line=token.line,
            )
        tokens.append(token)
    return tokenize.untokenize(tokens).decode("utf-8"), counts, trace


def restore_traced_identifier_tokens(
    source: str,
    mapping: dict[str, str],
    trace: dict[int, str],
) -> str:
    """Restaura somente os tokens alterados pela renomeação rastreada."""
    tokens = list(tokenize.tokenize(BytesIO(source.encode("utf-8")).readline))
    for index, old_name in trace.items():
        if index >= len(tokens):
            fail(f"Trace de token fora do intervalo: {index}")
        token = tokens[index]
        expected = mapping[old_name]
        if token.type != tokenize.NAME or token.string != expected:
            fail(
                "Trace de token divergente após renomeação: "
                f"índice={index}, esperado={expected!r}, atual={token.string!r}"
            )
        tokens[index] = tokenize.TokenInfo(
            type=token.type,
            string=old_name,
            start=token.start,
            end=token.end,
            line=token.line,
        )
    return tokenize.untokenize(tokens).decode("utf-8")


class ModuleBindingCollector(ast.NodeVisitor):
    """Coleta vínculos no escopo de módulo sem entrar em escopos internos."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)


def module_level_bound_names(source: str) -> set[str]:
    collector = ModuleBindingCollector()
    collector.visit(ast.parse(source))
    return collector.names


@dataclass(frozen=True)
class BindingEntry:
    name: str
    kind: str
    scope: str
    line: int
    ast_sha256: str


def _binding_assigned_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for child in node.elts:
            names.extend(_binding_assigned_names(child))
    return names


def _binding_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _binding_scope(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    names: list[str] = []
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(id(current))
    return ".".join(reversed(names)) or "<module>"


def collect_binding_entries(source: str) -> list[BindingEntry]:
    """Coleta vínculos em todos os escopos com a mesma semântica do inventário."""
    tree = ast.parse(source)
    parents = _binding_parent_map(tree)
    entries: list[BindingEntry] = []
    for node in ast.walk(tree):
        node_hash = sha256_bytes(
            ast.dump(node, include_attributes=False, annotate_fields=True).encode("utf-8")
        )
        scope = _binding_scope(node, parents)
        line = int(getattr(node, "lineno", 0) or 0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entries.append(BindingEntry(node.name, "function", scope, line, node_hash))
        elif isinstance(node, ast.ClassDef):
            entries.append(BindingEntry(node.name, "class", scope, line, node_hash))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _binding_assigned_names(target):
                    entries.append(BindingEntry(name, "assignment", scope, line, node_hash))
        elif isinstance(node, ast.AnnAssign):
            for name in _binding_assigned_names(node.target):
                entries.append(
                    BindingEntry(name, "annotated_assignment", scope, line, node_hash)
                )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                visible = alias.asname or alias.name.split(".")[-1]
                kind = "import_alias" if alias.asname else "import"
                entries.append(BindingEntry(visible, kind, scope, line, node_hash))
    return entries


def preflight_wave(
    *,
    source: str,
    mappings: tuple[tuple[str, str], ...],
    inventory: dict[str, Any],
    relative_path: str,
    label: str,
) -> tuple[str, Counter[str], dict[int, str], list[dict[str, Any]]]:
    """Valida todos os símbolos de uma onda e simula a reescrita antes de escrever."""
    candidate_by_name = {
        item["current_name"]: item
        for item in inventory["candidates"]
        if item.get("path") == relative_path
    }
    entries = collect_binding_entries(source)
    all_names = {name for pair in mappings for name in pair}
    before_counts = identifier_counts(source, all_names)
    errors: list[str] = []
    reports: list[dict[str, Any]] = []

    for old, new in mappings:
        item = candidate_by_name.get(old)
        if item is None:
            errors.append(f"{old}: ausente no inventário para {relative_path}")
            continue
        definition = item.get("definition", {})
        expected_kind = definition.get("definition_kind")
        expected_scope = definition.get("scope")
        expected_line = int(definition.get("line") or 0)
        expected_hash = definition.get("ast_sha256")
        origins = [
            entry
            for entry in entries
            if entry.name == old
            and entry.kind == expected_kind
            and entry.scope == expected_scope
            and entry.line == expected_line
            and entry.ast_sha256 == expected_hash
        ]
        if len(origins) != 1:
            nearby = [
                f"{entry.kind}@{entry.scope}:{entry.line}"
                for entry in entries
                if entry.name == old
            ]
            errors.append(
                f"{old}: definição inventariada não localizada de forma única "
                f"(encontradas={len(origins)}, candidatas={nearby})"
            )
        collisions = [
            entry
            for entry in entries
            if entry.name == new and entry.scope == expected_scope
        ]
        if collisions:
            detail = [f"{entry.kind}@{entry.scope}:{entry.line}" for entry in collisions]
            errors.append(f"{old} -> {new}: colisão no mesmo escopo: {detail}")
        if before_counts[old] < 1:
            errors.append(f"{old}: nenhum token NAME localizado")
        reports.append(
            {
                "old": old,
                "new": new,
                "definition_kind": expected_kind,
                "scope": expected_scope,
                "line": expected_line,
                "tokens_before": before_counts[old],
            }
        )

    if errors:
        fail(
            f"{label}: pré-validação integral falhou para {len(errors)} item(ns):\n- "
            + "\n- ".join(errors)
        )

    rewritten, counts, trace = rewrite_identifier_tokens_traced(source, dict(mappings))
    for old, new in mappings:
        if counts[old] != before_counts[old]:
            errors.append(
                f"{old}: substituições={counts[old]}; tokens esperados={before_counts[old]}"
            )
    if normalized_ast_sha256(source) != targeted_normalized_ast_sha256(
        rewritten, dict(mappings), trace
    ):
        errors.append("AST normalizada divergente após simulação")
    try:
        post_entries = collect_binding_entries(rewritten)
    except SyntaxError as exc:
        errors.append(f"fonte simulada não compila em AST: {exc}")
        post_entries = []
    for report in reports:
        old = report["old"]
        new = report["new"]
        if any(entry.name == old for entry in post_entries):
            errors.append(f"{old}: vínculo antigo permaneceu após simulação")
        migrated = [
            entry
            for entry in post_entries
            if entry.name == new
            and entry.kind == report["definition_kind"]
            and entry.scope == report["scope"]
            and entry.line == report["line"]
        ]
        if len(migrated) != 1:
            errors.append(
                f"{old} -> {new}: vínculo migrado não localizado de forma única "
                f"após simulação (encontrados={len(migrated)})"
            )
    if errors:
        fail(
            f"{label}: simulação integral falhou para {len(errors)} item(ns):\n- "
            + "\n- ".join(errors)
        )
    return rewritten, counts, trace, reports


def targeted_normalized_ast_sha256(
    rewritten_source: str,
    mapping: dict[str, str],
    trace: dict[int, str],
) -> str:
    restored = restore_traced_identifier_tokens(
        rewritten_source, mapping, trace
    )
    return normalized_ast_sha256(restored)


def rewrite_exact_names_in_python_contract(
    source: str,
    mapping: dict[str, str],
    *,
    replace_generic_prefix: bool = False,
) -> tuple[str, int]:
    """Atualiza nomes em tokens NAME e em literais de contrato, não em comentários."""
    tokens: list[tokenize.TokenInfo] = []
    changes = 0
    for token in tokenize.tokenize(BytesIO(source.encode("utf-8")).readline):
        new_string = token.string
        if token.type == tokenize.NAME and token.string in mapping:
            new_string = mapping[token.string]
        elif token.type == tokenize.STRING:
            for old, new in mapping.items():
                if old in new_string:
                    new_string = new_string.replace(old, new)
            if replace_generic_prefix:
                # Somente o literal genérico de prefixo; nomes protegidos completos
                # não são atingidos porque não são iguais a este literal.
                new_string = new_string.replace(
                    '"_ap003d_impl_"', '"_impl_"'
                ).replace("'_ap003d_impl_'", "'_impl_'")
        if new_string != token.string:
            changes += 1
            token = tokenize.TokenInfo(
                token.type, new_string, token.start, token.end, token.line
            )
        tokens.append(token)
    return tokenize.untokenize(tokens).decode("utf-8"), changes


class IdentifierNormalizer(ast.NodeTransformer):
    def __init__(self, reverse_mapping: dict[str, str]) -> None:
        self.reverse = reverse_mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.reverse.get(node.id, node.id)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node.attr = self.reverse.get(node.attr, node.attr)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self.reverse.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = self.reverse.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self.reverse.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self.reverse.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.asname:
            node.asname = self.reverse.get(node.asname, node.asname)
        return self.generic_visit(node)


def normalized_ast_sha256(source: str, reverse_mapping: dict[str, str] | None = None) -> str:
    tree = ast.parse(source)
    if reverse_mapping:
        tree = IdentifierNormalizer(reverse_mapping).visit(tree)  # type: ignore[assignment]
        ast.fix_missing_locations(tree)
    dump = ast.dump(tree, include_attributes=False, annotate_fields=True)
    return sha256_bytes(dump.encode("utf-8"))


def ast_definition_dump(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            matches.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                matches.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[-1]) == name for alias in node.names):
                matches.append(node)
    if len(matches) != 1:
        fail(f"Definição protegida {name!r} ambígua ou ausente: {len(matches)}")
    return ast.dump(matches[0], include_attributes=False, annotate_fields=True)


def validate_inventory(software_root: Path) -> dict[str, Any]:
    path = software_root / INVENTORY_JSON_REL
    data = load_json(path)
    if data.get("phase") != PHASE:
        fail("Inventário não pertence à AP-004C")
    if data.get("mode") != "internal-symbol-inventory-v1.3-read-only":
        fail(f"Modo do inventário divergente: {data.get('mode')!r}")
    if data.get("inventory_revision") != "1.3":
        fail(f"Revisão do inventário divergente: {data.get('inventory_revision')!r}")
    if data.get("git", {}).get("head") != EXPECTED_HEAD:
        fail("Inventário não está vinculado ao commit AP-004B esperado")
    stats = data.get("statistics", {})
    expected_stats = {
        "ready_wave_1_count": 7,
        "ready_wave_2_count": 13,
        "deferred_count": 49,
        "protected_count": 4,
        "destination_collision_count": 0,
    }
    for key, expected in expected_stats.items():
        if stats.get(key) != expected:
            fail(f"Contagem {key} divergente: {stats.get(key)} != {expected}")

    by_name = {item["current_name"]: item for item in data["candidates"]}
    for old, new in WAVE_1:
        item = by_name.get(old)
        if not item or item.get("suggested_name") != new:
            fail(f"Candidato onda 1 divergente: {old} -> {new}")
        if item.get("disposition") != "ready_local_ast_rename":
            fail(f"Disposição onda 1 divergente para {old}: {item.get('disposition')}")
        if item.get("path") != TOML_GENERATOR_REL.as_posix():
            fail(f"Arquivo onda 1 divergente para {old}: {item.get('path')}")
    for old, new in WAVE_2:
        item = by_name.get(old)
        if not item or item.get("suggested_name") != new:
            fail(f"Candidato onda 2 divergente: {old} -> {new}")
        if item.get("disposition") not in {
            "ready_contract_bound_ast_rename", "contract_update_required"
        }:
            fail(f"Disposição onda 2 divergente para {old}: {item.get('disposition')}")
        if item.get("path") != ORCHESTRATOR_REL.as_posix():
            fail(f"Arquivo onda 2 divergente para {old}: {item.get('path')}")

    protected = {(item["current_name"], item["path"]) for item in data["candidates"] if item.get("disposition") == "protected_xfail_out_of_scope"}
    if protected != set(PROTECTED_CONTROLS):
        fail(f"Controles protegidos divergentes: {sorted(protected)}")

    for record in data.get("source_manifest", []):
        file_path = software_root / record["path"]
        if not file_path.is_file():
            fail(f"Arquivo do manifesto ausente: {record['path']}")
        if sha256_path(file_path) != record["sha256"]:
            fail(f"Hash do manifesto divergente: {record['path']}")
    return data


@dataclass
class BackupRecord:
    relative: str
    existed: bool
    mode: int | None
    backup_path: Path | None


def create_backup(
    software_root: Path,
    paths: Iterable[str],
) -> tuple[Path, list[BackupRecord]]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        Path.home()
        / ".cache/academic-pipeline-refactor/backups/AP-004C-APPLICATION"
        / stamp
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    records: list[BackupRecord] = []
    for relative in sorted(set(paths)):
        source = software_root / relative
        if source.exists():
            target = backup_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append(
                BackupRecord(
                    relative=relative,
                    existed=True,
                    mode=stat.S_IMODE(source.stat().st_mode),
                    backup_path=target,
                )
            )
        else:
            records.append(
                BackupRecord(relative=relative, existed=False, mode=None, backup_path=None)
            )
    manifest = {
        "created_at_utc": utc_now(),
        "software_root": str(software_root),
        "records": [
            {
                "relative": item.relative,
                "existed": item.existed,
                "mode": item.mode,
                "backup_path": str(item.backup_path) if item.backup_path else None,
            }
            for item in records
        ],
    }
    (backup_root / "backup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return backup_root, records


def atomic_write_bytes(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str, mode: int | None = None) -> None:
    atomic_write_bytes(path, normalize_output(text).encode("utf-8"), mode=mode)


def rollback(software_root: Path, records: Iterable[BackupRecord]) -> None:
    errors: list[str] = []
    for record in reversed(list(records)):
        target = software_root / record.relative
        try:
            if record.existed:
                assert record.backup_path is not None
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record.backup_path, target)
                if record.mode is not None:
                    os.chmod(target, record.mode)
            elif target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        except OSError as exc:
            errors.append(f"{record.relative}: {exc}")
    if errors:
        print("[ERRO] Rollback incompleto:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)


def write_preserving_mode(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    atomic_write_text(path, text, mode=mode)


def replace_top_level_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        fail(f"Função contratual {name!r} ambígua ou ausente: {len(matches)}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    replacement_text = normalize_output(replacement)
    return "".join(lines[:start]) + replacement_text + "".join(lines[end:])


def insert_after_assignment(source: str, assignment_name: str, code: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if assignment_name in names:
            lines = source.splitlines(keepends=True)
            end = node.end_lineno or node.lineno
            return "".join(lines[:end]) + normalize_output(code) + "".join(lines[end:])
    fail(f"Atribuição {assignment_name!r} não encontrada")


def set_string_assignment(source: str, name: str, value: str) -> str:
    tree = ast.parse(source)
    matches: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            matches.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            matches.append(node)
    if len(matches) != 1:
        fail(f"Atribuição string {name!r} ambígua ou ausente: {len(matches)}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "".join(lines[:start]) + f"{name} = {value!r}\n" + "".join(lines[end:])


def set_literal_assignment(source: str, name: str, value: Any) -> str:
    tree = ast.parse(source)
    matches: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            matches.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            matches.append(node)
    if len(matches) != 1:
        fail(f"Atribuição literal {name!r} ambígua ou ausente: {len(matches)}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "".join(lines[:start]) + f"{name} = {value!r}\n" + "".join(lines[end:])


def set_dict_string_value(source: str, assignment: str, key: str, value: str) -> str:
    tree = ast.parse(source)
    target_node: ast.Dict | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment for target in node.targets
        ) and isinstance(node.value, ast.Dict):
            target_node = node.value
            break
    if target_node is None:
        fail(f"Dicionário {assignment!r} não encontrado")
    for key_node, value_node in zip(target_node.keys, target_node.values):
        if isinstance(key_node, ast.Constant) and key_node.value == key:
            if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, str):
                fail(f"Valor {assignment}[{key!r}] não é string")
            lines = source.splitlines(keepends=True)
            start_line = value_node.lineno - 1
            end_line = value_node.end_lineno or value_node.lineno
            start_col = value_node.col_offset
            end_col = value_node.end_col_offset
            if start_line != end_line - 1:
                fail(f"Valor multilinha inesperado em {assignment}[{key!r}]")
            line = lines[start_line]
            lines[start_line] = line[:start_col] + repr(value) + line[end_col:]
            return "".join(lines)
    fail(f"Chave {key!r} não encontrada em {assignment}")


def git_blob(repository_root: Path, commit: str, relative: str) -> bytes:
    repo_path = SOFTWARE_PREFIX + relative
    completed = subprocess.run(
        ("git", "show", f"{commit}:{repo_path}"),
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            f"Não foi possível ler {relative} em {commit}: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return completed.stdout


def compute_core_dump_hash(test_path: Path, orchestrator_path: Path) -> str:
    namespace = runpy.run_path(str(test_path))
    normalizer = namespace.get("_normalized_function_dump")
    core_name = namespace.get("CORE_NAME", "_ap003f_pipeline_core")
    if not callable(normalizer):
        fail("Helper _normalized_function_dump ausente no contrato AP-003F")
    tree = ast.parse(orchestrator_path.read_text(encoding="utf-8"))
    core = next(
        (
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == core_name
        ),
        None,
    )
    if core is None:
        fail(f"Core {core_name!r} ausente no orquestrador")
    return sha256_bytes(str(normalizer(core)).encode("utf-8"))


def durable_ap004b_application_contract(source: str) -> str:
    if "def _git_blob(" not in source:
        helper = '''

def _git_blob(commit: str, relative: str) -> bytes:
    repo_path = SOFTWARE_PREFIX + relative
    result = subprocess.run(
        ("git", "show", f"{commit}:{repo_path}"),
        cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout
'''
        source = insert_after_assignment(
            source, "FORBIDDEN_FULLTEXT_CANONICAL", helper
        )

    source = replace_top_level_function(
        source,
        "test_ap004b_module_paths_follow_approved_migration_policies",
        '''def test_ap004b_module_paths_follow_approved_migration_policies() -> None:
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
''',
    )
    source = replace_top_level_function(
        source,
        "test_ap004b_public_entrypoint_control_files_are_unchanged",
        '''def test_ap004b_public_entrypoint_control_files_are_unchanged() -> None:
    data = _data()
    for relative, expected in data["unchanged_control_files"].items():
        historical = _git_blob(EXPECTED_AP004B_COMMIT, relative)
        assert hashlib.sha256(historical).hexdigest() == expected["sha256_before"]
''',
    )
    source = replace_top_level_function(
        source,
        "test_ap004b_known_xfails_and_frozen_orchestrator_remain_unchanged",
        '''def test_ap004b_known_xfails_and_frozen_orchestrator_remain_unchanged() -> None:
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
''',
    )
    return source


def durable_ap004c_inventory_contract(source: str, expected_dirty: list[str]) -> str:
    if "AP004C_APPLICATION = ROOT /" not in source:
        source = insert_after_assignment(
            source,
            "INVENTORY",
            "AP004C_APPLICATION = ROOT / "
            + repr(APPLICATION_JSON_REL.as_posix()),
        )
    if "EXPECTED_AP004C_APPLICATION_SUBJECT" not in source:
        source = insert_after_assignment(
            source,
            "EXPECTED_SUBJECT",
            "EXPECTED_AP004C_APPLICATION_SUBJECT = "
            + repr(EXPECTED_AP004C_COMMIT_SUBJECT)
            + "\nEXPECTED_AP004C_APPLICATION_OUTPUTS = "
            + repr(expected_dirty),
        )
    else:
        source = set_string_assignment(
            source,
            "EXPECTED_AP004C_APPLICATION_SUBJECT",
            EXPECTED_AP004C_COMMIT_SUBJECT,
        )
    if "def _git_blob(" not in source:
        helper = '''\n\ndef _git_blob(commit: str, relative: str) -> bytes:\n    repo_path = SOFTWARE_PREFIX + relative\n    result = subprocess.run(\n        ("git", "show", f"{commit}:{repo_path}"),\n        cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE,\n        stderr=subprocess.PIPE, check=False,\n    )\n    assert result.returncode == 0, result.stderr.decode(errors="replace")\n    return result.stdout\n'''
        source = insert_after_assignment(source, "SOFTWARE_PREFIX", helper)

    source = replace_top_level_function(
        source,
        "test_ap004c_source_manifest_matches_current_baseline",
        '''def test_ap004c_source_manifest_matches_current_baseline() -> None:\n    data = _data()\n    application = json.loads(\n        AP004C_APPLICATION.read_text(encoding="utf-8")\n    )\n    baseline = application["inventory_baseline"]\n    assert baseline["source_manifest"] == data["source_manifest"]\n    preparatory = set(baseline["preparatory_dirty_paths"])\n    for record in data["source_manifest"]:\n        if record["path"] in preparatory:\n            continue\n        historical = _git_blob(BASELINE_HEAD, record["path"])\n        assert hashlib.sha256(historical).hexdigest() == record["sha256"]\n        if record["path"].endswith(".py"):\n            tree = ast.parse(\n                historical.decode("utf-8"), filename=record["path"]\n            )\n            actual = hashlib.sha256(\n                ast.dump(\n                    tree, include_attributes=False, annotate_fields=True\n                ).encode()\n            ).hexdigest()\n            assert actual == record["ast_sha256"]\n''',
    )
    source = replace_top_level_function(
        source,
        "test_ap004c_preserves_ap003_and_ap004b_control_files",
        '''def test_ap004c_preserves_ap003_and_ap004b_control_files() -> None:\n    data = _data()\n    manifest = {record["path"]: record for record in data["source_manifest"]}\n    required = {\n        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",\n        "tests/characterization/test_ap003g_stabilization_contract.py",\n        "docs/refactor/academic-pipeline/AP-003/ap003g_manifest.json",\n        "docs/refactor/academic-pipeline/AP-004/ap004b_module_file_application.json",\n        "tests/characterization/test_ap004b_module_file_application_contract.py",\n    }\n    assert required <= set(manifest)\n    historical = _git_blob(\n        BASELINE_HEAD,\n        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",\n    )\n    assert hashlib.sha256(historical).hexdigest() == (\n        "8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977"\n    )\n    current = (\n        ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"\n    ).read_text(encoding="utf-8")\n    assert "_refs_v6_strip_org" in current\n    assert "_ap003d_impl__refs_v6_strip_org" in current\n    for relative in (\n        "tests/characterization/test_ap004b_module_file_application_contract.py",\n        "tests/characterization/test_ap004b_module_file_inventory_contract.py",\n    ):\n        contract = (ROOT / relative).read_text(encoding="utf-8")\n        assert "EXPECTED_AP004B_COMMIT" in contract\n        assert "EXPECTED_AP004B_SUBJECT" in contract\n        assert "commit_scope_is_durable" in contract\n''',
    )
    source = replace_top_level_function(
        source,
        "test_ap004c_current_status_is_output_scope_or_clean_and_commit_is_durable",
        '''def test_ap004c_current_status_is_output_scope_or_clean_and_commit_is_durable() -> None:\n    result = _run("git", "status", "--porcelain=v1", "--untracked-files=all")\n    assert result.returncode == 0, result.stderr\n    actual = {\n        _status_path(line) for line in result.stdout.splitlines()\n        if line.strip() and not _ephemeral(_status_path(line))\n    }\n    expected = set(EXPECTED_AP004C_APPLICATION_OUTPUTS)\n    if actual:\n        assert actual == expected\n        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD\n    else:\n        result = _run(\n            "git", "log", "--format=%H%x00%s", f"{BASELINE_HEAD}..HEAD"\n        )\n        assert result.returncode == 0, result.stderr\n        matches = []\n        for line in result.stdout.splitlines():\n            if "\\x00" not in line:\n                continue\n            commit, subject = line.split("\\x00", 1)\n            if subject == EXPECTED_AP004C_APPLICATION_SUBJECT:\n                matches.append(commit)\n        assert len(matches) == 1\n        changed = _run(\n            "git", "diff-tree", "--no-commit-id", "--name-only",\n            "-r", matches[0],\n        )\n        assert changed.returncode == 0, changed.stderr\n        normalized = {\n            path[len(SOFTWARE_PREFIX):]\n            if path.startswith(SOFTWARE_PREFIX) else path\n            for path in changed.stdout.splitlines() if path\n        }\n        assert normalized == expected\n        assert _run(\n            "git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD"\n        ).returncode == 0\n''',
    )
    return source


def durable_ap003d_document_contract(source: str) -> str:
    implementation_aliases = {
        old[len("_ap003d_impl_"):]: new
        for old, new in WAVE_2
    }
    replacement = f'''def test_historical_helpers_are_thin_wrappers() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ORCHESTRATOR))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    expected_wrapper_counts = {{
        name: EXPECTED_HELPERS.count(name)
        for name in set(EXPECTED_HELPERS)
    }}
    implementation_aliases = {implementation_aliases!r}

    for helper, expected_wrapper_count in expected_wrapper_counts.items():
        matches = [
            node
            for node in functions
            if node.name == helper
        ]
        assert len(matches) >= expected_wrapper_count
        expected_impl = implementation_aliases.get(
            helper, f"_ap003d_impl_{{helper}}"
        )
        thin_wrappers = []
        for match in matches:
            calls = [
                ast.unparse(node.func)
                for node in ast.walk(match)
                if isinstance(node, ast.Call)
            ]
            if expected_impl in calls:
                thin_wrappers.append(match)
        assert len(thin_wrappers) == expected_wrapper_count
'''
    return replace_top_level_function(
        source,
        "test_historical_helpers_are_thin_wrappers",
        replacement,
    )


def durable_ap003g_stabilization_contract(
    source: str,
    *,
    manifest_hashes: dict[str, str],
    new_orchestrator_hash: str,
) -> str:
    source = set_dict_string_value(
        source, "EXPECTED_HASHES", "orchestrator", new_orchestrator_hash
    )
    if "EXPECTED_AP003G_PRODUCTION_HASHES" not in source:
        source = insert_after_assignment(
            source,
            "EXPECTED_HASHES",
            "EXPECTED_AP003G_PRODUCTION_HASHES = " + repr(manifest_hashes),
        )
    else:
        source = set_literal_assignment(
            source,
            "EXPECTED_AP003G_PRODUCTION_HASHES",
            manifest_hashes,
        )
    old = 'assert data["production_hashes"] == EXPECTED_HASHES'
    new = (
        'assert data["production_hashes"] '
        '== EXPECTED_AP003G_PRODUCTION_HASHES'
    )
    if old in source:
        source = source.replace(old, new, 1)
    elif new not in source:
        fail("Assert de production_hashes AP-003G não encontrado")
    return source


def build_application_data(
    *,
    git_state: dict[str, str],
    inventory: dict[str, Any],
    backup_root: Path,
    wave1_before: str,
    wave1_after: str,
    wave1_counts: Counter[str],
    wave1_trace: dict[int, str],
    wave2_before: str,
    wave2_after: str,
    wave2_counts: Counter[str],
    wave2_trace: dict[int, str],
    protected_before: dict[str, str],
    protected_after: dict[str, str],
    changed_contracts: list[str],
    changed_paths: list[str],
    tool_sha256: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "mode": MODE,
        "application_schema_version": APPLICATION_SCHEMA_VERSION,
        "application_revision": TOOL_REVISION,
        "generated_at_utc": utc_now(),
        "baseline": {
            "branch": git_state["branch"],
            "head": git_state["head"],
            "remote_head": git_state["remote_head"],
            "subject": git_state["subject"],
            "passed": BASELINE_PASSED,
            "xfailed": BASELINE_XFAILED,
        },
        "approval": {
            "inventory_mode": inventory["mode"],
            "inventory_revision": inventory["inventory_revision"],
            "approved": True,
            "approved_scope": "7 símbolos na onda 1 e 13 aliases na onda 2",
        },
        "inventory_baseline": {
            "source_manifest": inventory["source_manifest"],
            "preparatory_dirty_paths": sorted(PREPARATORY_DIRTY_PATHS),
            "validated_before_first_write": True,
        },
        "waves": {
            "wave_1": {
                "name": "local-private-safe",
                "path": TOML_GENERATOR_REL.as_posix(),
                "mappings": [
                    {"old": old, "new": new, "replacement_count": wave1_counts[old]}
                    for old, new in WAVE_1
                ],
                "source_sha256_before": sha256_bytes(wave1_before.encode("utf-8")),
                "source_sha256_after": sha256_bytes(wave1_after.encode("utf-8")),
                "ast_sha256_before": normalized_ast_sha256(wave1_before),
                "ast_sha256_after_normalized": targeted_normalized_ast_sha256(
                    wave1_after, dict(WAVE_1), wave1_trace
                ),
                "checkpoint": "passed",
            },
            "wave_2": {
                "name": "contract-bound-safe",
                "path": ORCHESTRATOR_REL.as_posix(),
                "mappings": [
                    {"old": old, "new": new, "replacement_count": wave2_counts[old]}
                    for old, new in WAVE_2
                ],
                "source_sha256_before": sha256_bytes(wave2_before.encode("utf-8")),
                "source_sha256_after": sha256_bytes(wave2_after.encode("utf-8")),
                "ast_sha256_before": normalized_ast_sha256(wave2_before),
                "ast_sha256_after_normalized": targeted_normalized_ast_sha256(
                    wave2_after, dict(WAVE_2), wave2_trace
                ),
                "checkpoint": "passed",
            },
        },
        "protected_controls": [
            {
                "name": name,
                "path": path,
                "ast_dump_sha256_before": sha256_bytes(protected_before[name].encode()),
                "ast_dump_sha256_after": sha256_bytes(protected_after[name].encode()),
            }
            for name, path in PROTECTED_CONTROLS
        ],
        "deferred": {
            "count": inventory["statistics"]["deferred_count"],
            "policy": "não alterados",
        },
        "contract_updates": changed_contracts,
        "scope": {
            "productive_changed_paths": [
                TOML_GENERATOR_REL.as_posix(), ORCHESTRATOR_REL.as_posix()
            ],
            "all_changed_paths": changed_paths,
            "application_artifacts": sorted(APPLICATION_ARTIFACT_PATHS),
            "known_xfails": list(KNOWN_XFAILS),
            "module_file_changes": False,
            "expected_commit_subject": EXPECTED_AP004C_COMMIT_SUBJECT,
        },
        "backup": {"path": str(backup_root)},
        "validation": {
            "py_compile": "pending",
            "git_diff_check": "pending",
            "application_contract": {"status": "pending"},
            "specific_suite": {"status": "pending"},
            "consolidated_suite": {"status": "pending"},
        },
        "tool": {
            "path": APPLICATION_TOOL_REL.as_posix(),
            "version": TOOL_VERSION,
            "revision": TOOL_REVISION,
            "sha256": tool_sha256,
        },
        "next_gate": {
            "blocked": True,
            "message": "Não consolidar sem revisão do diff e aprovação expressa.",
        },
    }


def build_report(data: dict[str, Any]) -> str:
    validation = data["validation"]
    lines = [
        "# AP-004C — aplicação produtiva de símbolos internos (v1.4)",
        "",
        "> Aplicação em duas ondas atômicas. Nenhum commit foi criado.",
        "",
        "## Base canônica",
        "",
        f"- Branch: `{data['baseline']['branch']}`.",
        f"- HEAD local/remoto: `{data['baseline']['head']}`.",
        f"- Inventário aprovado: `{data['approval']['inventory_mode']}`.",
        f"- Baseline: `{data['baseline']['passed']} passed, {data['baseline']['xfailed']} xfailed`.",
        "",
        "## Onda 1 — símbolos locais",
        "",
        f"Arquivo: `{data['waves']['wave_1']['path']}`.",
        "",
    ]
    for item in data["waves"]["wave_1"]["mappings"]:
        lines.append(f"- `{item['old']}` → `{item['new']}` ({item['replacement_count']} tokens).")
    lines.extend([
        "",
        "## Onda 2 — aliases vinculados a contratos",
        "",
        f"Arquivo: `{data['waves']['wave_2']['path']}`.",
        "",
    ])
    for item in data["waves"]["wave_2"]["mappings"]:
        lines.append(f"- `{item['old']}` → `{item['new']}` ({item['replacement_count']} tokens).")
    lines.extend([
        "",
        "A AST normalizada das duas ondas permanece idêntica ao baseline; somente os identificadores aprovados foram alterados.",
        "",
        "## Contratos atualizados",
        "",
    ])
    for path in data["contract_updates"]:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## Proteções",
        "",
        "- `_refs_v6_strip_org` preservado.",
        "- `_ap003d_impl__refs_v6_strip_org` preservado.",
        "- `WorkflowState._normalize` preservado.",
        "- `extract_org_abstracts` preservado.",
        f"- Símbolos adiados: **{data['deferred']['count']}**, sem alteração.",
        "- Nenhum módulo, arquivo ou diretório foi renomeado.",
        "",
        "## Validação",
        "",
        f"- `py_compile`: `{validation['py_compile']}`.",
        f"- `git diff --check`: `{validation['git_diff_check']}`.",
        f"- Contrato da aplicação: `{validation['application_contract'].get('summary', validation['application_contract']['status'])}`.",
        f"- Suíte específica: `{validation['specific_suite'].get('summary', validation['specific_suite']['status'])}`.",
        f"- Suíte consolidada: `{validation['consolidated_suite'].get('summary', validation['consolidated_suite']['status'])}`.",
        "",
        "## Estado",
        "",
        "A consolidação permanece bloqueada até revisão do diff e aprovação expressa.",
    ])
    return normalize_output("\n".join(lines))


def build_application_contract(data: dict[str, Any], tool_sha256: str) -> str:
    dirty = data["scope"]["all_changed_paths"]
    wave1 = list(WAVE_1)
    wave2 = list(WAVE_2)
    protected = list(PROTECTED_CONTROLS)
    return normalize_output(f'''from __future__ import annotations

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
APPLICATION = ROOT / {APPLICATION_JSON_REL.as_posix()!r}
TOOL = ROOT / {APPLICATION_TOOL_REL.as_posix()!r}
BASELINE_HEAD = {EXPECTED_HEAD!r}
EXPECTED_COMMIT_SUBJECT = {EXPECTED_AP004C_COMMIT_SUBJECT!r}
EXPECTED_TOOL_SHA256 = {tool_sha256!r}
EXPECTED_DIRTY_PATHS = {dirty!r}
WAVE_1 = {wave1!r}
WAVE_2 = {wave2!r}
PROTECTED = {protected!r}
SOFTWARE_PREFIX = {SOFTWARE_PREFIX!r}


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
    raw = raw.strip().strip('"').replace("\\\\", "/")
    return raw[len(SOFTWARE_PREFIX):] if raw.startswith(SOFTWARE_PREFIX) else raw


def _ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {{"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}}
    return any(part in ignored for part in parts) or path.endswith((".pyc", ".pyo"))


def _find_commit() -> str | None:
    result = _run("git", "log", "--format=%H%x00%s", f"{{BASELINE_HEAD}}..HEAD")
    assert result.returncode == 0, result.stderr
    matches = []
    for line in result.stdout.splitlines():
        if "\\x00" not in line:
            continue
        commit, subject = line.split("\\x00", 1)
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
    names = {{name for pair in WAVE_1 for name in pair}}
    counts = _identifier_counts(path, names)
    for old, new in WAVE_1:
        assert counts[old] == 0
        assert counts[new] >= 1


def test_ap004c_wave_2_old_identifiers_are_absent_and_new_are_present() -> None:
    path = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    names = {{name for pair in WAVE_2 for name in pair}}
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
    source = (ROOT / "tests/characterization/test_ap003g_stabilization_contract.py").read_text(encoding="utf-8")
    assert expected in source
    assert _sha256(ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py") == expected


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
                cfile=str(Path(tmp) / f"{{index}}.pyc"),
                doraise=True,
            )


def test_ap004c_validation_metadata_records_expected_suite() -> None:
    data = _data()
    consolidated = data["validation"]["consolidated_suite"]
    assert consolidated["status"] in {{"pending", "passed"}}
    if consolidated["status"] == "passed":
        assert consolidated["passed"] == {EXPECTED_CONSOLIDATED_PASSED}
        assert consolidated["xfailed"] == {BASELINE_XFAILED}


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
    actual = {{
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }}
    expected = set(EXPECTED_DIRTY_PATHS)
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
    assert count == {EXPECTED_APPLICATION_TESTS}
''')


def count_test_functions(source: str) -> int:
    tree = ast.parse(source)
    return sum(
        1 for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def parse_pytest_summary(result: CommandResult, *, label: str) -> dict[str, Any]:
    text = result.stdout + "\n" + result.stderr
    patterns = {
        "passed": r"(\d+) passed",
        "failed": r"(\d+) failed",
        "xfailed": r"(\d+) xfailed",
        "skipped": r"(\d+) skipped",
        "errors": r"(\d+) errors?",
    }
    counts = {
        name: max([int(value) for value in re.findall(pattern, text)] or [0])
        for name, pattern in patterns.items()
    }
    if not any(counts.values()) and result.returncode == 0:
        fail(f"Não foi possível interpretar o resumo pytest da suíte {label}.\n{text}")
    summary_parts = []
    for name in ("passed", "failed", "xfailed", "skipped", "errors"):
        if counts[name]:
            summary_parts.append(f"{counts[name]} {name}")
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        **counts,
        "summary": ", ".join(summary_parts),
    }


def run_pytest(
    software_root: Path,
    paths: Sequence[str],
    *,
    label: str,
    timeout: int = 1200,
) -> dict[str, Any]:
    result = run(
        ("pipenv", "run", "pytest", "-q", "-ra", *paths),
        cwd=software_root,
        check=False,
        timeout=timeout,
    )
    parsed = parse_pytest_summary(result, label=label)
    if result.returncode != 0:
        fail(f"Suíte {label} falhou:\n{result.stdout}{result.stderr}")
    return parsed


def py_compile_paths(software_root: Path, paths: Iterable[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="ap004c-compile-") as tmp:
        for index, relative in enumerate(sorted(set(paths))):
            if not relative.endswith(".py"):
                continue
            path = software_root / relative
            py_compile.compile(
                str(path), cfile=str(Path(tmp) / f"{index}.pyc"), doraise=True
            )


def git_diff_check(repository_root: Path) -> None:
    result = git(repository_root, "diff", "--check", check=False)
    if result.returncode != 0 or result.stdout.strip() or result.stderr.strip():
        fail(f"git diff --check falhou:\n{result.stdout}{result.stderr}")


def validate_final_status(repository_root: Path, expected: list[str]) -> None:
    unstaged, staged = status_paths(repository_root)
    if staged:
        fail(f"Nenhum arquivo pode estar staged: {staged}")
    if unstaged != sorted(expected):
        fail(
            "Estado final fora do escopo AP-004C aplicação.\n"
            f"Atual: {unstaged}\nEsperado: {sorted(expected)}"
        )


def candidate_paths_for_backup(software_root: Path) -> set[str]:
    paths = set(PREPARATORY_DIRTY_PATHS)
    paths.update({TOML_GENERATOR_REL.as_posix(), ORCHESTRATOR_REL.as_posix()})
    paths.update(APPLICATION_ARTIFACT_PATHS)
    paths.update({AP004B_APPLICATION_TEST_REL.as_posix(), INVENTORY_TEST_REL.as_posix()})
    contracts = software_root / "tests/characterization"
    for path in contracts.glob("test_ap003*.py"):
        paths.add(path.relative_to(software_root).as_posix())
    return paths


def apply() -> None:
    software_root, repository_root = resolve_roots()
    git_state = validate_git_state(software_root, repository_root)
    inventory = validate_inventory(software_root)

    # Pré-validação integral das duas ondas antes de qualquer backup ou escrita.
    wave1_path = software_root / TOML_GENERATOR_REL
    wave2_path = software_root / ORCHESTRATOR_REL
    wave1_before = wave1_path.read_text(encoding="utf-8")
    wave2_before = wave2_path.read_text(encoding="utf-8")
    wave1_after, wave1_counts, wave1_trace, wave1_preflight = preflight_wave(
        source=wave1_before,
        mappings=WAVE_1,
        inventory=inventory,
        relative_path=TOML_GENERATOR_REL.as_posix(),
        label="Onda 1",
    )
    wave2_after, wave2_counts, wave2_trace, wave2_preflight = preflight_wave(
        source=wave2_before,
        mappings=WAVE_2,
        inventory=inventory,
        relative_path=ORCHESTRATOR_REL.as_posix(),
        label="Onda 2",
    )
    if len(wave1_preflight) != 7 or len(wave2_preflight) != 13:
        fail("Pré-validação integral não confirmou os 20 símbolos aprovados")

    # Mapas usados somente após a pré-validação integral. Mantê-los explícitos
    # evita dependência de variáveis implícitas entre as duas ondas.
    wave1_map = dict(WAVE_1)
    wave2_map = dict(WAVE_2)
    if len(wave1_map) != 7 or len(wave2_map) != 13:
        fail("Mapas das ondas divergentes após a pré-validação integral")

    tool_source = normalize_output(Path(__file__).read_text(encoding="utf-8"))
    tool_sha256 = sha256_bytes(tool_source.encode("utf-8"))

    backup_paths = candidate_paths_for_backup(software_root)
    backup_root, backup_records = create_backup(software_root, backup_paths)

    try:
        # Provas de proteção antes de qualquer escrita.
        protected_before: dict[str, str] = {}
        for name, relative in PROTECTED_CONTROLS:
            protected_before[name] = ast_definition_dump(
                (software_root / relative).read_text(encoding="utf-8"), name
            )

        # Onda 1 — já integralmente simulada antes de qualquer escrita.
        write_preserving_mode(wave1_path, wave1_after)
        py_compile_paths(software_root, [TOML_GENERATOR_REL.as_posix()])

        # Onda 2 — já integralmente simulada antes de qualquer escrita.
        write_preserving_mode(wave2_path, wave2_after)

        changed_contracts: set[str] = set()

        # Atualização dirigida de contratos AP-003 que referenciem nomes exatos.
        for contract_path in sorted((software_root / "tests/characterization").glob("test_ap003*.py")):
            relative = contract_path.relative_to(software_root).as_posix()
            source = contract_path.read_text(encoding="utf-8")
            updated, changes = rewrite_exact_names_in_python_contract(
                source,
                wave2_map,
                replace_generic_prefix=(relative == AP003D_TEST_REL.as_posix()),
            )
            if changes:
                ast.parse(updated, filename=relative)
                write_preserving_mode(contract_path, updated)
                changed_contracts.add(relative)

        # O contrato AP-003D usa uma f-string de prefixo histórico; converte-se
        # para uma matriz explícita que preserva o alias protegido do xfail.
        ap003d_path = software_root / AP003D_TEST_REL
        source = ap003d_path.read_text(encoding="utf-8")
        updated = durable_ap003d_document_contract(source)
        if updated != source:
            ast.parse(updated, filename=str(ap003d_path))
            write_preserving_mode(ap003d_path, updated)
            changed_contracts.add(AP003D_TEST_REL.as_posix())

        # Rebaseline dirigido do hash do corpo do core AP-003F.
        ap003f_path = software_root / AP003F_TEST_REL
        if ap003f_path.is_file():
            source = ap003f_path.read_text(encoding="utf-8")
            new_core_hash = compute_core_dump_hash(ap003f_path, wave2_path)
            updated = set_string_assignment(source, "EXPECTED_CORE_DUMP_SHA256", new_core_hash)
            if updated != source:
                ast.parse(updated, filename=str(ap003f_path))
                write_preserving_mode(ap003f_path, updated)
                changed_contracts.add(AP003F_TEST_REL.as_posix())

        # Rebaseline corrente do orquestrador sem reescrever o manifesto
        # histórico da AP-003G. O contrato distingue os dois estados.
        ap003g_path = software_root / AP003G_TEST_REL
        if ap003g_path.is_file():
            source = ap003g_path.read_text(encoding="utf-8")
            manifest_path = (
                software_root
                / "docs/refactor/academic-pipeline/AP-003/ap003g_manifest.json"
            )
            manifest_hashes = load_json(manifest_path)["production_hashes"]
            if (
                manifest_hashes.get("orchestrator")
                != AP003G_HISTORICAL_ORCHESTRATOR_SHA256
            ):
                fail("Hash histórico do orquestrador AP-003G divergente")
            new_orchestrator_hash = sha256_path(wave2_path)
            updated = durable_ap003g_stabilization_contract(
                source,
                manifest_hashes=manifest_hashes,
                new_orchestrator_hash=new_orchestrator_hash,
            )
            if updated != source:
                ast.parse(updated, filename=str(ap003g_path))
                write_preserving_mode(ap003g_path, updated)
                changed_contracts.add(AP003G_TEST_REL.as_posix())

        # Contrato AP-004B passa a validar historicamente o controle congelado.
        ap004b_app_path = software_root / AP004B_APPLICATION_TEST_REL
        source = ap004b_app_path.read_text(encoding="utf-8")
        updated = durable_ap004b_application_contract(source)
        if updated != source:
            ast.parse(updated, filename=str(ap004b_app_path))
            write_preserving_mode(ap004b_app_path, updated)
            changed_contracts.add(AP004B_APPLICATION_TEST_REL.as_posix())

        protected_after: dict[str, str] = {}
        for name, relative in PROTECTED_CONTROLS:
            protected_after[name] = ast_definition_dump(
                (software_root / relative).read_text(encoding="utf-8"), name
            )
            if protected_after[name] != protected_before[name]:
                fail(f"Controle protegido foi alterado: {name}")

        # Descobre o escopo antes de gerar o contrato durável da própria fase.
        provisional_paths = set(PREPARATORY_DIRTY_PATHS)
        provisional_paths.update({TOML_GENERATOR_REL.as_posix(), ORCHESTRATOR_REL.as_posix()})
        provisional_paths.update(changed_contracts)
        provisional_paths.update(APPLICATION_ARTIFACT_PATHS)
        expected_dirty = sorted(provisional_paths)

        inventory_test_path = software_root / INVENTORY_TEST_REL
        source = inventory_test_path.read_text(encoding="utf-8")
        updated = durable_ap004c_inventory_contract(source, expected_dirty)
        if updated != source:
            ast.parse(updated, filename=str(inventory_test_path))
            write_preserving_mode(inventory_test_path, updated)
            changed_contracts.add(INVENTORY_TEST_REL.as_posix())

        final_paths = set(PREPARATORY_DIRTY_PATHS)
        final_paths.update({TOML_GENERATOR_REL.as_posix(), ORCHESTRATOR_REL.as_posix()})
        final_paths.update(changed_contracts)
        final_paths.update(APPLICATION_ARTIFACT_PATHS)
        changed_paths = sorted(final_paths)

        # Reaplica o contrato AP-004C com o conjunto final exato, caso tenha crescido.
        source = inventory_test_path.read_text(encoding="utf-8")
        if "EXPECTED_AP004C_APPLICATION_OUTPUTS" in source:
            tree = ast.parse(source)
            target = next(
                (
                    node for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "EXPECTED_AP004C_APPLICATION_OUTPUTS" for t in node.targets)
                ),
                None,
            )
            if target is not None:
                lines = source.splitlines(keepends=True)
                start = target.lineno - 1
                end = target.end_lineno or target.lineno
                source = "".join(lines[:start]) + f"EXPECTED_AP004C_APPLICATION_OUTPUTS = {changed_paths!r}\n" + "".join(lines[end:])
                ast.parse(source, filename=str(inventory_test_path))
                write_preserving_mode(inventory_test_path, source)

        application_data = build_application_data(
            git_state=git_state,
            inventory=inventory,
            backup_root=backup_root,
            wave1_before=wave1_before,
            wave1_after=wave1_after,
            wave1_counts=wave1_counts,
            wave1_trace=wave1_trace,
            wave2_before=wave2_before,
            wave2_after=wave2_after,
            wave2_counts=wave2_counts,
            wave2_trace=wave2_trace,
            protected_before=protected_before,
            protected_after=protected_after,
            changed_contracts=sorted(changed_contracts),
            changed_paths=changed_paths,
            tool_sha256=tool_sha256,
        )

        tool_path = software_root / APPLICATION_TOOL_REL
        atomic_write_text(tool_path, tool_source, mode=0o755)
        app_json_path = software_root / APPLICATION_JSON_REL
        atomic_write_text(
            app_json_path,
            json.dumps(application_data, ensure_ascii=False, indent=2),
        )
        app_report_path = software_root / APPLICATION_REPORT_REL
        atomic_write_text(app_report_path, build_report(application_data))
        app_contract_source = build_application_contract(application_data, tool_sha256)
        if count_test_functions(app_contract_source) != EXPECTED_APPLICATION_TESTS:
            fail(
                "Quantidade de testes do contrato de aplicação divergente: "
                f"{count_test_functions(app_contract_source)}"
            )
        app_test_path = software_root / APPLICATION_TEST_REL
        atomic_write_text(app_test_path, app_contract_source)

        py_compile_paths(software_root, changed_paths)
        git_diff_check(repository_root)
        validate_final_status(repository_root, changed_paths)

        app_contract_result = run_pytest(
            software_root,
            [APPLICATION_TEST_REL.as_posix()],
            label="AP-004C contrato da aplicação",
            timeout=900,
        )
        if app_contract_result["passed"] != EXPECTED_APPLICATION_TESTS:
            fail(
                "Contrato de aplicação AP-004C com contagem inesperada: "
                f"{app_contract_result}"
            )

        specific_paths = sorted(
            {
                APPLICATION_TEST_REL.as_posix(),
                INVENTORY_TEST_REL.as_posix(),
                AP004B_APPLICATION_TEST_REL.as_posix(),
                AP004B_INVENTORY_TEST_REL.as_posix(),
                *(
                    path.relative_to(software_root).as_posix()
                    for path in (software_root / "tests/characterization").glob("test_ap003*.py")
                ),
                "app_bundle/tests/test_entrypoints_orchestration_characterization.py",
                "app_bundle/tests/test_package_imports_entrypoints.py",
                "app_bundle/tests/test_rc10_configuration_characterization.py",
            }
        )
        specific_result = run_pytest(
            software_root,
            specific_paths,
            label="AP-004C específica",
            timeout=1200,
        )

        consolidated_result = run_pytest(
            software_root,
            ["app_bundle/tests", "tests"],
            label="AP-004C consolidada",
            timeout=1800,
        )
        if consolidated_result["passed"] != EXPECTED_CONSOLIDATED_PASSED:
            fail(
                "Suíte consolidada com total inesperado: "
                f"{consolidated_result['passed']} != {EXPECTED_CONSOLIDATED_PASSED}"
            )
        if consolidated_result["xfailed"] != BASELINE_XFAILED:
            fail(
                "Quantidade de xfails consolidada divergente: "
                f"{consolidated_result['xfailed']} != {BASELINE_XFAILED}"
            )

        application_data["validation"] = {
            "py_compile": "passed",
            "git_diff_check": "passed",
            "application_contract": app_contract_result,
            "specific_suite": specific_result,
            "consolidated_suite": consolidated_result,
        }
        atomic_write_text(
            app_json_path,
            json.dumps(application_data, ensure_ascii=False, indent=2),
        )
        atomic_write_text(app_report_path, build_report(application_data))

        # Validação final após registrar resultados.
        final_contract = run_pytest(
            software_root,
            [APPLICATION_TEST_REL.as_posix()],
            label="AP-004C contrato final",
            timeout=900,
        )
        if final_contract["passed"] != EXPECTED_APPLICATION_TESTS:
            fail("Contrato final AP-004C não permaneceu estável")
        py_compile_paths(software_root, changed_paths)
        git_diff_check(repository_root)
        validate_final_status(repository_root, changed_paths)

        print("[OK] Aplicação produtiva AP-004C concluída sem commit.")
        print(f"[OK] Branch: {git_state['branch']}")
        print(f"[OK] HEAD local/remoto: {git_state['head']}")
        print("[OK] Pré-validação integral das duas ondas: 20/20 símbolos")
        print("[OK] Onda 1 — símbolos locais renomeados: 7")
        print("[OK] Onda 2 — aliases contratuais renomeados: 13")
        print("[OK] Controles protegidos preservados: 4")
        print(f"[OK] Símbolos adiados preservados: {inventory['statistics']['deferred_count']}")
        print(f"[OK] Contratos atualizados: {len(changed_contracts)}")
        print(f"[OK] Relatório: {APPLICATION_REPORT_REL}")
        print(f"[OK] JSON: {APPLICATION_JSON_REL}")
        print(f"[OK] Teste: {APPLICATION_TEST_REL}")
        print(f"[OK] Ferramenta reexecutável: {APPLICATION_TOOL_REL}")
        print(f"[OK] Backup externo: {backup_root}")
        print(f"[OK] Contrato da aplicação: {app_contract_result['summary']}")
        print(f"[OK] Suíte específica: {specific_result['summary']}")
        print(f"[OK] Suíte consolidada: {consolidated_result['summary']}")
        print("[OK] Nenhum commit foi criado.")
        print("[BLOQUEIO] Não consolidar a AP-004C sem revisão do diff e aprovação expressa.")
    except Exception:
        rollback(software_root, backup_records)
        raise


def assert_no_unresolved_global_names(source: str) -> None:
    """Falha cedo quando o aplicador referencia um global inexistente.

    ``py_compile`` não detecta nomes resolvidos somente em runtime. Esta
    auditoria percorre a tabela de símbolos de todas as funções e bloqueia
    referências globais que não sejam importadas, definidas no módulo,
    built-ins ou nomes especiais fornecidos pelo interpretador.
    """
    table = symtable.symtable(source, "<ap004c-application>", "exec")
    module_defined = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if (
            symbol.is_assigned()
            or symbol.is_imported()
            or symbol.is_namespace()
            or symbol.is_parameter()
        )
    }
    allowed = set(dir(builtins)) | {
        "__file__",
        "__name__",
        "__package__",
        "__spec__",
        "__loader__",
        "__cached__",
        "__builtins__",
    }
    unresolved: set[tuple[str, int, str]] = set()

    def visit(current: symtable.SymbolTable) -> None:
        for symbol in current.get_symbols():
            if (
                symbol.is_referenced()
                and symbol.is_global()
                and symbol.get_name() not in module_defined
                and symbol.get_name() not in allowed
            ):
                unresolved.add(
                    (current.get_name(), current.get_lineno(), symbol.get_name())
                )
        for child in current.get_children():
            visit(child)

    visit(table)
    if unresolved:
        details = "; ".join(
            f"{scope}:{line}: {name}"
            for scope, line, name in sorted(unresolved)
        )
        fail(f"Referências globais não resolvidas no aplicador: {details}")


def self_test() -> None:
    assert_no_unresolved_global_names(Path(__file__).read_text(encoding="utf-8"))

    source = normalize_output('''\nfrom module import value as _ap003d_impl_output_paths\n\ndef _v5_is_local_document(path):\n    return _v5_is_local_document\n''')
    mapping = {
        "_ap003d_impl_output_paths": "_impl_output_paths",
        "_v5_is_local_document": "_is_local_document",
    }
    rewritten, counts, trace = rewrite_identifier_tokens_traced(source, mapping)
    assert counts == Counter({
        "_v5_is_local_document": 2,
        "_ap003d_impl_output_paths": 1,
    })
    assert normalized_ast_sha256(source) == targeted_normalized_ast_sha256(
        rewritten, mapping, trace
    )

    nested_only = normalize_output(
        "def outer():\n"
        "    def _ensure_reference_policy(value):\n"
        "        return value\n"
        "    return _ensure_reference_policy\n\n"
        "def _v5_ensure_reference_policy(value):\n"
        "    return value\n\n"
        "RESULT = _v5_ensure_reference_policy(True)\n"
    )
    bindings = module_level_bound_names(nested_only)
    assert "_v5_ensure_reference_policy" in bindings
    assert "_ensure_reference_policy" not in bindings
    nested_rewritten, nested_counts, nested_trace = rewrite_identifier_tokens_traced(
        nested_only, {"_v5_ensure_reference_policy": "_ensure_reference_policy"}
    )
    assert nested_counts["_v5_ensure_reference_policy"] == 2
    assert normalized_ast_sha256(nested_only) == targeted_normalized_ast_sha256(
        nested_rewritten,
        {"_v5_ensure_reference_policy": "_ensure_reference_policy"},
        nested_trace,
    )
    import_inside_function = normalize_output(
        "def configure():\n"
        "    from module import value as _ap003d_impl_output_paths\n"
        "    return _ap003d_impl_output_paths\n"
    )
    import_entries = collect_binding_entries(import_inside_function)
    assert any(
        entry.name == "_ap003d_impl_output_paths"
        and entry.kind == "import_alias"
        and entry.scope == "configure"
        for entry in import_entries
    )
    synthetic_inventory = {
        "candidates": [
            {
                "current_name": "_ap003d_impl_output_paths",
                "path": ORCHESTRATOR_REL.as_posix(),
                "definition": {
                    "definition_kind": "import_alias",
                    "scope": "configure",
                    "line": 2,
                    "ast_sha256": next(
                        entry.ast_sha256
                        for entry in import_entries
                        if entry.name == "_ap003d_impl_output_paths"
                    ),
                },
            }
        ]
    }
    simulated, simulated_counts, _sim_trace, reports = preflight_wave(
        source=import_inside_function,
        mappings=(("_ap003d_impl_output_paths", "_impl_output_paths"),),
        inventory=synthetic_inventory,
        relative_path=ORCHESTRATOR_REL.as_posix(),
        label="autoteste onda 2",
    )
    assert simulated_counts["_ap003d_impl_output_paths"] == 2
    assert len(reports) == 1
    assert "_impl_output_paths" in simulated

    # Exercita integralmente as duas ondas em memória. Isso evita que um nome
    # posterior da matriz só seja descoberto depois de corrigido o primeiro.
    wave1_synthetic = normalize_output(
        "\n".join(
            [
                f"def {old}(value=None):\n    return value"
                for old, _new in WAVE_1
            ]
            + [
                "WAVE1_VALUES = ("
                + ", ".join(old for old, _new in WAVE_1)
                + ",)"
            ]
        )
        + "\n"
    )
    wave1_entries = collect_binding_entries(wave1_synthetic)
    wave1_inventory = {
        "candidates": [
            {
                "current_name": old,
                "path": TOML_GENERATOR_REL.as_posix(),
                "definition": {
                    "definition_kind": next(
                        entry.kind for entry in wave1_entries if entry.name == old
                    ),
                    "scope": next(
                        entry.scope for entry in wave1_entries if entry.name == old
                    ),
                    "line": next(
                        entry.line for entry in wave1_entries if entry.name == old
                    ),
                    "ast_sha256": next(
                        entry.ast_sha256
                        for entry in wave1_entries
                        if entry.name == old
                    ),
                },
            }
            for old, _new in WAVE_1
        ]
    }
    wave1_full_after, wave1_full_counts, _wave1_full_trace, wave1_reports = (
        preflight_wave(
            source=wave1_synthetic,
            mappings=WAVE_1,
            inventory=wave1_inventory,
            relative_path=TOML_GENERATOR_REL.as_posix(),
            label="autoteste integral onda 1",
        )
    )
    assert len(wave1_reports) == len(WAVE_1) == 7
    assert set(wave1_full_counts) == {old for old, _new in WAVE_1}
    assert all(new in wave1_full_after for _old, new in WAVE_1)

    wave2_import_lines = [
        f"    from module_{index} import value as {old}"
        for index, (old, _new) in enumerate(WAVE_2, start=1)
    ]
    wave2_synthetic = normalize_output(
        "def configure():\n"
        + "\n".join(wave2_import_lines)
        + "\n    return ("
        + ", ".join(old for old, _new in WAVE_2)
        + ",)\n"
    )
    wave2_entries = collect_binding_entries(wave2_synthetic)
    wave2_inventory = {
        "candidates": [
            {
                "current_name": old,
                "path": ORCHESTRATOR_REL.as_posix(),
                "definition": {
                    "definition_kind": next(
                        entry.kind for entry in wave2_entries if entry.name == old
                    ),
                    "scope": next(
                        entry.scope for entry in wave2_entries if entry.name == old
                    ),
                    "line": next(
                        entry.line for entry in wave2_entries if entry.name == old
                    ),
                    "ast_sha256": next(
                        entry.ast_sha256
                        for entry in wave2_entries
                        if entry.name == old
                    ),
                },
            }
            for old, _new in WAVE_2
        ]
    }
    wave2_full_after, wave2_full_counts, _wave2_full_trace, wave2_reports = (
        preflight_wave(
            source=wave2_synthetic,
            mappings=WAVE_2,
            inventory=wave2_inventory,
            relative_path=ORCHESTRATOR_REL.as_posix(),
            label="autoteste integral onda 2",
        )
    )
    assert len(wave2_reports) == len(WAVE_2) == 13
    assert set(wave2_full_counts) == {old for old, _new in WAVE_2}
    assert all(new in wave2_full_after for _old, new in WAVE_2)

    # Regressões contratuais descobertas durante a aplicação real.
    ap003d_contract = (
        "from pathlib import Path\nimport ast\n"
        "ORCHESTRATOR = Path('x')\n"
        "EXPECTED_HELPERS = ['output_paths', '_refs_v6_strip_org']\n"
        "def test_historical_helpers_are_thin_wrappers() -> None:\n"
        "    pass\n"
    )
    ap003d_updated = durable_ap003d_document_contract(ap003d_contract)
    ast.parse(ap003d_updated)
    assert "_impl_output_paths" in ap003d_updated
    assert "_ap003d_impl_{helper}" in ap003d_updated

    ap003g_contract = (
        "EXPECTED_HASHES = {'orchestrator': 'old', 'parser': 'p'}\n"
        "def test_ap003g_manifest_freezes_final_contract() -> None:\n"
        "    data = {'production_hashes': {}}\n"
        '    assert data["production_hashes"] == EXPECTED_HASHES\n'
    )
    ap003g_updated = durable_ap003g_stabilization_contract(
        ap003g_contract,
        manifest_hashes={'orchestrator': 'historical', 'parser': 'p'},
        new_orchestrator_hash='current',
    )
    ast.parse(ap003g_updated)
    assert "EXPECTED_AP003G_PRODUCTION_HASHES" in ap003g_updated

    protected = "_ap003d_impl__refs_v6_strip_org"
    assert protected in rewrite_exact_names_in_python_contract(
        repr(protected), dict(WAVE_2)
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica a AP-004C em duas ondas atômicas, sem commit."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="executa somente autotestes internos e encerra",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        self_test()
        if args.self_test:
            print("[OK] Autotestes internos AP-004C aprovados.")
            return 0
        apply()
        return 0
    except ApplicationError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERRO] Falha inesperada: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inventário preparatório da AP-004E — superfícies de compatibilidade.

Este utilitário é deliberadamente não produtivo: ele somente lê o repositório e
escreve os artefatos de inventário, estratégia e caracterização previstos para
a AP-004E. Ele não renomeia, remove ou altera código de produção.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "ap004e.compatibility-inventory.v6"
EXPECTED_REPOSITORY = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline"
)
EXPECTED_BRANCH = "ap-refactor/03-orchestrator-decomposition"
EXPECTED_HEAD = "389f0ae526d12327a58ce23937225cf05b032566"
EXPECTED_SUBJECT = "refactor(academic-pipeline): consolidar marcadores de versão da AP-004D"
REMOTE_REF = "origin/ap-refactor/03-orchestrator-decomposition"
SOFTWARE_REL = Path("software/academic_pipeline_mppg")
AP004_DOCS_REL = Path("docs/refactor/academic-pipeline/AP-004")
REFACTOR_TOOLS_REL = Path("tools/refactor")
SCAN_ROOT_RELS = (SOFTWARE_REL, AP004_DOCS_REL, REFACTOR_TOOLS_REL)

TOOL_REL = Path("tools/refactor/ap004e_inventory_compatibility.py")
INVENTORY_MD_REL = Path(
    "docs/refactor/academic-pipeline/AP-004/AP-004E_COMPATIBILITY_INVENTORY.md"
)
STRATEGY_MD_REL = Path(
    "docs/refactor/academic-pipeline/AP-004/AP-004E_COMPATIBILITY_STRATEGY.md"
)
INVENTORY_JSON_REL = Path(
    "docs/refactor/academic-pipeline/AP-004/ap004e_compatibility_inventory.json"
)
TEST_REL = SOFTWARE_REL / Path(
    "tests/characterization/test_ap004e_compatibility_inventory_contract.py"
)
OUTPUT_RELS = (INVENTORY_MD_REL, STRATEGY_MD_REL, INVENTORY_JSON_REL, TEST_REL)
ALLOWED_DIRTY_RELS = frozenset((*OUTPUT_RELS, TOOL_REL))
SCAN_EXCLUDED_RELS = ALLOWED_DIRTY_RELS

PUBLIC_SURFACES = (
    "script direto: app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    "python -m academic_pipeline",
    "console script: academic-pipeline",
)
PROTECTED_SYMBOLS = (
    "_refs_v6_strip_org",
    "_ap003d_impl__refs_v6_strip_org",
    "WorkflowState._normalize",
    "extract_org_abstracts",
    "_ap003f_pipeline_core",
)
FROZEN_FILES = (
    str(
        SOFTWARE_REL
        / "app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_13.py"
    ),
    str(
        SOFTWARE_REL
        / "app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_14.py"
    ),
)
HISTORICAL_ORCHESTRATOR = str(
    SOFTWARE_REL / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)
CANONICAL_ORCHESTRATOR_BASENAME = "pipeline_orchestrator.py"

EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".cache",
        ".patch_backups",
        "patch_backups",
        ".backup",
        ".backups",
        "node_modules",
        "site-packages",
        "dist",
        "build",
        "output",
        "outputs",
        "backup",
        "backups",
        "coverage",
        "htmlcov",
        ".idea",
        ".vscode",
    }
)
EXCLUDED_FILE_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".class",
        ".jar",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".xlsx",
        ".xls",
        ".docx",
        ".pptx",
    }
)
TEXT_SUFFIXES = frozenset(
    {
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
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".service",
        ".desktop",
        ".env.example",
    }
)
COMPATIBILITY_WORDS = re.compile(
    r"\b(?:compat(?:ibility|ibilidade)?|legacy|legado|hist[oó]ric[oa]|"
    r"wrapper|alias|reexport|re-export|fachada|bridge|ponte|redirect|"
    r"redirecionamento|deprecated|deprecat|can[oô]nic[oa])\b",
    re.IGNORECASE,
)
VERSIONISH_NAME = re.compile(
    r"(?:^|_)(?:v\d+(?:_\d+)*|rc\d+(?:_\d+)*|pre_v?\d+|old|legacy|"
    r"historical|original|compat|wrapper)(?:_|$)",
    re.IGNORECASE,
)
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
ARROW_PAIR_RE = re.compile(
    r"`?([A-Za-z_][A-Za-z0-9_./-]*)`?\s*(?:→|->|⇒)\s*"
    r"`?([A-Za-z_][A-Za-z0-9_./-]*)`?"
)
STRUCTURAL_COMPATIBILITY_WORDS = re.compile(
    r"\b(?:alias|wrapper|bridge|ponte|reexport|re-export|fachada|redirect|"
    r"redirecionamento|deprecated|deprecat|legacy|legado|"
    r"(?:nome|m[oó]dulo|s[ií]mbolo|entrypoint)\s+(?:antigo|legado|hist[oó]rico|can[oô]nico))\b",
    re.IGNORECASE,
)
GENERIC_CONSUMER_NAMES = frozenset({
    "main",
    "module",
    "academic_pipeline",
})

REQUIRED_CLASSIFICATIONS = (
    "compatibilidade pública durável",
    "compatibilidade interna necessária",
    "compatibilidade transitória removível",
    "wrapper histórico congelado",
    "alias canônico necessário",
    "reexport necessário",
    "bridge de importação necessária",
    "entrypoint público preservado",
    "superfície sem consumidores",
    "consumidor apenas em teste",
    "consumidor apenas documental",
    "consumidor em snapshot, fixture ou manifesto histórico",
    "compatibilidade ligada a caminho físico fora do escopo",
    "compatibilidade protegida por decisão da AP-004B",
    "compatibilidade ligada aos três xfail",
    "item ambíguo que exige decisão manual",
    "colisão ou conflito de destino",
    "ocorrência residual sem efeito executável",
)


class InventoryError(RuntimeError):
    """Falha segura antes da publicação dos artefatos."""


@dataclasses.dataclass(frozen=True)
class Location:
    path: str
    line: int
    column: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Definition:
    path: str
    module: str
    name: str
    qualname: str
    kind: str
    line: int
    column: int
    scope: str
    is_private: bool
    docstring: str | None = None


@dataclasses.dataclass
class ImportRecord:
    path: str
    module: str
    imported_module: str
    imported_name: str | None
    local_name: str
    line: int
    column: int
    is_star: bool
    level: int


@dataclasses.dataclass
class AliasRecord:
    path: str
    module: str
    name: str
    target: str
    line: int
    column: int
    scope: str
    kind: str


@dataclasses.dataclass
class WrapperRecord:
    path: str
    module: str
    name: str
    qualname: str
    target: str
    line: int
    column: int
    scope: str
    forwarding: str
    docstring: str | None


@dataclasses.dataclass
class ReexportRecord:
    path: str
    module: str
    name: str
    source: str
    line: int
    in_all: bool
    package_facade: bool


@dataclasses.dataclass
class DynamicRecord:
    path: str
    module: str
    kind: str
    key: str | None
    target: str | None
    line: int
    column: int
    operational: bool
    scope: str
    container: str | None


@dataclasses.dataclass
class EntrypointRecord:
    path: str
    kind: str
    name: str
    target: str
    line: int


@dataclasses.dataclass
class CompatibilityNote:
    path: str
    line: int
    text: str
    source_kind: str


@dataclasses.dataclass
class FileAnalysis:
    path: str
    module: str
    is_package_init: bool
    definitions: list[Definition] = dataclasses.field(default_factory=list)
    imports: list[ImportRecord] = dataclasses.field(default_factory=list)
    aliases: list[AliasRecord] = dataclasses.field(default_factory=list)
    wrappers: list[WrapperRecord] = dataclasses.field(default_factory=list)
    reexports: list[ReexportRecord] = dataclasses.field(default_factory=list)
    dynamics: list[DynamicRecord] = dataclasses.field(default_factory=list)
    comments: list[CompatibilityNote] = dataclasses.field(default_factory=list)
    all_names: set[str] = dataclasses.field(default_factory=set)
    loaded_names: list[tuple[str, int, int]] = dataclasses.field(default_factory=list)
    loaded_attributes: list[tuple[str, int, int]] = dataclasses.field(default_factory=list)
    syntax_error: str | None = None
    module_facade: bool = False


@dataclasses.dataclass
class Reference:
    path: str
    line: int
    context: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Candidate:
    candidate_id: str
    current_name: str
    canonical_surface: str | None
    path: str
    line: int
    occurrence_type: str
    ast_scope: str | None
    definition: str | None
    target: str | None
    source: str
    evidence: list[str] = dataclasses.field(default_factory=list)
    classifications: list[str] = dataclasses.field(default_factory=list)
    internal_consumers: list[Reference] = dataclasses.field(default_factory=list)
    test_consumers: list[Reference] = dataclasses.field(default_factory=list)
    documentary_consumers: list[Reference] = dataclasses.field(default_factory=list)
    historical_consumers: list[Reference] = dataclasses.field(default_factory=list)
    imports_reexports: list[str] = dataclasses.field(default_factory=list)
    related_entrypoints: list[str] = dataclasses.field(default_factory=list)
    contract: str = "não determinado"
    external_consumer_status: str = "não demonstrado"
    risk: str = "médio"
    proposed_decision: str = "revisão manual"
    reason: str = "classificação preparatória pendente"
    application_wave: str = "decisão manual"
    deprecation_required: bool = False
    compatibility_after_ap004: str = "a decidir"
    preservation_or_removal_evidence: list[str] = dataclasses.field(default_factory=list)
    collision_targets: list[str] = dataclasses.field(default_factory=list)
    consumer_counts_total: dict[str, int] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["canonical_name"] = self.canonical_surface
        counts = self.consumer_counts_total or {
            "internal": len(self.internal_consumers),
            "test": len(self.test_consumers),
            "documentary": len(self.documentary_consumers),
            "historical": len(self.historical_consumers),
        }
        data["consumer_counts"] = {
            "internal": counts.get("internal", 0),
            "tests": counts.get("test", 0),
            "documentation": counts.get("documentary", 0),
            "historical": counts.get("historical", 0),
        }
        return data


class Git:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and proc.returncode != 0:
            raise InventoryError(
                f"git {' '.join(args)} falhou ({proc.returncode}): "
                f"{proc.stderr.strip()}"
            )
        return proc.stdout.strip()


class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, module: str, source: str) -> None:
        self.rel_path = rel_path
        self.module = module
        self.source = source
        self.analysis = FileAnalysis(
            path=rel_path,
            module=module,
            is_package_init=Path(rel_path).name == "__init__.py",
        )
        self.scope_stack: list[str] = [module]
        self.import_by_local: dict[str, ImportRecord] = {}

    @property
    def scope(self) -> str:
        return ".".join(part for part in self.scope_stack if part)

    def _qualname(self, name: str) -> str:
        return f"{self.scope}.{name}" if self.scope else name

    def _add_definition(
        self,
        node: ast.AST,
        name: str,
        kind: str,
        docstring: str | None = None,
    ) -> None:
        self.analysis.definitions.append(
            Definition(
                path=self.rel_path,
                module=self.module,
                name=name,
                qualname=self._qualname(name),
                kind=kind,
                line=getattr(node, "lineno", 0),
                column=getattr(node, "col_offset", 0),
                scope=self.scope,
                is_private=name.startswith("_"),
                docstring=docstring,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        doc = ast.get_docstring(node, clean=False)
        self._add_definition(node, node.name, "async function" if isinstance(node, ast.AsyncFunctionDef) else "function", doc)
        wrapper = detect_wrapper(node)
        if wrapper is not None:
            target, forwarding = wrapper
            self.analysis.wrappers.append(
                WrapperRecord(
                    path=self.rel_path,
                    module=self.module,
                    name=node.name,
                    qualname=self._qualname(node.name),
                    target=target,
                    line=node.lineno,
                    column=node.col_offset,
                    scope=self.scope,
                    forwarding=forwarding,
                    docstring=doc,
                )
            )
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        doc = ast.get_docstring(node, clean=False)
        self._add_definition(node, node.name, "class", doc)
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            record = ImportRecord(
                path=self.rel_path,
                module=self.module,
                imported_module=alias.name,
                imported_name=None,
                local_name=local,
                line=node.lineno,
                column=node.col_offset,
                is_star=False,
                level=0,
            )
            self.analysis.imports.append(record)
            self.import_by_local[local] = record
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        imported_module = ("." * node.level) + (node.module or "")
        for alias in node.names:
            local = alias.asname or alias.name
            record = ImportRecord(
                path=self.rel_path,
                module=self.module,
                imported_module=imported_module,
                imported_name=alias.name,
                local_name=local,
                line=node.lineno,
                column=node.col_offset,
                is_star=alias.name == "*",
                level=node.level,
            )
            self.analysis.imports.append(record)
            if alias.name != "*":
                self.import_by_local[local] = record
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        if any(is_name(target, "__all__") for target in node.targets):
            self.analysis.all_names.update(extract_string_collection(node.value))

        target_names = [name for target in node.targets for name in assigned_names(target)]
        value_name = expression_name(node.value)
        if value_name:
            for name in target_names:
                if name != value_name and name != "__all__":
                    self.analysis.aliases.append(
                        AliasRecord(
                            path=self.rel_path,
                            module=self.module,
                            name=name,
                            target=value_name,
                            line=node.lineno,
                            column=node.col_offset,
                            scope=self.scope,
                            kind="assignment alias",
                        )
                    )

        registry_target = any(
            re.search(
                r"(?:registry|registries|handlers?|dispatch|commands?|aliases?|"
                r"factories|plugins?|loaders?|resolvers?|entrypoints?|routes?|mapping)",
                name,
                re.IGNORECASE,
            )
            for name in target_names
        )
        if isinstance(node.value, ast.Dict) and registry_target:
            for key_node, value_node in zip(node.value.keys, node.value.values):
                key = literal_string(key_node)
                target = expression_name(value_node) or literal_string(value_node)
                if key is not None and target is not None:
                    self.analysis.dynamics.append(
                        DynamicRecord(
                            path=self.rel_path,
                            module=self.module,
                            kind="registry mapping",
                            key=key,
                            target=target,
                            line=node.lineno,
                            column=node.col_offset,
                            operational=True,
                            scope=self.scope,
                            container=target_names[0] if target_names else None,
                        )
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if is_name(node.target, "__all__") and node.value is not None:
            self.analysis.all_names.update(extract_string_collection(node.value))
        if node.value is not None:
            value_name = expression_name(node.value)
            if value_name:
                for name in assigned_names(node.target):
                    if name != value_name and name != "__all__":
                        self.analysis.aliases.append(
                            AliasRecord(
                                path=self.rel_path,
                                module=self.module,
                                name=name,
                                target=value_name,
                                line=node.lineno,
                                column=node.col_offset,
                                scope=self.scope,
                                kind="annotated assignment alias",
                            )
                        )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        if is_name(node.target, "__all__"):
            self.analysis.all_names.update(extract_string_collection(node.value))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            self.analysis.loaded_names.append((node.id, node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if isinstance(node.ctx, ast.Load):
            name = expression_name(node)
            if name:
                self.analysis.loaded_attributes.append(
                    (name, node.lineno, node.col_offset)
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = expression_name(node.func) or ""
        dynamic_kind: str | None = None
        if call_name in {"getattr", "setattr", "hasattr", "delattr"}:
            dynamic_kind = call_name
        elif call_name in {"globals", "locals", "vars"}:
            dynamic_kind = call_name
        elif call_name in {"__import__", "import_module", "importlib.import_module"}:
            dynamic_kind = "dynamic import"
        elif any(
            token in call_name.lower()
            for token in ("registry", "register", "resolve", "resolver", "dispatch")
        ):
            dynamic_kind = "registry/resolver call"
        if dynamic_kind:
            string_args = [literal_string(arg) for arg in node.args]
            key = next((item for item in string_args if item is not None), None)
            target = None
            for arg in node.args:
                name = expression_name(arg)
                if name and name != call_name:
                    target = name
                    break
            self.analysis.dynamics.append(
                DynamicRecord(
                    path=self.rel_path,
                    module=self.module,
                    kind=dynamic_kind,
                    key=key,
                    target=target,
                    line=node.lineno,
                    column=node.col_offset,
                    operational=True,
                    scope=self.scope,
                    container=call_name or None,
                )
            )
        self.generic_visit(node)

    def finalize(self, tree: ast.Module) -> FileAnalysis:
        imported = {record.local_name: record for record in self.analysis.imports}
        for local_name, record in imported.items():
            in_all = local_name in self.analysis.all_names
            package_facade = self.analysis.is_package_init
            project_relative = bool(
                record.level > 0
                or record.imported_module.startswith("academic_pipeline")
                or record.imported_module.startswith("app_bundle")
            )
            if in_all or record.is_star or (package_facade and project_relative):
                source = record.imported_module
                if record.imported_name:
                    source = f"{source}:{record.imported_name}"
                self.analysis.reexports.append(
                    ReexportRecord(
                        path=self.rel_path,
                        module=self.module,
                        name=local_name,
                        source=source,
                        line=record.line,
                        in_all=in_all,
                        package_facade=package_facade,
                    )
                )

        executable_nodes = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
                executable_nodes.append(node)
                continue
            if isinstance(node, ast.If) and is_main_guard(node.test):
                executable_nodes.append(node)
                continue
            executable_nodes.append(node)
        allowed_facade = all(
            isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
            for node in executable_nodes
        )
        has_redirect = any(
            record.is_star or record.imported_name is not None
            for record in self.analysis.imports
        )
        self.analysis.module_facade = allowed_facade and has_redirect
        return self.analysis


def is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    left = expression_name(node.left)
    right = literal_string(node.comparators[0])
    return left == "__name__" and right == "__main__" and isinstance(node.ops[0], ast.Eq)


def literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def expression_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        parent = expression_name(node.value)
        key = literal_string(node.slice)
        if parent and key is not None:
            return f"{parent}[{key!r}]"
    return None


def assigned_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        result: list[str] = []
        for element in node.elts:
            result.extend(assigned_names(element))
        return result
    return []


def extract_string_collection(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return {
            item.value
            for item in node.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return extract_string_collection(node.left) | extract_string_collection(node.right)
    return set()


def normalize_function_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            body = body[1:]
    return body


def detect_wrapper(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    body = normalize_function_body(node)
    if len(body) != 1:
        return None
    call: ast.Call | None = None
    if isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Call):
        call = body[0].value
    elif isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call):
        call = body[0].value
    elif isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Await) and isinstance(body[0].value.value, ast.Call):
        call = body[0].value.value
    if call is None:
        return None
    target = expression_name(call.func)
    if not target or target == node.name:
        return None

    param_names: list[str] = []
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        param_names.append(arg.arg)
    if node.args.vararg:
        param_names.append(node.args.vararg.arg)
    if node.args.kwarg:
        param_names.append(node.args.kwarg.arg)

    referenced = {
        sub.id
        for sub in ast.walk(call)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }
    forwarding = "integral" if referenced.issubset(set(param_names) | {target.split(".", 1)[0]}) else "adaptador"
    return target, forwarding


def module_name_from_path(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] in {"__init__", "__main__"}:
        parts.pop()
    project_root_names = ("academic_pipeline_mppg", "academic_pipeline_rc10_7_conformidade")
    project_root_index = next(
        (parts.index(name) for name in project_root_names if name in parts),
        None,
    )
    if project_root_index is not None:
        parts = parts[project_root_index + 1:]
    elif "tools" in parts:
        idx = parts.index("tools")
        parts = parts[idx:]
    return ".".join(parts)


def should_exclude(path: Path, repo_root: Path) -> bool:
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return True
    directory_parts = rel.parts if path.is_dir() else rel.parts[:-1]
    for part in directory_parts:
        lowered = part.lower()
        if part in EXCLUDED_DIR_NAMES:
            return True
        if (
            lowered.startswith(".patch_backup")
            or lowered.startswith("patch_backup")
            or lowered.startswith("backup_")
            or lowered.startswith("backups_")
            or lowered.endswith("_backup")
            or lowered.endswith("_backups")
        ):
            return True
    if path.is_file() and path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return True
    return False


def iter_files(repo_root: Path) -> Iterator[Path]:
    """Itera apenas sobre a raiz produtiva canônica e evidências da AP-004.

    Cópias históricas localizadas em outros diretórios de ``software/`` não são
    consumidoras nem superfícies da árvore canônica e, portanto, ficam fora do
    inventário. Os artefatos de testes, documentação e ``tools/refactor`` entram somente
    como evidência de consumo/documentação. A origem de candidatos é filtrada
    separadamente por ``is_candidate_origin_path``.
    """
    seen: set[Path] = set()
    for scan_rel in SCAN_ROOT_RELS:
        scan_root = repo_root / scan_rel
        if not scan_root.exists():
            continue
        for root, dirnames, filenames in os.walk(scan_root, followlinks=False):
            root_path = Path(root)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in EXCLUDED_DIR_NAMES
                and not should_exclude(root_path / name, repo_root)
                and not (root_path / name).is_symlink()
            )
            for filename in sorted(filenames):
                path = root_path / filename
                if path in seen:
                    continue
                seen.add(path)
                if path.is_symlink() or should_exclude(path, repo_root):
                    continue
                if path.relative_to(repo_root) in SCAN_EXCLUDED_RELS:
                    continue
                try:
                    if path.stat().st_size > 5_000_000:
                        continue
                except OSError:
                    continue
                if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
                    "Dockerfile",
                    "Makefile",
                    "Pipfile",
                    "Pipfile.lock",
                    "pyproject.toml",
                    "setup.cfg",
                    "setup.py",
                }:
                    yield path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def extract_explicit_compatibility_pair(text: str) -> tuple[str, str] | None:
    """Extrai somente relações nominais inequívocas antigo -> canônico."""
    arrow = ARROW_PAIR_RE.search(text)
    if arrow and arrow.group(1) != arrow.group(2):
        return arrow.group(1), arrow.group(2)

    quoted = r"`([A-Za-z_][A-Za-z0-9_./:-]*)`"
    patterns = (
        rf"\b(?:alias|wrapper|bridge|reexport|re-export|redirecionamento|fachada)\b"
        rf"[^\n]{{0,100}}?{quoted}[^\n]{{0,80}}?\b(?:para|como|to)\b"
        rf"[^\n]{{0,80}}?{quoted}",
        rf"\b(?:nome|m[oó]dulo|s[ií]mbolo|entrypoint)\s+(?:antigo|legado|hist[oó]rico)\b"
        rf"[^\n]{{0,100}}?{quoted}[^\n]{{0,120}}?"
        rf"\b(?:nome|m[oó]dulo|s[ií]mbolo|entrypoint)\s+can[oô]nico\b"
        rf"[^\n]{{0,100}}?{quoted}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1) != match.group(2):
            return match.group(1), match.group(2)
    return None


def compatibility_pair_is_specific(pair: tuple[str, str]) -> bool:
    """Exige ao menos um token com forma inequívoca de superfície técnica."""
    for token in pair:
        if (
            token.startswith("_")
            or any(marker in token for marker in ("_", ".", "/", "-", ":"))
            or VERSIONISH_NAME.search(token)
            or len(token) >= 18
        ):
            return True
    return False


def is_strong_compatibility_evidence(text: str) -> bool:
    """Aceita somente declaração nominal explícita e estrutural.

    Menções genéricas de compatibilidade não bastam: pares discursivos como
    ``documento -> chave`` são ignorados. Aceita-se uma palavra estrutural
    inequívoca (alias/wrapper/bridge/legado etc.) ou um token com forma de
    identificador/path/versionamento técnico.
    """
    pair = extract_explicit_compatibility_pair(text)
    if pair is None:
        return False
    return bool(
        STRUCTURAL_COMPATIBILITY_WORDS.search(text)
        or (COMPATIBILITY_WORDS.search(text) and compatibility_pair_is_specific(pair))
    )


def extract_compatibility_comments(rel_path: str, source: str) -> list[CompatibilityNote]:
    notes: list[CompatibilityNote] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and is_strong_compatibility_evidence(token.string):
                notes.append(
                    CompatibilityNote(
                        path=rel_path,
                        line=token.start[0],
                        text=token.string.strip()[:500],
                        source_kind="python comment",
                    )
                )
    except (tokenize.TokenError, IndentationError):
        pass
    return notes


def analyze_python(repo_root: Path, path: Path) -> FileAnalysis:
    rel = path.relative_to(repo_root).as_posix()
    source = read_text(path)
    module = module_name_from_path(repo_root, path)
    analyzer = PythonAnalyzer(rel_path=rel, module=module, source=source)
    analyzer.analysis.comments.extend(extract_compatibility_comments(rel, source))
    try:
        tree = ast.parse(source, filename=rel, type_comments=True)
    except SyntaxError as exc:
        analyzer.analysis.syntax_error = f"{exc.msg} (linha {exc.lineno})"
        return analyzer.analysis
    analyzer.visit(tree)
    return analyzer.finalize(tree)


def scan_non_python_notes(repo_root: Path, path: Path) -> list[CompatibilityNote]:
    rel = path.relative_to(repo_root).as_posix()
    if not rel.startswith(SOFTWARE_REL.as_posix() + "/"):
        return []
    if path.suffix.lower() not in {".toml", ".json", ".yaml", ".yml", ".ini", ".cfg", ".sh", ".bash", ".zsh", ".fish", ".service", ".desktop"}:
        return []
    source = read_text(path)
    notes: list[CompatibilityNote] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if is_strong_compatibility_evidence(line):
            notes.append(
                CompatibilityNote(
                    path=rel,
                    line=number,
                    text=line.strip()[:500],
                    source_kind=f"{path.suffix.lstrip('.') or path.name} metadata",
                )
            )
    return notes


def parse_pyproject(repo_root: Path, path: Path) -> list[EntrypointRecord]:
    rel = path.relative_to(repo_root).as_posix()
    try:
        data = tomllib.loads(read_text(path))
    except (tomllib.TOMLDecodeError, OSError):
        return []
    results: list[EntrypointRecord] = []
    project = data.get("project", {}) if isinstance(data, dict) else {}
    scripts = project.get("scripts", {}) if isinstance(project, dict) else {}
    gui_scripts = project.get("gui-scripts", {}) if isinstance(project, dict) else {}
    for kind, mapping in (("console script", scripts), ("gui script", gui_scripts)):
        if isinstance(mapping, dict):
            for name, target in sorted(mapping.items()):
                if isinstance(target, str):
                    results.append(
                        EntrypointRecord(rel, kind, str(name), target, find_line(read_text(path), str(name)))
                    )
    tool = data.get("tool", {}) if isinstance(data, dict) else {}
    poetry = tool.get("poetry", {}) if isinstance(tool, dict) else {}
    poetry_scripts = poetry.get("scripts", {}) if isinstance(poetry, dict) else {}
    if isinstance(poetry_scripts, dict):
        for name, target in sorted(poetry_scripts.items()):
            if isinstance(target, str):
                results.append(
                    EntrypointRecord(rel, "poetry script", str(name), target, find_line(read_text(path), str(name)))
                )
    return results


def parse_setup_cfg(repo_root: Path, path: Path) -> list[EntrypointRecord]:
    rel = path.relative_to(repo_root).as_posix()
    parser = configparser.ConfigParser()
    try:
        parser.read_string(read_text(path))
    except configparser.Error:
        return []
    results: list[EntrypointRecord] = []
    section = "options.entry_points"
    if parser.has_section(section):
        for kind, raw in parser.items(section):
            for line in raw.splitlines():
                if "=" not in line:
                    continue
                name, target = (part.strip() for part in line.split("=", 1))
                results.append(
                    EntrypointRecord(rel, kind, name, target, find_line(read_text(path), name))
                )
    return results


def parse_setup_py(repo_root: Path, path: Path) -> list[EntrypointRecord]:
    rel = path.relative_to(repo_root).as_posix()
    source = read_text(path)
    results: list[EntrypointRecord] = []
    for match in re.finditer(
        r"['\"]([A-Za-z0-9_.-]+)['\"]\s*=\s*['\"]([A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*)['\"]",
        source,
    ):
        results.append(
            EntrypointRecord(
                rel,
                "setup.py entrypoint",
                match.group(1),
                match.group(2),
                source[: match.start()].count("\n") + 1,
            )
        )
    return results


def find_line(source: str, needle: str) -> int:
    for number, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return number
    return 1


def discover_entrypoints(repo_root: Path, files: Sequence[Path]) -> list[EntrypointRecord]:
    entrypoints: list[EntrypointRecord] = []
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        if not rel.startswith(SOFTWARE_REL.as_posix() + "/"):
            continue
        if classify_path(rel) != "internal":
            continue
        if path.name == "pyproject.toml":
            entrypoints.extend(parse_pyproject(repo_root, path))
        elif path.name == "setup.cfg":
            entrypoints.extend(parse_setup_cfg(repo_root, path))
        elif path.name == "setup.py":
            entrypoints.extend(parse_setup_py(repo_root, path))
        elif path.name == "__main__.py":
            rel = path.relative_to(repo_root).as_posix()
            module = module_name_from_path(repo_root, path)
            entrypoints.append(
                EntrypointRecord(rel, "python -m", module.rsplit(".__main__", 1)[0] or module, f"{module}:main", 1)
            )
    unique: dict[tuple[str, str, str, str], EntrypointRecord] = {}
    for item in entrypoints:
        unique[(item.path, item.kind, item.name, item.target)] = item
    return sorted(unique.values(), key=lambda x: (x.path, x.line, x.name, x.target))



def is_candidate_origin_path(path: str) -> bool:
    """Retorna True apenas para superfícies produtivas da raiz canônica.

    Testes, documentação, ferramentas de refatoração, snapshots e backups podem
    fornecer evidência de consumo, mas nunca originam candidatos da AP-004E.
    """
    rel = Path(path)
    try:
        within_software = rel.relative_to(SOFTWARE_REL)
    except ValueError:
        return False

    parts = tuple(part.lower() for part in within_software.parts)
    if not parts:
        return False
    if any(
        part in {
            "tests", "test", "docs", "doc", "documentation", "fixtures",
            "fixture", "snapshots", "snapshot", "manifests", "manifest",
            "golden", "tools",
        }
        for part in parts[:-1]
    ):
        return False
    if any(
        part.startswith(".patch_backup")
        or part.startswith("patch_backup")
        or part in {"backup", "backups", "output", "outputs"}
        for part in parts[:-1]
    ):
        return False
    return True


def is_canonical_ap004_evidence_path(path: str) -> bool:
    """Aceita a documentação AP-004 no nível Git ou no pacote histórico."""
    rel = Path(path)
    accepted_roots = (AP004_DOCS_REL, SOFTWARE_REL / AP004_DOCS_REL)
    for root in accepted_roots:
        try:
            rel.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def is_executable_metadata_candidate_path(path: str) -> bool:
    if not is_candidate_origin_path(path):
        return False
    suffix = Path(path).suffix.lower()
    return suffix in {".toml", ".ini", ".cfg", ".yaml", ".yml", ".json", ".service", ".desktop"}


def classify_path(path: str) -> str:
    lower = path.lower()
    parts = tuple(part.lower() for part in Path(path).parts)
    name = Path(path).name.lower()
    if any(token in lower for token in ("snapshot", "fixture", "manifest", "golden")):
        return "historical"
    if any(part in {"tests", "test"} for part in parts) or name.startswith("test_"):
        return "test"
    if (
        any(part in {"docs", "doc", "documentation"} for part in parts)
        or any(part == "tools" for part in parts)
        or Path(path).suffix.lower() in {".md", ".rst", ".org"}
    ):
        return "documentary"
    if is_candidate_origin_path(path):
        return "internal"
    return "documentary"


def make_text_indexes(
    repo_root: Path, files: Sequence[Path]
) -> tuple[dict[str, list[Reference]], dict[str, str]]:
    """Constrói índice semântico de consumidores e preserva fontes documentais.

    Em Python, somente usos AST (``Load``), atributos, imports e strings
    operacionais entram no índice. Definições homônimas não são consumidores.
    Em arquivos não Python, a busca permanece textual e exata por identificador.
    """
    identifier_index: dict[str, list[Reference]] = defaultdict(list)
    sources: dict[str, str] = {}

    def add(name: str, rel: str, line: int, context: str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return
        bucket = identifier_index[name]
        key = (rel, line, context)
        if any((item.path, item.line, item.context) == key for item in bucket[-8:]):
            return
        if len(bucket) < 5000:
            bucket.append(Reference(rel, line, context[:320]))

    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        source = read_text(path)
        sources[rel] = source
        lines = source.splitlines()

        def context_at(line: int, kind: str) -> str:
            raw = lines[line - 1].strip() if 1 <= line <= len(lines) else ""
            return f"{kind}: {raw}"[:320]

        if path.suffix.lower() in {".py", ".pyi"}:
            try:
                tree = ast.parse(source, filename=rel, type_comments=True)
            except SyntaxError:
                # Erros de sintaxe produtivos são tratados em ``main``. Para
                # evidências externas, usa-se fallback textual conservador.
                for number, line in enumerate(lines, start=1):
                    for identifier in set(IDENTIFIER_RE.findall(line)):
                        add(identifier, rel, number, f"TEXT-FALLBACK: {line.strip()}")
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    add(node.id, rel, node.lineno, context_at(node.lineno, "AST-NAME"))
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    full = expression_name(node)
                    add(node.attr, rel, node.lineno, context_at(node.lineno, "AST-ATTR"))
                    if full:
                        leaf = full.rsplit(".", 1)[-1]
                        add(leaf, rel, node.lineno, context_at(node.lineno, "AST-ATTR"))
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name != "*":
                            add(
                                alias.asname or alias.name,
                                rel,
                                node.lineno,
                                context_at(node.lineno, "AST-IMPORT"),
                            )
                            add(
                                alias.name,
                                rel,
                                node.lineno,
                                context_at(node.lineno, "AST-IMPORT"),
                            )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        add(
                            alias.asname or alias.name.split(".", 1)[0],
                            rel,
                            node.lineno,
                            context_at(node.lineno, "AST-IMPORT"),
                        )
                        add(
                            alias.name.rsplit(".", 1)[-1],
                            rel,
                            node.lineno,
                            context_at(node.lineno, "AST-IMPORT"),
                        )
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # Strings são relevantes para getattr/registries/monkeypatch,
                    # mas nomes curtos e genéricos geram muitos falsos positivos.
                    for identifier in set(IDENTIFIER_RE.findall(node.value)):
                        if (
                            identifier.startswith("_")
                            or len(identifier) >= 12
                            or VERSIONISH_NAME.search(identifier)
                            or COMPATIBILITY_WORDS.search(identifier)
                        ):
                            add(
                                identifier,
                                rel,
                                getattr(node, "lineno", 1),
                                context_at(getattr(node, "lineno", 1), "AST-STRING"),
                            )
            continue

        for number, line in enumerate(lines, start=1):
            for identifier in set(IDENTIFIER_RE.findall(line)):
                add(identifier, rel, number, f"TEXT: {line.strip()}")

    for references in identifier_index.values():
        references.sort(key=lambda item: (item.path, item.line, item.context))
    return identifier_index, sources

def extract_ap004b_pairs(repo_root: Path, sources: dict[str, str]) -> list[tuple[str, str, str, int]]:
    """Extrai decisões AP-004B e deduplica o mesmo par entre cópias documentais."""
    selected: dict[tuple[str, str], tuple[int, str, int]] = {}
    canonical_prefix = AP004_DOCS_REL.as_posix() + "/"
    for rel, source in sources.items():
        if not is_canonical_ap004_evidence_path(rel):
            continue
        if "AP-004B" not in rel and "ap004b" not in rel.lower():
            continue
        rank = 0 if rel.startswith(canonical_prefix) else 1
        for number, line in enumerate(source.splitlines(), start=1):
            for match in ARROW_PAIR_RE.finditer(line):
                old, new = match.groups()
                if old == new:
                    continue
                key = (old, new)
                previous = selected.get(key)
                current = (rank, rel, number)
                if previous is None or current < previous:
                    selected[key] = current
    return sorted(
        (old, new, rel, line)
        for (old, new), (_rank, rel, line) in selected.items()
    )


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"AP004E-{digest[:12]}"


def ref_to_class(ref: Reference) -> str:
    return classify_path(ref.path)


def candidate_module_hint(candidate: Candidate) -> str | None:
    """Deriva o módulo Python da origem para filtrar nomes genéricos."""
    path = candidate.path
    markers = (
        "software/academic_pipeline_mppg/",
        "software/academic_pipeline_rc10_7_conformidade/",
    )
    marker = next((item for item in markers if item in path), None)
    if marker is None or not path.endswith((".py", ".pyi")):
        return None
    rel = path.split(marker, 1)[1]
    module_path = Path(rel)
    parts = list(module_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def generic_reference_is_explicit(
    candidate: Candidate,
    name: str,
    ref: Reference,
) -> bool:
    """Evita atribuir todos os ``main`` do repositório à mesma superfície."""
    if ref.path == candidate.path:
        return True

    context = ref.context
    module = candidate_module_hint(candidate)
    module_leaf = module.rsplit(".", 1)[-1] if module else None

    if candidate.occurrence_type == "entrypoint":
        if name == "academic_pipeline":
            return bool(
                re.search(r"python\s+-m\s+academic_pipeline\b", context)
                or "academic_pipeline:main" in context
                or "academic_pipeline/__main__.py" in context
            )
        return name in context and candidate.current_name in context

    if candidate.occurrence_type == "getattr":
        # A bridge dinâmica é definida no próprio ponto de reflexão; ocorrências
        # globais de ``main`` não são consumidoras dessa expectativa nominal.
        return False

    if context.startswith("AST-IMPORT:"):
        return bool(
            module
            and (module in context or (module_leaf and module_leaf in context))
            and name in context
        )
    if context.startswith("AST-ATTR:"):
        return bool(
            f".{name}" in context
            and module_leaf
            and module_leaf in context
        )
    if context.startswith(("TEXT:", "TEXT-FALLBACK:")):
        qualified = f"{module}.{name}" if module else None
        return bool(
            (qualified and qualified in context)
            or (module and module in context and name in context)
            or (candidate.path in context and name in context)
        )
    return False


def add_consumers(
    candidate: Candidate,
    identifier_index: dict[str, list[Reference]],
) -> None:
    """Vincula consumidores da superfície atual, nunca do destino canônico.

    Referências ao destino explicam a implementação do wrapper, mas não são
    consumidoras do nome legado. Versões anteriores misturavam esses conjuntos e inflava os
    totais, especialmente para wrappers que delegavam a ``_invoke_with_runtime``.
    """
    raw_names = {candidate.current_name}
    leaf = candidate.current_name.rsplit(".", 1)[-1]
    raw_names.add(leaf)
    if candidate.current_name.endswith(".py") or "/" in candidate.current_name:
        raw_names.add(Path(candidate.current_name).stem)
        raw_names.add(Path(candidate.current_name).name.removesuffix(".py"))

    names = {
        name
        for name in raw_names
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    }
    references: list[Reference] = []
    seen: set[tuple[str, int, str]] = set()
    for name in sorted(names):
        for ref in identifier_index.get(name, []):
            key = (ref.path, ref.line, ref.context)
            if key in seen:
                continue
            seen.add(key)
            if ref.path == candidate.path and ref.line == candidate.line:
                continue
            if name in GENERIC_CONSUMER_NAMES and not generic_reference_is_explicit(
                candidate, name, ref
            ):
                continue
            # Strings com nomes públicos curtos são fracas demais como prova.
            if ref.context.startswith("AST-STRING:") and not (
                name.startswith("_")
                or len(name) >= 12
                or VERSIONISH_NAME.search(name)
                or COMPATIBILITY_WORDS.search(name)
            ):
                continue
            references.append(ref)

    references.sort(key=lambda item: (item.path, item.line, item.context))
    buckets: dict[str, list[Reference]] = defaultdict(list)
    for ref in references:
        buckets[ref_to_class(ref)].append(ref)
    candidate.consumer_counts_total = {
        "internal": len(buckets["internal"]),
        "test": len(buckets["test"]),
        "documentary": len(buckets["documentary"]),
        "historical": len(buckets["historical"]),
    }
    evidence_cap = 12
    candidate.internal_consumers = buckets["internal"][:evidence_cap]
    candidate.test_consumers = buckets["test"][:evidence_cap]
    candidate.documentary_consumers = buckets["documentary"][:evidence_cap]
    candidate.historical_consumers = buckets["historical"][:evidence_cap]

def consumer_counts(candidate: Candidate) -> dict[str, int]:
    if candidate.consumer_counts_total:
        return dict(candidate.consumer_counts_total)
    return {
        "internal": len(candidate.internal_consumers),
        "test": len(candidate.test_consumers),
        "documentary": len(candidate.documentary_consumers),
        "historical": len(candidate.historical_consumers),
    }


def classify_candidate(candidate: Candidate) -> None:
    counts = consumer_counts(candidate)
    total = sum(counts.values())
    name = candidate.current_name
    path = candidate.path
    occurrence = candidate.occurrence_type
    classes: list[str] = []

    protected_match = any(
        name == protected
        or name.endswith(f".{protected}")
        or candidate.definition == protected
        for protected in PROTECTED_SYMBOLS
    )
    frozen_match = path in FROZEN_FILES
    ap004b = candidate.source == "AP-004B"
    is_entrypoint = occurrence == "entrypoint"
    is_reexport = occurrence == "reexport"
    is_import_bridge = occurrence in {"module facade", "redirect import"}
    public_name = not name.split(".")[-1].startswith("_")

    if is_entrypoint:
        classes.extend(["entrypoint público preservado", "compatibilidade pública durável"])
        candidate.contract = "superfície pública de execução"
        candidate.external_consumer_status = "suportado"
        candidate.risk = "crítico"
        candidate.proposed_decision = "preservar sem alteração"
        candidate.reason = "entrypoint público explicitamente protegido pela AP-003/AP-004"
        candidate.application_wave = "fora de remoção"
        candidate.deprecation_required = True
        candidate.compatibility_after_ap004 = "durável"
    elif frozen_match:
        classes.append("wrapper histórico congelado")
        candidate.contract = "artefato histórico congelado"
        candidate.external_consumer_status = "provável"
        candidate.risk = "alto"
        candidate.proposed_decision = "preservar congelado"
        candidate.reason = "arquivo explicitamente fora do escopo da AP-004E"
        candidate.application_wave = "fora de remoção"
        candidate.deprecation_required = False
        candidate.compatibility_after_ap004 = "durável até decisão específica"
    elif protected_match:
        classes.append("compatibilidade ligada aos três xfail" if name in PROTECTED_SYMBOLS[:4] or any(name.endswith(p) for p in PROTECTED_SYMBOLS[:4]) else "compatibilidade interna necessária")
        candidate.contract = "superfície protegida por caracterização histórica"
        candidate.external_consumer_status = "interno"
        candidate.risk = "crítico"
        candidate.proposed_decision = "preservar"
        candidate.reason = "símbolo protegido; alteração vedada nesta subfase"
        candidate.application_wave = "fora de remoção"
        candidate.deprecation_required = False
        candidate.compatibility_after_ap004 = "preservada"
    elif path == HISTORICAL_ORCHESTRATOR:
        classes.extend(["alias canônico necessário", "compatibilidade pública durável"])
        candidate.contract = "ponte entre orquestrador histórico e caminho canônico"
        candidate.external_consumer_status = "suportado ou provável"
        candidate.risk = "crítico"
        candidate.proposed_decision = "preservar"
        candidate.reason = "decisão arquitetural consolidada na AP-004B"
        candidate.application_wave = "fora de remoção"
        candidate.deprecation_required = True
        candidate.compatibility_after_ap004 = "durável"
    elif ap004b:
        classes.append("compatibilidade protegida por decisão da AP-004B")
        candidate.contract = "decisão nominal/estrutural da AP-004B"
        candidate.external_consumer_status = "potencialmente externo"
        candidate.risk = "alto"
        candidate.proposed_decision = "preservar"
        candidate.reason = "a AP-004E não pode reabrir decisão da AP-004B sem evidência estrutural nova"
        candidate.application_wave = "preservação"
        candidate.deprecation_required = public_name
        candidate.compatibility_after_ap004 = "preservada"
    elif is_reexport:
        classes.append("reexport necessário" if total > 0 or public_name else "item ambíguo que exige decisão manual")
        candidate.contract = "superfície de importação/reexportação"
        candidate.external_consumer_status = "provável" if public_name else "não demonstrado"
        candidate.risk = "alto" if public_name else "médio"
        candidate.proposed_decision = "preservar" if total > 0 else "revisar manualmente"
        candidate.reason = "reexport possui consumidores" if total > 0 else "ausência interna não elimina possíveis consumidores externos"
        candidate.application_wave = "preservação" if total > 0 else "decisão manual"
        candidate.deprecation_required = public_name
        candidate.compatibility_after_ap004 = "durável" if total > 0 else "a decidir"
    elif is_import_bridge:
        classes.append("bridge de importação necessária" if total > 0 else "item ambíguo que exige decisão manual")
        candidate.contract = "redirecionamento de importação"
        candidate.external_consumer_status = "provável" if public_name else "não demonstrado"
        candidate.risk = "alto" if public_name else "médio"
        candidate.proposed_decision = "preservar" if total > 0 else "revisar manualmente"
        candidate.reason = "bridge ainda consumida" if total > 0 else "nenhum consumidor interno encontrado, mas contrato externo não pode ser descartado"
        candidate.application_wave = "preservação" if total > 0 else "decisão manual"
        candidate.deprecation_required = public_name
        candidate.compatibility_after_ap004 = "a decidir"
    elif (
        occurrence == "getattr"
        and path.endswith("/academic_pipeline/legacy.py")
        and name == "main"
    ):
        classes.extend(["bridge de importação necessária", "compatibilidade interna necessária"])
        candidate.contract = "bridge explícita para o entrypoint do módulo legado"
        candidate.external_consumer_status = "interno"
        candidate.risk = "alto"
        candidate.proposed_decision = "preservar"
        candidate.reason = "load_legacy_module resolve deliberadamente o main do módulo histórico"
        candidate.application_wave = "preservação"
        candidate.deprecation_required = False
        candidate.compatibility_after_ap004 = "preservada"
    elif occurrence in {"registry mapping", "dynamic import", "registry/resolver call", "getattr", "setattr", "hasattr", "globals", "locals", "vars"}:
        classes.append("compatibilidade interna necessária")
        candidate.contract = "resolução dinâmica/registry operacional"
        candidate.external_consumer_status = "interno"
        candidate.risk = "alto"
        candidate.proposed_decision = "preservar até teste estrutural específico"
        candidate.reason = "a resolução dinâmica impede inferência segura por referências estáticas"
        candidate.application_wave = "decisão manual"
        candidate.deprecation_required = False
        candidate.compatibility_after_ap004 = "a decidir"
    elif occurrence == "compatibility metadata":
        if path.startswith("docs/"):
            classes.append("ocorrência residual sem efeito executável")
            candidate.contract = "registro documental"
            candidate.risk = "baixo"
            candidate.proposed_decision = "preservar como evidência"
            candidate.reason = "ocorrência documental separada do código executável"
            candidate.application_wave = "documentação"
            candidate.compatibility_after_ap004 = "documental"
        else:
            classes.append("item ambíguo que exige decisão manual")
            candidate.contract = "metadado potencialmente executável"
            candidate.risk = "médio"
            candidate.proposed_decision = "revisar sem substituição textual cega"
            candidate.reason = "comentário ou metadado exige classificação semântica"
            candidate.application_wave = "decisão manual"
    else:
        candidate.contract = "alias ou wrapper interno"
        candidate.external_consumer_status = "provável" if public_name else "não demonstrado"
        if counts["internal"] > 0:
            classes.append("compatibilidade interna necessária")
            candidate.risk = "alto"
            candidate.proposed_decision = "preservar ou migrar consumidores antes"
            candidate.reason = "há consumidores produtivos internos"
            candidate.application_wave = "migração prévia"
            candidate.deprecation_required = public_name
            candidate.compatibility_after_ap004 = "a decidir"
        elif counts["test"] > 0 and counts["documentary"] == 0 and counts["historical"] == 0:
            classes.append("consumidor apenas em teste")
            candidate.risk = "médio"
            candidate.proposed_decision = "revisar contrato de teste"
            candidate.reason = "a única evidência de consumo está na suíte"
            candidate.application_wave = "decisão manual"
            candidate.deprecation_required = public_name
        elif counts["documentary"] > 0 and counts["test"] == 0 and counts["historical"] == 0:
            classes.append("consumidor apenas documental")
            candidate.risk = "baixo" if not public_name else "médio"
            candidate.proposed_decision = "revisar documentação e contrato externo"
            candidate.reason = "não há consumidor executável interno"
            candidate.application_wave = "decisão manual"
            candidate.deprecation_required = public_name
        elif counts["historical"] > 0 and counts["test"] == 0 and counts["documentary"] == 0:
            classes.append("consumidor em snapshot, fixture ou manifesto histórico")
            candidate.risk = "médio"
            candidate.proposed_decision = "não reescrever história; revisar separadamente"
            candidate.reason = "referências estão somente em artefatos históricos"
            candidate.application_wave = "fora de reescrita histórica"
            candidate.deprecation_required = public_name
        elif total == 0:
            classes.append("superfície sem consumidores")
            if not public_name and candidate.occurrence_type in {"assignment alias", "annotated assignment alias", "function wrapper"}:
                classes.append("compatibilidade transitória removível")
                candidate.risk = "médio"
                candidate.proposed_decision = "candidato à remoção após aprovação"
                candidate.reason = "superfície privada sem consumidores detectados; requer gate manual"
                candidate.application_wave = "onda removível privada"
                candidate.deprecation_required = False
                candidate.compatibility_after_ap004 = "não necessária, se aprovada"
            else:
                classes.append("item ambíguo que exige decisão manual")
                candidate.risk = "alto" if public_name else "médio"
                candidate.proposed_decision = "preservar até análise externa"
                candidate.reason = "ausência interna isolada não comprova ausência de consumidores externos"
                candidate.application_wave = "decisão manual"
                candidate.deprecation_required = public_name
        else:
            classes.append("item ambíguo que exige decisão manual")
            candidate.risk = "médio"
            candidate.proposed_decision = "revisar"
            candidate.reason = "padrão de consumo misto"
            candidate.application_wave = "decisão manual"
            candidate.deprecation_required = public_name

    if candidate.collision_targets:
        classes.append("colisão ou conflito de destino")
        candidate.risk = "crítico"
        candidate.proposed_decision = "bloquear alteração automática"
        candidate.reason += "; há colisão de destino"
        candidate.application_wave = "bloqueada"

    candidate.classifications = sorted(set(classes))
    candidate.preservation_or_removal_evidence.extend(candidate.evidence)
    if counts["internal"]:
        candidate.preservation_or_removal_evidence.append(
            f"{counts['internal']} consumidor(es) produtivo(s) interno(s)"
        )
    if counts["test"]:
        candidate.preservation_or_removal_evidence.append(
            f"{counts['test']} consumidor(es) em teste"
        )
    if counts["documentary"]:
        candidate.preservation_or_removal_evidence.append(
            f"{counts['documentary']} consumidor(es) documental(is)"
        )
    if counts["historical"]:
        candidate.preservation_or_removal_evidence.append(
            f"{counts['historical']} consumidor(es) histórico(s)"
        )
    if total == 0:
        candidate.preservation_or_removal_evidence.append(
            "nenhum consumidor interno foi detectado; isso não exclui consumidor externo"
        )



def compatibility_name_signal(name: str | None, known_names: set[str]) -> bool:
    if not name:
        return False
    leaf = name.rsplit(".", 1)[-1]
    return bool(
        name in known_names
        or leaf in known_names
        or VERSIONISH_NAME.search(name)
        or COMPATIBILITY_WORDS.search(name)
    )


def dynamic_is_compatibility_candidate(
    dynamic: DynamicRecord,
    *,
    known_names: set[str],
    module_has_compatibility_role: bool,
) -> bool:
    """Distingue compatibilidade nominal de reflexão/dispatch ordinários."""
    key_signal = compatibility_name_signal(dynamic.key, known_names)
    target_signal = compatibility_name_signal(dynamic.target, known_names)
    container_signal = compatibility_name_signal(dynamic.container, known_names)

    if dynamic.kind == "registry mapping":
        # Registries de comandos, estados, formatos e opções são fluxo normal.
        # Só entram quando o próprio registry ou algum nome declara legado/alias.
        return key_signal or target_signal or container_signal

    if dynamic.kind in {"getattr", "setattr", "hasattr", "delattr"}:
        return key_signal or target_signal or module_has_compatibility_role

    if dynamic.kind == "dynamic import":
        return key_signal or target_signal or module_has_compatibility_role

    if dynamic.kind == "registry/resolver call":
        return key_signal or target_signal or container_signal

    # globals/locals/vars sem nome nominal não configuram bridge por si sós.
    return key_signal or target_signal

def build_candidates(
    analyses: Sequence[FileAnalysis],
    entrypoints: Sequence[EntrypointRecord],
    notes: Sequence[CompatibilityNote],
    ap004b_pairs: Sequence[tuple[str, str, str, int]],
    identifier_index: dict[str, list[Reference]],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    known_compatibility_names: set[str] = set(PROTECTED_SYMBOLS)
    for old_name, new_name, _path, _line in ap004b_pairs:
        known_compatibility_names.add(old_name)
        known_compatibility_names.add(new_name)

    def append(candidate: Candidate) -> None:
        add_consumers(candidate, identifier_index)
        candidates.append(candidate)

    for analysis in analyses:
        # Definições em testes, documentação e ferramentas são evidências de
        # consumo, não superfícies produtivas candidatas da AP-004E.
        if not is_candidate_origin_path(analysis.path):
            continue
        imports_by_local = {record.local_name: record for record in analysis.imports}

        if analysis.module_facade and analysis.reexports:
            targets = sorted({record.source for record in analysis.reexports})
            append(
                Candidate(
                    candidate_id=stable_id(analysis.path, "module facade", analysis.module),
                    current_name=analysis.module or Path(analysis.path).stem,
                    canonical_surface=", ".join(targets) if targets else None,
                    path=analysis.path,
                    line=1,
                    occurrence_type="module facade",
                    ast_scope=analysis.module,
                    definition="module facade",
                    target=", ".join(targets) if targets else None,
                    source="AST",
                    evidence=["módulo composto apenas por imports/reexports/atribuições"],
                )
            )

        for alias in analysis.aliases:
            import_target = imports_by_local.get(alias.target.split(".", 1)[0])
            source = "AST"
            occurrence = alias.kind
            evidence = [f"atribuição estrutural {alias.name} = {alias.target}"]
            if import_target is not None:
                occurrence = "redirect import"
                evidence.append(
                    f"destino deriva de import {import_target.imported_module}"
                )
            module_level = alias.scope == analysis.module
            compatibility_signal = (
                VERSIONISH_NAME.search(alias.name)
                or VERSIONISH_NAME.search(alias.target)
                or COMPATIBILITY_WORDS.search(alias.name)
                or COMPATIBILITY_WORDS.search(alias.target)
                or alias.name in analysis.all_names
                or import_target is not None
            )
            if module_level and compatibility_signal:
                append(
                    Candidate(
                        candidate_id=stable_id(alias.path, occurrence, alias.name, alias.target, str(alias.line)),
                        current_name=alias.name,
                        canonical_surface=alias.target,
                        path=alias.path,
                        line=alias.line,
                        occurrence_type=occurrence,
                        ast_scope=alias.scope,
                        definition=f"{alias.scope}.{alias.name}",
                        target=alias.target,
                        source=source,
                        evidence=evidence,
                    )
                )

        for wrapper in analysis.wrappers:
            module_level = wrapper.scope == analysis.module
            explicit_compatibility = (
                VERSIONISH_NAME.search(wrapper.name)
                or VERSIONISH_NAME.search(wrapper.target)
                or COMPATIBILITY_WORDS.search(wrapper.name)
                or COMPATIBILITY_WORDS.search(wrapper.target)
                or wrapper.name in analysis.all_names
                or (
                    wrapper.docstring is not None
                    and is_strong_compatibility_evidence(wrapper.docstring)
                )
            )
            if not (module_level and explicit_compatibility):
                continue
            evidence = [
                f"corpo delega para {wrapper.target}",
                f"encaminhamento {wrapper.forwarding}",
            ]
            if wrapper.docstring and is_strong_compatibility_evidence(wrapper.docstring):
                evidence.append("docstring declara compatibilidade")
            append(
                Candidate(
                    candidate_id=stable_id(wrapper.path, "function wrapper", wrapper.qualname, wrapper.target, str(wrapper.line)),
                    current_name=wrapper.name,
                    canonical_surface=wrapper.target,
                    path=wrapper.path,
                    line=wrapper.line,
                    occurrence_type="function wrapper",
                    ast_scope=wrapper.scope,
                    definition=wrapper.qualname,
                    target=wrapper.target,
                    source="AST",
                    evidence=evidence,
                )
            )

        for reexport in analysis.reexports:
            append(
                Candidate(
                    candidate_id=stable_id(reexport.path, "reexport", reexport.name, reexport.source, str(reexport.line)),
                    current_name=reexport.name,
                    canonical_surface=reexport.source,
                    path=reexport.path,
                    line=reexport.line,
                    occurrence_type="reexport",
                    ast_scope=analysis.module,
                    definition=f"{analysis.module}.{reexport.name}",
                    target=reexport.source,
                    source="AST",
                    evidence=[
                        "nome listado em __all__" if reexport.in_all else "reexport em fachada de pacote",
                        "arquivo __init__.py" if reexport.package_facade else "módulo explícito",
                    ],
                )
            )

        module_has_compatibility_role = bool(
            VERSIONISH_NAME.search(Path(analysis.path).stem)
            or COMPATIBILITY_WORDS.search(Path(analysis.path).stem)
            or VERSIONISH_NAME.search(analysis.module)
            or COMPATIBILITY_WORDS.search(analysis.module)
        )
        for dynamic in analysis.dynamics:
            if dynamic.kind != "registry mapping" and dynamic.key is None:
                continue
            if not dynamic_is_compatibility_candidate(
                dynamic,
                known_names=known_compatibility_names,
                module_has_compatibility_role=module_has_compatibility_role,
            ):
                continue
            current = dynamic.key or dynamic.target or dynamic.kind
            append(
                Candidate(
                    candidate_id=stable_id(
                        dynamic.path,
                        dynamic.kind,
                        dynamic.scope,
                        dynamic.container or "",
                        current,
                        str(dynamic.line),
                    ),
                    current_name=current,
                    canonical_surface=dynamic.target,
                    path=dynamic.path,
                    line=dynamic.line,
                    occurrence_type=dynamic.kind,
                    ast_scope=dynamic.scope,
                    definition=dynamic.container,
                    target=dynamic.target,
                    source="AST",
                    evidence=[
                        "resolução dinâmica com sinal nominal de compatibilidade",
                        f"container={dynamic.container}" if dynamic.container else "container não identificado",
                    ],
                )
            )

    for entry in entrypoints:
        append(
            Candidate(
                candidate_id=stable_id(entry.path, "entrypoint", entry.kind, entry.name, entry.target),
                current_name=entry.name,
                canonical_surface=entry.target,
                path=entry.path,
                line=entry.line,
                occurrence_type="entrypoint",
                ast_scope=None,
                definition=f"{entry.kind}: {entry.name}",
                target=entry.target,
                source="metadata",
                evidence=[f"entrypoint {entry.kind} resolve para {entry.target}"],
                related_entrypoints=[f"{entry.kind}: {entry.name} -> {entry.target}"],
            )
        )

    for note in notes:
        if not (
            is_candidate_origin_path(note.path)
            or is_executable_metadata_candidate_path(note.path)
        ):
            continue
        pair = extract_explicit_compatibility_pair(note.text)
        if pair is None:
            continue
        current, canonical = pair
        append(
            Candidate(
                candidate_id=stable_id(note.path, "compatibility metadata", str(note.line), note.text),
                current_name=current,
                canonical_surface=canonical,
                path=note.path,
                line=note.line,
                occurrence_type="compatibility metadata",
                ast_scope=None,
                definition=None,
                target=canonical,
                source=note.source_kind,
                evidence=[note.text],
            )
        )

    for old, new, path, line in ap004b_pairs:
        append(
            Candidate(
                candidate_id=stable_id(path, "AP-004B", old, new, str(line)),
                current_name=old,
                canonical_surface=new,
                path=path,
                line=line,
                occurrence_type="AP-004B compatibility decision",
                ast_scope=None,
                definition=old,
                target=new,
                source="AP-004B",
                evidence=[f"mapeamento AP-004B: {old} -> {new}"],
            )
        )

    canonical_orchestrator_paths = sorted(
        analysis.path
        for analysis in analyses
        if Path(analysis.path).name == CANONICAL_ORCHESTRATOR_BASENAME
    )
    canonical_orchestrator = (
        canonical_orchestrator_paths[0]
        if canonical_orchestrator_paths
        else CANONICAL_ORCHESTRATOR_BASENAME
    )
    append(
        Candidate(
            candidate_id=stable_id(HISTORICAL_ORCHESTRATOR, "historical orchestrator"),
            current_name="academic_pipeline_rc10.py",
            canonical_surface=canonical_orchestrator,
            path=HISTORICAL_ORCHESTRATOR,
            line=1,
            occurrence_type="module facade",
            ast_scope="app_bundle.scripts.pipeline.academic_pipeline_rc10",
            definition="orquestrador histórico",
            target=canonical_orchestrator,
            source="prompt canônico",
            evidence=[
                "arquivo histórico preservado; alias canônico consolidado na AP-004B",
                (
                    "caminho canônico localizado no repositório"
                    if canonical_orchestrator_paths
                    else "caminho canônico não localizado; registrado pelo nome aprovado"
                ),
            ],
        )
    )

    for frozen in FROZEN_FILES:
        append(
            Candidate(
                candidate_id=stable_id(frozen, "frozen historical wrapper"),
                current_name=Path(frozen).stem,
                canonical_surface=None,
                path=frozen,
                line=1,
                occurrence_type="historical frozen file",
                ast_scope=None,
                definition=Path(frozen).name,
                target=None,
                source="prompt canônico",
                evidence=["arquivo fulltext explicitamente congelado na AP-004B/AP-004E"],
            )
        )

    for protected in PROTECTED_SYMBOLS:
        path = HISTORICAL_ORCHESTRATOR
        append(
            Candidate(
                candidate_id=stable_id(path, "protected symbol", protected),
                current_name=protected,
                canonical_surface=protected,
                path=path,
                line=0,
                occurrence_type="protected symbol",
                ast_scope=None,
                definition=protected,
                target=protected,
                source="prompt canônico",
                evidence=["símbolo explicitamente protegido e fora do escopo de alteração"],
            )
        )

    deduped: dict[tuple[str, str, int, str, str | None], Candidate] = {}
    for candidate in candidates:
        key = (
            candidate.path,
            candidate.current_name,
            candidate.line,
            candidate.occurrence_type,
            candidate.target,
        )
        if key in deduped:
            existing = deduped[key]
            existing.evidence = sorted(set(existing.evidence + candidate.evidence))
            existing.related_entrypoints = sorted(
                set(existing.related_entrypoints + candidate.related_entrypoints)
            )
            continue
        deduped[key] = candidate

    result = list(deduped.values())
    surface_to_items: dict[tuple[str, str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in result:
        surface_to_items[(
            candidate.path,
            candidate.ast_scope or "",
            candidate.definition or "",
            candidate.current_name,
        )].append(candidate)
    for (_path, _scope, _definition, _surface), items in surface_to_items.items():
        targets = sorted({item.target for item in items if item.target})
        if len(targets) > 1:
            for item in items:
                item.collision_targets = [target for target in targets if target != item.target]

    for candidate in result:
        classify_candidate(candidate)
        candidate.imports_reexports = sorted(
            set(candidate.imports_reexports)
        )
    return sorted(
        result,
        key=lambda item: (
            item.path,
            item.line,
            item.occurrence_type,
            item.current_name,
            item.target or "",
        ),
    )


def validate_repo(repo_root: Path, *, fetch: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise InventoryError(f"repositório não existe: {repo_root}")
    git = Git(repo_root)
    actual_top = Path(git.run("rev-parse", "--show-toplevel")).resolve()
    if actual_top != repo_root:
        raise InventoryError(
            f"--repo-root deve apontar para a raiz Git: informado={repo_root}, real={actual_top}"
        )
    if fetch:
        git.run("fetch", "origin")
    branch = git.run("branch", "--show-current")
    head = git.run("rev-parse", "HEAD")
    subject = git.run("show", "-s", "--format=%s", "HEAD")
    remote = git.run("rev-parse", REMOTE_REF)
    divergence = git.run("rev-list", "--left-right", "--count", f"HEAD...{REMOTE_REF}")
    status_lines = git.run("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    dirty: list[str] = []
    allowed_dirty: list[str] = []
    for line in status_lines:
        if not line:
            continue
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        rel = Path(raw_path)
        if rel in ALLOWED_DIRTY_RELS:
            allowed_dirty.append(line)
        else:
            dirty.append(line)

    errors: list[str] = []
    if branch != EXPECTED_BRANCH:
        errors.append(f"branch atual={branch!r}; esperado={EXPECTED_BRANCH!r}")
    if head != EXPECTED_HEAD:
        errors.append(f"HEAD={head}; esperado={EXPECTED_HEAD}")
    if subject != EXPECTED_SUBJECT:
        errors.append(f"mensagem={subject!r}; esperada={EXPECTED_SUBJECT!r}")
    if remote != EXPECTED_HEAD:
        errors.append(f"{REMOTE_REF}={remote}; esperado={EXPECTED_HEAD}")
    if divergence.replace("\t", " ").strip() != "0 0":
        errors.append(f"divergência={divergence!r}; esperada='0 0'")
    if dirty:
        errors.append("árvore possui mudanças fora dos artefatos AP-004E permitidos: " + "; ".join(dirty))
    if errors:
        restoration = (
            "cd " + str(repo_root) + " &&\n\n"
            "git fetch origin &&\n\n"
            "git switch " + EXPECTED_BRANCH + " &&\n\n"
            "git status --short --branch &&\n\n"
            "git rev-list --left-right --count HEAD..." + REMOTE_REF
        )
        raise InventoryError(
            "baseline AP-004D inválida:\n- "
            + "\n- ".join(errors)
            + "\n\nComando seguro de diagnóstico/restauração (não descarta mudanças):\n"
            + restoration
        )

    commit_time = git.run("show", "-s", "--format=%cI", "HEAD")
    return {
        "repository": str(repo_root),
        "branch": branch,
        "head": head,
        "subject": subject,
        "remote_ref": REMOTE_REF,
        "remote_head": remote,
        "divergence": [0, 0],
        "tree_clean_except_ap004e_outputs": True,
        "permitted_dirty_paths": sorted(path.as_posix() for path in ALLOWED_DIRTY_RELS),
        "commit_time": commit_time,
        "remote_verified": True,
    }


def transaction_write(repo_root: Path, outputs: dict[Path, str]) -> tuple[Path, list[str]]:
    backup_dir = Path(tempfile.mkdtemp(prefix="ap004e_inventory_backup_"))
    previous: dict[Path, Path | None] = {}
    changed: list[str] = []
    try:
        for rel, content in outputs.items():
            destination = repo_root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup_path: Path | None = None
            if destination.exists():
                backup_path = backup_dir / rel
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup_path)
            previous[rel] = backup_path

            normalized = content.replace("\r\n", "\n")
            if destination.exists() and read_text(destination).replace("\r\n", "\n") == normalized:
                continue
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(normalized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, destination)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            changed.append(rel.as_posix())
        return backup_dir, changed
    except Exception:
        for rel, backup_path in previous.items():
            destination = repo_root / rel
            if backup_path is None:
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, destination)
        raise


def fingerprint_payload(payload: dict[str, Any]) -> str:
    normalized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def build_inventory_payload(
    *,
    baseline: dict[str, Any],
    files: Sequence[Path],
    analyses: Sequence[FileAnalysis],
    notes: Sequence[CompatibilityNote],
    entrypoints: Sequence[EntrypointRecord],
    candidates: Sequence[Candidate],
    repo_root: Path,
) -> dict[str, Any]:
    classification_counts: Counter[str] = Counter()
    occurrence_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    consumer_totals: Counter[str] = Counter()
    for candidate in candidates:
        classification_counts.update(candidate.classifications)
        occurrence_counts[candidate.occurrence_type] += 1
        risk_counts[candidate.risk] += 1
        decision_counts[candidate.proposed_decision] += 1
        consumer_totals.update(consumer_counts(candidate))

    syntax_errors = [
        {"path": analysis.path, "error": analysis.syntax_error}
        for analysis in analyses
        if analysis.syntax_error
    ]
    source_fingerprint_entries = []
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        if Path(rel) in OUTPUT_RELS:
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        source_fingerprint_entries.append((rel, digest))
    source_tree_fingerprint = fingerprint_payload(
        {"files": source_fingerprint_entries}
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "AP-004E",
        "title": "Consolidação das superfícies de compatibilidade transitória",
        "baseline": baseline,
        "generation": {
            "deterministic_timestamp": baseline["commit_time"],
            "tool": TOOL_REL.as_posix(),
            "source_tree_fingerprint": source_tree_fingerprint,
            "read_only_product_scan": True,
            "productive_code_changed": False,
            "applicator_created": False,
            "commit_created": False,
            "push_performed": False,
            "integration_performed": False,
        },
        "scope": {
            "software_root": SOFTWARE_REL.as_posix(),
            "scan_roots": [path.as_posix() for path in SCAN_ROOT_RELS],
            "candidate_root": SOFTWARE_REL.as_posix(),
            "candidate_origin_policy": "produção canônica excluindo tests/docs/tools/backups",
            "evidence_roots": [path.as_posix() for path in SCAN_ROOT_RELS],
            "historical_software_copies_excluded": True,
            "embedded_patch_backups_excluded": True,
            "ordinary_dynamic_operations_excluded": True,
            "collision_scope_includes_dynamic_container": True,
            "test_and_tool_definitions_are_consumers_only": True,
            "consumer_index_uses_python_ast_loads": True,
            "canonical_target_references_are_not_counted_as_legacy_consumers": True,
            "consumer_count_keys": ["internal", "tests", "documentation", "historical"],
            "files_scanned": len(files),
            "python_files_analyzed": len(analyses),
            "compatibility_notes": len(notes),
            "entrypoints": len(entrypoints),
            "excluded_directory_names": sorted(EXCLUDED_DIR_NAMES),
            "excluded_binary_suffixes": sorted(EXCLUDED_FILE_SUFFIXES),
            "protected_symbols": list(PROTECTED_SYMBOLS),
            "frozen_files": list(FROZEN_FILES),
            "public_surfaces": list(PUBLIC_SURFACES),
        },
        "classification_model": {
            "required_classifications": list(REQUIRED_CLASSIFICATIONS),
            "absence_of_internal_consumers_is_not_removal_proof": True,
            "public_or_distributed_surfaces_presumed_external": True,
            "blind_text_replacement_for_python_forbidden": True,
        },
        "summary": {
            "item_count": len(candidates),
            "classification_counts": dict(sorted(classification_counts.items())),
            "occurrence_type_counts": dict(sorted(occurrence_counts.items())),
            "risk_counts": dict(sorted(risk_counts.items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "consumer_reference_totals": {
                "internal": consumer_totals.get("internal", 0),
                "tests": consumer_totals.get("test", 0),
                "documentation": consumer_totals.get("documentary", 0),
                "historical": consumer_totals.get("historical", 0),
            },
            "syntax_errors": len(syntax_errors),
            "manual_decision_items": sum(
                1
                for candidate in candidates
                if "item ambíguo que exige decisão manual" in candidate.classifications
                or candidate.application_wave == "decisão manual"
            ),
            "removal_candidates": sum(
                1
                for candidate in candidates
                if "compatibilidade transitória removível" in candidate.classifications
            ),
            "blocked_items": sum(
                1 for candidate in candidates if candidate.application_wave == "bloqueada"
            ),
        },
        "syntax_errors": syntax_errors,
        "entrypoints": [dataclasses.asdict(item) for item in entrypoints],
        "items": [candidate.as_dict() for candidate in candidates],
        "gate": {
            "inventory_approval_required": True,
            "productive_applicator_allowed": False,
            "productive_changes_allowed": False,
            "commit_allowed": False,
            "push_allowed": False,
            "integration_allowed": False,
            "message": "[BLOQUEIO] Não criar nem executar aplicador produtivo antes da aprovação expressa do inventário AP-004E.",
        },
    }
    contract_basis = {
        "schema_version": payload["schema_version"],
        "baseline": payload["baseline"],
        "scope": payload["scope"],
        "classification_model": payload["classification_model"],
        "summary": payload["summary"],
        "entrypoints": payload["entrypoints"],
        "items": payload["items"],
        "gate": payload["gate"],
    }
    payload["contract_fingerprint"] = fingerprint_payload(contract_basis)
    return payload


def md_escape(value: Any) -> str:
    text = str(value if value is not None else "—")
    return text.replace("|", "\\|").replace("\n", " ")


def render_inventory_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    baseline = payload["baseline"]
    lines = [
        "# AP-004E — Inventário de superfícies de compatibilidade",
        "",
        "> Inventário preparatório, reproduzível e sem alteração de código produtivo.",
        "",
        "## Baseline validada",
        "",
        f"- Branch: `{baseline['branch']}`",
        f"- Commit local: `{baseline['head']}`",
        f"- Commit remoto: `{baseline['remote_head']}`",
        f"- Divergência: `{baseline['divergence'][0]} {baseline['divergence'][1]}`",
        f"- Assunto: `{baseline['subject']}`",
        f"- Fingerprint do inventário: `{payload['contract_fingerprint']}`",
        "",
        "## Gate vigente",
        "",
        "```text",
        "[BLOQUEIO] Não criar nem executar aplicador produtivo.",
        "[BLOQUEIO] Não alterar código produtivo.",
        "[BLOQUEIO] Não criar commit.",
        "[BLOQUEIO] Não realizar push.",
        "[BLOQUEIO] Não integrar na branch refactor/academic-pipeline.",
        "```",
        "",
        "## Resumo",
        "",
        f"- Arquivos lidos: **{payload['scope']['files_scanned']}**",
        f"- Arquivos Python analisados por AST: **{payload['scope']['python_files_analyzed']}**",
        f"- Itens inventariados: **{summary['item_count']}**",
        f"- Itens para decisão manual: **{summary['manual_decision_items']}**",
        f"- Candidatos preparatórios à remoção: **{summary['removal_candidates']}**",
        f"- Itens bloqueados por colisão: **{summary['blocked_items']}**",
        f"- Erros de sintaxe encontrados: **{summary['syntax_errors']}**",
        "",
        "## Método",
        "",
        "O inventário combina AST, tokenização de comentários, resolução de imports e `__all__`, "
        "entrypoints de empacotamento, fachadas de módulos, aliases, wrappers simples, registries, "
        "resolução dinâmica, metadados executáveis e busca separada de consumidores produtivos, "
        "testes, documentação e artefatos históricos. A ausência de consumidor interno não é "
        "tratada como prova suficiente para remoção de superfície pública ou distribuída.",
        "",
        "## Contagens por classificação",
        "",
        "| Classificação | Quantidade |",
        "|---|---:|",
    ]
    for key, value in summary["classification_counts"].items():
        lines.append(f"| {md_escape(key)} | {value} |")

    lines.extend(
        [
            "",
            "## Itens inventariados",
            "",
            "| ID | Superfície atual | Canônica/destino | Arquivo:linha | Tipo | Consumidores I/T/D/H | Risco | Decisão proposta | Classificação |",
            "|---|---|---|---|---|---:|---|---|---|",
        ]
    )
    for item in payload["items"]:
        raw_counts = item.get("consumer_counts", {})
        counts = {
            "I": raw_counts.get("internal", 0),
            "T": raw_counts.get("tests", 0),
            "D": raw_counts.get("documentation", 0),
            "H": raw_counts.get("historical", 0),
        }
        count_text = "/".join(str(counts[key]) for key in ("I", "T", "D", "H"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{md_escape(item['candidate_id'])}`",
                    f"`{md_escape(item['current_name'])}`",
                    f"`{md_escape(item['canonical_surface'])}`",
                    f"`{md_escape(item['path'])}:{item['line']}`",
                    md_escape(item["occurrence_type"]),
                    count_text,
                    md_escape(item["risk"]),
                    md_escape(item["proposed_decision"]),
                    md_escape("; ".join(item["classifications"])),
                ]
            )
            + " |"
        )

    if payload["syntax_errors"]:
        lines.extend(["", "## Arquivos Python não analisados por AST", ""])
        for error in payload["syntax_errors"]:
            lines.append(f"- `{error['path']}`: {error['error']}")

    lines.extend(
        [
            "",
            "## Leitura dos consumidores",
            "",
            "- **I**: consumidor produtivo interno.",
            "- **T**: consumidor em teste.",
            "- **D**: consumidor documental.",
            "- **H**: consumidor em snapshot, fixture, manifesto ou artefato histórico.",
            "",
            "A listagem completa das evidências e referências está em "
            "`ap004e_compatibility_inventory.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_strategy_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    by_wave: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload["items"]:
        by_wave[item["application_wave"]].append(item)
    lines = [
        "# AP-004E — Estratégia para superfícies de compatibilidade",
        "",
        "## Princípio de decisão",
        "",
        "A AP-004E preserva por padrão superfícies públicas, distribuídas, protegidas ou "
        "dinamicamente resolvidas. Remoção só pode ser proposta quando houver evidência estrutural "
        "positiva, migração prévia dos consumidores e aprovação expressa. Ausência de referência "
        "interna, isoladamente, não autoriza remoção.",
        "",
        "## Ordem proposta",
        "",
        "1. Preservar entrypoints públicos, símbolos protegidos, arquivos congelados e decisões da AP-004B.",
        "2. Validar bridges, aliases canônicos e reexports com consumidores produtivos.",
        "3. Investigar registries, `getattr`, imports dinâmicos e strings operacionais.",
        "4. Separar referências exclusivas de testes, documentação e artefatos históricos.",
        "5. Submeter os candidatos privados sem consumidores a aprovação nominal.",
        "6. Bloquear qualquer item com colisão ou conflito de destino.",
        "7. Somente após aprovação, criar aplicador estrutural com AST, pré-validação integral, backup externo e rollback.",
        "",
        "## Ondas preparatórias",
        "",
    ]
    for wave in sorted(by_wave):
        items = by_wave[wave]
        lines.extend(
            [
                f"### {wave}",
                "",
                f"Quantidade: **{len(items)}**",
                "",
            ]
        )
        for item in items[:100]:
            lines.append(
                f"- `{item['candidate_id']}` — `{item['current_name']}` em "
                f"`{item['path']}:{item['line']}`: {item['proposed_decision']}. "
                f"Motivo: {item['reason']}"
            )
        if len(items) > 100:
            lines.append(f"- … {len(items) - 100} item(ns) adicionais no JSON.")
        lines.append("")

    lines.extend(
        [
            "## Critérios para eventual aplicador",
            "",
            "O aplicador futuro não está autorizado nesta preparação. Quando autorizado, deverá:",
            "",
            "- validar todos os candidatos e ondas antes da primeira escrita;",
            "- usar AST ou transformação estrutural equivalente em Python;",
            "- tratar strings, comentários e metadados somente após classificação semântica;",
            "- criar backup fora do repositório;",
            "- escrever atomicamente e restaurar integralmente em falha;",
            "- recusar estado parcialmente aplicado;",
            "- executar `py_compile`, `git diff --check`, testes específicos e a suíte consolidada;",
            "- preservar exatamente os três `xfail`, sem `xpass`;",
            "- não criar commit, não publicar e não integrar automaticamente.",
            "",
            "## Gate",
            "",
            "```text",
            "[BLOQUEIO] Inventário ainda não aprovado.",
            "[BLOQUEIO] Aplicador produtivo não criado.",
            "[BLOQUEIO] Alterações produtivas não autorizadas.",
            "```",
            "",
            f"Fingerprint contratual: `{payload['contract_fingerprint']}`",
            "",
            f"Itens totais: **{summary['item_count']}**; decisões manuais: "
            f"**{summary['manual_decision_items']}**; candidatos preparatórios à remoção: "
            f"**{summary['removal_candidates']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def render_characterization_test(payload: dict[str, Any]) -> str:
    counts = payload["summary"]["classification_counts"]
    expected_counts_literal = repr(dict(sorted(counts.items())))
    protected_literal = repr(list(PROTECTED_SYMBOLS))
    frozen_literal = repr(list(FROZEN_FILES))
    return f'''"""Contrato congelado do inventário AP-004E.

Gerado por tools/refactor/ap004e_inventory_compatibility.py.
Não editar manualmente: regenere o inventário após decisão explícita.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_HEAD = {EXPECTED_HEAD!r}
EXPECTED_SCHEMA = {SCHEMA_VERSION!r}
EXPECTED_FINGERPRINT = {payload['contract_fingerprint']!r}
EXPECTED_ITEM_COUNT = {payload['summary']['item_count']}
EXPECTED_CLASSIFICATION_COUNTS = {expected_counts_literal}
EXPECTED_PROTECTED_SYMBOLS = {protected_literal}
EXPECTED_FROZEN_FILES = {frozen_literal}


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        marker = parent / "docs/refactor/academic-pipeline/AP-004/ap004e_compatibility_inventory.json"
        if marker.is_file():
            return parent
    raise AssertionError("não foi possível localizar a raiz do repositório")


def _load_inventory() -> dict:
    path = _repo_root() / "docs/refactor/academic-pipeline/AP-004/ap004e_compatibility_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_basis(data: dict) -> str:
    basis = {{
        "schema_version": data["schema_version"],
        "baseline": data["baseline"],
        "scope": data["scope"],
        "classification_model": data["classification_model"],
        "summary": data["summary"],
        "entrypoints": data["entrypoints"],
        "items": data["items"],
        "gate": data["gate"],
    }}
    encoded = json.dumps(
        basis,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_ap004e_inventory_contract_is_frozen() -> None:
    data = _load_inventory()
    assert data["schema_version"] == EXPECTED_SCHEMA
    assert data["baseline"]["head"] == EXPECTED_HEAD
    assert data["baseline"]["remote_head"] == EXPECTED_HEAD
    assert data["baseline"]["divergence"] == [0, 0]
    assert data["summary"]["item_count"] == EXPECTED_ITEM_COUNT
    assert data["summary"]["classification_counts"] == EXPECTED_CLASSIFICATION_COUNTS
    assert data["contract_fingerprint"] == EXPECTED_FINGERPRINT
    assert _fingerprint_basis(data) == EXPECTED_FINGERPRINT


def test_ap004e_protected_and_frozen_surfaces_are_present() -> None:
    data = _load_inventory()
    assert data["scope"]["protected_symbols"] == EXPECTED_PROTECTED_SYMBOLS
    assert data["scope"]["frozen_files"] == EXPECTED_FROZEN_FILES
    current_names = {{item["current_name"] for item in data["items"]}}
    for symbol in EXPECTED_PROTECTED_SYMBOLS:
        assert symbol in current_names
    for path in EXPECTED_FROZEN_FILES:
        assert Path(path).stem in current_names


def test_ap004e_gate_blocks_productive_actions() -> None:
    data = _load_inventory()
    assert data["generation"]["read_only_product_scan"] is True
    assert data["generation"]["productive_code_changed"] is False
    assert data["generation"]["applicator_created"] is False
    assert data["generation"]["commit_created"] is False
    assert data["generation"]["push_performed"] is False
    assert data["generation"]["integration_performed"] is False
    assert data["gate"]["inventory_approval_required"] is True
    assert data["gate"]["productive_applicator_allowed"] is False
    assert data["gate"]["productive_changes_allowed"] is False
    assert data["gate"]["commit_allowed"] is False
    assert data["gate"]["push_allowed"] is False
    assert data["gate"]["integration_allowed"] is False
'''


def prepare_outputs(payload: dict[str, Any]) -> dict[Path, str]:
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return {
        INVENTORY_MD_REL: render_inventory_markdown(payload),
        STRATEGY_MD_REL: render_strategy_markdown(payload),
        INVENTORY_JSON_REL: json_text,
        TEST_REL: render_characterization_test(payload),
    }


def print_summary(
    *,
    repo_root: Path,
    payload: dict[str, Any],
    backup_dir: Path | None,
    changed: Sequence[str],
    dry_run: bool,
) -> None:
    summary = payload["summary"]
    print("=== AP-004E — INVENTÁRIO PREPARATÓRIO ===")
    print(f"Repositório: {repo_root}")
    print(f"Branch: {payload['baseline']['branch']}")
    print(f"HEAD: {payload['baseline']['head']}")
    print(f"Arquivos lidos: {payload['scope']['files_scanned']}")
    print(f"Arquivos Python/AST: {payload['scope']['python_files_analyzed']}")
    print(f"Itens: {summary['item_count']}")
    print(f"Decisão manual: {summary['manual_decision_items']}")
    print(f"Candidatos preparatórios à remoção: {summary['removal_candidates']}")
    print(f"Bloqueados por colisão: {summary['blocked_items']}")
    print(f"Fingerprint: {payload['contract_fingerprint']}")
    if dry_run:
        print("Modo: DRY-RUN; nenhum artefato foi escrito.")
    else:
        print(f"Backup externo: {backup_dir}")
        if changed:
            print("Arquivos criados/atualizados:")
            for rel in changed:
                print(f"- {rel}")
        else:
            print("Arquivos já estavam idênticos; nenhuma reescrita necessária.")
    print("[BLOQUEIO] Não criar nem executar aplicador produtivo.")
    print("[BLOQUEIO] Não alterar código produtivo.")
    print("[BLOQUEIO] Não criar commit, push ou integração.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventaria superfícies de compatibilidade da AP-004E sem alterar código produtivo."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=EXPECTED_REPOSITORY,
        help=f"raiz Git (padrão: {EXPECTED_REPOSITORY})",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="não executar git fetch origin antes de validar a baseline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="executar inventário e validações sem escrever artefatos",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        baseline = validate_repo(repo_root, fetch=not args.no_fetch)
        files = list(iter_files(repo_root))
        candidate_origin_files = [
            path
            for path in files
            if is_candidate_origin_path(path.relative_to(repo_root).as_posix())
        ]
        analyses: list[FileAnalysis] = []
        notes: list[CompatibilityNote] = []
        for path in candidate_origin_files:
            if path.suffix.lower() in {".py", ".pyi"}:
                analysis = analyze_python(repo_root, path)
                analyses.append(analysis)
                notes.extend(analysis.comments)
            else:
                notes.extend(scan_non_python_notes(repo_root, path))

        syntax_errors = [analysis for analysis in analyses if analysis.syntax_error]
        if syntax_errors:
            details = "; ".join(
                f"{analysis.path}: {analysis.syntax_error}"
                for analysis in syntax_errors
            )
            raise InventoryError(
                "há erro(s) de sintaxe Python dentro da raiz produtiva canônica: "
                + details
            )

        entrypoints = discover_entrypoints(repo_root, candidate_origin_files)
        identifier_index, sources = make_text_indexes(repo_root, files)
        ap004b_pairs = extract_ap004b_pairs(repo_root, sources)
        candidates = build_candidates(
            analyses,
            entrypoints,
            notes,
            ap004b_pairs,
            identifier_index,
        )
        payload = build_inventory_payload(
            baseline=baseline,
            files=files,
            analyses=analyses,
            notes=notes,
            entrypoints=entrypoints,
            candidates=candidates,
            repo_root=repo_root,
        )
        outputs = prepare_outputs(payload)
        backup_dir: Path | None = None
        changed: list[str] = []
        if not args.dry_run:
            backup_dir, changed = transaction_write(repo_root, outputs)
        print_summary(
            repo_root=repo_root,
            payload=payload,
            backup_dir=backup_dir,
            changed=changed,
            dry_run=args.dry_run,
        )
        return 0
    except InventoryError as exc:
        print(f"ERRO SEGURO: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERRO INESPERADO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

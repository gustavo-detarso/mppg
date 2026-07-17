#!/usr/bin/env python3
"""Inventário preparatório de consumidores e dependências da AP-005A."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence


BASE_COMMIT = "f45c123bc692b80f4796b701fe71019630dba2f5"
BASE_BRANCH = "refactor/academic-pipeline"
TARGET_BRANCH = "ap-refactor/04-consumer-canonicalization"

SOFTWARE_ROOT = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)
AP004E_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-004/"
    "ap004e_compatibility_inventory.json"
)

JSON_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005a_consumer_dependency_inventory.json"
)
INVENTORY_MD_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005A_CONSUMER_DEPENDENCY_INVENTORY.md"
)
STRATEGY_MD_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005A_CONSUMER_MIGRATION_STRATEGY.md"
)
TOOL_REL = pathlib.PurePosixPath(
    "tools/refactor/ap005a_inventory_consumers.py"
)
TEST_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "tests/characterization/"
    "test_ap005a_consumer_dependency_inventory_contract.py"
)

ALLOWED_OUTPUTS = [
    str(TOOL_REL),
    str(JSON_REL),
    str(INVENTORY_MD_REL),
    str(STRATEGY_MD_REL),
    str(TEST_REL),
]

GENERIC_NAMES = {
    "main",
    "module",
    "path",
    "root",
    "documento",
    "article",
    "artigo",
    "chave",
    "prisma",
    "academic_pipeline",
}

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cfg",
    ".csv",
    ".el",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".org",
    ".py",
    ".pyproject",
    ".r",
    ".rst",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

MAX_EVIDENCE_PER_CATEGORY = 40


@dataclass(frozen=True)
class GitBlob:
    mode: str
    kind: str
    oid: str
    path: pathlib.PurePosixPath


@dataclass(frozen=True)
class ImportRecord:
    line: int
    local_name: str
    qualified_name: str
    module_name: str
    kind: str
    context: str


@dataclass
class PythonAnalysis:
    path: pathlib.PurePosixPath
    module_name: str | None
    source: str
    tree: ast.AST
    imports: list[ImportRecord] = field(default_factory=list)
    module_aliases: dict[str, str] = field(default_factory=dict)
    imported_names: dict[str, str] = field(default_factory=dict)
    graph_targets: set[str] = field(default_factory=set)
    dynamic_operations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Evidence:
    category: str
    path: str
    line: int
    kind: str
    confidence: str
    context: str

    def key(self) -> tuple[Any, ...]:
        return (
            self.category,
            self.path,
            self.line,
            self.kind,
            self.confidence,
            self.context,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "path": self.path,
            "line": self.line,
            "kind": self.kind,
            "confidence": self.confidence,
            "context": self.context,
        }


class GitRepository:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self._blob_cache: dict[str, bytes] = {}

    def run(
        self,
        *arguments: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        completed = subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if check and completed.returncode != 0:
            sys.stderr.buffer.write(completed.stderr)
            raise SystemExit(
                f"Falha em git {' '.join(arguments)}: "
                f"código {completed.returncode}"
            )

        return completed

    def text(self, *arguments: str) -> str:
        return self.run(*arguments).stdout.decode("utf-8").strip()

    def blobs_at(self, revision: str) -> list[GitBlob]:
        raw = self.run(
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            revision,
        ).stdout

        result: list[GitBlob] = []

        for record in raw.split(b"\0"):
            if not record:
                continue

            metadata, encoded_path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split()
            path = pathlib.PurePosixPath(
                encoded_path.decode(
                    "utf-8",
                    errors="surrogateescape",
                )
            )
            result.append(
                GitBlob(
                    mode=mode,
                    kind=kind,
                    oid=oid,
                    path=path,
                )
            )

        return result

    def blob_bytes(self, oid: str) -> bytes:
        if oid not in self._blob_cache:
            self._blob_cache[oid] = self.run(
                "cat-file",
                "blob",
                oid,
            ).stdout
        return self._blob_cache[oid]

    def commit_time(self, revision: str) -> str:
        return self.text(
            "show",
            "-s",
            "--format=%cI",
            revision,
        )

    def subject(self, revision: str) -> str:
        return self.text(
            "show",
            "-s",
            "--format=%s",
            revision,
        )


def repository_root() -> pathlib.Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if completed.returncode != 0:
        raise SystemExit(
            "A ferramenta deve ser executada dentro do worktree Git."
        )

    return pathlib.Path(completed.stdout.strip()).resolve()


def is_under(
    path: pathlib.PurePosixPath,
    root: pathlib.PurePosixPath,
) -> bool:
    return path.parts[: len(root.parts)] == root.parts


def module_from_path(
    path: pathlib.PurePosixPath,
) -> str | None:
    if not is_under(path, SOFTWARE_ROOT):
        return None

    relative = path.relative_to(SOFTWARE_ROOT)

    if relative.suffix != ".py":
        return None

    if not relative.parts:
        return None

    if relative.parts[0] not in {
        "academic_pipeline",
        "app_bundle",
    }:
        return None

    parts = list(relative.parts)
    filename = pathlib.PurePosixPath(parts[-1])

    if filename.name == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = filename.stem

    return ".".join(parts) if parts else None



def resolve_relative_module(
    current_module: str | None,
    is_package: bool,
    level: int,
    imported_module: str | None,
) -> str:
    """Resolve imports absolutos e relativos sem misturá-los."""

    imported_module = imported_module or ""

    # level == 0 representa um import absoluto. O módulo deve
    # permanecer exatamente como declarado no código-fonte.
    if level == 0:
        return imported_module

    if not current_module:
        return imported_module

    current_parts = current_module.split(".")

    package_parts = (
        current_parts
        if is_package
        else current_parts[:-1]
    )

    trim = max(level - 1, 0)

    if trim:
        package_parts = package_parts[:-trim]

    if imported_module:
        package_parts = [
            *package_parts,
            *imported_module.split("."),
        ]

    return ".".join(
        part
        for part in package_parts
        if part
    )

def ast_context(
    source: str,
    node: ast.AST,
) -> str:
    segment = ast.get_source_segment(source, node)

    if not segment:
        segment = type(node).__name__

    normalized = " ".join(segment.strip().split())

    if len(normalized) > 240:
        normalized = normalized[:237] + "..."

    return normalized


def analyze_python(
    path: pathlib.PurePosixPath,
    source: str,
) -> PythonAnalysis:
    tree = ast.parse(source, filename=str(path))
    module_name = module_from_path(path)
    is_package = path.name == "__init__.py"

    analysis = PythonAnalysis(
        path=path,
        module_name=module_name,
        source=source,
        tree=tree,
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]

                analysis.module_aliases[local_name] = alias.name
                analysis.graph_targets.add(alias.name)
                analysis.imports.append(
                    ImportRecord(
                        line=node.lineno,
                        local_name=local_name,
                        qualified_name=alias.name,
                        module_name=alias.name,
                        kind="AST-IMPORT-MODULE",
                        context=ast_context(source, node),
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            base_module = resolve_relative_module(
                module_name,
                is_package,
                node.level,
                node.module,
            )

            if base_module:
                analysis.graph_targets.add(base_module)

            for alias in node.names:
                if alias.name == "*":
                    qualified = f"{base_module}.*"
                    local_name = "*"
                else:
                    qualified = (
                        f"{base_module}.{alias.name}"
                        if base_module
                        else alias.name
                    )
                    local_name = alias.asname or alias.name

                analysis.imported_names[local_name] = qualified
                analysis.imports.append(
                    ImportRecord(
                        line=node.lineno,
                        local_name=local_name,
                        qualified_name=qualified,
                        module_name=base_module,
                        kind="AST-IMPORT-NAME",
                        context=ast_context(source, node),
                    )
                )

        elif isinstance(node, ast.Call):
            operation: dict[str, Any] | None = None

            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                operation = {
                    "kind": "getattr",
                    "line": node.lineno,
                    "symbol": node.args[1].value,
                    "context": ast_context(source, node),
                }

                if isinstance(node.args[0], ast.Name):
                    operation["object_name"] = node.args[0].id
                    operation["object_module"] = (
                        analysis.module_aliases.get(
                            node.args[0].id
                        )
                    )

            elif (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                operation = {
                    "kind": "__import__",
                    "line": node.lineno,
                    "module": node.args[0].value,
                    "context": ast_context(source, node),
                }

            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                operation = {
                    "kind": "importlib.import_module",
                    "line": node.lineno,
                    "module": node.args[0].value,
                    "context": ast_context(source, node),
                }

            if operation is not None:
                analysis.dynamic_operations.append(operation)

    return analysis



def category_for(path: pathlib.PurePosixPath) -> str:
    parts = path.parts
    rendered = f"/{path}/"

    if (
        "tests" in parts
        or any(
            part.startswith("test_")
            for part in parts
        )
    ):
        return "tests"

    historical_directory_names = {
        ".patch_backups",
        "__pycache__",
        "backups",
        "build",
        "dist",
        "execucoes_anteriores",
        "fixtures",
        "output",
        "output_pesquisa",
        "outputs",
        "snapshots",
    }

    if (
        "/AP-003/" in rendered
        or "/AP-004/" in rendered
        or any(
            part in historical_directory_names
            for part in parts
        )
        or (
            len(parts) >= 2
            and parts[:2] == ("tools", "refactor")
        )
        or (
            is_under(path, SOFTWARE_ROOT)
            and len(
                path.relative_to(
                    SOFTWARE_ROOT
                ).parts
            )
            >= 2
            and path.relative_to(
                SOFTWARE_ROOT
            ).parts[:2]
            == ("tools", "refactor")
        )
    ):
        return "historical"

    if (
        "docs" in parts
        or path.suffix.lower()
        in {".md", ".org", ".rst"}
    ):
        return "documentation"

    if is_under(path, SOFTWARE_ROOT):
        return "internal"

    return "historical"

def normalize_module(
    module_name: str,
    known_modules: set[str],
    basename_index: dict[str, set[str]],
) -> str | None:
    if module_name in known_modules:
        return module_name

    candidates = basename_index.get(module_name)

    if candidates and len(candidates) == 1:
        return next(iter(candidates))

    parts = module_name.split(".")

    while parts:
        candidate = ".".join(parts)

        if candidate in known_modules:
            return candidate

        parts.pop()

    return None


def strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index

        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in sorted(graph.get(node, set())):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(
                    lowlinks[node],
                    lowlinks[target],
                )
            elif target in on_stack:
                lowlinks[node] = min(
                    lowlinks[node],
                    indices[target],
                )

        if lowlinks[node] == indices[node]:
            component: list[str] = []

            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)

                if member == node:
                    break

            components.append(sorted(component))

    for node in sorted(graph):
        if node not in indices:
            visit(node)

    return sorted(
        components,
        key=lambda component: (
            -len(component),
            component,
        ),
    )


def item_source_module(
    item: dict[str, Any],
) -> str | None:
    ast_scope = item.get("ast_scope")

    if isinstance(ast_scope, str) and ast_scope:
        return ast_scope

    raw_path = item.get("path")

    if isinstance(raw_path, str):
        return module_from_path(
            pathlib.PurePosixPath(raw_path)
        )

    return None


def item_leaf_and_owner(
    item: dict[str, Any],
) -> tuple[str, str | None]:
    current = str(item.get("current_name") or "")

    if "." in current and "/" not in current:
        owner, leaf = current.rsplit(".", 1)
        return leaf, owner

    return current, None


def candidate_qualified_names(
    item: dict[str, Any],
    source_module: str | None,
    leaf: str,
) -> set[str]:
    candidates: set[str] = set()
    definition = item.get("definition")

    if isinstance(definition, str):
        normalized = definition.replace(":", ".")

        if " " not in normalized:
            candidates.add(normalized)

    if source_module and leaf:
        candidates.add(f"{source_module}.{leaf}")

    return candidates



def text_terms_for_item(
    item: dict[str, Any],
    source_module: str | None,
    leaf: str,
    owner: str | None,
) -> list[str]:
    """Retorna somente termos textuais semanticamente específicos."""

    current = str(item.get("current_name") or "")
    occurrence = str(
        item.get("occurrence_type") or ""
    )
    terms: set[str] = set()

    if occurrence == "entrypoint":
        if current == "academic_pipeline":
            terms.add("python -m academic_pipeline")
        elif current == "academic-pipeline":
            terms.add(
                'academic-pipeline = '
                '"academic_pipeline.cli:main"'
            )

        return sorted(
            terms,
            key=lambda term: (-len(term), term),
        )

    definition = item.get("definition")

    if (
        isinstance(definition, str)
        and len(definition) >= 4
        and " " not in definition
    ):
        terms.add(definition)

    if owner and leaf:
        terms.add(f"{owner}.{leaf}")

    if (
        current
        and current.lower() not in GENERIC_NAMES
    ):
        terms.add(current)

    if (
        source_module
        and leaf
        and leaf.lower() not in GENERIC_NAMES
    ):
        terms.add(f"{source_module}.{leaf}")
        terms.add(f"{source_module}:{leaf}")

    raw_path = item.get("path")

    if (
        occurrence
        in {
            "module facade",
            "historical frozen file",
            "AP-004B compatibility decision",
        }
        and isinstance(raw_path, str)
    ):
        terms.add(raw_path)
        terms.add(
            pathlib.PurePosixPath(raw_path).name
        )

    return sorted(
        (
            term
            for term in terms
            if term and len(term) >= 3
        ),
        key=lambda term: (-len(term), term),
    )

def line_context(line: str) -> str:
    normalized = " ".join(line.strip().split())

    if len(normalized) > 240:
        normalized = normalized[:237] + "..."

    return normalized


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", term):
        return re.compile(
            rf"(?<![A-Za-z0-9_]){escaped}"
            rf"(?![A-Za-z0-9_])"
        )

    return re.compile(escaped)


def add_evidence(
    collection: dict[tuple[Any, ...], Evidence],
    evidence: Evidence,
) -> None:
    collection[evidence.key()] = evidence


def recommended_action(
    item: dict[str, Any],
    counts: dict[str, int],
    has_dynamic: bool,
    in_cycle: bool,
) -> str:
    wave = item.get("application_wave")

    if wave == "fora de remoção":
        return "preservar sem transformação na AP-005"

    if wave == "preservação":
        return "preservar e formalizar o contrato observado"

    if counts["internal"] > 0:
        qualifiers: list[str] = []

        if has_dynamic:
            qualifiers.append("resolução dinâmica")

        if in_cycle:
            qualifiers.append("ciclo de importação")

        suffix = (
            f"; tratar também {', '.join(qualifiers)}"
            if qualifiers
            else ""
        )

        return (
            "migrar consumidores internos para a superfície "
            f"canônica antes de revisar o wrapper{suffix}"
        )

    if counts["tests"] > 0:
        return (
            "auditar testes e contratos; nenhum consumidor "
            "produtivo interno confirmado"
        )

    return (
        "preservar provisoriamente e auditar consumidores "
        "externos não observáveis"
    )


def migration_priority(
    item: dict[str, Any],
    counts: dict[str, int],
    has_dynamic: bool,
    in_cycle: bool,
) -> str:
    if item.get("application_wave") != "migração prévia":
        return "não aplicável"

    if (
        item.get("risk") == "crítico"
        or has_dynamic
        or in_cycle
    ):
        return "alta"

    if counts["internal"] >= 2:
        return "média"

    return "baixa"


def canonical_payload_bytes(
    payload: dict[str, Any],
) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def contract_fingerprint(
    payload: dict[str, Any],
) -> str:
    copy = dict(payload)
    copy.pop("contract_fingerprint", None)

    return hashlib.sha256(
        canonical_payload_bytes(copy)
    ).hexdigest()


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_inventory_markdown(
    payload: dict[str, Any],
) -> str:
    baseline = payload["baseline"]
    summary = payload["summary"]
    scope = payload["scope"]

    lines = [
        "# AP-005A — Inventário de consumidores e dependências",
        "",
        "> Inventário preparatório, reproduzível e sem alteração "
        "de código produtivo.",
        "",
        "## Baseline",
        "",
        f"- Branch de trabalho: `{baseline['target_branch']}`",
        f"- Commit-base: `{baseline['source_commit']}`",
        f"- Branch-base: `{baseline['source_branch']}`",
        f"- Assunto: `{baseline['source_subject']}`",
        f"- Data do commit-base: `{baseline['source_commit_time']}`",
        f"- Fingerprint do contrato: "
        f"`{payload['contract_fingerprint']}`",
        "",
        "## Gate vigente",
        "",
        "```text",
        "[BLOQUEIO] Não alterar código produtivo.",
        "[BLOQUEIO] Não criar aplicador produtivo.",
        "[BLOQUEIO] Não remover wrappers, aliases, fachadas ou "
        "reexports.",
        "[BLOQUEIO] Não criar commit ou realizar push antes da "
        "aprovação expressa do inventário.",
        "```",
        "",
        "## Resumo",
        "",
        f"- Superfícies herdadas da AP-004E: "
        f"**{summary['surface_count']}**",
        f"- Superfícies em migração prévia: "
        f"**{summary['migration_wave_count']}**",
        f"- Superfícies congeladas ausentes: "
        f"**{summary['frozen_missing_count']}**",
        f"- Arquivos do corpus lidos: "
        f"**{scope['files_read']}**",
        f"- Arquivos Python analisados por AST: "
        f"**{scope['python_files_analyzed']}**",
        f"- Erros de sintaxe: **{summary['syntax_errors']}**",
        f"- Componentes cíclicos: "
        f"**{summary['cyclic_component_count']}**",
        f"- Superfícies com resolução dinâmica: "
        f"**{summary['items_with_dynamic_consumers']}**",
        f"- Superfícies com ambiguidades registradas: "
        f"**{summary['items_with_ambiguities']}**",
        f"- Candidatos autorizados à remoção: "
        f"**{summary['removal_candidates']}**",
        "",
        "## Método",
        "",
        "- O corpus é lido diretamente dos blobs do commit-base.",
        "- Backups, outputs, ambientes virtuais e diretórios "
        "excluídos pela AP-004E não são abertos.",
        "- Imports, nomes carregados, atributos e operações "
        "dinâmicas são analisados por AST.",
        "- Referências textuais são usadas apenas como evidência "
        "documental, histórica ou de metadados.",
        "- Referências ao destino canônico não são contadas como "
        "consumo do nome legado.",
        "- Ausência de consumidor interno não é tratada como prova "
        "de remoção.",
        "",
        "## Contagens por onda",
        "",
        "| Onda | Quantidade |",
        "|---|---:|",
    ]

    for key, value in sorted(
        summary["application_wave_counts"].items()
    ):
        lines.append(f"| {markdown_escape(key)} | {value} |")

    lines.extend(
        [
            "",
            "## Ciclos de importação",
            "",
        ]
    )

    if payload["import_graph"]["cyclic_components"]:
        for component in payload["import_graph"][
            "cyclic_components"
        ]:
            lines.append(
                "- " + " → ".join(f"`{name}`" for name in component)
            )
    else:
        lines.append("- Nenhum componente cíclico observado.")

    lines.extend(
        [
            "",
            "## Superfícies inventariadas",
            "",
            "| ID | Superfície | Origem | Onda | I/T/D/H | "
            "Dinâmica | Ciclo | Prioridade | Ação proposta |",
            "|---|---|---|---|---:|---|---|---|---|",
        ]
    )

    for item in payload["items"]:
        counts = item["consumer_counts"]
        count_text = (
            f"{counts['internal']}/"
            f"{counts['tests']}/"
            f"{counts['documentation']}/"
            f"{counts['historical']}"
        )
        cycle_text = (
            "sim"
            if item["cycle_membership"]
            else "não"
        )

        lines.append(
            "| "
            f"`{item['source_candidate_id']}` | "
            f"`{markdown_escape(item['current_name'])}` | "
            f"`{markdown_escape(item['path'])}:"
            f"{item['line']}` | "
            f"{markdown_escape(item['application_wave'])} | "
            f"{count_text} | "
            f"{'sim' if item['dynamic_consumers'] else 'não'} | "
            f"{cycle_text} | "
            f"{markdown_escape(item['migration_priority'])} | "
            f"{markdown_escape(item['recommended_action'])} |"
        )

    lines.extend(
        [
            "",
            "## Política de aprovação",
            "",
            "Este inventário não autoriza transformação produtiva. "
            "Após a auditoria nominal e a correção de falsos "
            "positivos, será necessária aprovação expressa antes "
            "de qualquer aplicador ou migração.",
            "",
        ]
    )

    return "\n".join(lines)


def render_strategy_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload["summary"]

    lines = [
        "# AP-005A — Estratégia de migração de consumidores",
        "",
        "## Princípio",
        "",
        "A AP-005A apenas descreve dependências e propõe a ordem de "
        "migração. Nenhuma superfície pode ser removida, alterada "
        "ou depreciada nesta etapa.",
        "",
        "## Estado observado",
        "",
        f"- Superfícies totais: **{summary['surface_count']}**",
        f"- Migração prévia: "
        f"**{summary['migration_wave_count']}**",
        f"- Com consumidores internos observados: "
        f"**{summary['items_with_internal_consumers']}**",
        f"- Com consumidores dinâmicos: "
        f"**{summary['items_with_dynamic_consumers']}**",
        f"- Associadas a ciclos: "
        f"**{summary['items_in_cycles']}**",
        f"- Sem consumidor interno observado: "
        f"**{summary['items_without_internal_consumers']}**",
        "",
        "## Ondas propostas",
        "",
        "### Onda 0 — Preservação obrigatória",
        "",
        "- entrypoints públicos;",
        "- wrappers históricos congelados;",
        "- superfícies ligadas aos três `xfail`;",
        "- decisões arquiteturais protegidas da AP-004B;",
        "- bridges dinâmicas ainda necessárias.",
        "",
        "### Onda 1 — Imports internos diretos",
        "",
        "Migrar imports semanticamente resolvidos para módulos e "
        "símbolos canônicos, preservando os contratos públicos.",
        "",
        "### Onda 2 — Cluster de orquestração PRISMA",
        "",
        "Tratar em conjunto os wrappers `*_impl_001` e sua relação "
        "com `_invoke_with_runtime`, evitando substituições "
        "textuais e verificando ciclos.",
        "",
        "### Onda 3 — Aliases do gerador TOML",
        "",
        "Migrar os consumidores dos aliases `_original` somente "
        "após caracterização focada do fluxo interativo.",
        "",
        "### Onda 4 — Reexports e fachadas",
        "",
        "Formalizar a API pública em `__init__.py`, fachadas e "
        "reexports. Ausência de consumidor interno não autoriza "
        "remoção de superfície distribuída.",
        "",
        "### Onda 5 — Revisão pós-migração",
        "",
        "Somente depois da migração integral, reexecutar o "
        "inventário e submeter nominalmente qualquer candidato à "
        "preservação, depreciação ou remoção.",
        "",
        "## Gates obrigatórios",
        "",
        "1. Auditorar todas as evidências de baixa confiança.",
        "2. Confirmar ciclos e imports dinâmicos.",
        "3. Aprovar expressamente o inventário AP-005A.",
        "4. Criar testes de caracterização focados.",
        "5. Criar aplicador apenas quando houver transformação "
        "estrutural comprovadamente segura.",
        "6. Executar testes focados e a suíte canônica.",
        "7. Revisar o diff antes de commit ou push.",
        "",
        "## Bloqueios",
        "",
        "```text",
        "alteração produtiva = bloqueada",
        "aplicador produtivo = bloqueado",
        "remoção = bloqueada",
        "commit = bloqueado até aprovação do inventário",
        "push = bloqueado até aprovação do inventário",
        "```",
        "",
    ]

    return "\n".join(lines)


def build_inventory(
    repo: GitRepository,
) -> dict[str, Any]:
    current_head = repo.text("rev-parse", "HEAD")

    ancestor = repo.run(
        "merge-base",
        "--is-ancestor",
        BASE_COMMIT,
        current_head,
        check=False,
    )

    if ancestor.returncode != 0:
        raise SystemExit(
            f"O commit-base {BASE_COMMIT} não é ancestral de HEAD."
        )

    blobs = repo.blobs_at(BASE_COMMIT)
    blobs_by_path = {
        blob.path: blob
        for blob in blobs
    }

    ap004e_blob = blobs_by_path.get(AP004E_REL)

    if ap004e_blob is None:
        raise SystemExit(
            f"Inventário AP-004E ausente em {BASE_COMMIT}."
        )

    ap004e_raw = repo.blob_bytes(ap004e_blob.oid)
    ap004e = json.loads(ap004e_raw.decode("utf-8"))

    if ap004e.get("schema_version") != (
        "ap004e.compatibility-inventory.v6"
    ):
        raise SystemExit(
            "Schema inesperado no inventário AP-004E."
        )

    inherited_items = ap004e.get("items")

    if not isinstance(inherited_items, list) or len(
        inherited_items
    ) != 64:
        raise SystemExit(
            "O inventário AP-004E não contém 64 superfícies."
        )

    scope = ap004e.get("scope", {})
    scan_roots = tuple(
        pathlib.PurePosixPath(value)
        for value in scope.get("scan_roots", [])
    )
    excluded_dirs = {
        str(value)
        for value in scope.get(
            "excluded_directory_names",
            [],
        )
    }
    excluded_suffixes = {
        str(value).lower()
        for value in scope.get(
            "excluded_binary_suffixes",
            [],
        )
    }

    if not scan_roots:
        raise SystemExit(
            "A política de raízes da AP-004E está ausente."
        )

    eligible: list[GitBlob] = []

    for blob in blobs:
        if not any(
            is_under(blob.path, root)
            for root in scan_roots
        ):
            continue

        if any(
            part in excluded_dirs
            for part in blob.path.parts
        ):
            continue

        lowered = str(blob.path).lower()

        if any(
            lowered.endswith(suffix)
            for suffix in excluded_suffixes
        ):
            continue

        eligible.append(blob)

    text_by_path: dict[pathlib.PurePosixPath, str] = {}
    decode_errors: list[dict[str, Any]] = []

    for blob in eligible:
        suffix = blob.path.suffix.lower()

        if suffix not in TEXT_SUFFIXES:
            continue

        raw = repo.blob_bytes(blob.oid)

        try:
            text_by_path[blob.path] = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            decode_errors.append(
                {
                    "path": str(blob.path),
                    "error": str(exc),
                }
            )

    python_analyses: list[PythonAnalysis] = []
    syntax_errors: list[dict[str, Any]] = []

    for path, source in sorted(
        text_by_path.items(),
        key=lambda pair: str(pair[0]),
    ):
        if path.suffix.lower() != ".py":
            continue

        try:
            python_analyses.append(
                analyze_python(path, source)
            )
        except SyntaxError as exc:
            syntax_errors.append(
                {
                    "path": str(path),
                    "line": exc.lineno,
                    "error": exc.msg,
                }
            )

    if syntax_errors:
        raise SystemExit(
            "O corpus Python elegível contém erros de sintaxe."
        )

    known_modules = {
        analysis.module_name
        for analysis in python_analyses
        if analysis.module_name
    }
    basename_index: dict[str, set[str]] = (
        collections.defaultdict(set)
    )

    for module in known_modules:
        basename_index[module.rsplit(".", 1)[-1]].add(module)

    required_modules = {
        "academic_pipeline.legacy",
        "academic_pipeline.prisma_generic_orchestration",
        (
            "app_bundle.scripts.pipeline."
            "academic_pipeline_rc10"
        ),
        (
            "app_bundle.scripts.pipeline."
            "article_workflow"
        ),
    }

    missing_required_modules = (
        required_modules - known_modules
    )

    if missing_required_modules:
        raise SystemExit(
            "O índice de módulos produtivos está incompleto: "
            + ", ".join(
                sorted(missing_required_modules)
            )
        )

    graph: dict[str, set[str]] = {
        module: set()
        for module in known_modules
    }

    for analysis in python_analyses:
        if not analysis.module_name:
            continue

        for target in analysis.graph_targets:
            normalized = normalize_module(
                target,
                known_modules,
                basename_index,
            )

            if normalized:
                graph[analysis.module_name].add(normalized)

    components = strongly_connected_components(graph)
    cyclic_components = [
        component
        for component in components
        if len(component) > 1
        or (
            len(component) == 1
            and component[0] in graph.get(component[0], set())
        )
    ]

    cycle_by_module: dict[str, list[str]] = {}

    for component in cyclic_components:
        for module in component:
            cycle_by_module[module] = component

    result_items: list[dict[str, Any]] = []

    for inherited in inherited_items:
        source_module = item_source_module(inherited)
        leaf, owner = item_leaf_and_owner(inherited)
        qualified_names = candidate_qualified_names(
            inherited,
            source_module,
            leaf,
        )
        evidence_map: dict[
            tuple[Any, ...],
            Evidence,
        ] = {}

        for analysis in python_analyses:
            category = category_for(analysis.path)

            for record in analysis.imports:
                if (
                    record.qualified_name in qualified_names
                    or (
                        source_module
                        and record.module_name == source_module
                        and record.local_name == leaf
                    )
                ):
                    add_evidence(
                        evidence_map,
                        Evidence(
                            category=category,
                            path=str(analysis.path),
                            line=record.line,
                            kind=record.kind,
                            confidence="alta",
                            context=record.context,
                        ),
                    )

            for node in ast.walk(analysis.tree):
                if isinstance(node, ast.Name) and isinstance(
                    node.ctx,
                    ast.Load,
                ):
                    bound = analysis.imported_names.get(
                        node.id
                    )

                    if bound in qualified_names:
                        add_evidence(
                            evidence_map,
                            Evidence(
                                category=category,
                                path=str(analysis.path),
                                line=node.lineno,
                                kind="AST-NAME-IMPORTED",
                                confidence="alta",
                                context=ast_context(
                                    analysis.source,
                                    node,
                                ),
                            ),
                        )

                    elif (
                        source_module
                        and analysis.module_name == source_module
                        and node.id == leaf
                        and node.lineno
                        != int(inherited.get("line") or 0)
                        and leaf.lower() not in GENERIC_NAMES
                    ):
                        add_evidence(
                            evidence_map,
                            Evidence(
                                category=category,
                                path=str(analysis.path),
                                line=node.lineno,
                                kind="AST-NAME-LOCAL",
                                confidence="média",
                                context=ast_context(
                                    analysis.source,
                                    node,
                                ),
                            ),
                        )

                elif isinstance(node, ast.Attribute):
                    if node.attr != leaf:
                        continue

                    matched = False

                    if isinstance(node.value, ast.Name):
                        value_name = node.value.id
                        imported_module = (
                            analysis.module_aliases.get(
                                value_name
                            )
                        )

                        if (
                            source_module
                            and imported_module == source_module
                        ):
                            matched = True

                        if owner:
                            imported_owner = (
                                analysis.imported_names.get(
                                    value_name
                                )
                            )

                            if imported_owner and (
                                imported_owner.endswith(
                                    f".{owner}"
                                )
                            ):
                                matched = True

                    if matched:
                        add_evidence(
                            evidence_map,
                            Evidence(
                                category=category,
                                path=str(analysis.path),
                                line=node.lineno,
                                kind="AST-ATTRIBUTE",
                                confidence="alta",
                                context=ast_context(
                                    analysis.source,
                                    node,
                                ),
                            ),
                        )

            for dynamic in analysis.dynamic_operations:
                matched = False
                occurrence_type = str(
                    inherited.get("occurrence_type") or ""
                )
                inherited_path = str(
                    inherited.get("path") or ""
                )

                if (
                    dynamic.get("kind") == "getattr"
                    and dynamic.get("symbol") == leaf
                ):
                    object_module = dynamic.get(
                        "object_module"
                    )

                    same_declared_bridge = (
                        occurrence_type == "getattr"
                        and str(analysis.path)
                        == inherited_path
                    )

                    same_source_module = (
                        source_module is not None
                        and analysis.module_name
                        == source_module
                    )

                    if (
                        same_declared_bridge
                        or object_module == source_module
                        or (
                            same_source_module
                            and int(
                                dynamic.get("line") or 0
                            )
                            == int(
                                inherited.get("line") or 0
                            )
                        )
                    ):
                        matched = True

                elif (
                    source_module is not None
                    and "module" in dynamic
                    and dynamic.get("module")
                    == source_module
                ):
                    matched = True

                if matched:
                    add_evidence(
                        evidence_map,
                        Evidence(
                            category=category,
                            path=str(analysis.path),
                            line=int(dynamic["line"]),
                            kind=(
                                "AST-DYNAMIC-"
                                + str(dynamic["kind"])
                            ),
                            confidence="alta",
                            context=str(dynamic["context"]),
                        ),
                    )

        terms = text_terms_for_item(
            inherited,
            source_module,
            leaf,
            owner,
        )
        patterns = [
            (term, term_pattern(term))
            for term in terms
        ]

        for path, source in text_by_path.items():
            if path.suffix.lower() == ".py":
                continue

            category = category_for(path)

            # Inventários históricos podem incorporar fontes
            # completas e milhares de repetições. Não representam
            # consumidores atuais.
            if (
                category == "historical"
                and path.suffix.lower() == ".json"
            ):
                continue

            hits_by_term: dict[str, int] = (
                collections.defaultdict(int)
            )

            for line_number, line in enumerate(
                source.splitlines(),
                start=1,
            ):
                for term, pattern in patterns:
                    if not pattern.search(line):
                        continue

                    maximum_hits = (
                        1
                        if category == "historical"
                        else 3
                    )

                    if (
                        hits_by_term[term]
                        >= maximum_hits
                    ):
                        continue

                    if (
                        category == "internal"
                        and path.name
                        == "pyproject.toml"
                    ):
                        confidence = "alta"
                    elif (
                        category == "internal"
                        and path.suffix.lower()
                        in {".sh", ".toml", ".yaml", ".yml"}
                        and (
                            "/" in term
                            or "python -m " in term
                            or " = " in term
                        )
                    ):
                        confidence = "média"
                    else:
                        confidence = "baixa"

                    hits_by_term[term] += 1

                    add_evidence(
                        evidence_map,
                        Evidence(
                            category=category,
                            path=str(path),
                            line=line_number,
                            kind="TEXT-EXACT",
                            confidence=confidence,
                            context=(
                                f"{term}: "
                                f"{line_context(line)}"
                            ),
                        ),
                    )
                    break

        evidences = sorted(
            evidence_map.values(),
            key=lambda evidence: evidence.key(),
        )

        grouped: dict[str, list[Evidence]] = {
            category: []
            for category in (
                "internal",
                "tests",
                "documentation",
                "historical",
            )
        }

        for evidence in evidences:
            grouped[evidence.category].append(evidence)

        counts = {
            category: len(values)
            for category, values in grouped.items()
        }
        files = {
            category: sorted(
                {
                    evidence.path
                    for evidence in values
                }
            )
            for category, values in grouped.items()
        }

        dynamic_consumers = [
            evidence.as_dict()
            for evidence in evidences
            if "DYNAMIC" in evidence.kind
        ]
        cycle_membership = (
            cycle_by_module.get(source_module or "")
        )


        ambiguities: list[str] = []
        contractual_notes: list[str] = []

        occurrence_type = str(
            inherited.get("occurrence_type") or ""
        )
        application_wave = str(
            inherited.get("application_wave") or ""
        )

        public_contract_surface = (
            application_wave
            in {"preservação", "fora de remoção"}
            and occurrence_type
            in {
                "entrypoint",
                "module facade",
                "AP-004B compatibility decision",
                "historical frozen file",
            }
        )

        structural_evidence = [
            evidence
            for evidence in evidences
            if evidence.kind.startswith("AST-")
            and evidence.confidence == "alta"
        ]

        unresolved_low_confidence = [
            evidence
            for evidence in evidences
            if evidence.confidence == "baixa"
            and evidence.category
            in {"internal", "tests"}
        ]

        if public_contract_surface:
            if occurrence_type == "entrypoint":
                contractual_notes.append(
                    "entrypoint público preservado; ausência "
                    "de consumidor interno não reduz seu contrato"
                )

            elif occurrence_type == "module facade":
                contractual_notes.append(
                    "fachada pública preservada por consumidores "
                    "operacionais e testes de compatibilidade"
                )

            elif occurrence_type == (
                "AP-004B compatibility decision"
            ):
                contractual_notes.append(
                    "decisão de compatibilidade da AP-004B; "
                    "evidências operacionais não constituem "
                    "ambiguidade de resolução"
                )

            elif occurrence_type == (
                "historical frozen file"
            ):
                contractual_notes.append(
                    "wrapper histórico congelado preservado "
                    "por contrato"
                )

        if (
            leaf.lower() in GENERIC_NAMES
            and not structural_evidence
            and not public_contract_surface
        ):
            ambiguities.append(
                "nome genérico sem evidência estrutural "
                "suficiente"
            )

        if (
            unresolved_low_confidence
            and not public_contract_surface
        ):
            ambiguities.append(
                "há evidência operacional ou de teste de baixa "
                "confiança sem confirmação estrutural"
            )

        if (
            inherited.get("application_wave")
            == "migração prévia"
            and counts["internal"] == 0
        ):
            ambiguities.append(
                "onda de migração prévia sem consumidor "
                "produtivo interno confirmado"
            )

        raw_path = pathlib.PurePosixPath(
            str(inherited.get("path") or "")
        )

        frozen_missing = (
            raw_path
            in {
                pathlib.PurePosixPath(value)
                for value in scope.get("frozen_files", [])
            }
            and raw_path not in blobs_by_path
        )

        if frozen_missing:
            note = (
                "wrapper histórico congelado ausente por contrato"
            )

            if note not in contractual_notes:
                contractual_notes.append(note)

        action = recommended_action(
            inherited,
            counts,
            bool(dynamic_consumers),
            bool(cycle_membership),
        )

        result_items.append(
            {
                "source_candidate_id": inherited[
                    "candidate_id"
                ],
                "current_name": inherited.get(
                    "current_name"
                ),
                "canonical_surface": inherited.get(
                    "canonical_surface"
                ),
                "definition": inherited.get("definition"),
                "path": inherited.get("path"),
                "line": inherited.get("line"),
                "occurrence_type": inherited.get(
                    "occurrence_type"
                ),
                "source_module": source_module,
                "application_wave": inherited.get(
                    "application_wave"
                ),
                "risk": inherited.get("risk"),
                "inherited_classifications": inherited.get(
                    "classifications",
                    [],
                ),
                "inherited_external_consumer_status": (
                    inherited.get("external_consumer_status")
                ),
                "consumer_counts": counts,
                "consumer_files": files,
                "internal_consumers": [
                    evidence.as_dict()
                    for evidence in grouped["internal"][
                        :MAX_EVIDENCE_PER_CATEGORY
                    ]
                ],
                "test_consumers": [
                    evidence.as_dict()
                    for evidence in grouped["tests"][
                        :MAX_EVIDENCE_PER_CATEGORY
                    ]
                ],
                "documentary_consumers": [
                    evidence.as_dict()
                    for evidence in grouped["documentation"][
                        :MAX_EVIDENCE_PER_CATEGORY
                    ]
                ],
                "historical_consumers": [
                    evidence.as_dict()
                    for evidence in grouped["historical"][
                        :MAX_EVIDENCE_PER_CATEGORY
                    ]
                ],
                "evidence_truncated": {
                    category: max(
                        0,
                        len(grouped[category])
                        - MAX_EVIDENCE_PER_CATEGORY,
                    )
                    for category in grouped
                },
                "dynamic_consumers": dynamic_consumers,
                "cycle_membership": cycle_membership,
                "frozen_missing": frozen_missing,
                "contractual_notes": contractual_notes,
                "ambiguities": ambiguities,
                "migration_priority": migration_priority(
                    inherited,
                    counts,
                    bool(dynamic_consumers),
                    bool(cycle_membership),
                ),
                "recommended_action": action,
                "removal_eligibility": (
                    "bloqueada na AP-005A"
                ),
                "post_migration_review_required": (
                    inherited.get("application_wave")
                    == "migração prévia"
                ),
            }
        )

    result_items.sort(
        key=lambda item: item["source_candidate_id"]
    )

    wave_counts = collections.Counter(
        item["application_wave"]
        for item in result_items
    )
    occurrence_counts = collections.Counter(
        item["occurrence_type"]
        for item in result_items
    )
    consumer_totals = {
        category: sum(
            item["consumer_counts"][category]
            for item in result_items
        )
        for category in (
            "internal",
            "tests",
            "documentation",
            "historical",
        )
    }

    corpus_fingerprint = hashlib.sha256()

    for blob in sorted(
        eligible,
        key=lambda value: str(value.path),
    ):
        corpus_fingerprint.update(
            str(blob.path).encode("utf-8")
        )
        corpus_fingerprint.update(b"\0")
        corpus_fingerprint.update(
            blob.oid.encode("ascii")
        )
        corpus_fingerprint.update(b"\0")

    payload: dict[str, Any] = {
        "schema_version": (
            "ap005a.consumer-dependency-inventory.v3"
        ),
        "phase": "AP-005A",
        "title": (
            "Inventário de consumidores e dependências"
        ),
        "baseline": {
            "target_branch": TARGET_BRANCH,
            "source_branch": BASE_BRANCH,
            "source_commit": BASE_COMMIT,
            "source_commit_time": repo.commit_time(
                BASE_COMMIT
            ),
            "source_subject": repo.subject(BASE_COMMIT),
            "ap004e_schema": ap004e.get(
                "schema_version"
            ),
            "ap004e_contract_fingerprint": ap004e.get(
                "contract_fingerprint"
            ),
            "ap004e_sha256": hashlib.sha256(
                ap004e_raw
            ).hexdigest(),
        },
        "gate": {
            "inventory_approval_required": True,
            "productive_changes_allowed": False,
            "productive_applicator_allowed": False,
            "removal_allowed": False,
            "commit_allowed": False,
            "push_allowed": False,
            "integration_allowed": False,
            "message": (
                "[BLOQUEIO] A AP-005A é exclusivamente "
                "preparatória."
            ),
        },
        "scope": {
            "source_revision": BASE_COMMIT,
            "source_is_immutable_git_tree": True,
            "scan_roots": [
                str(root)
                for root in scan_roots
            ],
            "excluded_directory_names": sorted(
                excluded_dirs
            ),
            "excluded_binary_suffixes": sorted(
                excluded_suffixes
            ),
            "tracked_files_at_source_revision": len(
                blobs
            ),
            "eligible_files": len(eligible),
            "files_read": len(text_by_path),
            "python_files_analyzed": len(
                python_analyses
            ),
            "decode_errors": decode_errors,
            "syntax_errors": syntax_errors,
            "corpus_fingerprint": (
                corpus_fingerprint.hexdigest()
            ),
            "python_ast_used": True,
            "dynamic_operations_analyzed": True,
            "canonical_target_references_not_counted_as_legacy": (
                True
            ),
            "external_consumers_not_directly_observable": True,
            "allowed_outputs": ALLOWED_OUTPUTS,
        },
        "import_graph": {
            "module_count": len(graph),
            "edge_count": sum(
                len(targets)
                for targets in graph.values()
            ),
            "cyclic_components": cyclic_components,
        },
        "summary": {
            "surface_count": len(result_items),
            "migration_wave_count": wave_counts[
                "migração prévia"
            ],
            "application_wave_counts": dict(
                sorted(wave_counts.items())
            ),
            "occurrence_type_counts": dict(
                sorted(occurrence_counts.items())
            ),
            "consumer_reference_totals": consumer_totals,
            "items_with_internal_consumers": sum(
                item["consumer_counts"]["internal"] > 0
                for item in result_items
            ),
            "items_without_internal_consumers": sum(
                item["consumer_counts"]["internal"] == 0
                for item in result_items
            ),
            "items_with_dynamic_consumers": sum(
                bool(item["dynamic_consumers"])
                for item in result_items
            ),
            "items_in_cycles": sum(
                bool(item["cycle_membership"])
                for item in result_items
            ),
            "items_with_ambiguities": sum(
                bool(item["ambiguities"])
                for item in result_items
            ),
            "frozen_missing_count": sum(
                item["frozen_missing"]
                for item in result_items
            ),
            "cyclic_component_count": len(
                cyclic_components
            ),
            "syntax_errors": len(syntax_errors),
            "decode_errors": len(decode_errors),
            "removal_candidates": 0,
            "productive_files_changed": 0,
        },
        "items": result_items,
        "generation": {
            "tool": str(TOOL_REL),
            "read_only_source_scan": True,
            "productive_code_changed": False,
            "applicator_created": False,
            "commit_created": False,
            "push_performed": False,
            "integration_performed": False,
        },
    }

    payload["contract_fingerprint"] = (
        contract_fingerprint(payload)
    )

    return payload


def json_text(payload: dict[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def atomic_write(
    path: pathlib.Path,
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(content)
        temporary = pathlib.Path(stream.name)

    os.replace(temporary, path)


def generated_outputs(
    payload: dict[str, Any],
) -> dict[pathlib.PurePosixPath, str]:
    return {
        JSON_REL: json_text(payload),
        INVENTORY_MD_REL: render_inventory_markdown(
            payload
        ),
        STRATEGY_MD_REL: render_strategy_markdown(
            payload
        ),
    }


def write_outputs(
    root: pathlib.Path,
    outputs: dict[pathlib.PurePosixPath, str],
) -> None:
    for relative, content in outputs.items():
        atomic_write(root / relative, content)


def check_outputs(
    root: pathlib.Path,
    outputs: dict[pathlib.PurePosixPath, str],
) -> None:
    mismatches: list[str] = []

    for relative, expected in outputs.items():
        path = root / relative

        if not path.is_file():
            mismatches.append(
                f"ausente: {relative}"
            )
            continue

        actual = path.read_text(encoding="utf-8")

        if actual != expected:
            mismatches.append(
                f"divergente: {relative}"
            )

    if mismatches:
        for mismatch in mismatches:
            print(mismatch, file=sys.stderr)

        raise SystemExit(
            "Os artefatos AP-005A não correspondem à "
            "regeneração determinística."
        )


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    action = parser.add_mutually_exclusive_group(
        required=True
    )
    action.add_argument(
        "--write",
        action="store_true",
        help="grava os artefatos preparatórios",
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="compara os artefatos com a regeneração",
    )

    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_arguments(arguments)
    root = repository_root()
    repo = GitRepository(root)
    payload = build_inventory(repo)
    outputs = generated_outputs(payload)

    if args.write:
        write_outputs(root, outputs)
        print(
            "Inventário AP-005A gerado com sucesso."
        )
    else:
        check_outputs(root, outputs)
        print(
            "Inventário AP-005A reproduzido sem divergências."
        )

    summary = payload["summary"]

    print(
        f"superfícies={summary['surface_count']} "
        f"migração_prévia="
        f"{summary['migration_wave_count']} "
        f"ciclos={summary['cyclic_component_count']} "
        f"dinâmicas="
        f"{summary['items_with_dynamic_consumers']} "
        f"ambiguidades="
        f"{summary['items_with_ambiguities']}"
    )
    print(
        f"fingerprint={payload['contract_fingerprint']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

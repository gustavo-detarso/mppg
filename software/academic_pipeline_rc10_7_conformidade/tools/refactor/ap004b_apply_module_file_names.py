#!/usr/bin/env python3
"""AP-004B — aplicador produtivo de nomes de módulos e arquivos.

Aplicador conservador vinculado ao inventário AP-004B v1.6 aprovado. Ele:

- valida diretório, branch, HEAD local/remoto e árvore preparatória exata;
- valida integralmente os manifests e os sete consumidores acionáveis do
  inventário v1.6;
- cria três módulos canônicos como cópias byte a byte das implementações;
- cria `pipeline_orchestrator.py` como alias canônico do orquestrador histórico;
- converte somente os outros três caminhos históricos em wrappers-loader;
- preserva byte a byte o orquestrador histórico congelado pela AP-003G;
- atualiza somente cinco ocorrências aprovadas em quatro consumidores;
- preserva integralmente os executores full-text v1_13 e v1_14;
- torna durável o contrato de inventário AP-004B;
- gera relatório, JSON, ferramenta reexecutável e contrato de aplicação;
- cria backup externo, usa escrita atômica e executa rollback integral;
- executa py_compile, git diff --check, suíte específica e suíte consolidada.

Execute a partir da raiz do software e mantenha este arquivo fora do
repositório, por exemplo em ~/Downloads.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import py_compile
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tokenize
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
EXPECTED_HEAD = "6de61fc9741035187836460d97da6d672708998a"
EXPECTED_AP004A_SUBJECT = (
    "chore(academic-pipeline): consolidar inventário de nomes da AP-004A"
)
SOFTWARE_PREFIX = "software/academic_pipeline_rc10_7_conformidade/"

PHASE = "AP-004B"
MODE = "module-file-application-v1.4"
TOOL_VERSION = 1
TOOL_REVISION = "1.4"
APPLICATION_SCHEMA_VERSION = 1
BASELINE_PASSED = 431
BASELINE_XFAILED = 3
EXPECTED_APPLICATION_TESTS = 17
EXPECTED_CONSOLIDATED_PASSED = BASELINE_PASSED + EXPECTED_APPLICATION_TESTS

DOC_DIR = Path("docs/refactor/academic-pipeline/AP-004")
INVENTORY_REPORT_REL = DOC_DIR / "AP-004B_MODULE_FILE_INVENTORY.md"
INVENTORY_STRATEGY_REL = DOC_DIR / "AP-004B_MODULE_FILE_STRATEGY.md"
INVENTORY_JSON_REL = DOC_DIR / "ap004b_module_file_inventory.json"
INVENTORY_TEST_REL = Path(
    "tests/characterization/test_ap004b_module_file_inventory_contract.py"
)
INVENTORY_TOOL_REL = Path("tools/refactor/ap004b_inventory_modules.py")
AP004A_TEST_REL = Path(
    "tests/characterization/test_ap004a_naming_inventory_contract.py"
)

APPLICATION_REPORT_REL = DOC_DIR / "AP-004B_MODULE_FILE_APPLICATION.md"
APPLICATION_JSON_REL = DOC_DIR / "ap004b_module_file_application.json"
APPLICATION_TEST_REL = Path(
    "tests/characterization/test_ap004b_module_file_application_contract.py"
)
APPLICATION_TOOL_REL = Path("tools/refactor/ap004b_apply_module_file_names.py")

PREPARATORY_DIRTY_PATHS = {
    AP004A_TEST_REL.as_posix(),
    INVENTORY_REPORT_REL.as_posix(),
    INVENTORY_STRATEGY_REL.as_posix(),
    INVENTORY_JSON_REL.as_posix(),
    INVENTORY_TEST_REL.as_posix(),
    INVENTORY_TOOL_REL.as_posix(),
}

CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "key": "pipeline_orchestrator",
        "historical": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "canonical": "app_bundle/scripts/pipeline/pipeline_orchestrator.py",
    },
    {
        "key": "toml_generator",
        "historical": (
            "app_bundle/scripts/pipeline/"
            "academic_pipeline_toml_generator_v0_3_1.py"
        ),
        "canonical": (
            "app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py"
        ),
    },
    {
        "key": "prisma_ai_prescreen_configurator",
        "historical": "configurar_pretriagem_ia_prisma_v16.py",
        "canonical": "configurar_pretriagem_ia_prisma.py",
    },
    {
        "key": "article_diagnostic_log",
        "historical": "gerar_log_diagnostico_artigo_v1_18.py",
        "canonical": "gerar_log_diagnostico_artigo.py",
    },
)

FULLTEXT_PATHS = (
    "executar_artigo_longo_fulltext_v1_13.py",
    "executar_artigo_longo_fulltext_v1_14.py",
)
FORBIDDEN_FULLTEXT_CANONICAL = "executar_artigo_longo_fulltext.py"

CONSUMER_REPLACEMENTS: tuple[dict[str, Any], ...] = (
    {
        "candidate_key": "pipeline_orchestrator",
        "path": "app_bundle/scripts/pipeline/academic_pipeline_gui.py",
        "line": 63,
        "old": "academic_pipeline_rc10.py",
        "new": "pipeline_orchestrator.py",
        "kind": "python_string_reference",
        "call_selector": "HERE.with_name",
    },
    {
        "candidate_key": "pipeline_orchestrator",
        "path": (
            "app_bundle/scripts/pipeline/"
            "academic_pipeline_toml_generator_interativo.py"
        ),
        "line": 4215,
        "old": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "new": "app_bundle/scripts/pipeline/pipeline_orchestrator.py",
        "kind": "python_path_assignment",
        "assignment_selector": "command_lines",
    },
    {
        "candidate_key": "pipeline_orchestrator",
        "path": (
            "app_bundle/scripts/pipeline/"
            "academic_pipeline_toml_generator_interativo.py"
        ),
        "line": 4216,
        "old": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "new": "app_bundle/scripts/pipeline/pipeline_orchestrator.py",
        "kind": "python_path_assignment",
        "assignment_selector": "command_lines",
    },
    {
        "candidate_key": "pipeline_orchestrator",
        "path": "app_bundle/scripts/pipeline/academic_pipeline_tui.py",
        "line": 39,
        "old": "academic_pipeline_rc10.py",
        "new": "pipeline_orchestrator.py",
        "kind": "python_string_reference",
        "call_selector": "HERE.with_name",
    },
    {
        "candidate_key": "pipeline_orchestrator",
        "path": "app_bundle/scripts/pipeline/prisma_congelar_artigo.py",
        "line": 186,
        "old": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "new": "app_bundle/scripts/pipeline/pipeline_orchestrator.py",
        "kind": "python_path_assignment",
        "assignment_selector": "pipeline",
    },
)

DEFERRED_ACTIONABLE: tuple[dict[str, Any], ...] = (
    {
        "candidate_key": "pipeline_orchestrator",
        "path": "executar_artigo_longo_fulltext_v1_13.py",
        "line": 9,
        "old": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    },
    {
        "candidate_key": "pipeline_orchestrator",
        "path": "executar_artigo_longo_fulltext_v1_14.py",
        "line": 9,
        "old": "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    },
)

PRODUCTIVE_CHANGED_PATHS = {
    *(item["canonical"] for item in CANDIDATES),
    *(
        item["historical"]
        for item in CANDIDATES
        if item["key"] != "pipeline_orchestrator"
    ),
    *(item["path"] for item in CONSUMER_REPLACEMENTS),
}
APPLICATION_ARTIFACT_PATHS = {
    APPLICATION_REPORT_REL.as_posix(),
    APPLICATION_JSON_REL.as_posix(),
    APPLICATION_TEST_REL.as_posix(),
    APPLICATION_TOOL_REL.as_posix(),
}
EXPECTED_DIRTY_PATHS = (
    PREPARATORY_DIRTY_PATHS
    | PRODUCTIVE_CHANGED_PATHS
    | APPLICATION_ARTIFACT_PATHS
)
APPLICATION_WRITE_PATHS = (
    PRODUCTIVE_CHANGED_PATHS
    | APPLICATION_ARTIFACT_PATHS
    | {INVENTORY_TEST_REL.as_posix()}
)

SPECIFIC_TEST_PATHS = (
    INVENTORY_TEST_REL.as_posix(),
    APPLICATION_TEST_REL.as_posix(),
    "app_bundle/tests/test_entrypoints_orchestration_characterization.py",
    "app_bundle/tests/test_package_imports_entrypoints.py",
    "app_bundle/tests/test_official_package_entrypoint.py",
    "app_bundle/tests/test_package_imports_prisma_core.py",
    "app_bundle/tests/test_package_imports_rendering.py",
    "app_bundle/tests/test_package_imports_support_services.py",
    "app_bundle/tests/test_packaging_metadata.py",
    "app_bundle/tests/test_rc10_configuration_characterization.py",
)

KNOWN_XFAIL_SYMBOLS = (
    "_refs_v6_strip_org",
    "extract_org_abstracts",
    "WorkflowState._normalize",
)


class ApplicationError(RuntimeError):
    """Erro controlado do aplicador AP-004B."""


def fail(message: str) -> NoReturn:
    raise ApplicationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def ast_sha256_bytes(data: bytes, *, filename: str) -> str:
    source = data.decode("utf-8")
    tree = ast.parse(source, filename=filename)
    dump = ast.dump(tree, include_attributes=False, annotate_fields=True)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return text.rstrip() + "\n"


def run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        tuple(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if check and result.returncode != 0:
        fail(
            f"Comando falhou ({result.returncode}): {' '.join(args)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


def git(repository_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(("git", *args), cwd=repository_root, check=check)


def software_relative(raw: str) -> str:
    path = raw.strip().strip('"').replace("\\", "/")
    if path.startswith(SOFTWARE_PREFIX):
        return path[len(SOFTWARE_PREFIX):]
    return path


def status_path(line: str) -> str:
    raw = line[2:].lstrip() if len(line) >= 2 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return software_relative(raw)


def ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return any(part in ignored for part in parts) or path.endswith((".pyc", ".pyo"))


def git_status_paths(repository_root: Path) -> tuple[set[str], set[str], list[str]]:
    result = git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    paths: set[str] = set()
    staged: set[str] = set()
    lines: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = status_path(line)
        if ephemeral(path):
            continue
        lines.append(line)
        paths.add(path)
        index_status = line[0] if line else " "
        if index_status not in {" ", "?"}:
            staged.add(path)
    return paths, staged, lines


def validate_git_state(
    software_root: Path,
    *,
    skip_remote_check: bool,
) -> tuple[Path, dict[str, Any]]:
    if software_root != EXPECTED_SOFTWARE_ROOT:
        fail(
            "Diretório incorreto. Execute em:\n"
            f"{EXPECTED_SOFTWARE_ROOT}\nAtual: {software_root}"
        )
    repository_root_raw = run(
        ("git", "rev-parse", "--show-toplevel"), cwd=software_root
    ).stdout.strip()
    repository_root = Path(repository_root_raw).resolve()
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
        fail(f"HEAD divergente: {head}; esperado: {EXPECTED_HEAD}")
    if not skip_remote_check:
        git(repository_root, "fetch", "origin")
        remote = git(repository_root, "rev-parse", EXPECTED_REMOTE_REF).stdout.strip()
        if remote != head:
            fail(f"HEAD remoto divergente: local={head}, remoto={remote}")
    else:
        remote = None
    paths, staged, lines = git_status_paths(repository_root)
    if staged:
        fail(
            "Há arquivos staged; o aplicador exige índice vazio:\n"
            + "\n".join(sorted(staged))
        )
    if paths != PREPARATORY_DIRTY_PATHS:
        fail(
            "Árvore preparatória AP-004B divergente.\n"
            f"Esperado: {sorted(PREPARATORY_DIRTY_PATHS)}\n"
            f"Atual: {sorted(paths)}\n"
            + "\n".join(lines)
        )
    return repository_root, {
        "branch": branch,
        "head": head,
        "remote_head": remote,
        "initial_status_paths": sorted(paths),
        "initial_status_lines": lines,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Não foi possível ler JSON {path}: {exc}")


def validate_manifest(
    software_root: Path,
    manifest: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(manifest, dict) or not manifest:
        fail(f"Manifesto {label} ausente ou vazio.")
    for relative, record in manifest.items():
        path = software_root / relative
        if not path.is_file():
            fail(f"Arquivo do manifesto {label} ausente: {relative}")
        actual = sha256_path(path)
        expected = record.get("sha256")
        if actual != expected:
            fail(
                f"Hash divergente no manifesto {label}: {relative}\n"
                f"Esperado: {expected}\nAtual: {actual}"
            )
        if path.suffix == ".py" and record.get("ast_sha256"):
            ast_actual = ast_sha256_bytes(path.read_bytes(), filename=relative)
            if ast_actual != record["ast_sha256"]:
                fail(f"Hash AST divergente no manifesto {label}: {relative}")


def action_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("candidate_key"),
        record.get("consumer_path"),
        record.get("line"),
        record.get("kind"),
        tuple(record.get("matched_tokens", [])),
        record.get("semantic_category"),
    )


def expected_action_identity(spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        spec["candidate_key"],
        spec["path"],
        spec["line"],
        spec.get("kind"),
        (spec["old"],),
        "actionable_productive",
    )


def validate_inventory(
    software_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    inventory_path = software_root / INVENTORY_JSON_REL
    data = load_json(inventory_path)
    required = {
        "phase": "AP-004B",
        "mode": "module-file-inventory-v1.6-read-only",
        "inventory_schema_version": 2,
        "tool_revision": "1.6",
        "inventory_revision": "1.6",
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            fail(
                f"Inventário AP-004B divergente em {key}: "
                f"{data.get(key)!r} != {expected!r}"
            )
    if data.get("git", {}).get("head") != EXPECTED_HEAD:
        fail("Inventário AP-004B não está vinculado ao HEAD aprovado.")
    stats = data.get("statistics", {})
    expected_stats = {
        "candidates": 6,
        "raw_candidate_reference_records": 297,
        "deduplicated_candidate_reference_records": 269,
        "effective_consumer_records": 31,
        "effective_consumer_files": 15,
        "actionable_productive_records": 7,
        "compatibility_contract_records": 24,
        "destination_collisions": 1,
    }
    for key, expected in expected_stats.items():
        if stats.get(key) != expected:
            fail(
                f"Estatística AP-004B divergente em {key}: "
                f"{stats.get(key)!r} != {expected!r}"
            )
    validation = data.get("validation", {})
    if validation.get("specific_suite", {}).get("passed") != 13:
        fail("Inventário AP-004B não registra 13 testes específicos aprovados.")
    consolidated = validation.get("consolidated_suite", {})
    if consolidated.get("passed") != BASELINE_PASSED or consolidated.get("xfailed") != BASELINE_XFAILED:
        fail(
            "Inventário AP-004B não registra baseline consolidado "
            f"{BASELINE_PASSED} passed, {BASELINE_XFAILED} xfailed."
        )
    candidate_map = {item["key"]: item for item in data.get("candidates", [])}
    for spec in CANDIDATES:
        item = candidate_map.get(spec["key"])
        if item is None:
            fail(f"Candidato ausente no inventário: {spec['key']}")
        if item.get("current_path") != spec["historical"]:
            fail(f"Caminho histórico divergente: {spec['key']}")
        if item.get("proposed_path") != spec["canonical"]:
            fail(f"Caminho canônico divergente: {spec['key']}")
        if item.get("classification") != "renomeação com compatibilidade":
            fail(f"Classificação divergente: {spec['key']}")
        source = item.get("source", {})
        historical = software_root / spec["historical"]
        if source.get("sha256") != sha256_path(historical):
            fail(f"Hash candidato divergente: {spec['historical']}")
    collision = data.get("destination_collisions", [{}])[0]
    if (
        collision.get("decision") != "suspended-manual-review-required"
        or collision.get("suspended_target") != FORBIDDEN_FULLTEXT_CANONICAL
        or collision.get("origins") != list(FULLTEXT_PATHS)
    ):
        fail("Colisão full-text não permanece suspensa como aprovado.")
    records = [
        item
        for item in data.get("reference_records", [])
        if item.get("semantic_category") == "actionable_productive"
    ]
    actual_identities = {action_identity(item) for item in records}
    expected_selected = {expected_action_identity(item) for item in CONSUMER_REPLACEMENTS}
    expected_deferred = {
        (
            item["candidate_key"],
            item["path"],
            item["line"],
            "python_string_reference",
            (item["old"],),
            "actionable_productive",
        )
        for item in DEFERRED_ACTIONABLE
    }
    if actual_identities != expected_selected | expected_deferred:
        fail(
            "Conjunto dos sete consumidores produtivos diverge do escopo aprovado.\n"
            f"Esperado: {sorted(expected_selected | expected_deferred)}\n"
            f"Atual: {sorted(actual_identities)}"
        )
    validate_manifest(
        software_root,
        data.get("source_manifest", {}),
        label="source_manifest",
    )
    validate_manifest(
        software_root,
        data.get("control_manifest", {}),
        label="control_manifest",
    )
    tool_record = data.get("tool", {})
    inventory_tool = software_root / INVENTORY_TOOL_REL
    if tool_record.get("sha256") and tool_record["sha256"] != sha256_path(inventory_tool):
        fail("Ferramenta AP-004B v1.6 no repositório diverge do inventário.")
    for relative in PREPARATORY_DIRTY_PATHS:
        if not (software_root / relative).is_file():
            fail(f"Artefato preparatório ausente: {relative}")
    if not (
        "EXPECTED_AP004A_SUBJECT"
        in (software_root / AP004A_TEST_REL).read_text(encoding="utf-8")
    ):
        fail("Contrato AP-004A não está na forma durável produzida pela v1.6.")
    if (software_root / FORBIDDEN_FULLTEXT_CANONICAL).exists():
        fail(f"Destino full-text suspenso já existe: {FORBIDDEN_FULLTEXT_CANONICAL}")
    for spec in CANDIDATES:
        if (software_root / spec["canonical"]).exists():
            fail(f"Destino canônico já existe: {spec['canonical']}")
    ap004a_commit = git(
        repository_root,
        "log",
        "--format=%H%x09%s",
        "-n",
        "20",
    ).stdout
    expected_line = f"{EXPECTED_HEAD}\t{EXPECTED_AP004A_SUBJECT}"
    if expected_line not in ap004a_commit:
        fail("Commit AP-004A aprovado não foi localizado no histórico recente.")
    return data


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def absolute_offset(offsets: Sequence[int], position: tuple[int, int]) -> int:
    line, column = position
    if line < 1 or line > len(offsets):
        fail(f"Posição token inválida: {position}")
    return offsets[line - 1] + column


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _assignment_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for child in target.elts:
            result.update(_assignment_target_names(child))
        return result
    return set()


def _string_nodes_in_assignment(
    tree: ast.AST,
    *,
    target_name: str,
    old: str,
) -> list[ast.Constant]:
    selected: list[ast.Constant] = []
    for node in ast.walk(tree):
        value: ast.AST | None = None
        names: set[str] = set()
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_assignment_target_names(target))
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names.update(_assignment_target_names(node.target))
            value = node.value
        if target_name not in names or value is None:
            continue
        for child in ast.walk(value):
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and old in child.value
            ):
                selected.append(child)
    return sorted(selected, key=lambda item: (item.lineno, item.col_offset))


def _string_nodes_in_call(
    tree: ast.AST,
    *,
    call_name: str,
    old: str,
) -> list[ast.Constant]:
    selected: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _dotted_name(node.func) != call_name:
            continue
        for child in [*node.args, *(kw.value for kw in node.keywords)]:
            for descendant in ast.walk(child):
                if (
                    isinstance(descendant, ast.Constant)
                    and isinstance(descendant.value, str)
                    and old in descendant.value
                ):
                    selected.append(descendant)
    return sorted(selected, key=lambda item: (item.lineno, item.col_offset))


def _char_column_from_utf8(line: str, byte_column: int) -> int:
    encoded = line.encode("utf-8")
    if byte_column < 0 or byte_column > len(encoded):
        fail(f"Coluna AST inválida: {byte_column}")
    try:
        return len(encoded[:byte_column].decode("utf-8"))
    except UnicodeDecodeError as exc:
        fail(f"Coluna AST divide caractere UTF-8: {byte_column}: {exc}")


def _ast_absolute_offset(
    source_lines: Sequence[str],
    offsets: Sequence[int],
    position: tuple[int, int],
) -> int:
    line, byte_column = position
    if line < 1 or line > len(source_lines):
        fail(f"Posição AST inválida: {position}")
    char_column = _char_column_from_utf8(source_lines[line - 1], byte_column)
    return offsets[line - 1] + char_column


def replace_approved_string_literals(
    source: str,
    *,
    relative: str,
    replacements: Sequence[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Substitui apenas literais aprovados por seletores AST duráveis.

    Números de linha são mantidos como evidência do inventário, não como chave
    primária. Isso evita falhas em strings multilinha, cujos tokens podem começar
    antes da linha em que o caminho foi encontrado pelo inventariador.
    """
    tree = ast.parse(source, filename=relative)
    offsets = line_offsets(source)
    source_lines = source.splitlines(keepends=True)
    if not source_lines:
        source_lines = [""]

    edits: list[tuple[int, int, str, dict[str, Any], ast.Constant]] = []
    evidence: list[dict[str, Any]] = []
    claimed: set[tuple[int, int, int, int]] = set()

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for spec in replacements:
        selector_kind = ""
        selector_value = ""
        if spec.get("assignment_selector"):
            selector_kind = "assignment"
            selector_value = str(spec["assignment_selector"])
        elif spec.get("call_selector"):
            selector_kind = "call"
            selector_value = str(spec["call_selector"])
        else:
            selector_kind = "line"
            selector_value = str(spec["line"])
        groups.setdefault(
            (selector_kind, selector_value, spec["old"], spec["new"]), []
        ).append(spec)

    for (selector_kind, selector_value, old, new), specs in groups.items():
        if selector_kind == "assignment":
            nodes = _string_nodes_in_assignment(
                tree, target_name=selector_value, old=old
            )
        elif selector_kind == "call":
            nodes = _string_nodes_in_call(tree, call_name=selector_value, old=old)
        else:
            expected_line = int(selector_value)
            nodes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and old in node.value
                and node.lineno <= expected_line <= (node.end_lineno or node.lineno)
            ]
            nodes.sort(key=lambda item: (item.lineno, item.col_offset))

        specs_sorted = sorted(specs, key=lambda item: int(item["line"]))
        if len(nodes) != len(specs_sorted):
            fail(
                "Seleção AST não única em "
                f"{relative} para {old!r} via "
                f"{selector_kind}={selector_value!r}; "
                f"esperadas: {len(specs_sorted)}, encontradas: {len(nodes)}; "
                f"linhas AST: {[node.lineno for node in nodes]}"
            )

        for spec, node in zip(specs_sorted, nodes):
            assert node.end_lineno is not None and node.end_col_offset is not None
            identity = (
                node.lineno,
                node.col_offset,
                node.end_lineno,
                node.end_col_offset,
            )
            if identity in claimed:
                fail(
                    f"Literal AST selecionado mais de uma vez em {relative}: "
                    f"{identity}"
                )
            claimed.add(identity)
            old_value = node.value
            if old_value.count(old) != 1:
                fail(
                    "Literal AST contém quantidade inesperada do caminho antigo "
                    f"em {relative}:{node.lineno}: {old_value.count(old)}"
                )
            new_value = old_value.replace(old, new)
            start = _ast_absolute_offset(
                source_lines, offsets, (node.lineno, node.col_offset)
            )
            end = _ast_absolute_offset(
                source_lines,
                offsets,
                (node.end_lineno, node.end_col_offset),
            )
            old_literal = source[start:end]
            raw_occurrences = old_literal.count(old)
            if raw_occurrences != 1:
                fail(
                    "Segmento-fonte AST não contém exatamente uma ocorrência "
                    f"do caminho antigo em {relative}:{node.lineno}; "
                    f"encontradas: {raw_occurrences}; segmento={old_literal!r}"
                )
            # Em strings comuns, o segmento inclui prefixo e aspas. Em f-strings,
            # ast.Constant representa somente a parte literal de ast.JoinedStr,
            # sem aspas nem expressões interpoladas. A substituição textual no
            # intervalo exato do nó preserva ambos os formatos sem tentar
            # ast.literal_eval(), que não aceita ast.JoinedStr.
            replacement_literal = old_literal.replace(old, new)
            edits.append((start, end, replacement_literal, spec, node))
            evidence.append({
                "path": relative,
                "inventory_line": spec["line"],
                "actual_line": node.lineno,
                "selector_kind": selector_kind,
                "selector_value": selector_value,
                "old_value": old_value,
                "new_value": new_value,
                "old_literal": old_literal,
                "new_literal": replacement_literal,
            })

    ordered = sorted(edits, key=lambda item: item[0])
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] > current[0]:
            fail(f"Edições AST sobrepostas em {relative}.")

    result = source
    for start, end, replacement_literal, _, _ in sorted(edits, reverse=True):
        result = result[:start] + replacement_literal + result[end:]
    ast.parse(result, filename=relative)
    if len(edits) != len(replacements):
        fail(f"Quantidade de edições divergente em {relative}.")
    return result, evidence


def validate_internal_string_rewriter() -> None:
    """Prova local dos seletores em literal comum e em f-string."""
    old = "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    new = "app_bundle/scripts/pipeline/pipeline_orchestrator.py"
    synthetic = normalize_text(
        "command_lines = [\n"
        f'    f"pipenv run python {old} --config {{config_path}}",\n'
        f"    f'pipenv run python {old} --config {{other_config}}',\n"
        "]\n"
        f'pipeline = Path("{old}")\n'
    )
    replacements = (
        {
            "candidate_key": "pipeline_orchestrator",
            "path": "synthetic.py",
            "line": 2,
            "old": old,
            "new": new,
            "kind": "python_path_assignment",
            "assignment_selector": "command_lines",
        },
        {
            "candidate_key": "pipeline_orchestrator",
            "path": "synthetic.py",
            "line": 3,
            "old": old,
            "new": new,
            "kind": "python_path_assignment",
            "assignment_selector": "command_lines",
        },
        {
            "candidate_key": "pipeline_orchestrator",
            "path": "synthetic.py",
            "line": 5,
            "old": old,
            "new": new,
            "kind": "python_path_assignment",
            "assignment_selector": "pipeline",
        },
    )
    updated, evidence = replace_approved_string_literals(
        synthetic, relative="synthetic.py", replacements=replacements
    )
    if len(evidence) != 3:
        fail(
            "Autoteste do reescritor AST retornou quantidade divergente de "
            f"evidências: {len(evidence)}"
        )
    if updated.count(new) != 3 or old in updated:
        fail("Autoteste do reescritor AST não migrou os três caminhos.")
    if "{config_path}" not in updated or "{other_config}" not in updated:
        fail("Autoteste do reescritor AST alterou expressões de f-string.")
    ast.parse(updated, filename="synthetic.py")


def canonical_alias_source(historical_filename: str) -> str:
    return normalize_text(
        f'''#!/usr/bin/env python3
# Alias canônico AP-004B para {historical_filename}.
# O orquestrador histórico permanece congelado byte a byte pela AP-003G.

from __future__ import annotations

import pathlib as _ap004b_alias_pathlib

_ap004b_alias_historical = _ap004b_alias_pathlib.Path(__file__).with_name(
    {historical_filename!r}
)
_ap004b_alias_source = _ap004b_alias_historical.read_bytes()
exec(
    compile(
        _ap004b_alias_source,
        str(_ap004b_alias_historical),
        "exec",
    ),
    globals(),
    globals(),
)

del _ap004b_alias_source
del _ap004b_alias_historical
del _ap004b_alias_pathlib
'''
    )


def wrapper_source(canonical_filename: str) -> str:
    return normalize_text(
        f'''#!/usr/bin/env python3
# Wrapper transitório AP-004B para {canonical_filename}.
# A implementação canônica é executada no namespace histórico.

from __future__ import annotations

import pathlib as _ap004b_compat_pathlib

_ap004b_compat_canonical = _ap004b_compat_pathlib.Path(__file__).with_name(
    {canonical_filename!r}
)
_ap004b_compat_source = _ap004b_compat_canonical.read_bytes()
exec(
    compile(
        _ap004b_compat_source,
        str(_ap004b_compat_canonical),
        "exec",
    ),
    globals(),
    globals(),
)

del _ap004b_compat_source
del _ap004b_compat_canonical
del _ap004b_compat_pathlib
'''
    )


@dataclass
class BackupRecord:
    path: Path
    existed: bool
    data: bytes | None
    mode: int | None


def create_backups(
    paths: Iterable[Path],
    *,
    software_root: Path,
) -> tuple[Path, list[BackupRecord]]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        Path.home()
        / ".cache/academic-pipeline-refactor/backups/AP-004B-APPLICATION"
        / stamp
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    records: list[BackupRecord] = []
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        existed = path.exists()
        data = path.read_bytes() if existed else None
        mode = stat.S_IMODE(path.stat().st_mode) if existed else None
        records.append(BackupRecord(path=path, existed=existed, data=data, mode=mode))
        relative = path.relative_to(software_root)
        metadata = backup_root / (relative.as_posix() + ".metadata.json")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(
                {
                    "path": relative.as_posix(),
                    "existed": existed,
                    "mode": mode,
                    "sha256": sha256_bytes(data) if data is not None else None,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if data is not None:
            payload = backup_root / relative
            payload.parent.mkdir(parents=True, exist_ok=True)
            payload.write_bytes(data)
    return backup_root, records


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def rollback(records: Sequence[BackupRecord]) -> None:
    errors: list[str] = []
    for record in reversed(records):
        try:
            if record.existed:
                assert record.data is not None
                atomic_write_bytes(record.path, record.data, mode=record.mode)
            elif record.path.exists():
                record.path.unlink()
        except Exception as exc:  # pragma: no cover - contingência operacional
            errors.append(f"{record.path}: {exc}")
    if errors:
        print("[ERRO] Rollback incompleto:\n" + "\n".join(errors), file=sys.stderr)


def current_mode(path: Path, default: int = 0o644) -> int:
    if path.exists():
        return stat.S_IMODE(path.stat().st_mode)
    return default


def build_durable_inventory_contract(
    *,
    inventory: dict[str, Any],
    expected_dirty_paths: Sequence[str],
) -> str:
    candidates = [item["current_path"] for item in inventory["candidates"]]
    source = f'''from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parents[1]
INVENTORY = ROOT / {INVENTORY_JSON_REL.as_posix()!r}
TOOL = ROOT / {INVENTORY_TOOL_REL.as_posix()!r}
EXPECTED_HEAD = {EXPECTED_HEAD!r}
EXPECTED_CANDIDATES = {candidates!r}
EXPECTED_DIRTY_PATHS = {sorted(expected_dirty_paths)!r}
SOFTWARE_PREFIX = {SOFTWARE_PREFIX!r}
EFFECTIVE = {{"actionable_productive", "compatibility_contract"}}
EXCLUDED = {{
    "historical_immutable", "physical_directory_reference",
    "protected_operational", "contextual_non_actionable",
}}


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
    raw = raw.strip().strip('"').replace("\\\\", "/")
    return raw[len(SOFTWARE_PREFIX):] if raw.startswith(SOFTWARE_PREFIX) else raw


def _ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {{"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}}
    return any(part in ignored for part in parts) or path.endswith((".pyc", ".pyo"))


def _baseline_bytes(relative: str) -> bytes:
    object_name = f"{{EXPECTED_HEAD}}:{{SOFTWARE_PREFIX}}{{relative}}"
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
    actual = {{
        (item["consumer_path"], item["line"], tuple(item.get("matched_tokens", [])))
        for item in data["reference_records"]
        if item["semantic_category"] == "actionable_productive"
    }}
    expected = {{
        (item["path"], item["line"], (item["old"],))
        for item in {list(CONSUMER_REPLACEMENTS) + list(DEFERRED_ACTIONABLE)!r}
    }}
    assert actual == expected


def test_ap004b_v1_6_collision_remains_suspended() -> None:
    collision = _data()["destination_collisions"][0]
    assert collision["decision"] == "suspended-manual-review-required"
    assert collision["suspended_target"] == {FORBIDDEN_FULLTEXT_CANONICAL!r}
    assert collision["origins"] == {list(FULLTEXT_PATHS)!r}


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
    assert set(data["control_manifest"]) == {{
        "docs/refactor/academic-pipeline/AP-004/AP-004_NAMING_CONVENTION.md",
        "docs/refactor/academic-pipeline/AP-004/ap004a_naming_inventory.json",
    }}
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
    assert _data()["protected"]["known_xfails"] == {list(KNOWN_XFAIL_SYMBOLS)!r}


def test_ap004b_v1_6_inventory_artifacts_and_tool_remain_available() -> None:
    data = _data()
    assert INVENTORY.is_file()
    assert TOOL.is_file()
    assert data["tool"]["sha256"] == _sha256(TOOL)


def test_ap004b_v1_6_current_status_is_application_scope_or_clean() -> None:
    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    actual = {{
        _status_path(line) for line in status.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }}
    assert actual == set(EXPECTED_DIRTY_PATHS) or actual == set()


def test_ap004b_v1_6_generated_python_compiles() -> None:
    ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))
    with tempfile.TemporaryDirectory(prefix="ap004b-inventory-durable-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)
'''
    source = normalize_text(source)
    tree = ast.parse(source, filename=INVENTORY_TEST_REL.as_posix())
    count = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in tree.body
    )
    if count != 13:
        fail(f"Contrato durável AP-004B deveria ter 13 testes; gerados: {count}")
    return source


def build_application_contract(
    *,
    tool_sha256: str,
    expected_dirty_paths: Sequence[str],
) -> str:
    source = f'''from __future__ import annotations

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
APPLICATION = ROOT / {APPLICATION_JSON_REL.as_posix()!r}
TOOL = ROOT / {APPLICATION_TOOL_REL.as_posix()!r}
EXPECTED_HEAD = {EXPECTED_HEAD!r}
EXPECTED_TOOL_SHA256 = {tool_sha256!r}
EXPECTED_DIRTY_PATHS = {sorted(expected_dirty_paths)!r}
SOFTWARE_PREFIX = {SOFTWARE_PREFIX!r}
CANDIDATES = {list(CANDIDATES)!r}
REPLACEMENTS = {list(CONSUMER_REPLACEMENTS)!r}
FULLTEXT_PATHS = {list(FULLTEXT_PATHS)!r}
FORBIDDEN_FULLTEXT_CANONICAL = {FORBIDDEN_FULLTEXT_CANONICAL!r}


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
    raw = raw.strip().strip('"').replace("\\\\", "/")
    return raw[len(SOFTWARE_PREFIX):] if raw.startswith(SOFTWARE_PREFIX) else raw


def _ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {{"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}}
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
            assert _sha256(historical) == item["source_sha256_before"]
            assert item["historical_sha256_after"] == item["source_sha256_before"]
            assert _sha256(canonical) == item["canonical_sha256_after"]
            source = canonical.read_text(encoding="utf-8")
            assert "Alias canônico AP-004B" in source
            assert "academic_pipeline_rc10.py" in source
        else:
            assert item["migration_policy"] == "canonical-copy-with-historical-wrapper"
            assert _sha256(canonical) == item["source_sha256_before"]
            assert item["canonical_sha256_after"] == item["source_sha256_before"]
            tree = ast.parse(canonical.read_text(encoding="utf-8"), filename=str(canonical))
            dump = ast.dump(tree, include_attributes=False, annotate_fields=True)
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
        names = {{node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}}
        calls = {{
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }}
        assert "globals" in calls
        assert "compile" in calls
        assert "exec" in calls
        prefix = "_ap004b_alias_" if item["key"] == "pipeline_orchestrator" else "_ap004b_compat_"
        assert any(name.startswith(prefix) for name in names)


def test_ap004b_consumer_replacements_are_exact() -> None:
    data = _data()
    assert len(data["consumer_replacements"]) == 5
    for item in data["consumer_replacements"]:
        source = (ROOT / item["path"]).read_text(encoding="utf-8")
        line = source.splitlines()[item["line"] - 1]
        assert item["new"] in line
        assert item["old"] not in line
        ast.parse(source, filename=item["path"])


def test_ap004b_only_five_approved_runtime_occurrences_were_migrated() -> None:
    data = _data()
    actual = {{(item["path"], item["line"], item["old"], item["new"]) for item in data["consumer_replacements"]}}
    expected = {{(item["path"], item["line"], item["old"], item["new"]) for item in REPLACEMENTS}}
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
        assert _sha256(ROOT / relative) == expected["sha256_before"]


def test_ap004b_legacy_module_and_compatibility_contracts_remain() -> None:
    data = _data()
    assert _sha256(ROOT / "academic_pipeline/legacy.py") == data["unchanged_control_files"]["academic_pipeline/legacy.py"]["sha256_before"]
    assert data["scope"]["compatibility_contract_records_preserved"] == 24
    assert (ROOT / {AP004A_TEST_REL.as_posix()!r}).is_file()
    assert (ROOT / {INVENTORY_TEST_REL.as_posix()!r}).is_file()


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
                cfile=str(Path(tmp) / f"{{index}}.pyc"),
                doraise=True,
            )


def test_ap004b_known_xfails_and_frozen_orchestrator_remain_unchanged() -> None:
    data = _data()
    assert data["protected"]["known_xfails"] == {list(KNOWN_XFAIL_SYMBOLS)!r}
    orchestrator = next(item for item in data["module_migrations"] if item["key"] == "pipeline_orchestrator")
    assert orchestrator["historical_sha256_after"] == orchestrator["source_sha256_before"]
    assert _sha256(ROOT / orchestrator["historical_path"]) == orchestrator["source_sha256_before"]


def test_ap004b_git_diff_is_limited_to_approved_scope() -> None:
    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert status.returncode == 0
    actual = {{
        _status_path(line) for line in status.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }}
    assert actual == set(EXPECTED_DIRTY_PATHS) or actual == set()


def test_ap004b_application_artifacts_are_coherent() -> None:
    data = _data()
    assert TOOL.is_file()
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    assert data["tool"]["sha256"] == EXPECTED_TOOL_SHA256
    assert (ROOT / {APPLICATION_REPORT_REL.as_posix()!r}).is_file()
    assert data["scope"]["forbidden_fulltext_target"] == FORBIDDEN_FULLTEXT_CANONICAL


def test_ap004b_inventory_contract_is_durable_and_compiles() -> None:
    path = ROOT / {INVENTORY_TEST_REL.as_posix()!r}
    source = path.read_text(encoding="utf-8")
    assert "source_manifest_matches_baseline_commit" in source
    assert "current_status_is_application_scope_or_clean" in source
    ast.parse(source, filename=str(path))
    with tempfile.TemporaryDirectory(prefix="ap004b-inventory-contract-") as tmp:
        py_compile.compile(str(path), cfile=str(Path(tmp) / "inventory.pyc"), doraise=True)


def test_ap004b_ap004a_contract_remains_durable_and_compiles() -> None:
    path = ROOT / {AP004A_TEST_REL.as_posix()!r}
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
'''
    source = normalize_text(source)
    tree = ast.parse(source, filename=APPLICATION_TEST_REL.as_posix())
    count = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in tree.body
    )
    if count != EXPECTED_APPLICATION_TESTS:
        fail(
            f"Contrato de aplicação deveria ter {EXPECTED_APPLICATION_TESTS} "
            f"testes; gerados: {count}"
        )
    return source


def build_application_data(
    *,
    git_state: dict[str, Any],
    inventory: dict[str, Any],
    tool_source: bytes,
    module_migrations: list[dict[str, Any]],
    consumer_evidence: list[dict[str, Any]],
    fulltext_before: dict[str, dict[str, Any]],
    unchanged_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "mode": MODE,
        "application_schema_version": APPLICATION_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "tool": {
            "version": TOOL_VERSION,
            "revision": TOOL_REVISION,
            "path": APPLICATION_TOOL_REL.as_posix(),
            "sha256": sha256_bytes(tool_source),
        },
        "baseline": {
            "head": EXPECTED_HEAD,
            "branch": EXPECTED_BRANCH,
            "remote_ref": EXPECTED_REMOTE_REF,
            "git_state": git_state,
            "consolidated": {
                "passed": BASELINE_PASSED,
                "xfailed": BASELINE_XFAILED,
            },
        },
        "approval": {
            "approved": True,
            "inventory_revision": inventory["inventory_revision"],
            "inventory_mode": inventory["mode"],
            "inventory_sha256": sha256_path(EXPECTED_SOFTWARE_ROOT / INVENTORY_JSON_REL),
            "decision": (
                "orquestrador histórico congelado com alias canônico; três módulos "
                "canônicos com wrappers; cinco ocorrências produtivas migradas; "
                "dois consumidores full-text adiados"
            ),
        },
        "scope": {
            "module_migrations": 4,
            "selected_actionable_records": 5,
            "deferred_actionable_records": 2,
            "compatibility_contract_records_preserved": 24,
            "productive_changed_paths": sorted(PRODUCTIVE_CHANGED_PATHS),
            "application_artifacts": sorted(APPLICATION_ARTIFACT_PATHS),
            "expected_dirty_paths": sorted(EXPECTED_DIRTY_PATHS),
            "forbidden_fulltext_target": FORBIDDEN_FULLTEXT_CANONICAL,
            "functional_change": "proibida",
            "cli_semantics_change": "proibida",
            "physical_directory_rename": "proibida até AP-006",
        },
        "module_migrations": module_migrations,
        "consumer_replacements": consumer_evidence,
        "deferred_fulltext": fulltext_before,
        "unchanged_control_files": unchanged_controls,
        "protected": {
            "known_xfails": list(KNOWN_XFAIL_SYMBOLS),
            "fulltext_collision": "suspended-manual-review-required",
            "physical_directory": "academic_pipeline_rc10_7_conformidade",
        },
        "validation": {
            "py_compile": "pending",
            "git_diff_check": "pending",
            "specific_suite": {"status": "pending"},
            "consolidated_suite": {"status": "pending"},
        },
        "next_gate": {
            "commit_allowed": False,
            "requires_explicit_consolidation_approval": True,
            "integration_allowed": False,
        },
    }


def build_report(data: dict[str, Any]) -> str:
    migrations = "\n".join(
        f"- `{item['historical_path']}` → `{item['canonical_path']}` "
        f"(`{item['migration_policy']}`)."
        for item in data["module_migrations"]
    )
    replacements = "\n".join(
        f"- `{item['path']}:{item['line']}`: `{item['old']}` → `{item['new']}`."
        for item in data["consumer_replacements"]
    )
    validation = data["validation"]
    return normalize_text(
        f'''# AP-004B — aplicação produtiva de módulos e arquivos (v1.4)

> Aplicação vinculada ao inventário AP-004B v1.6 aprovado. Nenhum commit foi criado.

## Base canônica

- Branch: `{EXPECTED_BRANCH}`.
- HEAD local/remoto: `{EXPECTED_HEAD}`.
- Baseline: `{BASELINE_PASSED} passed, {BASELINE_XFAILED} xfailed`.
- Mudança funcional: **não**.
- Mudança de semântica CLI: **não**.

## Módulos canônicos e wrappers

{migrations}

O orquestrador `academic_pipeline_rc10.py` permanece byte a byte intacto para
preservar os contratos congelados da AP-003G; `pipeline_orchestrator.py` é um
alias canônico que executa essa implementação no próprio namespace. Nos outros
três casos, o caminho canônico é cópia da implementação e o caminho histórico
é um loader transitório. Símbolos privados, monkeypatches, argumentos, código
de saída e guarda `__main__` permanecem preservados.

## Consumidores migrados

{replacements}

Foram alteradas exatamente cinco ocorrências em quatro arquivos. Os dois usos
internos de `executar_artigo_longo_fulltext_v1_13.py` e
`executar_artigo_longo_fulltext_v1_14.py` permaneceram intocados e continuam
funcionais pelo orquestrador histórico preservado `academic_pipeline_rc10.py`.

## Colisão full-text

- `executar_artigo_longo_fulltext.py`: **não criado**.
- `v1_13` e `v1_14`: preservados byte a byte.
- Decisão: `suspended-manual-review-required`.

## Proteções

- `academic-pipeline` e `python -m academic_pipeline` preservados.
- `academic_pipeline/legacy.py` preservado.
- 24 contratos de compatibilidade preservados.
- Diretório físico reservado à AP-006.
- Três xfails históricos mantidos sem correção.

## Validação

- `py_compile`: `{validation['py_compile']}`.
- `git diff --check`: `{validation['git_diff_check']}`.
- Suíte específica: `{validation['specific_suite'].get('summary', validation['specific_suite'].get('status'))}`.
- Suíte consolidada: `{validation['consolidated_suite'].get('summary', validation['consolidated_suite'].get('status'))}`.

## Estado

A consolidação permanece bloqueada até revisão do diff e aprovação expressa.
'''
    )


def parse_pytest_summary(output: str) -> dict[str, Any]:
    patterns = {
        "passed": r"(\d+) passed",
        "failed": r"(\d+) failed",
        "xfailed": r"(\d+) xfailed",
        "errors": r"(\d+) errors?",
        "skipped": r"(\d+) skipped",
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        result[key] = int(matches[-1]) if matches else 0
    summary_lines = [line.strip() for line in output.splitlines() if " in " in line and "=" in line]
    result["summary"] = summary_lines[-1].strip("= ") if summary_lines else output.strip().splitlines()[-1]
    return result


def run_specific_suite(software_root: Path) -> dict[str, Any]:
    result = run(
        ("pipenv", "run", "pytest", "-q", "-ra", *SPECIFIC_TEST_PATHS),
        cwd=software_root,
        check=False,
    )
    output = result.stdout + result.stderr
    parsed = parse_pytest_summary(output)
    if result.returncode != 0:
        fail(f"Suíte específica AP-004B aplicação falhou:\n{output}")
    if parsed["failed"] or parsed["errors"]:
        fail(f"Suíte específica registrou falhas/erros:\n{output}")
    parsed["status"] = "passed"
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
    )
    output = result.stdout + result.stderr
    parsed = parse_pytest_summary(output)
    if result.returncode != 0:
        fail(f"Suíte consolidada AP-004B aplicação falhou:\n{output}")
    if (
        parsed["passed"] != EXPECTED_CONSOLIDATED_PASSED
        or parsed["xfailed"] != BASELINE_XFAILED
        or parsed["failed"] != 0
        or parsed["errors"] != 0
    ):
        fail(
            "Contagem consolidada divergente.\n"
            f"Esperado: {EXPECTED_CONSOLIDATED_PASSED} passed, "
            f"{BASELINE_XFAILED} xfailed\nAtual: {parsed['summary']}"
        )
    parsed["status"] = "passed"
    return parsed


def compile_paths(paths: Iterable[Path]) -> None:
    with tempfile.TemporaryDirectory(prefix="ap004b-apply-pyc-") as tmp:
        for index, path in enumerate(sorted(set(paths), key=lambda item: item.as_posix())):
            try:
                py_compile.compile(
                    str(path),
                    cfile=str(Path(tmp) / f"{index}.pyc"),
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                fail(f"py_compile falhou em {path}: {exc.msg}")


def validate_final_status(repository_root: Path) -> None:
    paths, staged, lines = git_status_paths(repository_root)
    if staged:
        fail("O aplicador criou alterações staged, o que é proibido.")
    if paths != EXPECTED_DIRTY_PATHS:
        fail(
            "Escopo final divergente.\n"
            f"Esperado: {sorted(EXPECTED_DIRTY_PATHS)}\n"
            f"Atual: {sorted(paths)}\n"
            + "\n".join(lines)
        )


def validate_no_trailing_whitespace(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file() or path.suffix not in {".py", ".md", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip() != line:
                fail(f"Whitespace final em {path}:{number}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplicador produtivo AP-004B de nomes de módulos e arquivos."
    )
    parser.add_argument(
        "--skip-remote-check",
        action="store_true",
        help="Uso excepcional offline; não comprova sincronização remota.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    software_root = Path.cwd().resolve()
    tool_path = Path(__file__).resolve()
    tool_source = tool_path.read_bytes()
    ast.parse(tool_source.decode("utf-8"), filename=str(tool_path))
    validate_internal_string_rewriter()

    repository_root, git_state = validate_git_state(
        software_root,
        skip_remote_check=args.skip_remote_check,
    )
    if args.skip_remote_check:
        print(
            "[AVISO] Verificação remota ignorada; a execução não comprova "
            "sincronização com o GitHub.",
            file=sys.stderr,
        )
    inventory = validate_inventory(software_root, repository_root)

    module_migrations: list[dict[str, Any]] = []
    fulltext_before = {
        relative: {
            "sha256_before": sha256_path(software_root / relative),
            "ast_sha256_before": ast_sha256_bytes(
                (software_root / relative).read_bytes(), filename=relative
            ),
            "decision": "untouched-deferred-high-risk",
        }
        for relative in FULLTEXT_PATHS
    }
    unchanged_control_paths = (
        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
        "pyproject.toml",
        "academic_pipeline/__main__.py",
        "academic_pipeline/cli.py",
        "academic_pipeline/legacy.py",
        "academic_pipeline/prisma_generic_orchestration.py",
    )
    unchanged_controls = {
        relative: {
            "sha256_before": sha256_path(software_root / relative),
            "ast_sha256_before": (
                ast_sha256_bytes((software_root / relative).read_bytes(), filename=relative)
                if relative.endswith(".py")
                else None
            ),
        }
        for relative in unchanged_control_paths
    }

    write_bytes: dict[Path, tuple[bytes, int]] = {}
    for spec in CANDIDATES:
        historical = software_root / spec["historical"]
        canonical = software_root / spec["canonical"]
        source_bytes = historical.read_bytes()
        source_mode = current_mode(historical)
        source_ast = ast_sha256_bytes(source_bytes, filename=spec["historical"])

        if spec["key"] == "pipeline_orchestrator":
            alias = canonical_alias_source(historical.name).encode("utf-8")
            alias_ast = ast_sha256_bytes(alias, filename=spec["canonical"])
            write_bytes[canonical] = (alias, source_mode)
            module_migrations.append({
                "key": spec["key"],
                "historical_path": spec["historical"],
                "canonical_path": spec["canonical"],
                "canonical_filename": canonical.name,
                "source_sha256_before": sha256_bytes(source_bytes),
                "source_ast_sha256_before": source_ast,
                "historical_sha256_after": sha256_bytes(source_bytes),
                "historical_ast_sha256_after": source_ast,
                "canonical_sha256_after": sha256_bytes(alias),
                "canonical_ast_sha256_after": alias_ast,
                "wrapper_sha256_after": None,
                "wrapper_ast_sha256_after": None,
                "mode": oct(source_mode),
                "migration_policy": "canonical-alias-over-frozen-historical",
                "wrapper_policy": "historical-source-frozen-by-AP-003G",
            })
            continue

        wrapper = wrapper_source(canonical.name).encode("utf-8")
        wrapper_ast = ast_sha256_bytes(wrapper, filename=spec["historical"])
        write_bytes[canonical] = (source_bytes, source_mode)
        write_bytes[historical] = (wrapper, source_mode)
        module_migrations.append({
            "key": spec["key"],
            "historical_path": spec["historical"],
            "canonical_path": spec["canonical"],
            "canonical_filename": canonical.name,
            "source_sha256_before": sha256_bytes(source_bytes),
            "source_ast_sha256_before": source_ast,
            "historical_sha256_after": sha256_bytes(wrapper),
            "historical_ast_sha256_after": wrapper_ast,
            "canonical_sha256_after": sha256_bytes(source_bytes),
            "canonical_ast_sha256_after": source_ast,
            "wrapper_sha256_after": sha256_bytes(wrapper),
            "wrapper_ast_sha256_after": wrapper_ast,
            "mode": oct(source_mode),
            "migration_policy": "canonical-copy-with-historical-wrapper",
            "wrapper_policy": "exec-canonical-source-in-legacy-namespace-until-AP-004E",
        })

    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in CONSUMER_REPLACEMENTS:
        grouped.setdefault(spec["path"], []).append(spec)
    consumer_evidence: list[dict[str, Any]] = []
    for relative, replacements in grouped.items():
        path = software_root / relative
        original = path.read_text(encoding="utf-8")
        updated, evidence = replace_approved_string_literals(
            original,
            relative=relative,
            replacements=replacements,
        )
        write_bytes[path] = (updated.encode("utf-8"), current_mode(path))
        for spec, record in zip(replacements, evidence):
            consumer_evidence.append({
                "candidate_key": spec["candidate_key"],
                "path": relative,
                "line": spec["line"],
                "actual_literal_line": record["actual_line"],
                "inventory_line": spec["line"],
                "selector_kind": record["selector_kind"],
                "selector_value": record["selector_value"],
                "kind": spec["kind"],
                "old": spec["old"],
                "new": spec["new"],
                "sha256_before": sha256_bytes(original.encode("utf-8")),
                "sha256_after": sha256_bytes(updated.encode("utf-8")),
                "literal_before": record["old_literal"],
                "literal_after": record["new_literal"],
            })

    application_data = build_application_data(
        git_state=git_state,
        inventory=inventory,
        tool_source=tool_source,
        module_migrations=module_migrations,
        consumer_evidence=consumer_evidence,
        fulltext_before=fulltext_before,
        unchanged_controls=unchanged_controls,
    )
    inventory_contract = build_durable_inventory_contract(
        inventory=inventory,
        expected_dirty_paths=sorted(EXPECTED_DIRTY_PATHS),
    )
    application_contract = build_application_contract(
        tool_sha256=sha256_bytes(tool_source),
        expected_dirty_paths=sorted(EXPECTED_DIRTY_PATHS),
    )
    artifact_texts: dict[Path, str] = {
        software_root / INVENTORY_TEST_REL: inventory_contract,
        software_root / APPLICATION_TOOL_REL: tool_source.decode("utf-8"),
        software_root / APPLICATION_TEST_REL: application_contract,
        software_root / APPLICATION_JSON_REL: json.dumps(
            application_data, ensure_ascii=False, indent=2
        ) + "\n",
        software_root / APPLICATION_REPORT_REL: build_report(application_data),
    }

    all_write_paths = set(write_bytes) | set(artifact_texts)
    backup_root, backups = create_backups(
        all_write_paths,
        software_root=software_root,
    )
    try:
        for path, (data, mode) in write_bytes.items():
            atomic_write_bytes(path, data, mode=mode)
        for path, text in artifact_texts.items():
            mode = current_mode(path, 0o755 if path == software_root / APPLICATION_TOOL_REL else 0o644)
            atomic_write_text(path, text, mode=mode)

        # Verificações pós-escrita antes de executar testes.
        for item in module_migrations:
            historical = software_root / item["historical_path"]
            canonical = software_root / item["canonical_path"]
            if item["key"] == "pipeline_orchestrator":
                if sha256_path(historical) != item["source_sha256_before"]:
                    fail("Orquestrador histórico congelado foi alterado.")
                if sha256_path(canonical) != item["canonical_sha256_after"]:
                    fail(f"Alias canônico divergente: {item['canonical_path']}")
            else:
                if sha256_path(canonical) != item["source_sha256_before"]:
                    fail(f"Cópia canônica divergente: {item['canonical_path']}")
                if sha256_path(historical) != item["wrapper_sha256_after"]:
                    fail(f"Wrapper divergente: {item['historical_path']}")
        for relative, record in fulltext_before.items():
            if sha256_path(software_root / relative) != record["sha256_before"]:
                fail(f"Executor full-text foi alterado indevidamente: {relative}")
        if (software_root / FORBIDDEN_FULLTEXT_CANONICAL).exists():
            fail(f"Destino full-text proibido foi criado: {FORBIDDEN_FULLTEXT_CANONICAL}")
        for relative, record in unchanged_controls.items():
            if sha256_path(software_root / relative) != record["sha256_before"]:
                fail(f"Controle público foi alterado indevidamente: {relative}")

        compile_targets = [
            software_root / relative
            for relative in PRODUCTIVE_CHANGED_PATHS
        ] + [
            software_root / INVENTORY_TEST_REL,
            software_root / APPLICATION_TEST_REL,
            software_root / APPLICATION_TOOL_REL,
            software_root / AP004A_TEST_REL,
            software_root / INVENTORY_TOOL_REL,
        ]
        compile_paths(compile_targets)
        generated_whitespace_paths = set(artifact_texts) | {
            software_root / item["historical"] for item in CANDIDATES
        }
        validate_no_trailing_whitespace(generated_whitespace_paths)
        diff_check = git(repository_root, "diff", "--check", check=False)
        if diff_check.returncode != 0:
            fail(f"git diff --check falhou:\n{diff_check.stdout}{diff_check.stderr}")
        validate_final_status(repository_root)

        application_data["validation"]["py_compile"] = "passed"
        application_data["validation"]["git_diff_check"] = "passed"
        specific = run_specific_suite(software_root)
        application_data["validation"]["specific_suite"] = specific
        consolidated = run_consolidated_suite(software_root)
        application_data["validation"]["consolidated_suite"] = consolidated

        # Registra o resultado efetivo; contratos não dependem desse campo.
        atomic_write_text(
            software_root / APPLICATION_JSON_REL,
            json.dumps(application_data, ensure_ascii=False, indent=2) + "\n",
            mode=current_mode(software_root / APPLICATION_JSON_REL),
        )
        atomic_write_text(
            software_root / APPLICATION_REPORT_REL,
            build_report(application_data),
            mode=current_mode(software_root / APPLICATION_REPORT_REL),
        )
        validate_no_trailing_whitespace(
            (software_root / APPLICATION_JSON_REL, software_root / APPLICATION_REPORT_REL)
        )
        diff_check = git(repository_root, "diff", "--check", check=False)
        if diff_check.returncode != 0:
            fail(f"git diff --check final falhou:\n{diff_check.stdout}{diff_check.stderr}")
        validate_final_status(repository_root)
    except Exception:
        rollback(backups)
        raise

    print("[OK] Aplicação produtiva AP-004B concluída sem commit.")
    print(f"[OK] Branch: {git_state['branch']}")
    print(f"[OK] HEAD local/remoto: {git_state['head']}")
    print("[OK] Módulos canônicos/aliases criados: 4")
    print("[OK] Orquestrador histórico AP-003G preservado: 1")
    print("[OK] Wrappers históricos criados: 3")
    print("[OK] Ocorrências produtivas migradas: 5 em 4 arquivos")
    print("[OK] Consumidores full-text preservados: 2")
    print(f"[OK] Destino full-text não criado: {FORBIDDEN_FULLTEXT_CANONICAL}")
    print(f"[OK] Relatório: {APPLICATION_REPORT_REL}")
    print(f"[OK] JSON: {APPLICATION_JSON_REL}")
    print(f"[OK] Teste: {APPLICATION_TEST_REL}")
    print(f"[OK] Ferramenta reexecutável: {APPLICATION_TOOL_REL}")
    print(f"[OK] Backup externo: {backup_root}")
    print(
        "[OK] Suíte específica: "
        f"{application_data['validation']['specific_suite']['summary']}"
    )
    print(
        "[OK] Suíte consolidada: "
        f"{application_data['validation']['consolidated_suite']['summary']}"
    )
    print("[OK] Nenhum commit foi criado.")
    print("[BLOQUEIO] Não consolidar a AP-004B sem revisão do diff e aprovação expressa.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApplicationError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        raise SystemExit(1)

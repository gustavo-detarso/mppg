#!/usr/bin/env python3
"""Regenera o inventário estrutural e os snapshots da AP-003A.

A ferramenta é deliberadamente somente-leitura para código produtivo. As únicas
saídas gravadas ficam em docs/ e tests/characterization/snapshots/.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
WORKTREE = ROOT.parents[1]
ORCHESTRATOR = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
WRAPPER_NAME = "_original_main_before_prisma_artigo_generico_wrapper"
PRODUCTIVE_SEARCH_ROOTS = (ROOT / "academic_pipeline", ROOT / "app_bundle")
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".patch_backups", "backups", "build", "dist"}
DOC_DIR = ROOT / "docs/refactor/academic-pipeline/AP-003"
JSON_PATH = DOC_DIR / "ap003a_orchestrator_inventory.json"
REPORT_PATH = DOC_DIR / "AP-003A_ORCHESTRATOR_MAP.md"
SNAPSHOT_DIR = ROOT / "tests/characterization/snapshots/ap003a"
DIRECT_HELP_PATH = SNAPSHOT_DIR / "direct_script_help.txt"
PACKAGE_HELP_PATH = SNAPSHOT_DIR / "package_module_help.txt"


def fail(message: str) -> None:
    raise RuntimeError(message)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    result = [item.arg for item in node.args.posonlyargs]
    result.extend(item.arg for item in node.args.args)
    if node.args.vararg:
        result.append("*" + node.args.vararg.arg)
    result.extend(item.arg for item in node.args.kwonlyargs)
    if node.args.kwarg:
        result.append("**" + node.args.kwarg.arg)
    return result


def calls(node: ast.AST) -> list[str]:
    result: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = dotted_name(child.func)
            if name:
                result.add(name)
    return sorted(result)


def productive_python_files() -> list[Path]:
    files: list[Path] = []
    for base in PRODUCTIVE_SEARCH_ROOTS:
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            relative = path.relative_to(ROOT)
            if not any(part in EXCLUDED_PARTS for part in relative.parts):
                files.append(path)
    return sorted(set(files))


def wrapper_symbol_occurrences() -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for path in productive_python_files():
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if WRAPPER_NAME not in source:
            continue
        relative = str(path.relative_to(ROOT))
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            for line_number, line in enumerate(source.splitlines(), start=1):
                if WRAPPER_NAME in line:
                    occurrences.append({"path": relative, "line": line_number, "kind": "text"})
            continue
        for node in ast.walk(tree):
            kind: str | None = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == WRAPPER_NAME:
                kind = "function-definition"
            elif isinstance(node, ast.Name) and node.id == WRAPPER_NAME:
                kind = f"name-{node.ctx.__class__.__name__.lower()}"
            elif isinstance(node, ast.alias) and (node.name == WRAPPER_NAME or node.asname == WRAPPER_NAME):
                kind = "import-alias"
            elif isinstance(node, ast.Attribute) and node.attr == WRAPPER_NAME:
                kind = "attribute"
            if kind is not None:
                occurrences.append({"path": relative, "line": getattr(node, "lineno", None), "kind": kind})
    unique = {(item["path"], item.get("line"), item["kind"]): item for item in occurrences}
    return [unique[key] for key in sorted(unique)]


def function_record(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "name": node.name,
        "line_start": node.lineno,
        "line_end": getattr(node, "end_lineno", node.lineno),
        "arguments": arguments(node),
        "calls": calls(node),
        "docstring": ast.get_docstring(node),
    }


def classify(record: dict[str, Any]) -> list[str]:
    text = " ".join(
        [record["name"], *(record["calls"] or []), record.get("docstring") or ""]
    ).lower()
    rules = {
        "parser-e-argumentos": ("argparse", "argumentparser", "parse_args", "add_argument", "parser"),
        "despacho-de-comandos": ("dispatch", "command", "comando", "subcommand", "handler", "executar"),
        "orquestracao-documental": ("document", "docx", "pdf", "org", "pandoc", "latex", "abnt"),
        "prisma-e-artigo-generico": ("prisma", "artigo", "article", "systematic", "revis"),
        "entrypoint-e-fluxo-principal": ("main", "entrypoint", "cli"),
    }
    groups = [group for group, keys in rules.items() if any(key in text for key in keys)]
    return groups or ["nao-classificado"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        fail(result.stderr or result.stdout)
    return result.stdout.strip()


def build_inventory() -> dict[str, Any]:
    if not ORCHESTRATOR.is_file():
        fail(f"Orquestrador ausente: {ORCHESTRATOR}")
    tree = ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"), filename=str(ORCHESTRATOR))
    functions = [
        function_record(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for record in functions:
        record["candidate_groups"] = classify(record)
    mains = [record for record in functions if record["name"] == "main"]
    wrappers = [record for record in functions if record["name"] == WRAPPER_NAME]
    if len(mains) != 2:
        fail(f"Trava AP-003A: esperados 2 main(); encontrados {len(mains)}")

    guards = []
    argparse_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            rendered = ast.unparse(node.test)
            if "__name__" in rendered and "__main__" in rendered:
                guards.append(
                    {
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "test": rendered,
                        "calls": calls(node),
                    }
                )
        if isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name and (
                name.endswith("ArgumentParser")
                or name.endswith("add_argument")
                or name.endswith("add_subparsers")
                or name.endswith("add_parser")
            ):
                argparse_calls.append({"line": node.lineno, "call": name})

    import tomllib

    pyproject_path = ROOT / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject.get("project") or {}
    return {
        "schema_version": 1,
        "phase": "AP-003A",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "worktree_root": str(WORKTREE),
            "software_root": str(ROOT),
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "remote_integration_ref": "origin/refactor/academic-pipeline",
            "remote_integration_commit": git("rev-parse", "origin/refactor/academic-pipeline"),
        },
        "orchestrator": {
            "path": str(ORCHESTRATOR.relative_to(ROOT)),
            "sha256": sha256(ORCHESTRATOR),
            "line_count": len(ORCHESTRATOR.read_text(encoding="utf-8").splitlines()),
            "functions": functions,
            "main_definitions": mains,
            "historical_wrapper": wrappers,
            "wrapper_symbol": {
                "name": WRAPPER_NAME,
                "canonical_top_level_definition_count": len(wrappers),
                "productive_occurrences": wrapper_symbol_occurrences(),
            },
            "main_guards": sorted(guards, key=lambda item: item["line_start"]),
            "argparse_calls": sorted(argparse_calls, key=lambda item: item["line"]),
        },
        "entrypoints": {
            "package_main_exists": (ROOT / "academic_pipeline/__main__.py").is_file(),
            "project_name": project.get("name"),
            "project_version": project.get("version"),
            "console_scripts": project.get("scripts") or {},
        },
    }


def render_report(data: dict[str, Any]) -> str:
    repo = data["repository"]
    orch = data["orchestrator"]
    entry = data["entrypoints"]
    lines = [
        "# AP-003A — Inventário e mapa estrutural do orquestrador",
        "",
        "> Documento gerado por análise AST. Nenhum módulo produtivo foi alterado nesta subfase.",
        "",
        "## Identificação do baseline",
        "",
        f"- Branch: `{repo['branch']}`",
        f"- HEAD: `{repo['head']}`",
        f"- Base remota: `{repo['remote_integration_ref']}` em `{repo['remote_integration_commit']}`",
        f"- Orquestrador: `{orch['path']}`",
        f"- SHA-256: `{orch['sha256']}`",
        f"- Linhas físicas: {orch['line_count']}",
        "",
        "## Contratos estruturais preservados até a AP-003F",
        "",
        f"- Definições `main()` de nível superior: **{len(orch['main_definitions'])}**",
        f"- Definições de nível superior do wrapper `{WRAPPER_NAME}` no orquestrador: **{len(orch['historical_wrapper'])}**",
        f"- Ocorrências produtivas do símbolo histórico: **{len(orch['wrapper_symbol']['productive_occurrences'])}**",
        "- A AP-003A registra a forma real do wrapper sem exigir que ele seja uma função no arquivo canônico.",
        f"- Guardas de execução direta: **{len(orch['main_guards'])}**",
        "",
        "## Entrypoints e empacotamento",
        "",
        f"- `academic_pipeline/__main__.py`: **{'presente' if entry['package_main_exists'] else 'ausente'}**",
        f"- Projeto: `{entry['project_name']}`",
        f"- Versão: `{entry['project_version']}`",
        f"- Scripts: `{json.dumps(entry['console_scripts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Funções de nível superior",
        "",
        "| Intervalo | Função | Argumentos | Grupos candidatos | Chamadas distintas |",
        "|---:|---|---|---|---:|",
    ]
    for function in orch["functions"]:
        lines.append(
            f"| {function['line_start']}-{function['line_end']} | `{function['name']}` | "
            f"`{', '.join(function['arguments'])}` | {', '.join(function['candidate_groups'])} | "
            f"{len(function['calls'])} |"
        )
    lines.extend(
        [
            "",
            "## Superfície de argumentos",
            "",
            f"Chamadas relacionadas a `argparse`: **{len(orch['argparse_calls'])}**.",
            "",
            "## Leitura para as próximas subfases",
            "",
            "- **AP-003B:** confirmar e extrair parser e argumentos.",
            "- **AP-003C:** confirmar a tabela comando → handler antes da extração do despacho.",
            "- **AP-003D:** isolar a orquestração documental.",
            "- **AP-003E:** isolar PRISMA e artigo genérico após confirmar a forma real do wrapper histórico.",
            "- **AP-003F:** atualizar deliberadamente as travas dos dois `main()` e unificar o fluxo.",
            "",
        ]
    )
    return "\n".join(lines)


def normalize(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    for source, replacement in (
        (str(ROOT), "<SOFTWARE_ROOT>"),
        (str(WORKTREE), "<WORKTREE_ROOT>"),
        (str(Path(sys.executable)), "<PYTHON>"),
        (str(Path(sys.executable).resolve()), "<PYTHON>"),
    ):
        value = value.replace(source, replacement)
    return "\n".join(line.rstrip() for line in value.split("\n")).strip() + "\n"


def capture(command: Sequence[str]) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.update({"COLUMNS": "120", "LINES": "40", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1", "TERM": "dumb"})
    result = subprocess.run(
        list(command), cwd=ROOT, env=env, text=True, capture_output=True, timeout=60, check=False
    )
    captured = (
        f"# command: {' '.join(command)}\n"
        f"# returncode: {result.returncode}\n"
        "# stdout\n"
        f"{result.stdout}"
        "# stderr\n"
        f"{result.stderr}"
    )
    if result.returncode != 0:
        fail("Falha na captura de --help:\n" + normalize(captured))
    return normalize(captured)


def main() -> int:
    data = build_inventory()
    atomic_write(JSON_PATH, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write(REPORT_PATH, render_report(data))
    atomic_write(DIRECT_HELP_PATH, capture([sys.executable, str(ORCHESTRATOR), "--help"]))
    atomic_write(PACKAGE_HELP_PATH, capture([sys.executable, "-m", "academic_pipeline", "--help"]))
    print(f"[OK] Inventário JSON: {JSON_PATH.relative_to(ROOT)}")
    print(f"[OK] Mapa Markdown : {REPORT_PATH.relative_to(ROOT)}")
    print(f"[OK] Snapshots CLI : {SNAPSHOT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

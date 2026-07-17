#!/usr/bin/env python3
"""Inventário reproduzível das facades e superfícies públicas da AP-005D."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "ap005d.facade-inventory.v1"
BASELINE_COMMIT = "78f3be0fce0dd8f79e55729a7111a9359c9edb8d"

PROJECT_REL = Path("software/academic_pipeline_rc10_7_conformidade")
INVENTORY_REL = Path(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005d_facade_inventory.json"
)

SELF_EXCLUDED = {
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005d_facade_inventory_contract.py"
    ),
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "app_bundle.scripts.pipeline.article_workflow": {
        "path": "app_bundle/scripts/pipeline/article_workflow/__init__.py",
        "classification": [
            "facade_reexport_publico_verdadeiro",
            "caminho_publico_preservado",
        ],
        "decision": "preserve_unchanged",
        "false_positive_names": [],
    },
    "academic_pipeline": {
        "path": "academic_pipeline/__init__.py",
        "classification": [
            "superficie_compatibilidade_preservada",
            "falso_positivo_do_inventario",
            "caminho_publico_preservado",
        ],
        "decision": "preserve_unchanged",
        "false_positive_names": ["Sequence", "annotations"],
    },
    "academic_pipeline.cli_parser": {
        "path": "academic_pipeline/cli_parser.py",
        "classification": [
            "modulo_canonico_com_all_declarativo",
            "nao_e_facade",
            "consumidores_internos_preservados",
        ],
        "decision": "preserve_unchanged",
        "false_positive_names": [],
    },
    "academic_pipeline.command_dispatch": {
        "path": "academic_pipeline/command_dispatch.py",
        "classification": [
            "modulo_canonico_com_all_declarativo",
            "nao_e_facade",
            "consumidores_internos_preservados",
        ],
        "decision": "preserve_unchanged",
        "false_positive_names": [],
    },
    "academic_pipeline.document_orchestration": {
        "path": "academic_pipeline/document_orchestration.py",
        "classification": [
            "modulo_canonico_com_all_declarativo",
            "nao_e_facade",
            "consumidores_internos_preservados",
        ],
        "decision": "preserve_unchanged",
        "false_positive_names": [],
    },
    "academic_pipeline.prisma_generic_orchestration": {
        "path": "academic_pipeline/prisma_generic_orchestration.py",
        "classification": [
            "modulo_canonico_com_all_declarativo",
            "nao_e_facade",
            "consumidores_internos_preservados",
        ],
        "decision": "preserve_unchanged",
        "false_positive_names": [],
    },
}


class InventoryError(RuntimeError):
    """Falha contratual do inventário AP-005D."""


def run_git(
    root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def repository_root(explicit: str | None) -> Path:
    root = (
        Path(explicit).expanduser().resolve()
        if explicit
        else Path(__file__).resolve().parents[2]
    )

    if not (root / ".git").exists():
        raise InventoryError(f"Repositório Git não encontrado: {root}")

    return root


def ensure_baseline_policy(root: Path) -> None:
    run_git(root, "cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}")

    result = run_git(
        root,
        "merge-base",
        "--is-ancestor",
        BASELINE_COMMIT,
        "HEAD",
        check=False,
    )

    if result.returncode != 0:
        raise InventoryError(
            "O HEAD atual não descende da baseline AP-005D"
        )


def tracked_python_files(root: Path) -> list[Path]:
    project_prefix = PurePosixPath(PROJECT_REL.as_posix())
    result = run_git(root, "ls-files", "-z")
    files: list[Path] = []

    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue

        relative_text = os.fsdecode(raw).replace("\\", "/")

        if relative_text in SELF_EXCLUDED:
            continue

        relative = PurePosixPath(relative_text)

        if relative.suffix != ".py":
            continue

        try:
            inside_project = relative.relative_to(project_prefix)
        except ValueError:
            continue

        if inside_project.parts and inside_project.parts[0] == "backups":
            continue

        files.append(root / Path(*relative.parts))

    return sorted(
        files,
        key=lambda path: path.relative_to(root).as_posix(),
    )


def project_relative(root: Path, path: Path) -> PurePosixPath:
    return PurePosixPath(
        path.relative_to(root / PROJECT_REL).as_posix()
    )


def module_name(root: Path, path: Path) -> str | None:
    parts = list(project_relative(root, path).parts)

    if not parts:
        return None

    if parts[-1] == "__init__.py":
        parts.pop()
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    else:
        return None

    return ".".join(parts) if parts else None


def is_test(root: Path, path: Path) -> bool:
    relative = project_relative(root, path)
    return (
        "tests" in {part.lower() for part in relative.parts}
        or relative.name.startswith("test_")
    )


def is_patch_backup(root: Path, path: Path) -> bool:
    relative = project_relative(root, path)
    return bool(
        relative.parts and relative.parts[0] == ".patch_backups"
    )


def read_python(path: Path) -> str:
    with tokenize.open(path) as handle:
        return handle.read()


def resolve_relative_module(
    source_module: str | None,
    source_is_package: bool,
    raw_module: str | None,
    level: int,
) -> str:
    raw_module = raw_module or ""

    if level == 0:
        return raw_module

    if not source_module:
        return raw_module

    package = (
        source_module
        if source_is_package
        else source_module.rpartition(".")[0]
    )

    package_parts = [part for part in package.split(".") if part]
    keep = max(0, len(package_parts) - (level - 1))
    resolved = package_parts[:keep]

    if raw_module:
        resolved.extend(raw_module.split("."))

    return ".".join(resolved)


def imports_for(
    tree: ast.Module,
    source_module: str | None,
    source_is_package: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(
                    {
                        "kind": "import",
                        "module": alias.name,
                        "name": None,
                        "asname": alias.asname,
                        "level": 0,
                        "lineno": node.lineno,
                    }
                )

        elif isinstance(node, ast.ImportFrom):
            resolved = resolve_relative_module(
                source_module,
                source_is_package,
                node.module,
                node.level,
            )

            for alias in node.names:
                records.append(
                    {
                        "kind": "from",
                        "module": resolved,
                        "name": alias.name,
                        "asname": alias.asname,
                        "level": node.level,
                        "lineno": node.lineno,
                    }
                )

    return records


def top_level_bindings(
    tree: ast.Module,
    source_module: str | None,
    source_is_package: bool,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                result[local] = {
                    "kind": "import",
                    "module": alias.name,
                    "name": None,
                    "asname": alias.asname,
                    "level": 0,
                    "lineno": node.lineno,
                }

        elif isinstance(node, ast.ImportFrom):
            resolved = resolve_relative_module(
                source_module,
                source_is_package,
                node.module,
                node.level,
            )

            for alias in node.names:
                local = alias.asname or alias.name
                result[local] = {
                    "kind": "from",
                    "module": resolved,
                    "name": alias.name,
                    "asname": alias.asname,
                    "level": node.level,
                    "lineno": node.lineno,
                }

    return result


def all_declaration(
    tree: ast.Module,
) -> tuple[bool, bool, list[str]]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue

        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return True, True, []

        values: list[str] = []

        for element in value.elts:
            if not (
                isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            ):
                return True, True, []

            values.append(element.value)

        return True, False, values

    return False, False, []


def local_definitions(tree: ast.Module) -> set[str]:
    result: set[str] = set()

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            result.add(node.name)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result.add(target.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                result.add(node.target.id)

    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def build_inventory(root: Path) -> dict[str, Any]:
    ensure_baseline_policy(root)

    project = root / PROJECT_REL
    paths = tracked_python_files(root)

    if len(paths) != 145:
        raise InventoryError(
            "O universo AP-005D deveria conter 145 "
            f"arquivos; obtidos {len(paths)}"
        )

    parsed = {
        path: ast.parse(read_python(path), filename=str(path))
        for path in paths
    }

    consumers: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path, tree in parsed.items():
        source_module = module_name(root, path)
        source_package = path.name == "__init__.py"

        if is_test(root, path):
            scope = "test"
        elif is_patch_backup(root, path):
            scope = "patch_backup"
        else:
            scope = "ordinary"

        for record in imports_for(
            tree,
            source_module,
            source_package,
        ):
            for candidate in CANDIDATES:
                if record["module"] != candidate:
                    continue

                consumers[candidate].append(
                    {
                        "source_file": path.relative_to(root).as_posix(),
                        "source_module": source_module,
                        "source_scope": scope,
                        "kind": record["kind"],
                        "imported_name": record["name"],
                        "asname": record["asname"],
                        "level": record["level"],
                        "lineno": record["lineno"],
                    }
                )

    candidate_records: list[dict[str, Any]] = []

    for module, policy in CANDIDATES.items():
        path = project / policy["path"]

        if path not in parsed:
            raise InventoryError(f"Candidato fora do universo: {path}")

        tree = parsed[path]
        package = path.name == "__init__.py"
        all_present, dynamic_all, exports = all_declaration(tree)
        bindings = top_level_bindings(tree, module, package)
        definitions = local_definitions(tree)

        reexports: list[dict[str, Any]] = []
        local_exports: list[str] = []
        unresolved_exports: list[str] = []

        for exported in exports:
            if exported in bindings:
                reexports.append(
                    {
                        "exported_name": exported,
                        "source": bindings[exported],
                    }
                )
            elif exported in definitions:
                local_exports.append(exported)
            else:
                unresolved_exports.append(exported)

        by_scope = {
            scope: sorted(
                [
                    edge
                    for edge in consumers[module]
                    if edge["source_scope"] == scope
                ],
                key=lambda edge: (
                    edge["source_file"],
                    edge["lineno"],
                    edge["imported_name"] or "",
                ),
            )
            for scope in ("ordinary", "test", "patch_backup")
        }

        candidate_records.append(
            {
                "module": module,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "is_package_initializer": package,
                "all_present": all_present,
                "dynamic_all": dynamic_all,
                "all": exports,
                "local_exports": sorted(local_exports),
                "reexports": reexports,
                "unresolved_exports": sorted(unresolved_exports),
                "consumers": by_scope,
                "classification": policy["classification"],
                "decision": policy["decision"],
                "false_positive_names": policy[
                    "false_positive_names"
                ],
            }
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "baseline_commit": BASELINE_COMMIT,
        "head_policy": "baseline_commit_must_be_ancestor_of_head",
        "source_manifest": "git ls-files",
        "scope": {
            "auditable_python_files": len(paths),
            "excluded": [
                "project backups/",
                *sorted(SELF_EXCLUDED),
            ],
        },
        "decision": {
            "productive_changes_required": False,
            "facades_to_preserve": [
                "app_bundle.scripts.pipeline.article_workflow",
                "academic_pipeline",
            ],
            "canonical_modules_not_facades": [
                "academic_pipeline.cli_parser",
                "academic_pipeline.command_dispatch",
                "academic_pipeline.document_orchestration",
                "academic_pipeline.prisma_generic_orchestration",
            ],
            "deferred_scope": (
                "Any broad package-path migration belongs outside AP-005D."
            ),
        },
        "candidates": candidate_records,
        "summary": {
            "candidate_count": len(candidate_records),
            "true_facade_count": 1,
            "public_package_surface_count": 1,
            "canonical_non_facade_count": 4,
            "productive_changes_required": False,
        },
    }

    payload["fingerprint"] = canonical_fingerprint(payload)
    return payload


def inventory_path(root: Path) -> Path:
    return root / INVENTORY_REL


def write_inventory(root: Path, payload: dict[str, Any]) -> None:
    destination = inventory_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def check_inventory(root: Path, payload: dict[str, Any]) -> None:
    destination = inventory_path(root)

    if not destination.is_file():
        raise InventoryError(f"Inventário não encontrado: {destination}")

    committed = json.loads(destination.read_text(encoding="utf-8"))

    if committed != payload:
        raise InventoryError("O inventário AP-005D está desatualizado")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera ou verifica o inventário de facades da AP-005D."
    )
    parser.add_argument("--root", help="Raiz do repositório.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repository_root(args.root)
    payload = build_inventory(root)

    if args.write:
        write_inventory(root, payload)
    else:
        check_inventory(root, payload)

    print(f"schema={payload['schema']}")
    print(f"baseline_commit={payload['baseline_commit']}")
    print(
        "auditable_python_files="
        f"{payload['scope']['auditable_python_files']}"
    )
    print(f"candidates={payload['summary']['candidate_count']}")
    print(f"true_facades={payload['summary']['true_facade_count']}")
    print(
        "canonical_non_facades="
        f"{payload['summary']['canonical_non_facade_count']}"
    )
    print(
        "productive_changes_required="
        f"{payload['summary']['productive_changes_required']}"
    )
    print(f"fingerprint={payload['fingerprint']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InventoryError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        raise SystemExit(1)

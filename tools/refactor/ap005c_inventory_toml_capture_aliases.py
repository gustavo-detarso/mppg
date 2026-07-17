#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys
from collections.abc import Iterator, Sequence
from typing import Any


SCHEMA_VERSION = (
    "ap005c.toml-capture-alias-inventory.v1"
)

BASELINE_COMMIT = (
    "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
)

SOURCE_PLAN_FINGERPRINT = (
    "e659a91460dd5058ba6e49942454c26650eb4455e42f1d6e2ce450125f6284c8"
)

MODULE_SHA256 = (
    "7b3ff44794275df2a3470796e78a25c3c87ca2c44f93fac6ec18eee397c89beb"
)

PROJECT_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)

MODULE_REL = (
    PROJECT_REL
    / "app_bundle/scripts/pipeline/"
    "academic_pipeline_toml_generator_interativo.py"
)

SOURCE_PLAN_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005b_consumer_canonicalization_plan.json"
)

INVENTORY_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005c_toml_capture_alias_inventory.json"
)

STRATEGY_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005C_TOML_CAPTURE_ALIAS_STRATEGY.md"
)

EXPECTED = {
    "AP004E-054764be4586": {
        "legacy_alias": (
            "_original_ensure_reference_policy"
        ),
        "captured_expression": (
            "_WizInputController."
            "_ensure_reference_policy"
        ),
        "canonical_capture_name": (
            "_captured_wiz_input_"
            "ensure_reference_policy"
        ),
        "source_line": 4976,
        "reference_lines": [4983],
    },
    "AP004E-5fa6e68ff3fc": {
        "legacy_alias": (
            "_wiz_disable_references_original"
        ),
        "captured_expression": (
            "_wiz_disable_references"
        ),
        "canonical_capture_name": (
            "_captured_wiz_disable_references"
        ),
        "source_line": 4993,
        "reference_lines": [4997],
    },
    "AP004E-936e788786e4": {
        "legacy_alias": "_render_toml_original",
        "captured_expression": "render_toml",
        "canonical_capture_name": (
            "_captured_render_toml"
        ),
        "source_line": 4958,
        "reference_lines": [4963],
    },
    "AP004E-c3f6df07093a": {
        "legacy_alias": (
            "_collect_outputs_and_options_original"
        ),
        "captured_expression": (
            "collect_outputs_and_options"
        ),
        "canonical_capture_name": (
            "_captured_collect_outputs_and_options"
        ),
        "source_line": 4907,
        "reference_lines": [
            4913,
            4917,
            4944,
        ],
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def resolve(
    root: pathlib.Path,
    relative: pathlib.PurePosixPath,
) -> pathlib.Path:
    root = root.resolve()
    path = (root / relative).resolve()

    try:
        path.relative_to(root)
    except ValueError as error:
        raise SystemExit(
            f"Caminho fora da raiz: {relative}"
        ) from error

    return path



def baseline_file_bytes(
    root: pathlib.Path,
    relative: pathlib.PurePosixPath,
) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{BASELINE_COMMIT}:{relative}",
        ],
        check=False,
        capture_output=True,
    )

    if result.returncode != 0:
        raise SystemExit(
            "Não foi possível ler a pré-imagem "
            f"{relative} em {BASELINE_COMMIT}: "
            + result.stderr.decode(
                "utf-8",
                errors="replace",
            )
        )

    return result.stdout


def walk_json(
    value: Any,
) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value

        for child in value.values():
            yield from walk_json(child)

    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def parent_map(
    tree: ast.AST,
) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {}

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            result[child] = node

    return result


def ancestors(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> list[ast.AST]:
    result: list[ast.AST] = []
    current = parents.get(node)

    while current is not None:
        result.append(current)
        current = parents.get(current)

    return result


_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def is_module_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    return not any(
        isinstance(ancestor, _SCOPE_NODES)
        for ancestor in ancestors(node, parents)
        if not isinstance(ancestor, ast.Module)
    )


def assignment_targets(
    node: ast.AST,
) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)

    if isinstance(node, ast.AnnAssign):
        return [node.target]

    if isinstance(node, ast.NamedExpr):
        return [node.target]

    return []


def assignment_value(
    node: ast.AST,
) -> ast.expr | None:
    if isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
            ast.NamedExpr,
        ),
    ):
        return node.value

    return None


def source_line(
    lines: list[str],
    line: int,
) -> str:
    return lines[line - 1].strip()


def enclosure(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> list[str]:
    result: list[str] = []

    for ancestor in reversed(
        ancestors(node, parents)
    ):
        if isinstance(ancestor, ast.If):
            result.append(
                f"If@{ancestor.lineno}"
            )

        elif isinstance(ancestor, ast.Try):
            result.append(
                f"Try@{ancestor.lineno}"
            )

        elif isinstance(ancestor, ast.With):
            result.append(
                f"With@{ancestor.lineno}"
            )

        elif isinstance(ancestor, ast.Match):
            result.append(
                f"Match@{ancestor.lineno}"
            )

        elif isinstance(
            ancestor,
            (
                ast.For,
                ast.AsyncFor,
                ast.While,
            ),
        ):
            result.append(
                f"{type(ancestor).__name__}@"
                f"{ancestor.lineno}"
            )

    return result


def module_name_events(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    lines: list[str],
    name: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            if (
                node.name == name
                and is_module_scope(
                    node,
                    parents,
                )
            ):
                events.append(
                    {
                        "line": node.lineno,
                        "kind": type(node).__name__,
                        "source": source_line(
                            lines,
                            node.lineno,
                        ),
                    }
                )

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.NamedExpr,
            ),
        ):
            if not is_module_scope(
                node,
                parents,
            ):
                continue

            for target in assignment_targets(node):
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                ):
                    events.append(
                        {
                            "line": node.lineno,
                            "kind": type(node).__name__,
                            "source": source_line(
                                lines,
                                node.lineno,
                            ),
                        }
                    )

    return sorted(
        {
            (
                event["line"],
                event["kind"],
                event["source"],
            )
            for event in events
        }
    )


def normalize_events(
    events: list[
        tuple[int, str, str]
    ],
) -> list[dict[str, Any]]:
    return [
        {
            "line": line,
            "kind": kind,
            "source": source,
        }
        for line, kind, source in events
    ]


def attribute_events(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    lines: list[str],
    owner: str,
    member: str,
) -> list[dict[str, Any]]:
    events: set[
        tuple[int, str, str]
    ] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == owner
            and is_module_scope(
                node,
                parents,
            )
        ):
            for child in node.body:
                if (
                    isinstance(
                        child,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    )
                    and child.name == member
                ):
                    events.add(
                        (
                            child.lineno,
                            "ClassMethodDef",
                            source_line(
                                lines,
                                child.lineno,
                            ),
                        )
                    )

        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.NamedExpr,
            ),
        ):
            if not is_module_scope(
                node,
                parents,
            ):
                continue

            for target in assignment_targets(node):
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(
                        target.value,
                        ast.Name,
                    )
                    and target.value.id == owner
                    and target.attr == member
                ):
                    events.add(
                        (
                            node.lineno,
                            "AttributeAssign",
                            source_line(
                                lines,
                                node.lineno,
                            ),
                        )
                    )

    return normalize_events(
        sorted(events)
    )


def binding_events(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    lines: list[str],
    expression: str,
) -> list[dict[str, Any]]:
    if "." not in expression:
        raw = module_name_events(
            tree,
            parents,
            lines,
            expression,
        )

        return normalize_events(raw)

    owner, member = expression.split(
        ".",
        1,
    )

    return attribute_events(
        tree,
        parents,
        lines,
        owner,
        member,
    )


def plan_records(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    objects = list(walk_json(payload))
    result: dict[str, dict[str, Any]] = {}

    for candidate_id, expected in EXPECTED.items():
        matches = []

        for item in objects:
            values = {
                value
                for value in item.values()
                if isinstance(value, str)
            }

            if (
                candidate_id in values
                and expected["legacy_alias"]
                in values
            ):
                matches.append(item)

        if not matches:
            raise SystemExit(
                f"Registro do plano ausente: "
                f"{candidate_id}"
            )

        selected = max(
            matches,
            key=len,
        )

        if selected["source_candidate_id"] != (
            candidate_id
        ):
            raise SystemExit(
                f"Candidate ID divergente: "
                f"{candidate_id}"
            )

        if selected["current_name"] != (
            expected["legacy_alias"]
        ):
            raise SystemExit(
                f"Alias divergente: {candidate_id}"
            )

        if selected["canonical_target"] != {
            "captured_expression": (
                expected["captured_expression"]
            ),
            "kind": "captured_previous_binding",
            "requires_new_export": False,
        }:
            raise SystemExit(
                f"Target divergente: {candidate_id}"
            )

        if selected["ap005b_disposition"] != (
            "deferred_to_ap005c"
        ):
            raise SystemExit(
                f"Disposição divergente: "
                f"{candidate_id}"
            )

        result[candidate_id] = selected

    return result


def build_inventory(
    root: pathlib.Path,
) -> dict[str, Any]:
    module_path = resolve(
        root,
        MODULE_REL,
    )

    plan_path = resolve(
        root,
        SOURCE_PLAN_REL,
    )

    module_bytes = baseline_file_bytes(
        root,
        MODULE_REL,
    )

    if sha256_bytes(module_bytes) != MODULE_SHA256:
        raise SystemExit(
            "Hash do módulo produtivo divergente."
        )

    plan = json.loads(
        plan_path.read_text(encoding="utf-8")
    )

    if plan["contract_fingerprint"] != (
        SOURCE_PLAN_FINGERPRINT
    ):
        raise SystemExit(
            "Fingerprint do plano AP-005B "
            "divergente."
        )

    records = plan_records(plan)

    source = module_bytes.decode("utf-8")
    lines = source.splitlines()

    tree = ast.parse(
        source,
        filename=str(module_path),
    )

    parents = parent_map(tree)

    assignments: dict[
        str,
        list[ast.AST],
    ] = {
        item["legacy_alias"]: []
        for item in EXPECTED.values()
    }

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.NamedExpr,
            ),
        ):
            continue

        if not is_module_scope(
            node,
            parents,
        ):
            continue

        for target in assignment_targets(node):
            if (
                isinstance(target, ast.Name)
                and target.id in assignments
            ):
                assignments[target.id].append(
                    node
                )

    references: dict[
        str,
        list[int],
    ] = {
        item["legacy_alias"]: []
        for item in EXPECTED.values()
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in references
        ):
            references[node.id].append(
                node.lineno
            )

    entries: list[dict[str, Any]] = []

    for candidate_id in sorted(EXPECTED):
        expected = EXPECTED[candidate_id]
        alias = expected["legacy_alias"]

        matches = assignments[alias]

        if len(matches) != 1:
            raise SystemExit(
                f"{alias}: atribuições={len(matches)}"
            )

        assignment = matches[0]
        node_value = assignment_value(
            assignment
        )

        if node_value is None:
            raise SystemExit(
                f"{alias}: atribuição sem valor"
            )

        expression = ast.unparse(node_value)

        if expression != (
            expected["captured_expression"]
        ):
            raise SystemExit(
                f"{alias}: expressão divergente"
            )

        if assignment.lineno != (
            expected["source_line"]
        ):
            raise SystemExit(
                f"{alias}: linha divergente"
            )

        actual_references = sorted(
            references[alias]
        )

        if actual_references != (
            expected["reference_lines"]
        ):
            raise SystemExit(
                f"{alias}: referências divergentes"
            )

        events = binding_events(
            tree,
            parents,
            lines,
            expression,
        )

        previous = [
            event
            for event in events
            if event["line"] < assignment.lineno
        ]

        later = [
            event
            for event in events
            if event["line"] > assignment.lineno
        ]

        if not previous or not later:
            raise SystemExit(
                f"{alias}: ordem de captura inválida"
            )

        plan_record = records[candidate_id]

        entries.append(
            {
                "source_candidate_id": candidate_id,
                "legacy_alias": alias,
                "canonical_capture_name": (
                    expected[
                        "canonical_capture_name"
                    ]
                ),
                "captured_expression": expression,
                "source_line": assignment.lineno,
                "source": source_line(
                    lines,
                    assignment.lineno,
                ),
                "enclosure": enclosure(
                    assignment,
                    parents,
                ),
                "previous_bindings": previous,
                "later_bindings": later,
                "reference_lines": actual_references,
                "reference_count": len(
                    actual_references
                ),
                "risk": plan_record["risk"],
                "removal_allowed": False,
                "requires_new_export": False,
                "direct_substitution_allowed": False,
                "legacy_alias_policy": (
                    "preserve_as_compatibility_alias"
                ),
                "consumer_migration_policy": (
                    "migrate_internal_consumers_to_"
                    "canonical_capture_name"
                ),
                "runtime_identity_contract": (
                    "legacy_alias_and_canonical_capture_"
                    "must_reference_same_previous_binding"
                ),
                "ordering_contract": (
                    "capture_after_previous_binding_and_"
                    "before_redefinition"
                ),
            }
        )

    inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": BASELINE_COMMIT,
        "source_plan": str(SOURCE_PLAN_REL),
        "source_plan_fingerprint": (
            SOURCE_PLAN_FINGERPRINT
        ),
        "productive_module": str(MODULE_REL),
        "productive_module_sha256": (
            MODULE_SHA256
        ),
        "application_strategy": {
            "batch": "AP-005C.1",
            "atomic": True,
            "productive_application_status": (
                "blocked_pending_characterization_"
                "and_explicit_approval"
            ),
            "operation": (
                "introduce_explicit_capture_symbols_"
                "preserve_legacy_aliases_and_migrate_"
                "internal_consumers"
            ),
            "direct_substitution_by_current_name": (
                "prohibited"
            ),
            "legacy_alias_removal": "prohibited",
            "new_public_exports": False,
        },
        "summary": {
            "aliases": len(entries),
            "previous_bindings_confirmed": sum(
                bool(entry["previous_bindings"])
                for entry in entries
            ),
            "later_redefinitions_confirmed": sum(
                bool(entry["later_bindings"])
                for entry in entries
            ),
            "productive_references": sum(
                entry["reference_count"]
                for entry in entries
            ),
            "removal_allowed": sum(
                bool(entry["removal_allowed"])
                for entry in entries
            ),
            "direct_substitution_allowed": sum(
                bool(
                    entry[
                        "direct_substitution_allowed"
                    ]
                )
                for entry in entries
            ),
        },
        "entries": entries,
    }

    fingerprint_payload = dict(inventory)

    inventory["contract_fingerprint"] = (
        sha256_bytes(
            canonical_bytes(
                fingerprint_payload
            )
        )
    )

    return inventory


def render_strategy(
    inventory: dict[str, Any],
) -> str:
    lines = [
        "# AP-005C — Estratégia para aliases de captura do gerador TOML",
        "",
        "## Baseline",
        "",
        f"- Commit: `{inventory['baseline_commit']}`",
        f"- Módulo: `{inventory['productive_module']}`",
        f"- SHA-256: `{inventory['productive_module_sha256']}`",
        f"- Fingerprint do inventário: `{inventory['contract_fingerprint']}`",
        "",
        "## Diagnóstico",
        "",
        (
            "Os quatro símbolos adiados pela AP-005B não são aliases "
            "redundantes. Cada um captura um binding anterior antes de "
            "uma redefinição posterior no mesmo módulo."
        ),
        "",
        (
            "A substituição direta pelo nome corrente é proibida porque "
            "mudaria a cadeia de patches ou introduziria recursão."
        ),
        "",
        "## Estratégia selecionada",
        "",
        (
            "A AP-005C.1 deverá introduzir nomes canônicos explícitos "
            "para as quatro capturas, manter os aliases históricos "
            "apontando para o mesmo objeto anterior e migrar somente os "
            "consumidores internos para os novos nomes."
        ),
        "",
        "A aplicação deverá ser atômica no único módulo produtivo.",
        "",
        "Não haverá novo export público.",
        "",
        "A remoção dos aliases históricos permanece proibida.",
        "",
        "## Mapeamento",
        "",
        "| Alias histórico | Captura canônica planejada | Binding capturado | Usos |",
        "|---|---|---|---:|",
    ]

    for entry in inventory["entries"]:
        lines.append(
            f"| `{entry['legacy_alias']}` "
            f"| `{entry['canonical_capture_name']}` "
            f"| `{entry['captured_expression']}` "
            f"| {entry['reference_count']} |"
        )

    lines.extend(
        [
            "",
            "## Contratos obrigatórios",
            "",
            "1. A captura canônica deve ocorrer na mesma posição relativa.",
            "2. O alias histórico e a captura canônica devem apontar para o mesmo binding anterior.",
            "3. O binding corrente redefinido deve continuar distinto da captura.",
            "4. Todos os seis consumidores internos devem migrar de forma atômica.",
            "5. Nenhum alias histórico pode ser removido nesta fase.",
            "6. Nenhum novo símbolo deve ser exportado publicamente.",
            "7. A suíte canônica deve permanecer sem regressões.",
            "",
            "## Bloqueio produtivo",
            "",
            (
                "Nenhum aplicador produtivo está autorizado por este "
                "documento. A geração do aplicador dependerá da aprovação "
                "dos contratos de caracterização e de uma nova auditoria "
                "da pré-imagem."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def generated_files(
    root: pathlib.Path,
) -> dict[pathlib.Path, bytes]:
    inventory = build_inventory(root)

    inventory_text = (
        json.dumps(
            inventory,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    strategy_text = render_strategy(
        inventory
    )

    return {
        resolve(
            root,
            INVENTORY_REL,
        ): inventory_text.encode("utf-8"),
        resolve(
            root,
            STRATEGY_REL,
        ): strategy_text.encode("utf-8"),
    }


def write_files(
    files: dict[pathlib.Path, bytes],
) -> None:
    for path, content in files.items():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(content)


def check_files(
    files: dict[pathlib.Path, bytes],
) -> None:
    failures: list[str] = []

    for path, expected in files.items():
        if not path.is_file():
            failures.append(
                f"ausente: {path}"
            )
            continue

        actual = path.read_bytes()

        if actual != expected:
            failures.append(
                f"divergente: {path}"
            )

    if failures:
        raise SystemExit(
            "\n".join(failures)
        )


def verify_git_baseline(
    root: pathlib.Path,
) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    head = result.stdout.strip()

    if head != BASELINE_COMMIT:
        raise SystemExit(
            f"HEAD divergente: {head}"
        )


def parse_arguments(
    arguments: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventaria os aliases de captura "
            "adiados para a AP-005C."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        type=pathlib.Path,
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--write",
        action="store_true",
    )

    mode.add_argument(
        "--check",
        action="store_true",
    )

    return parser.parse_args(arguments)


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    args = parse_arguments(arguments)
    root = args.root.resolve()

    if not root.is_dir():
        raise SystemExit(
            f"Raiz inexistente: {root}"
        )

    verify_git_baseline(root)

    inventory = build_inventory(root)
    files = generated_files(root)

    if args.write:
        write_files(files)
        action = "gravados"
    else:
        check_files(files)
        action = "verificados"

    print(
        f"schema={inventory['schema_version']}"
    )
    print(
        f"fingerprint="
        f"{inventory['contract_fingerprint']}"
    )
    print(
        f"aliases="
        f"{inventory['summary']['aliases']}"
    )
    print(
        f"referências produtivas="
        f"{inventory['summary']['productive_references']}"
    )
    print(
        f"remoção permitida="
        f"{inventory['summary']['removal_allowed']}"
    )
    print(
        f"substituição direta permitida="
        f"{inventory['summary']['direct_substitution_allowed']}"
    )
    print(f"arquivos {action}=2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

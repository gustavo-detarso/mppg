from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[4]

BASELINE_COMMIT = (
    "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
)

PROJECT = (
    ROOT
    / "software/"
    "academic_pipeline_rc10_7_conformidade"
)

MODULE = (
    PROJECT
    / "app_bundle/scripts/pipeline/"
    "academic_pipeline_toml_generator_interativo.py"
)

INVENTORY = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005c_toml_capture_alias_inventory.json"
)

EXPECTED_ASSIGNMENTS = {
    "_original_ensure_reference_policy": (
        4976,
        "_WizInputController._ensure_reference_policy",
    ),
    "_wiz_disable_references_original": (
        4993,
        "_wiz_disable_references",
    ),
    "_render_toml_original": (
        4958,
        "render_toml",
    ),
    "_collect_outputs_and_options_original": (
        4907,
        "collect_outputs_and_options",
    ),
}

EXPECTED_REFERENCES = {
    "_original_ensure_reference_policy": [4983],
    "_wiz_disable_references_original": [4997],
    "_render_toml_original": [4963],
    "_collect_outputs_and_options_original": [
        4913,
        4917,
        4944,
    ],
}


def baseline_module_bytes() -> bytes:
    relative = MODULE.relative_to(ROOT)

    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{BASELINE_COMMIT}:{relative}",
        ],
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, (
        result.stdout,
        result.stderr,
    )

    return result.stdout


def source_tree() -> tuple[str, ast.Module]:
    source = baseline_module_bytes().decode("utf-8")

    return (
        source,
        ast.parse(
            source,
            filename=(
                f"{BASELINE_COMMIT}:{MODULE.relative_to(ROOT)}"
            ),
        ),
    )


def parents(
    tree: ast.AST,
) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {}

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            result[child] = node

    return result


def ancestors(
    node: ast.AST,
    parent: dict[ast.AST, ast.AST],
) -> list[ast.AST]:
    result: list[ast.AST] = []
    current = parent.get(node)

    while current is not None:
        result.append(current)
        current = parent.get(current)

    return result


def is_module_scope(
    node: ast.AST,
    parent: dict[ast.AST, ast.AST],
) -> bool:
    scope_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
    )

    return not any(
        isinstance(item, scope_nodes)
        for item in ancestors(node, parent)
        if not isinstance(item, ast.Module)
    )


def targets(
    node: ast.AST,
) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)

    if isinstance(node, ast.AnnAssign):
        return [node.target]

    return []


def load_inventory() -> dict[str, Any]:
    return json.loads(
        INVENTORY.read_text(encoding="utf-8")
    )


def test_alias_assignments_capture_exact_bindings() -> None:
    _, tree = source_tree()
    parent = parents(tree)

    found: dict[
        str,
        tuple[int, str],
    ] = {}

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        if not is_module_scope(
            node,
            parent,
        ):
            continue

        for target in targets(node):
            if (
                isinstance(target, ast.Name)
                and target.id in EXPECTED_ASSIGNMENTS
            ):
                found[target.id] = (
                    node.lineno,
                    ast.unparse(node.value),
                )

    assert found == EXPECTED_ASSIGNMENTS


def test_alias_reference_lines_are_frozen() -> None:
    _, tree = source_tree()

    found = {
        alias: []
        for alias in EXPECTED_REFERENCES
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in found
        ):
            found[node.id].append(node.lineno)

    found = {
        alias: sorted(lines)
        for alias, lines in found.items()
    }

    assert found == EXPECTED_REFERENCES


def test_every_capture_has_previous_and_later_binding() -> None:
    payload = load_inventory()

    for entry in payload["entries"]:
        assert entry["previous_bindings"]
        assert entry["later_bindings"]

        assert max(
            item["line"]
            for item in entry["previous_bindings"]
        ) < entry["source_line"]

        assert min(
            item["line"]
            for item in entry["later_bindings"]
        ) > entry["source_line"]


def test_capture_order_prohibits_current_name_substitution() -> None:
    payload = load_inventory()

    for entry in payload["entries"]:
        assert (
            entry["direct_substitution_allowed"]
            is False
        )

        assert entry["ordering_contract"] == (
            "capture_after_previous_binding_and_"
            "before_redefinition"
        )

        assert entry["runtime_identity_contract"] == (
            "legacy_alias_and_canonical_capture_"
            "must_reference_same_previous_binding"
        )


def test_legacy_aliases_are_preservation_contracts() -> None:
    payload = load_inventory()

    for entry in payload["entries"]:
        assert entry["removal_allowed"] is False

        assert entry["legacy_alias_policy"] == (
            "preserve_as_compatibility_alias"
        )

        assert entry[
            "consumer_migration_policy"
        ] == (
            "migrate_internal_consumers_to_"
            "canonical_capture_name"
        )

        assert entry["requires_new_export"] is False


def test_productive_module_hash_and_syntax_are_frozen() -> None:
    data = baseline_module_bytes()

    assert hashlib.sha256(data).hexdigest() == (
        "7b3ff44794275df2a3470796e78a25c3c"
        "87ca2c44f93fac6ec18eee397c89beb"
    )

    compile(
        data,
        str(MODULE),
        "exec",
    )

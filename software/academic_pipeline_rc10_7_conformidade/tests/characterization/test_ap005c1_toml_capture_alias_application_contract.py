from __future__ import annotations

import ast
import hashlib
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[4]

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

INVENTORY_TOOL = (
    ROOT
    / "tools/refactor/"
    "ap005c_inventory_toml_capture_aliases.py"
)

EXPECTED_MODULE_HASH = "9d627348fcdc3b9ec727abb3c2862eb26b11bbd1d1bc744958d892f9f4afa7f9"

MAPPING = {
    "_original_ensure_reference_policy": (
        "_captured_wiz_input_ensure_reference_policy",
        "_WizInputController._ensure_reference_policy",
        1,
    ),
    "_wiz_disable_references_original": (
        "_captured_wiz_disable_references",
        "_wiz_disable_references",
        1,
    ),
    "_render_toml_original": (
        "_captured_render_toml",
        "render_toml",
        1,
    ),
    "_collect_outputs_and_options_original": (
        "_captured_collect_outputs_and_options",
        "collect_outputs_and_options",
        3,
    ),
}


def parse_module() -> tuple[str, ast.Module]:
    source = MODULE.read_text(encoding="utf-8")

    return source, ast.parse(
        source,
        filename=str(MODULE),
    )


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


def is_module_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    scopes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
    )

    return not any(
        isinstance(item, scopes)
        for item in ancestors(node, parents)
        if not isinstance(item, ast.Module)
    )


def targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)

    if isinstance(node, ast.AnnAssign):
        return [node.target]

    return []


def module_assignments() -> dict[str, tuple[int, str]]:
    _, tree = parse_module()
    parents = parent_map(tree)
    wanted = set(MAPPING)

    for canonical, _, _ in MAPPING.values():
        wanted.add(canonical)

    result: dict[str, tuple[int, str]] = {}

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        if not is_module_scope(node, parents):
            continue

        for target in targets(node):
            if (
                isinstance(target, ast.Name)
                and target.id in wanted
            ):
                result[target.id] = (
                    node.lineno,
                    ast.unparse(node.value),
                )

    return result


def loaded_names() -> dict[str, list[int]]:
    _, tree = parse_module()

    wanted = set(MAPPING)

    for canonical, _, _ in MAPPING.values():
        wanted.add(canonical)

    result = {
        name: []
        for name in wanted
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in result
        ):
            result[node.id].append(node.lineno)

    return {
        name: sorted(lines)
        for name, lines in result.items()
    }


def test_productive_module_hash_matches_postimage() -> None:
    assert hashlib.sha256(
        MODULE.read_bytes()
    ).hexdigest() == EXPECTED_MODULE_HASH


def test_canonical_capture_assignments_exist() -> None:
    assignments = module_assignments()

    for legacy, (
        canonical,
        captured_expression,
        _,
    ) in MAPPING.items():
        canonical_line, canonical_value = (
            assignments[canonical]
        )

        legacy_line, legacy_value = (
            assignments[legacy]
        )

        assert canonical_value == captured_expression
        assert legacy_value == canonical
        assert canonical_line < legacy_line


def test_internal_consumers_use_canonical_names() -> None:
    loaded = loaded_names()

    for legacy, (
        canonical,
        _,
        expected_references,
    ) in MAPPING.items():
        assert loaded[legacy] == []
        assert len(loaded[canonical]) == (
            expected_references + 1
        )


def test_six_productive_consumers_were_migrated() -> None:
    loaded = loaded_names()

    consumer_references = sum(
        len(loaded[canonical]) - 1
        for canonical, _, _ in MAPPING.values()
    )

    assert consumer_references == 6


def test_legacy_aliases_remain_compatibility_bindings() -> None:
    assignments = module_assignments()

    for legacy, (
        canonical,
        _,
        _,
    ) in MAPPING.items():
        assert assignments[legacy][1] == canonical


def test_no_new_public_export_was_added() -> None:
    _, tree = parse_module()

    exported: set[str] = set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        if not any(
            isinstance(target, ast.Name)
            and target.id == "__all__"
            for target in node.targets
        ):
            continue

        value = ast.literal_eval(node.value)
        exported.update(value)

    canonical_names = {
        canonical
        for canonical, _, _ in MAPPING.values()
    }

    assert not (
        exported
        & canonical_names
    )


def test_preimage_inventory_remains_reproducible() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(INVENTORY_TOOL),
            "--root",
            str(ROOT),
            "--check",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, (
        result.stdout,
        result.stderr,
    )

    assert "aliases=4" in result.stdout
    assert "referências produtivas=6" in result.stdout


def test_productive_module_compiles() -> None:
    compile(
        MODULE.read_bytes(),
        str(MODULE),
        "exec",
    )

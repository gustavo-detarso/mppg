#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import os
import pathlib
import subprocess
import tempfile
from collections.abc import Sequence
from typing import Callable


BASELINE_COMMIT = (
    "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
)

PROJECT_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)

MODULE_REL = (
    PROJECT_REL
    / "app_bundle/scripts/pipeline/"
    "academic_pipeline_toml_generator_interativo.py"
)

INVENTORY_TOOL_REL = pathlib.PurePosixPath(
    "tools/refactor/"
    "ap005c_inventory_toml_capture_aliases.py"
)

SEMANTICS_TEST_REL = (
    PROJECT_REL
    / "tests/characterization/"
    "test_ap005c_toml_capture_alias_"
    "semantics_characterization.py"
)

APPLICATION_TEST_REL = (
    PROJECT_REL
    / "tests/characterization/"
    "test_ap005c1_toml_capture_alias_"
    "application_contract.py"
)

PRE_HASHES = {
    MODULE_REL: (
        "7b3ff44794275df2a3470796e78a25c3c"
        "87ca2c44f93fac6ec18eee397c89beb"
    ),
    INVENTORY_TOOL_REL: (
        "92bdb295883182aeaf7c31a9bf9237e7f"
        "67399a43d50e5d608780adb7fd83c8e"
    ),
    SEMANTICS_TEST_REL: (
        "b7c2d15a3f220bfb0da99a42d75279f4"
        "c13920cdeeef06b34e972db7994003d4"
    ),
}

CANONICAL_MAPPING = {
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def replace_exact(
    text: str,
    old: str,
    new: str,
    *,
    count: int = 1,
) -> str:
    actual = text.count(old)

    if actual != count:
        raise SystemExit(
            f"Pré-imagem inesperada para {old!r}: "
            f"ocorrências={actual}; esperado={count}"
        )

    return text.replace(
        old,
        new,
        count,
    )


def transform_module(text: str) -> str:
    text = replace_exact(
        text,
        (
            "_collect_outputs_and_options_original "
            "= collect_outputs_and_options"
        ),
        (
            "_captured_collect_outputs_and_options "
            "= collect_outputs_and_options\n"
            "_collect_outputs_and_options_original "
            "= _captured_collect_outputs_and_options"
        ),
    )

    text = replace_exact(
        text,
        "_collect_outputs_and_options_original(data)",
        "_captured_collect_outputs_and_options(data)",
        count=3,
    )

    text = replace_exact(
        text,
        "_render_toml_original = render_toml",
        (
            "_captured_render_toml = render_toml\n"
            "_render_toml_original = "
            "_captured_render_toml"
        ),
    )

    text = replace_exact(
        text,
        "_render_toml_original(data)",
        "_captured_render_toml(data)",
    )

    text = replace_exact(
        text,
        (
            "    _original_ensure_reference_policy "
            "= _WizInputController."
            "_ensure_reference_policy"
        ),
        (
            "    _captured_wiz_input_"
            "ensure_reference_policy = "
            "_WizInputController."
            "_ensure_reference_policy\n"
            "    _original_ensure_reference_policy "
            "= _captured_wiz_input_"
            "ensure_reference_policy"
        ),
    )

    text = replace_exact(
        text,
        "_original_ensure_reference_policy(self)",
        (
            "_captured_wiz_input_"
            "ensure_reference_policy(self)"
        ),
    )

    text = replace_exact(
        text,
        (
            "_wiz_disable_references_original "
            "= _wiz_disable_references"
        ),
        (
            "_captured_wiz_disable_references "
            "= _wiz_disable_references\n"
            "_wiz_disable_references_original "
            "= _captured_wiz_disable_references"
        ),
    )

    text = replace_exact(
        text,
        "_wiz_disable_references_original(text)",
        "_captured_wiz_disable_references(text)",
    )

    return text


def transform_inventory_tool(text: str) -> str:
    helper = '''
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


'''

    text = replace_exact(
        text,
        "    return path\n\n\ndef walk_json(\n",
        "    return path\n\n\n" + helper + "def walk_json(\n",
    )

    text = replace_exact(
        text,
        "    module_bytes = module_path.read_bytes()\n",
        (
            "    module_bytes = baseline_file_bytes(\n"
            "        root,\n"
            "        MODULE_REL,\n"
            "    )\n"
        ),
    )

    return text


def transform_semantics_test(text: str) -> str:
    text = replace_exact(
        text,
        (
            "import pathlib\n"
            "from typing import Any\n"
        ),
        (
            "import pathlib\n"
            "import subprocess\n"
            "from typing import Any\n"
        ),
    )

    text = replace_exact(
        text,
        (
            "ROOT = pathlib.Path(__file__).resolve().parents[4]\n\n"
            "PROJECT = (\n"
        ),
        (
            "ROOT = pathlib.Path(__file__).resolve().parents[4]\n\n"
            "BASELINE_COMMIT = (\n"
            '    "9372de8f621c9012a28d4c4a9a64e252a398bdf3"\n'
            ")\n\n"
            "PROJECT = (\n"
        ),
    )

    old_source_tree = '''def source_tree() -> tuple[str, ast.Module]:
    source = MODULE.read_text(encoding="utf-8")

    return (
        source,
        ast.parse(
            source,
            filename=str(MODULE),
        ),
    )
'''

    new_source_tree = '''def baseline_module_bytes() -> bytes:
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
'''

    text = replace_exact(
        text,
        old_source_tree,
        new_source_tree,
    )

    text = replace_exact(
        text,
        "    data = MODULE.read_bytes()\n",
        "    data = baseline_module_bytes()\n",
    )

    return text


def application_test_text(
    module_post_hash: str,
) -> str:
    return f'''from __future__ import annotations

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

EXPECTED_MODULE_HASH = "{module_post_hash}"

MAPPING = {{
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
}}


def parse_module() -> tuple[str, ast.Module]:
    source = MODULE.read_text(encoding="utf-8")

    return source, ast.parse(
        source,
        filename=str(MODULE),
    )


def parent_map(
    tree: ast.AST,
) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {{}}

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

    result: dict[str, tuple[int, str]] = {{}}

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

    result = {{
        name: []
        for name in wanted
    }}

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in result
        ):
            result[node.id].append(node.lineno)

    return {{
        name: sorted(lines)
        for name, lines in result.items()
    }}


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

    canonical_names = {{
        canonical
        for canonical, _, _ in MAPPING.values()
    }}

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
'''


def transformed_files(
    root: pathlib.Path,
) -> dict[pathlib.Path, bytes]:
    module_path = resolve(root, MODULE_REL)
    tool_path = resolve(root, INVENTORY_TOOL_REL)
    semantics_path = resolve(
        root,
        SEMANTICS_TEST_REL,
    )

    module_pre = module_path.read_bytes()
    tool_pre = tool_path.read_bytes()
    semantics_pre = semantics_path.read_bytes()

    for relative, data in (
        (MODULE_REL, module_pre),
        (INVENTORY_TOOL_REL, tool_pre),
        (SEMANTICS_TEST_REL, semantics_pre),
    ):
        actual = sha256_bytes(data)
        expected = PRE_HASHES[relative]

        if actual != expected:
            raise SystemExit(
                f"Pré-imagem divergente: {relative}; "
                f"esperado={expected}; encontrado={actual}"
            )

    module_post = transform_module(
        module_pre.decode("utf-8")
    ).encode("utf-8")

    tool_post = transform_inventory_tool(
        tool_pre.decode("utf-8")
    ).encode("utf-8")

    semantics_post = transform_semantics_test(
        semantics_pre.decode("utf-8")
    ).encode("utf-8")

    application_test = application_test_text(
        sha256_bytes(module_post)
    ).encode("utf-8")

    return {
        module_path: module_post,
        tool_path: tool_post,
        semantics_path: semantics_post,
        resolve(
            root,
            APPLICATION_TEST_REL,
        ): application_test,
    }


def unified_diff(
    path: pathlib.Path,
    before: bytes,
    after: bytes,
) -> str:
    return "".join(
        difflib.unified_diff(
            before.decode("utf-8").splitlines(
                keepends=True
            ),
            after.decode("utf-8").splitlines(
                keepends=True
            ),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def atomic_write(
    path: pathlib.Path,
    data: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        temporary,
        path,
    )


def verify_post(
    root: pathlib.Path,
) -> None:
    module_path = resolve(root, MODULE_REL)
    tool_path = resolve(root, INVENTORY_TOOL_REL)
    semantics_path = resolve(
        root,
        SEMANTICS_TEST_REL,
    )
    application_path = resolve(
        root,
        APPLICATION_TEST_REL,
    )

    for path in (
        module_path,
        tool_path,
        semantics_path,
        application_path,
    ):
        if not path.is_file():
            raise SystemExit(
                f"Pós-imagem ausente: {path}"
            )

        compile(
            path.read_bytes(),
            str(path),
            "exec",
        )

    module_source = module_path.read_text(
        encoding="utf-8"
    )

    for legacy, (
        canonical,
        captured,
        references,
    ) in CANONICAL_MAPPING.items():
        if (
            f"{canonical} = {captured}"
            not in module_source
        ):
            raise SystemExit(
                f"Captura canônica ausente: "
                f"{canonical}"
            )

        if (
            f"{legacy} = {canonical}"
            not in module_source
        ):
            raise SystemExit(
                f"Alias de compatibilidade ausente: "
                f"{legacy}"
            )

        if module_source.count(
            f"{legacy}("
        ) != 0:
            raise SystemExit(
                f"Consumidor legado restante: "
                f"{legacy}"
            )

        if module_source.count(
            f"{canonical}("
        ) != references:
            raise SystemExit(
                f"Contagem de consumidores canônicos "
                f"divergente: {canonical}"
            )

    tool_source = tool_path.read_text(
        encoding="utf-8"
    )

    if (
        "def baseline_file_bytes("
        not in tool_source
    ):
        raise SystemExit(
            "Leitura reproduzível da pré-imagem "
            "não foi instalada."
        )

    if (
        "module_bytes = baseline_file_bytes("
        not in tool_source
    ):
        raise SystemExit(
            "Inventariador ainda lê o módulo corrente."
        )

    semantics_source = semantics_path.read_text(
        encoding="utf-8"
    )

    if (
        "def baseline_module_bytes()"
        not in semantics_source
    ):
        raise SystemExit(
            "Caracterização da pré-imagem não foi "
            "congelada no baseline."
        )

    if (
        "data = baseline_module_bytes()"
        not in semantics_source
    ):
        raise SystemExit(
            "Hash de pré-imagem ainda depende do "
            "worktree corrente."
        )

    application_source = application_path.read_text(
        encoding="utf-8"
    )

    module_hash = sha256_bytes(
        module_path.read_bytes()
    )

    if (
        f'EXPECTED_MODULE_HASH = "{module_hash}"'
        not in application_source
    ):
        raise SystemExit(
            "Contrato de pós-imagem possui hash "
            "divergente."
        )

    print(
        f"módulo pós={module_hash}"
    )
    print(
        f"inventariador pós="
        f"{sha256_bytes(tool_path.read_bytes())}"
    )
    print(
        f"caracterização pré-imagem pós="
        f"{sha256_bytes(semantics_path.read_bytes())}"
    )
    print(
        f"contrato aplicação="
        f"{sha256_bytes(application_path.read_bytes())}"
    )
    print("aliases históricos preservados=4")
    print("capturas canônicas presentes=4")
    print("consumidores canônicos=6")
    print("consumidores legados restantes=0")
    print("AP-005C.1 pós-aplicação=aprovada")


def parse_arguments(
    arguments: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aplicador transacional AP-005C.1 "
            "para aliases de captura TOML."
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
        "--check",
        action="store_true",
    )

    mode.add_argument(
        "--apply",
        action="store_true",
    )

    mode.add_argument(
        "--verify-post",
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

    head = subprocess.run(
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
    ).stdout.strip()

    if head != BASELINE_COMMIT:
        raise SystemExit(
            f"HEAD divergente: {head}"
        )

    if args.verify_post:
        verify_post(root)
        return 0

    application_path = resolve(
        root,
        APPLICATION_TEST_REL,
    )

    if application_path.exists():
        raise SystemExit(
            "Contrato de aplicação já existe na "
            "pré-imagem."
        )

    files = transformed_files(root)

    print(f"baseline={BASELINE_COMMIT}")
    print("lote=AP-005C.1")
    print("arquivos previstos=4")

    for path, post in files.items():
        relative = path.relative_to(root)

        if path.is_file():
            pre = path.read_bytes()

            print(
                f"pré={sha256_bytes(pre)} "
                f"pós={sha256_bytes(post)} "
                f"{relative}"
            )

            print(
                unified_diff(
                    relative,
                    pre,
                    post,
                )
            )
        else:
            print(
                f"pré=ausente "
                f"pós={sha256_bytes(post)} "
                f"{relative}"
            )

    if args.check:
        print(
            "AP-005C.1 dry-run=aprovado; "
            "arquivos escritos=0"
        )
        return 0

    for path, data in files.items():
        atomic_write(
            path,
            data,
        )

    verify_post(root)

    print(
        "AP-005C.1 aplicação transacional="
        "concluída"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


BASE_COMMIT = (
    "6ef568b250390e12dc2e86b86a8c530188604a28"
)

BATCH_NAME = "AP-005B2.3"

PLAN_FINGERPRINT = (
    "c5c6ab8734707cdf792cef3aa3b81ecb67b4b9aa17015bd1e2b83dcdf7122664"
)

PROJECT_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)

PLAN_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005b2_prisma_adapter_batches.json"
)

PRISMA_REL = (
    PROJECT_REL
    / "academic_pipeline/prisma_generic_orchestration.py"
)

RC10_REL = (
    PROJECT_REL
    / "app_bundle/scripts/pipeline/"
    "academic_pipeline_rc10.py"
)

AP003G_REL = (
    PROJECT_REL
    / "tests/characterization/"
    "test_ap003g_stabilization_contract.py"
)

EXPECTED_PRE_HASHES = {
    PRISMA_REL: (
        "d91349d7e3fd8cb0cf66541480d4da0c3b6db4d1cea2269bb23ae48921bc6e5c"
    ),
    RC10_REL: (
        "4809c5a9ff95f3297781ac5b943b6495ea7af712e101308f520e0c10a73a6a0a"
    ),
    AP003G_REL: (
        "f55548e40c697127e7482d3fea98dbcbfbf6a998466d3b559ee38c6ff179b7e5"
    ),
}

APPLIED_BATCHES_AFTER = frozenset(
    {
        "AP-005B2.1",
        "AP-005B2.2",
        "AP-005B2.3",
    }
)


@dataclass(frozen=True)
class RenderedFile:
    relative: pathlib.PurePosixPath
    path: pathlib.Path
    before: bytes
    after: bytes
    mode: int


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve(
    root: pathlib.Path,
    relative: pathlib.PurePosixPath,
) -> pathlib.Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()

    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise SystemExit(
            f"Caminho fora da raiz: {relative}"
        ) from error

    return path


def replace_once(
    source: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = source.count(old)

    if count != 1:
        raise SystemExit(
            f"{label}: pré-imagem encontrada "
            f"{count} vezes; esperado=1"
        )

    return source.replace(old, new, 1)


def assignment(
    tree: ast.Module,
    name: str,
) -> ast.Assign | ast.AnnAssign:
    matches: list[
        ast.Assign | ast.AnnAssign
    ] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == name
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
        raise SystemExit(
            f"{name}: assignments={len(matches)}; "
            "esperado=1"
        )

    return matches[0]


def literal_assignment(
    tree: ast.Module,
    name: str,
) -> Any:
    node = assignment(tree, name)
    value = node.value

    if value is None:
        raise SystemExit(
            f"{name}: assignment sem valor."
        )

    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "frozenset"
        and len(value.args) == 1
    ):
        value = value.args[0]

    return ast.literal_eval(value)


def segment(
    source: str,
    node: ast.AST,
) -> str:
    value = ast.get_source_segment(
        source,
        node,
    )

    if value is None:
        raise SystemExit(
            "Trecho AST não recuperado na linha "
            f"{getattr(node, 'lineno', '?')}."
        )

    return value


def top_level_names(
    tree: ast.Module,
) -> set[str]:
    names: set[str] = set()

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            names.add(node.name)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        ):
            names.add(node.target.id)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(
                    alias.asname
                    or alias.name.split(".")[0]
                )

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(
                    alias.asname or alias.name
                )

    return names


def top_functions(
    tree: ast.Module,
) -> dict[
    str,
    ast.FunctionDef | ast.AsyncFunctionDef,
]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    }


def effective_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    body = list(function.body)

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(
            body[0].value,
            ast.Constant,
        )
        and isinstance(
            body[0].value.value,
            str,
        )
    ):
        body = body[1:]

    return body


def return_call(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call:
    body = effective_body(function)

    if (
        len(body) != 1
        or not isinstance(body[0], ast.Return)
        or not isinstance(
            body[0].value,
            ast.Call,
        )
    ):
        raise SystemExit(
            f"{function.name}: esperado "
            "return-call simples."
        )

    return body[0].value


def forwarding(
    arguments: ast.arguments,
) -> str:
    values: list[str] = []

    for argument in (
        *arguments.posonlyargs,
        *arguments.args,
    ):
        values.append(argument.arg)

    if arguments.vararg is not None:
        values.append(
            f"*{arguments.vararg.arg}"
        )

    for argument in arguments.kwonlyargs:
        values.append(
            f"{argument.arg}={argument.arg}"
        )

    if arguments.kwarg is not None:
        values.append(
            f"**{arguments.kwarg.arg}"
        )

    return ", ".join(values)


def header(
    name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    keyword = (
        "async def"
        if isinstance(
            function,
            ast.AsyncFunctionDef,
        )
        else "def"
    )

    returns = ""

    if function.returns is not None:
        returns = (
            f" -> {ast.unparse(function.returns)}"
        )

    return (
        f"{keyword} {name}("
        f"{ast.unparse(function.args)})"
        f"{returns}:"
    )


def load_plan(
    root: pathlib.Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    path = resolve(root, PLAN_REL)

    if not path.is_file():
        raise SystemExit(
            f"Plano AP-005B2 ausente: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload["schema_version"] != (
        "ap005b2.prisma-adapter-batches.v1"
    ):
        raise SystemExit(
            "Schema inesperado no plano AP-005B2."
        )

    if payload["contract_fingerprint"] != (
        PLAN_FINGERPRINT
    ):
        raise SystemExit(
            "Fingerprint inesperado no plano AP-005B2."
        )

    selected = [
        entry
        for entry in payload["entries"]
        if entry["batch"] == BATCH_NAME
    ]

    if len(selected) != 9:
        raise SystemExit(
            f"{BATCH_NAME}: entries={len(selected)}; "
            "esperado=9"
        )

    return payload, selected


def render_prisma(
    source: str,
    entries: list[dict[str, Any]],
) -> str:
    tree = ast.parse(
        source,
        filename=str(PRISMA_REL),
    )

    functions = top_functions(tree)
    names = top_level_names(tree)

    exports = list(
        literal_assignment(tree, "__all__")
    )

    protected = list(
        literal_assignment(
            tree,
            "_PROTECTED_RUNTIME_NAMES",
        )
    )

    candidates = [
        entry["candidate_name"]
        for entry in entries
    ]

    collisions = sorted(
        set(candidates) & names
    )

    if collisions:
        raise SystemExit(
            f"Adapters já existentes: {collisions}"
        )

    rendered = source

    for entry in sorted(
        entries,
        key=lambda item: item[
            "wrapper_line_baseline"
        ],
        reverse=True,
    ):
        wrapper_name = entry["wrapper_name"]
        candidate_name = entry["candidate_name"]
        body_name = entry["body_function"]

        wrapper = functions.get(wrapper_name)

        if wrapper is None:
            raise SystemExit(
                f"Wrapper ausente: {wrapper_name}"
            )

        if wrapper.decorator_list:
            raise SystemExit(
                f"Wrapper decorado: {wrapper_name}"
            )

        call = return_call(wrapper)

        if (
            not isinstance(call.func, ast.Name)
            or call.func.id
            != "_invoke_with_runtime"
        ):
            raise SystemExit(
                f"{wrapper_name} não chama "
                "_invoke_with_runtime."
            )

        if (
            len(call.args) < 2
            or ast.unparse(call.args[0])
            != body_name
            or ast.unparse(call.args[1])
            != "runtime"
        ):
            raise SystemExit(
                f"Body/runtime inesperado em "
                f"{wrapper_name}."
            )

        old_wrapper = segment(
            source,
            wrapper,
        )

        candidate_source = (
            f"{header(candidate_name, wrapper)}\n"
            f"    return {ast.unparse(call)}"
        )

        wrapper_source = (
            f"{header(wrapper_name, wrapper)}\n"
            f"    return {candidate_name}("
            f"{forwarding(wrapper.args)})"
        )

        rendered = replace_once(
            rendered,
            old_wrapper,
            candidate_source
            + "\n\n"
            + wrapper_source,
            f"wrapper {wrapper_name}",
        )

    protected_node = assignment(
        tree,
        "_PROTECTED_RUNTIME_NAMES",
    )

    old_protected = segment(
        source,
        protected_node,
    )

    if set(candidates) & set(protected):
        raise SystemExit(
            "Um adapter já consta dos nomes "
            "protegidos."
        )

    rendered = replace_once(
        rendered,
        old_protected,
        (
            "_PROTECTED_RUNTIME_NAMES = "
            f"frozenset("
            f"{tuple([*protected, *candidates])!r}"
            f")"
        ),
        "_PROTECTED_RUNTIME_NAMES",
    )

    exports_node = assignment(
        tree,
        "__all__",
    )

    old_exports = segment(
        source,
        exports_node,
    )

    if set(candidates) & set(exports):
        raise SystemExit(
            "Um adapter já consta de __all__."
        )

    rendered = replace_once(
        rendered,
        old_exports,
        f"__all__ = {[*exports, *candidates]!r}",
        "__all__",
    )

    compile(
        rendered,
        str(PRISMA_REL),
        "exec",
    )

    return rendered


def render_rc10(
    source: str,
    entries: list[dict[str, Any]],
) -> str:
    rendered = source

    for entry in entries:
        old_import = (
            "from academic_pipeline."
            "prisma_generic_orchestration import "
            f"{entry['wrapper_name']} as "
            f"{entry['rc10_local_alias']}"
        )

        new_import = (
            "from academic_pipeline."
            "prisma_generic_orchestration import "
            f"{entry['candidate_name']} as "
            f"{entry['rc10_local_alias']}"
        )

        rendered = replace_once(
            rendered,
            old_import,
            new_import,
            f"import RC10 {entry['wrapper_name']}",
        )

    compile(
        rendered,
        str(RC10_REL),
        "exec",
    )

    return rendered


def replace_hash_key(
    assignment_source: str,
    *,
    key: str,
    expected_old: str,
    new_value: str,
) -> str:
    pattern = re.compile(
        rf"(?P<prefix>['\"]{re.escape(key)}['\"]"
        rf"\s*:\s*['\"])"
        rf"{re.escape(expected_old)}"
        rf"(?P<suffix>['\"])"
    )

    updated, count = pattern.subn(
        lambda match: (
            match.group("prefix")
            + new_value
            + match.group("suffix")
        ),
        assignment_source,
        count=1,
    )

    if count != 1:
        raise SystemExit(
            f"EXPECTED_HASHES[{key!r}]: "
            f"ocorrências={count}; esperado=1"
        )

    return updated


def render_ap003g(
    source: str,
    *,
    prisma_hash: str,
    rc10_hash: str,
) -> str:
    tree = ast.parse(
        source,
        filename=str(AP003G_REL),
    )

    node = assignment(
        tree,
        "EXPECTED_HASHES",
    )

    current = literal_assignment(
        tree,
        "EXPECTED_HASHES",
    )

    if current["prisma"] != (
        EXPECTED_PRE_HASHES[PRISMA_REL]
    ):
        raise SystemExit(
            "EXPECTED_HASHES['prisma'] "
            "divergente."
        )

    if current["orchestrator"] != (
        EXPECTED_PRE_HASHES[RC10_REL]
    ):
        raise SystemExit(
            "EXPECTED_HASHES['orchestrator'] "
            "divergente."
        )

    old_assignment = segment(
        source,
        node,
    )

    new_assignment = replace_hash_key(
        old_assignment,
        key="prisma",
        expected_old=(
            EXPECTED_PRE_HASHES[PRISMA_REL]
        ),
        new_value=prisma_hash,
    )

    new_assignment = replace_hash_key(
        new_assignment,
        key="orchestrator",
        expected_old=(
            EXPECTED_PRE_HASHES[RC10_REL]
        ),
        new_value=rc10_hash,
    )

    rendered = replace_once(
        source,
        old_assignment,
        new_assignment,
        "EXPECTED_HASHES",
    )

    verified_tree = ast.parse(
        rendered,
        filename=str(AP003G_REL),
    )

    verified = literal_assignment(
        verified_tree,
        "EXPECTED_HASHES",
    )

    if verified["prisma"] != prisma_hash:
        raise SystemExit(
            "Hash PRISMA não gravado na chave."
        )

    if verified["orchestrator"] != rc10_hash:
        raise SystemExit(
            "Hash RC10 não gravado na chave."
        )

    compile(
        rendered,
        str(AP003G_REL),
        "exec",
    )

    return rendered


def render_all(
    root: pathlib.Path,
) -> list[RenderedFile]:
    _, entries = load_plan(root)

    paths = {
        relative: resolve(root, relative)
        for relative in EXPECTED_PRE_HASHES
    }

    for path in paths.values():
        if (
            not path.is_file()
            or path.is_symlink()
        ):
            raise SystemExit(
                f"Arquivo inválido: {path}"
            )

    before = {
        relative: path.read_bytes()
        for relative, path in paths.items()
    }

    for relative, expected in (
        EXPECTED_PRE_HASHES.items()
    ):
        actual = digest(before[relative])

        if actual != expected:
            raise SystemExit(
                f"Hash pré-aplicação divergente "
                f"em {relative}: esperado={expected}; "
                f"encontrado={actual}"
            )

    prisma_after = render_prisma(
        before[PRISMA_REL].decode("utf-8"),
        entries,
    ).encode("utf-8")

    rc10_after = render_rc10(
        before[RC10_REL].decode("utf-8"),
        entries,
    ).encode("utf-8")

    ap003g_after = render_ap003g(
        before[AP003G_REL].decode("utf-8"),
        prisma_hash=digest(prisma_after),
        rc10_hash=digest(rc10_after),
    ).encode("utf-8")

    after = {
        PRISMA_REL: prisma_after,
        RC10_REL: rc10_after,
        AP003G_REL: ap003g_after,
    }

    return [
        RenderedFile(
            relative=relative,
            path=paths[relative],
            before=before[relative],
            after=after[relative],
            mode=stat.S_IMODE(
                paths[relative].stat().st_mode
            ),
        )
        for relative in (
            PRISMA_REL,
            RC10_REL,
            AP003G_REL,
        )
    ]


def atomic_write(
    path: pathlib.Path,
    data: bytes,
    mode: int,
) -> None:
    descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
    )

    temporary = pathlib.Path(
        temporary_name
    )

    try:
        with os.fdopen(
            descriptor,
            "wb",
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.chmod(
            temporary,
            mode,
        )

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def apply_transaction(
    rendered: Sequence[RenderedFile],
) -> None:
    backups = {
        item.path: (
            item.before,
            item.mode,
        )
        for item in rendered
    }

    try:
        for item in rendered:
            atomic_write(
                item.path,
                item.after,
                item.mode,
            )

    except BaseException:
        restoration_errors: list[str] = []

        for path, (
            content,
            mode,
        ) in backups.items():
            try:
                atomic_write(
                    path,
                    content,
                    mode,
                )

            except BaseException as error:
                restoration_errors.append(
                    f"{path}: {error}"
                )

        if restoration_errors:
            raise RuntimeError(
                "Falha na aplicação e na "
                "restauração:\n"
                + "\n".join(
                    restoration_errors
                )
            )

        raise


def rc10_imports(
    tree: ast.Module,
) -> dict[
    tuple[str, str],
    list[str],
]:
    result: dict[
        tuple[str, str],
        list[str],
    ] = {}

    for function in tree.body:
        if not isinstance(
            function,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        for node in ast.walk(function):
            if not isinstance(
                node,
                ast.ImportFrom,
            ):
                continue

            if node.module != (
                "academic_pipeline."
                "prisma_generic_orchestration"
            ):
                continue

            for alias in node.names:
                key = (
                    function.name,
                    alias.asname or alias.name,
                )

                result.setdefault(
                    key,
                    [],
                ).append(alias.name)

    return result


def verify_post(
    root: pathlib.Path,
) -> None:
    payload, selected = load_plan(root)

    prisma_path = resolve(
        root,
        PRISMA_REL,
    )

    rc10_path = resolve(
        root,
        RC10_REL,
    )

    ap003g_path = resolve(
        root,
        AP003G_REL,
    )

    prisma_source = prisma_path.read_text(
        encoding="utf-8"
    )

    rc10_source = rc10_path.read_text(
        encoding="utf-8"
    )

    ap003g_source = ap003g_path.read_text(
        encoding="utf-8"
    )

    prisma_tree = ast.parse(
        prisma_source,
        filename=str(PRISMA_REL),
    )

    rc10_tree = ast.parse(
        rc10_source,
        filename=str(RC10_REL),
    )

    ap003g_tree = ast.parse(
        ap003g_source,
        filename=str(AP003G_REL),
    )

    functions = top_functions(
        prisma_tree
    )

    exports = set(
        literal_assignment(
            prisma_tree,
            "__all__",
        )
    )

    protected = set(
        literal_assignment(
            prisma_tree,
            "_PROTECTED_RUNTIME_NAMES",
        )
    )

    imports = rc10_imports(
        rc10_tree
    )

    batch_presence: dict[
        str,
        tuple[int, int],
    ] = {}

    for batch in payload["batches"]:
        entries = [
            entry
            for entry in payload["entries"]
            if entry["batch"] == batch["batch"]
        ]

        present = sum(
            entry["candidate_name"]
            in functions
            for entry in entries
        )

        batch_presence[
            batch["batch"]
        ] = (
            present,
            len(entries),
        )

    expected_presence = {
        "AP-005B2.1": (6, 6),
        "AP-005B2.2": (10, 10),
        "AP-005B2.3": (9, 9),
        "AP-005B2.4": (0, 6),
    }

    if batch_presence != expected_presence:
        raise SystemExit(
            "Estado atômico divergente: "
            f"{batch_presence}"
        )

    for entry in payload["entries"]:
        wrapper_name = entry["wrapper_name"]
        candidate_name = entry["candidate_name"]
        body_name = entry["body_function"]
        batch_name = entry["batch"]

        wrapper = functions.get(wrapper_name)
        body = functions.get(body_name)

        if wrapper is None or body is None:
            raise SystemExit(
                f"Wrapper/body ausente: "
                f"{wrapper_name}/{body_name}"
            )

        if wrapper_name not in exports:
            raise SystemExit(
                f"{wrapper_name} ausente de __all__."
            )

        if wrapper_name not in protected:
            raise SystemExit(
                f"{wrapper_name} desprotegido."
            )

        if body_name not in protected:
            raise SystemExit(
                f"{body_name} desprotegido."
            )

        import_key = (
            entry["rc10_consumer_function"],
            entry["rc10_local_alias"],
        )

        wrapper_call = return_call(wrapper)

        if batch_name in APPLIED_BATCHES_AFTER:
            candidate = functions.get(
                candidate_name
            )

            if candidate is None:
                raise SystemExit(
                    f"Adapter ausente: "
                    f"{candidate_name}"
                )

            if ast.unparse(
                wrapper.args
            ) != ast.unparse(
                candidate.args
            ):
                raise SystemExit(
                    "Assinaturas divergentes: "
                    f"{wrapper_name}/"
                    f"{candidate_name}"
                )

            candidate_call = return_call(
                candidate
            )

            if (
                not isinstance(
                    candidate_call.func,
                    ast.Name,
                )
                or candidate_call.func.id
                != "_invoke_with_runtime"
            ):
                raise SystemExit(
                    f"{candidate_name} não chama "
                    "_invoke_with_runtime."
                )

            if (
                ast.unparse(
                    candidate_call.args[0]
                )
                != body_name
                or ast.unparse(
                    candidate_call.args[1]
                )
                != "runtime"
            ):
                raise SystemExit(
                    f"Body/runtime divergente em "
                    f"{candidate_name}."
                )

            if (
                not isinstance(
                    wrapper_call.func,
                    ast.Name,
                )
                or wrapper_call.func.id
                != candidate_name
            ):
                raise SystemExit(
                    f"{wrapper_name} não delega "
                    f"a {candidate_name}."
                )

            if candidate_name not in exports:
                raise SystemExit(
                    f"{candidate_name} ausente "
                    "de __all__."
                )

            if candidate_name not in protected:
                raise SystemExit(
                    f"{candidate_name} "
                    "desprotegido."
                )

            if imports.get(
                import_key
            ) != [candidate_name]:
                raise SystemExit(
                    f"Import RC10 divergente: "
                    f"{import_key}."
                )

        else:
            if candidate_name in functions:
                raise SystemExit(
                    "Adapter futuro aplicado cedo: "
                    f"{candidate_name}"
                )

            if (
                not isinstance(
                    wrapper_call.func,
                    ast.Name,
                )
                or wrapper_call.func.id
                != "_invoke_with_runtime"
            ):
                raise SystemExit(
                    f"Wrapper futuro alterado: "
                    f"{wrapper_name}"
                )

            if imports.get(
                import_key
            ) != [wrapper_name]:
                raise SystemExit(
                    f"Import futuro alterado: "
                    f"{import_key}."
                )

    prisma_hash = digest(
        prisma_path.read_bytes()
    )

    rc10_hash = digest(
        rc10_path.read_bytes()
    )

    ap003g_hash = digest(
        ap003g_path.read_bytes()
    )

    expected_hashes = literal_assignment(
        ap003g_tree,
        "EXPECTED_HASHES",
    )

    if expected_hashes["prisma"] != prisma_hash:
        raise SystemExit(
            "EXPECTED_HASHES['prisma'] "
            "divergente."
        )

    if (
        expected_hashes["orchestrator"]
        != rc10_hash
    ):
        raise SystemExit(
            "EXPECTED_HASHES['orchestrator'] "
            "divergente."
        )

    if len(selected) != 9:
        raise SystemExit(
            "Seleção AP-005B2.3 divergente."
        )

    print(f"PRISMA pós={prisma_hash}")
    print(f"RC10 pós={rc10_hash}")
    print(f"AP003G pós={ap003g_hash}")

    print(
        "estado atômico="
        "B2.1 6/6; B2.2 10/10; "
        "B2.3 9/9; B2.4 0/6"
    )

    print(
        f"{BATCH_NAME} pós-aplicação=aprovada"
    )


def print_diff(
    rendered: Sequence[RenderedFile],
) -> None:
    for item in rendered:
        before_lines = item.before.decode(
            "utf-8"
        ).splitlines(
            keepends=True
        )

        after_lines = item.after.decode(
            "utf-8"
        ).splitlines(
            keepends=True
        )

        sys.stdout.writelines(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=(
                    f"a/{item.relative}"
                ),
                tofile=(
                    f"b/{item.relative}"
                ),
            )
        )


def parse_arguments(
    arguments: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aplicador transacional AP-005B2.3."
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

    if args.verify_post:
        verify_post(root)
        return 0

    rendered = render_all(root)

    print(f"baseline={BASE_COMMIT}")
    print(f"lote={BATCH_NAME}")
    print(
        f"arquivos previstos={len(rendered)}"
    )

    for item in rendered:
        print(
            f"pré={digest(item.before)} "
            f"pós={digest(item.after)} "
            f"{item.relative}"
        )

    print("\n=== DIFF PROPOSTO ===")

    print_diff(rendered)

    if args.check:
        print(
            f"\n{BATCH_NAME} dry-run=aprovado; "
            "arquivos escritos=0"
        )
        return 0

    apply_transaction(rendered)
    verify_post(root)

    print(
        f"\n{BATCH_NAME} aplicação "
        "transacional=concluída"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

BATCH_NAME = "AP-005B2.1"

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
        "f250487a7787c967a0bad0ac38d5dbe210ff63981078d3c65e1d77655ff5f072"
    ),
    RC10_REL: (
        "b7d2e0c8039e0a35ef1ffde343fa315dd15670728fe099fb1dd2c5c7b3fe517d"
    ),
    AP003G_REL: (
        "495160a1a419c8700d7fb43db6886d5f0c162caec585c1c4715846ef374c9168"
    ),
}

EXPECTED_POST_HASHES = {
    PRISMA_REL: (
        "c04a2951ccf92f8353c4bca69d99925d7a4cbe39bd2b97327ecc6a1b748fd6b8"
    ),
    RC10_REL: (
        "207c750e1111b9539cbf65cf64cb622b5a15570e7bbb7733ad5d334f03e62c66"
    ),
    AP003G_REL: (
        "a03892ccc954466b7fd156822248dce245b973421d612b6839eb4e8dd1274540"
    ),
}


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
    matches: list[ast.Assign | ast.AnnAssign] = []

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


def segment(source: str, node: ast.AST) -> str:
    value = ast.get_source_segment(source, node)

    if value is None:
        raise SystemExit(
            "Trecho AST não recuperado na linha "
            f"{getattr(node, 'lineno', '?')}."
        )

    return value


def top_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    }


def effective_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    body = list(function.body)

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
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
        or not isinstance(body[0].value, ast.Call)
    ):
        raise SystemExit(
            f"{function.name}: esperado return-call simples."
        )

    return body[0].value


def forwarding(arguments: ast.arguments) -> str:
    values: list[str] = []

    for argument in (
        *arguments.posonlyargs,
        *arguments.args,
    ):
        values.append(argument.arg)

    if arguments.vararg is not None:
        values.append(f"*{arguments.vararg.arg}")

    for argument in arguments.kwonlyargs:
        values.append(
            f"{argument.arg}={argument.arg}"
        )

    if arguments.kwarg is not None:
        values.append(f"**{arguments.kwarg.arg}")

    return ", ".join(values)


def header(
    name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    keyword = (
        "async def"
        if isinstance(function, ast.AsyncFunctionDef)
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


def load_entries(
    root: pathlib.Path,
) -> list[dict[str, Any]]:
    path = resolve(root, PLAN_REL)

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

    entries = [
        entry
        for entry in payload["entries"]
        if entry["batch"] == BATCH_NAME
    ]

    if len(entries) != 6:
        raise SystemExit(
            f"{BATCH_NAME}: entries={len(entries)}; "
            "esperado=6"
        )

    return entries


def render_prisma(
    source: str,
    entries: list[dict[str, Any]],
) -> str:
    tree = ast.parse(
        source,
        filename=str(PRISMA_REL),
    )

    functions = top_functions(tree)
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
        set(candidates) & set(functions)
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
            or call.func.id != "_invoke_with_runtime"
        ):
            raise SystemExit(
                f"{wrapper_name} não chama o invoker."
            )

        if (
            len(call.args) < 2
            or ast.unparse(call.args[0]) != body_name
            or ast.unparse(call.args[1]) != "runtime"
        ):
            raise SystemExit(
                f"Body/runtime inesperado em {wrapper_name}."
            )

        old = segment(source, wrapper)

        candidate = (
            f"{header(candidate_name, wrapper)}\n"
            f"    return {ast.unparse(call)}"
        )

        compatibility_wrapper = (
            f"{header(wrapper_name, wrapper)}\n"
            f"    return {candidate_name}("
            f"{forwarding(wrapper.args)})"
        )

        rendered = replace_once(
            rendered,
            old,
            candidate + "\n\n" + compatibility_wrapper,
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
            "Adapter já consta dos nomes protegidos."
        )

    rendered = replace_once(
        rendered,
        old_protected,
        (
            "_PROTECTED_RUNTIME_NAMES = "
            f"frozenset({tuple([*protected, *candidates])!r})"
        ),
        "_PROTECTED_RUNTIME_NAMES",
    )

    exports_node = assignment(tree, "__all__")
    old_exports = segment(source, exports_node)

    if set(candidates) & set(exports):
        raise SystemExit(
            "Adapter já consta de __all__."
        )

    rendered = replace_once(
        rendered,
        old_exports,
        f"__all__ = {[*exports, *candidates]!r}",
        "__all__",
    )

    compile(rendered, str(PRISMA_REL), "exec")

    return rendered


def render_rc10(
    source: str,
    entries: list[dict[str, Any]],
) -> str:
    rendered = source

    for entry in entries:
        old = (
            "from academic_pipeline."
            "prisma_generic_orchestration import "
            f"{entry['wrapper_name']} as "
            f"{entry['rc10_local_alias']}"
        )

        new = (
            "from academic_pipeline."
            "prisma_generic_orchestration import "
            f"{entry['candidate_name']} as "
            f"{entry['rc10_local_alias']}"
        )

        rendered = replace_once(
            rendered,
            old,
            new,
            f"import RC10 {entry['wrapper_name']}",
        )

    compile(rendered, str(RC10_REL), "exec")

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

    node = assignment(tree, "EXPECTED_HASHES")
    current = literal_assignment(
        tree,
        "EXPECTED_HASHES",
    )

    if current["prisma"] != (
        EXPECTED_PRE_HASHES[PRISMA_REL]
    ):
        raise SystemExit(
            "EXPECTED_HASHES['prisma'] divergente."
        )

    if current["orchestrator"] != (
        EXPECTED_PRE_HASHES[RC10_REL]
    ):
        raise SystemExit(
            "EXPECTED_HASHES['orchestrator'] divergente."
        )

    old = segment(source, node)

    new = replace_hash_key(
        old,
        key="prisma",
        expected_old=EXPECTED_PRE_HASHES[PRISMA_REL],
        new_value=prisma_hash,
    )

    new = replace_hash_key(
        new,
        key="orchestrator",
        expected_old=EXPECTED_PRE_HASHES[RC10_REL],
        new_value=rc10_hash,
    )

    rendered = replace_once(
        source,
        old,
        new,
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
            "Hash PRISMA não gravado na chave correta."
        )

    if verified["orchestrator"] != rc10_hash:
        raise SystemExit(
            "Hash RC10 não gravado na chave correta."
        )

    compile(rendered, str(AP003G_REL), "exec")

    return rendered


def render_all(
    root: pathlib.Path,
) -> list[RenderedFile]:
    entries = load_entries(root)

    paths = {
        relative: resolve(root, relative)
        for relative in EXPECTED_PRE_HASHES
    }

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
                f"Hash pré-aplicação divergente em "
                f"{relative}: {actual}"
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

    for relative, expected in (
        EXPECTED_POST_HASHES.items()
    ):
        actual = digest(after[relative])

        if actual != expected:
            raise SystemExit(
                f"Hash pós-aplicação divergente em "
                f"{relative}: esperado={expected}; "
                f"encontrado={actual}"
            )

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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )

    temporary = pathlib.Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.chmod(temporary, mode)
        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def apply_transaction(
    rendered: Sequence[RenderedFile],
) -> None:
    backups = {
        item.path: (item.before, item.mode)
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

        for path, (data, mode) in backups.items():
            try:
                atomic_write(path, data, mode)
            except BaseException as error:
                restoration_errors.append(
                    f"{path}: {error}"
                )

        if restoration_errors:
            raise RuntimeError(
                "Falha na aplicação e restauração:\n"
                + "\n".join(restoration_errors)
            )

        raise


def rc10_imports(
    tree: ast.Module,
) -> dict[tuple[str, str], list[str]]:
    result: dict[
        tuple[str, str],
        list[str],
    ] = {}

    for function in tree.body:
        if not isinstance(
            function,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        for node in ast.walk(function):
            if not isinstance(node, ast.ImportFrom):
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
    entries = load_entries(root)

    for relative, expected in (
        EXPECTED_POST_HASHES.items()
    ):
        actual = digest(
            resolve(root, relative).read_bytes()
        )

        if actual != expected:
            raise SystemExit(
                f"Hash pós-aplicação divergente em "
                f"{relative}: {actual}"
            )

    prisma_source = resolve(
        root,
        PRISMA_REL,
    ).read_text(encoding="utf-8")

    rc10_source = resolve(
        root,
        RC10_REL,
    ).read_text(encoding="utf-8")

    ap003g_source = resolve(
        root,
        AP003G_REL,
    ).read_text(encoding="utf-8")

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

    functions = top_functions(prisma_tree)

    exports = set(
        literal_assignment(prisma_tree, "__all__")
    )

    protected = set(
        literal_assignment(
            prisma_tree,
            "_PROTECTED_RUNTIME_NAMES",
        )
    )

    imports = rc10_imports(rc10_tree)

    for entry in entries:
        wrapper = functions.get(
            entry["wrapper_name"]
        )
        candidate = functions.get(
            entry["candidate_name"]
        )
        body = functions.get(
            entry["body_function"]
        )

        if wrapper is None or candidate is None or body is None:
            raise SystemExit(
                f"Símbolos ausentes para "
                f"{entry['wrapper_name']}."
            )

        if ast.unparse(wrapper.args) != (
            ast.unparse(candidate.args)
        ):
            raise SystemExit(
                f"Assinaturas divergentes: "
                f"{entry['wrapper_name']}."
            )

        candidate_call = return_call(candidate)
        wrapper_call = return_call(wrapper)

        if (
            not isinstance(candidate_call.func, ast.Name)
            or candidate_call.func.id
            != "_invoke_with_runtime"
        ):
            raise SystemExit(
                f"{entry['candidate_name']} não chama "
                "_invoke_with_runtime."
            )

        if (
            ast.unparse(candidate_call.args[0])
            != entry["body_function"]
            or ast.unparse(candidate_call.args[1])
            != "runtime"
        ):
            raise SystemExit(
                f"Body/runtime divergente em "
                f"{entry['candidate_name']}."
            )

        if (
            not isinstance(wrapper_call.func, ast.Name)
            or wrapper_call.func.id
            != entry["candidate_name"]
        ):
            raise SystemExit(
                f"{entry['wrapper_name']} não delega "
                "ao adapter."
            )

        for name in (
            entry["wrapper_name"],
            entry["candidate_name"],
        ):
            if name not in exports:
                raise SystemExit(
                    f"{name} ausente de __all__."
                )

            if name not in protected:
                raise SystemExit(
                    f"{name} desprotegido."
                )

        if entry["body_function"] not in protected:
            raise SystemExit(
                f"{entry['body_function']} desprotegido."
            )

        import_key = (
            entry["rc10_consumer_function"],
            entry["rc10_local_alias"],
        )

        if imports.get(import_key) != [
            entry["candidate_name"]
        ]:
            raise SystemExit(
                f"Import RC10 divergente: {import_key}."
            )

    expected_hashes = literal_assignment(
        ap003g_tree,
        "EXPECTED_HASHES",
    )

    if expected_hashes["prisma"] != (
        EXPECTED_POST_HASHES[PRISMA_REL]
    ):
        raise SystemExit(
            "EXPECTED_HASHES['prisma'] divergente."
        )

    if expected_hashes["orchestrator"] != (
        EXPECTED_POST_HASHES[RC10_REL]
    ):
        raise SystemExit(
            "EXPECTED_HASHES['orchestrator'] divergente."
        )

    print(
        f"PRISMA pós="
        f"{EXPECTED_POST_HASHES[PRISMA_REL]}"
    )
    print(
        f"RC10 pós="
        f"{EXPECTED_POST_HASHES[RC10_REL]}"
    )
    print(
        f"AP003G pós="
        f"{EXPECTED_POST_HASHES[AP003G_REL]}"
    )
    print(
        f"{BATCH_NAME} pós-aplicação=aprovada"
    )


def print_diff(
    rendered: Sequence[RenderedFile],
) -> None:
    for item in rendered:
        before = item.before.decode(
            "utf-8"
        ).splitlines(keepends=True)

        after = item.after.decode(
            "utf-8"
        ).splitlines(keepends=True)

        sys.stdout.writelines(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{item.relative}",
                tofile=f"b/{item.relative}",
            )
        )


def parse_arguments(
    arguments: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aplicador transacional AP-005B2.1 v3."
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

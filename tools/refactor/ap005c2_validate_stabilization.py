#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any


SCHEMA_VERSION = (
    "ap005c2.stabilization-manifest.v1"
)

BASELINE_COMMIT = (
    "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
)

TARGET_BRANCH = (
    "ap-refactor/04-consumer-canonicalization"
)

CLOSURE_COMMIT = 'b8cb7ba3a3175ac79799b78a5d0678224076ef80'

POST_COMMIT_REPAIR_FILES = (
    'tools/refactor/ap005c_inventory_toml_capture_aliases.py',
    'tools/refactor/ap005c2_validate_stabilization.py',
    'tools/refactor/ap005c3_validate_closure.py',
    'docs/refactor/academic-pipeline/AP-005/ap005c2_stabilization_manifest.json',
    'docs/refactor/academic-pipeline/AP-005/AP-005C2_STABILIZATION_VALIDATION.md',
    'docs/refactor/academic-pipeline/AP-005/ap005c3_closure_manifest.json',
    'docs/refactor/academic-pipeline/AP-005/AP-005C_CLOSURE_REPORT.md',
    'software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c2_stabilization_contract.py',
    'software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c3_closure_contract.py',
)


PROJECT_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)

MODULE_REL = (
    PROJECT_REL
    / "app_bundle/scripts/pipeline/"
    "academic_pipeline_toml_generator_interativo.py"
)

STRATEGY_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005C_TOML_CAPTURE_ALIAS_STRATEGY.md"
)

INVENTORY_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005c_toml_capture_alias_inventory.json"
)

INVENTORY_TEST_REL = (
    PROJECT_REL
    / "tests/characterization/"
    "test_ap005c_toml_capture_alias_inventory_contract.py"
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

INVENTORY_TOOL_REL = pathlib.PurePosixPath(
    "tools/refactor/"
    "ap005c_inventory_toml_capture_aliases.py"
)

APPLICATOR_REL = pathlib.PurePosixPath(
    "tools/refactor/"
    "ap005c1_apply_toml_capture_aliases.py"
)

STABILIZER_REL = pathlib.PurePosixPath(
    "tools/refactor/"
    "ap005c2_validate_stabilization.py"
)

MANIFEST_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005c2_stabilization_manifest.json"
)

REPORT_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005C2_STABILIZATION_VALIDATION.md"
)

STABILIZATION_TEST_REL = (
    PROJECT_REL
    / "tests/characterization/"
    "test_ap005c2_stabilization_contract.py"
)

CORE_HASHES = {
    str(MODULE_REL): (
        "9d627348fcdc3b9ec727abb3c2862eb26"
        "b11bbd1d1bc744958d892f9f4afa7f9"
    ),
    str(INVENTORY_TOOL_REL): (
        'aed2b3859c124052b0ffa2d0b6a309f6485af3af18added6504c1d65c7fb8137'
    ),
    str(INVENTORY_REL): (
        "f97714602a8c0d076d54819ec429ad8c"
        "492768e1754ea2a326e4e3f71dfc5f63"
    ),
    str(STRATEGY_REL): (
        "4e01b0596b7f55033074f775ababaff48"
        "f84fbef46312d490c4a05e5c7792e6c"
    ),
    str(INVENTORY_TEST_REL): (
        "afbd6003479a49b9633f54f41032ddf2"
        "906a82147073728b675642e7c695c170"
    ),
    str(SEMANTICS_TEST_REL): (
        "031a7f56feab2fca7d6729bb5ed117f9"
        "2abe26a05f80866ca9218d4b539f4795"
    ),
    str(APPLICATION_TEST_REL): (
        "a17498273c482fa6e5855eb78cce2ed1"
        "adc595f7718299e6d2cb3419fca2c7e3"
    ),
    str(APPLICATOR_REL): (
        "4be0bc2bc9de73513f7743be8489a750"
        "006dc58c835c21e74cf0f231f07f4a68"
    ),
}

CANDIDATE_FILES = (
    str(STRATEGY_REL),
    str(INVENTORY_REL),
    str(REPORT_REL),
    str(MANIFEST_REL),
    str(MODULE_REL),
    str(APPLICATION_TEST_REL),
    str(INVENTORY_TEST_REL),
    str(SEMANTICS_TEST_REL),
    str(STABILIZATION_TEST_REL),
    str(APPLICATOR_REL),
    str(INVENTORY_TOOL_REL),
    str(STABILIZER_REL),
)

RECOGNIZED_DOWNSTREAM_CLOSURE_FILES = (
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005C_CLOSURE_REPORT.md",
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005c3_closure_manifest.json",
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c3_closure_contract.py"
    ),
    "tools/refactor/ap005c3_validate_closure.py",
)

SYMBOLS = {
    "_original_ensure_reference_policy": {
        "canonical": (
            "_captured_wiz_input_ensure_reference_policy"
        ),
        "captured_expression": (
            "_WizInputController._ensure_reference_policy"
        ),
        "consumer_count": 1,
    },
    "_wiz_disable_references_original": {
        "canonical": (
            "_captured_wiz_disable_references"
        ),
        "captured_expression": (
            "_wiz_disable_references"
        ),
        "consumer_count": 1,
    },
    "_render_toml_original": {
        "canonical": "_captured_render_toml",
        "captured_expression": "render_toml",
        "consumer_count": 1,
    },
    "_collect_outputs_and_options_original": {
        "canonical": (
            "_captured_collect_outputs_and_options"
        ),
        "captured_expression": (
            "collect_outputs_and_options"
        ),
        "consumer_count": 3,
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
    relative: str | pathlib.PurePosixPath,
) -> pathlib.Path:
    root = root.resolve()
    path = (root / pathlib.PurePosixPath(relative)).resolve()

    try:
        path.relative_to(root)
    except ValueError as error:
        raise SystemExit(
            f"Caminho fora da raiz: {relative}"
        ) from error

    return path


def git(
    root: pathlib.Path,
    *arguments: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            *arguments,
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise SystemExit(
            result.stderr.strip()
            or result.stdout.strip()
            or f"git {' '.join(arguments)} falhou"
        )

    return result.stdout


def verify_hashes(
    root: pathlib.Path,
) -> None:
    for relative, expected in CORE_HASHES.items():
        path = resolve(root, relative)

        if not path.is_file():
            raise SystemExit(
                f"Arquivo ausente: {relative}"
            )

        actual = sha256_bytes(
            path.read_bytes()
        )

        if actual != expected:
            raise SystemExit(
                f"Hash divergente: {relative}; "
                f"esperado={expected}; encontrado={actual}"
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


def module_scope(
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


def assignment_targets(
    node: ast.AST,
) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)

    if isinstance(node, ast.AnnAssign):
        return [node.target]

    return []


def inspect_symbols(
    root: pathlib.Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    module_path = resolve(
        root,
        MODULE_REL,
    )

    tree = ast.parse(
        module_path.read_text(encoding="utf-8"),
        filename=str(module_path),
    )

    parents = parent_map(tree)

    wanted = set(SYMBOLS)

    for item in SYMBOLS.values():
        wanted.add(item["canonical"])

    assignments: dict[
        str,
        tuple[int, str],
    ] = {}

    loads: dict[str, list[int]] = {
        name: []
        for name in wanted
    }

    exported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ) and module_scope(node, parents):
            for target in assignment_targets(node):
                if (
                    isinstance(target, ast.Name)
                    and target.id in wanted
                ):
                    assignments[target.id] = (
                        node.lineno,
                        ast.unparse(node.value),
                    )

        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in loads
        ):
            loads[node.id].append(node.lineno)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        if not any(
            isinstance(target, ast.Name)
            and target.id == "__all__"
            for target in node.targets
        ):
            continue

        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue

        if isinstance(value, (list, tuple)):
            exported.update(
                item
                for item in value
                if isinstance(item, str)
            )

    entries: list[dict[str, Any]] = []
    total_consumers = 0

    for legacy in sorted(SYMBOLS):
        contract = SYMBOLS[legacy]
        canonical = contract["canonical"]
        captured = contract["captured_expression"]
        expected_consumers = contract["consumer_count"]

        if canonical not in assignments:
            raise SystemExit(
                f"Captura canônica ausente: {canonical}"
            )

        if legacy not in assignments:
            raise SystemExit(
                f"Alias histórico ausente: {legacy}"
            )

        canonical_line, canonical_value = (
            assignments[canonical]
        )

        legacy_line, legacy_value = (
            assignments[legacy]
        )

        if canonical_value != captured:
            raise SystemExit(
                f"Binding canônico divergente: {canonical}"
            )

        if legacy_value != canonical:
            raise SystemExit(
                f"Alias histórico divergente: {legacy}"
            )

        if canonical_line >= legacy_line:
            raise SystemExit(
                f"Ordem captura/alias divergente: {legacy}"
            )

        if loads[legacy]:
            raise SystemExit(
                f"Consumidor legado restante: {legacy}"
            )

        consumer_lines = sorted(
            loads[canonical]
        )

        consumer_count = (
            len(consumer_lines) - 1
        )

        if consumer_count != expected_consumers:
            raise SystemExit(
                f"Consumidores canônicos divergentes: "
                f"{canonical}; esperado={expected_consumers}; "
                f"encontrado={consumer_count}"
            )

        if canonical in exported:
            raise SystemExit(
                f"Export público inesperado: {canonical}"
            )

        total_consumers += consumer_count

        entries.append(
            {
                "legacy_alias": legacy,
                "canonical_capture": canonical,
                "captured_expression": captured,
                "canonical_assignment_line": canonical_line,
                "legacy_assignment_line": legacy_line,
                "canonical_load_lines": consumer_lines,
                "productive_consumer_count": consumer_count,
                "legacy_load_lines": sorted(loads[legacy]),
                "publicly_exported": False,
            }
        )

    summary = {
        "canonical_captures": len(entries),
        "legacy_aliases_preserved": len(entries),
        "canonical_consumers": total_consumers,
        "legacy_consumers_remaining": sum(
            len(loads[legacy])
            for legacy in SYMBOLS
        ),
        "new_public_exports": 0,
    }

    if summary != {
        "canonical_captures": 4,
        "legacy_aliases_preserved": 4,
        "canonical_consumers": 6,
        "legacy_consumers_remaining": 0,
        "new_public_exports": 0,
    }:
        raise SystemExit(
            f"Resumo estrutural divergente: {summary}"
        )

    return summary, entries


def verify_workspace(
    root: pathlib.Path,
) -> None:
    head = git(
        root,
        "rev-parse",
        "HEAD",
    ).strip()

    branch = git(
        root,
        "branch",
        "--show-current",
    ).strip()

    if branch != TARGET_BRANCH:
        raise SystemExit(
            f"Branch divergente: {branch}"
        )

    modified = sorted(
        line
        for line in git(
            root,
            "diff",
            "--name-only",
        ).splitlines()
        if line
    )

    staged = sorted(
        line
        for line in git(
            root,
            "diff",
            "--cached",
            "--name-only",
        ).splitlines()
        if line
    )

    untracked = sorted(
        line
        for line in git(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
        if line
    )

    if staged:
        raise SystemExit(
            f"Staging inesperado: {staged}"
        )

    if head == BASELINE_COMMIT:
        if modified != [str(MODULE_REL)]:
            raise SystemExit(
                f"Rastreados divergentes: {modified}"
            )

        required_untracked = {
            relative
            for relative in CANDIDATE_FILES
            if relative != str(MODULE_REL)
        }

        recognized_downstream = set(
            RECOGNIZED_DOWNSTREAM_CLOSURE_FILES
        )

        actual_untracked = set(untracked)

        missing = sorted(
            required_untracked - actual_untracked
        )

        unexpected = sorted(
            actual_untracked
            - required_untracked
            - recognized_downstream
        )

        if missing or unexpected:
            raise SystemExit(
                "Conjunto não rastreado divergente.\n"
                f"Obrigatórios ausentes: {missing}\n"
                f"Não reconhecidos: {unexpected}\n"
                f"Encontrado: {untracked}"
            )
    else:
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                CLOSURE_COMMIT,
                "HEAD",
            ],
            check=False,
            text=True,
            capture_output=True,
        )

        if ancestor.returncode != 0:
            raise SystemExit(
                "Commit de encerramento AP-005C "
                f"não é ancestral de HEAD: {head}"
            )

        allowed = set(
            POST_COMMIT_REPAIR_FILES
        )

        unexpected = sorted(
            set(modified) - allowed
        )

        if unexpected:
            raise SystemExit(
                "Alterações pós-commit não reconhecidas: "
                f"{unexpected}"
            )

        if untracked:
            raise SystemExit(
                "Arquivos não rastreados inesperados "
                f"no modo pós-commit: {untracked}"
            )

    git(
        root,
        "diff",
        "--check",
    )



def productive_diff(
    root: pathlib.Path,
) -> dict[str, Any]:
    head = git(
        root,
        "rev-parse",
        "HEAD",
    ).strip()

    if head == BASELINE_COMMIT:
        arguments = (
            "diff",
            "--numstat",
            "--",
            str(MODULE_REL),
        )
    else:
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                CLOSURE_COMMIT,
                "HEAD",
            ],
            check=False,
            text=True,
            capture_output=True,
        )

        if ancestor.returncode != 0:
            raise SystemExit(
                "Commit de encerramento AP-005C "
                f"não é ancestral de HEAD: {head}"
            )

        arguments = (
            "diff",
            "--numstat",
            f"{BASELINE_COMMIT}..{CLOSURE_COMMIT}",
            "--",
            str(MODULE_REL),
        )

    output = git(
        root,
        *arguments,
    ).strip()

    fields = output.split("\t")

    if len(fields) != 3:
        raise SystemExit(
            f"Numstat inesperado: {output!r}"
        )

    insertions, deletions, relative = fields

    result = {
        "files": 1,
        "insertions": int(insertions),
        "deletions": int(deletions),
        "path": relative,
    }

    if result != {
        "files": 1,
        "insertions": 14,
        "deletions": 10,
        "path": str(MODULE_REL),
    }:
        raise SystemExit(
            f"Diff produtivo divergente: {result}"
        )

    return result



def build_manifest(
    root: pathlib.Path,
) -> dict[str, Any]:
    verify_hashes(root)

    symbol_summary, entries = inspect_symbols(
        root
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": BASELINE_COMMIT,
        "branch": TARGET_BRANCH,
        "productive_module": str(MODULE_REL),
        "core_file_hashes": dict(
            sorted(CORE_HASHES.items())
        ),
        "productive_diff": productive_diff(root),
        "symbol_contract": symbol_summary,
        "entries": entries,
        "candidate_files": sorted(CANDIDATE_FILES),
        "candidate_file_count": len(
            CANDIDATE_FILES
        ),
        "test_gates": {
            "legacy_related_tests": 106,
            "ap005c_tests": 24,
            "focused_regression": 53,
            "canonical_suite_passed": 532,
            "canonical_suite_xfailed": 3,
        },
        "staging": 0,
        "commit": 0,
        "push": 0,
        "next_phase": (
            "AP-005C.3 — encerramento formal, "
            "auditoria do manifesto, commit isolado "
            "e publicação"
        ),
    }

    manifest["contract_fingerprint"] = (
        sha256_bytes(
            canonical_bytes(manifest)
        )
    )

    return manifest


def render_report(
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# AP-005C.2 — Validação de estabilização",
        "",
        "## Baseline",
        "",
        f"- Commit: `{manifest['baseline_commit']}`",
        f"- Branch: `{manifest['branch']}`",
        (
            "- Fingerprint: "
            f"`{manifest['contract_fingerprint']}`"
        ),
        "",
        "## Resultado estrutural",
        "",
        (
            "- Capturas canônicas: "
            f"{manifest['symbol_contract']['canonical_captures']}/4"
        ),
        (
            "- Aliases históricos preservados: "
            f"{manifest['symbol_contract']['legacy_aliases_preserved']}/4"
        ),
        (
            "- Consumidores internos canônicos: "
            f"{manifest['symbol_contract']['canonical_consumers']}/6"
        ),
        (
            "- Consumidores legados restantes: "
            f"{manifest['symbol_contract']['legacy_consumers_remaining']}"
        ),
        (
            "- Novos exports públicos: "
            f"{manifest['symbol_contract']['new_public_exports']}"
        ),
        "",
        "## Diff produtivo",
        "",
        (
            f"- Arquivos: "
            f"{manifest['productive_diff']['files']}"
        ),
        (
            f"- Inserções: "
            f"{manifest['productive_diff']['insertions']}"
        ),
        (
            f"- Remoções: "
            f"{manifest['productive_diff']['deletions']}"
        ),
        (
            f"- Módulo: "
            f"`{manifest['productive_diff']['path']}`"
        ),
        "",
        "## Gates de estabilização",
        "",
        (
            "- Testes legados relacionados: "
            f"{manifest['test_gates']['legacy_related_tests']} passed"
        ),
        (
            "- Testes AP-005C: "
            f"{manifest['test_gates']['ap005c_tests']} passed"
        ),
        (
            "- Regressão focada: "
            f"{manifest['test_gates']['focused_regression']} passed"
        ),
        (
            "- Suíte canônica: "
            f"{manifest['test_gates']['canonical_suite_passed']} passed, "
            f"{manifest['test_gates']['canonical_suite_xfailed']} xfailed"
        ),
        "",
        "## Manifesto candidato",
        "",
        (
            f"Total de arquivos candidatos: "
            f"{manifest['candidate_file_count']}"
        ),
        "",
    ]

    for index, relative in enumerate(
        manifest["candidate_files"],
        start=1,
    ):
        lines.append(
            f"{index}. `{relative}`"
        )

    lines.extend(
        [
            "",
            "## Decisão",
            "",
            (
                "A AP-005C.1 está estabilizada. "
                "Não foram identificadas regressões, "
                "consumidores legados ou alterações "
                "colaterais fora do módulo previsto."
            ),
            "",
            (
                "A consolidação permanece bloqueada até "
                "a AP-005C.3 realizar a auditoria final "
                "do manifesto e receber autorização "
                "explícita para commit e publicação."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def generated_files(
    root: pathlib.Path,
) -> dict[pathlib.Path, bytes]:
    manifest = build_manifest(root)

    manifest_text = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    report_text = render_report(manifest)

    return {
        resolve(
            root,
            MANIFEST_REL,
        ): manifest_text.encode("utf-8"),
        resolve(
            root,
            REPORT_REL,
        ): report_text.encode("utf-8"),
    }


def write_files(
    files: dict[pathlib.Path, bytes],
) -> None:
    for path, data in files.items():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(data)


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

        if path.read_bytes() != expected:
            failures.append(
                f"divergente: {path}"
            )

    if failures:
        raise SystemExit(
            "\n".join(failures)
        )


def parse_arguments(
    arguments: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida e documenta a estabilização "
            "da AP-005C.1."
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

    files = generated_files(root)

    if args.write:
        write_files(files)
        action = "gravados"
    else:
        check_files(files)
        action = "verificados"

    verify_workspace(root)

    manifest = build_manifest(root)

    print(
        f"schema={manifest['schema_version']}"
    )
    print(
        f"fingerprint="
        f"{manifest['contract_fingerprint']}"
    )
    print(
        f"capturas canônicas="
        f"{manifest['symbol_contract']['canonical_captures']}"
    )
    print(
        f"aliases preservados="
        f"{manifest['symbol_contract']['legacy_aliases_preserved']}"
    )
    print(
        f"consumidores canônicos="
        f"{manifest['symbol_contract']['canonical_consumers']}"
    )
    print(
        f"consumidores legados="
        f"{manifest['symbol_contract']['legacy_consumers_remaining']}"
    )
    print(
        f"arquivos candidatos="
        f"{manifest['candidate_file_count']}"
    )
    print(f"arquivos {action}=2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

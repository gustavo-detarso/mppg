#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any


SCHEMA_VERSION = "ap005c3.closure-manifest.v1"

BASELINE_COMMIT = (
    "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
)

TARGET_BRANCH = (
    "ap-refactor/04-consumer-canonicalization"
)

PROJECT_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)

MODULE_REL = (
    PROJECT_REL
    / "app_bundle/scripts/pipeline/"
    "academic_pipeline_toml_generator_interativo.py"
)

C2_MANIFEST_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005c2_stabilization_manifest.json"
)

C2_FINGERPRINT = (
    "9cfc858992cdb30343d02d6526eb36ae"
    "6e8f2cc82fecf762f6849673022528f1"
)

C3_MANIFEST_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005c3_closure_manifest.json"
)

C3_REPORT_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005C_CLOSURE_REPORT.md"
)

C3_VALIDATOR_REL = pathlib.PurePosixPath(
    "tools/refactor/"
    "ap005c3_validate_closure.py"
)

C3_TEST_REL = (
    PROJECT_REL
    / "tests/characterization/"
    "test_ap005c3_closure_contract.py"
)

PRE_CLOSURE_HASHES = {
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "AP-005C2_STABILIZATION_VALIDATION.md"
    ): (
        "f956ba70312eb7009d8f9e654f3623e2"
        "f5e57afcb0785a7bdd9021e30e75c27e"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "AP-005C_TOML_CAPTURE_ALIAS_STRATEGY.md"
    ): (
        "4e01b0596b7f55033074f775ababaff48"
        "f84fbef46312d490c4a05e5c7792e6c"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "ap005c2_stabilization_manifest.json"
    ): (
        "c4df6410dfc48b4f30e225710b060dab"
        "77fdd30fae4e1f389062b9aad662f4d3"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "ap005c_toml_capture_alias_inventory.json"
    ): (
        "f97714602a8c0d076d54819ec429ad8c"
        "492768e1754ea2a326e4e3f71dfc5f63"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "app_bundle/scripts/pipeline/"
        "academic_pipeline_toml_generator_interativo.py"
    ): (
        "9d627348fcdc3b9ec727abb3c2862eb26"
        "b11bbd1d1bc744958d892f9f4afa7f9"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c1_toml_capture_alias_application_contract.py"
    ): (
        "a17498273c482fa6e5855eb78cce2ed1"
        "adc595f7718299e6d2cb3419fca2c7e3"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c2_stabilization_contract.py"
    ): (
        "39b23ccd2494660c14f35049b7c40a85"
        "f32e154b8942726e36b917f41d437846"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c_toml_capture_alias_inventory_contract.py"
    ): (
        "afbd6003479a49b9633f54f41032ddf2"
        "906a82147073728b675642e7c695c170"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c_toml_capture_alias_"
        "semantics_characterization.py"
    ): (
        "031a7f56feab2fca7d6729bb5ed117f9"
        "2abe26a05f80866ca9218d4b539f4795"
    ),
    (
        "tools/refactor/"
        "ap005c1_apply_toml_capture_aliases.py"
    ): (
        "4be0bc2bc9de73513f7743be8489a750"
        "006dc58c835c21e74cf0f231f07f4a68"
    ),
    (
        "tools/refactor/"
        "ap005c2_validate_stabilization.py"
    ): (
        "0eca7bf94cfada9fa2685db50df629e6"
        "4811384c7b0dc0f1906c121c7667e443"
    ),
    (
        "tools/refactor/"
        "ap005c_inventory_toml_capture_aliases.py"
    ): (
        "d3a59884d0a6262adb1e07593bb476f7"
        "c4fce05587e65742563f8911184a98f8"
    ),
}

CANDIDATE_FILES = sorted(
    [
        *PRE_CLOSURE_HASHES,
        str(C3_MANIFEST_REL),
        str(C3_REPORT_REL),
        str(C3_VALIDATOR_REL),
        str(C3_TEST_REL),
    ]
)


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
    path = (
        root / pathlib.PurePosixPath(relative)
    ).resolve()

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
        ["git", "-C", str(root), *arguments],
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


def load_c2_manifest(
    root: pathlib.Path,
) -> dict[str, Any]:
    payload = json.loads(
        resolve(
            root,
            C2_MANIFEST_REL,
        ).read_text(encoding="utf-8")
    )

    if payload["contract_fingerprint"] != (
        C2_FINGERPRINT
    ):
        raise SystemExit(
            "Fingerprint AP-005C.2 divergente."
        )

    return payload


def verify_hashes(
    root: pathlib.Path,
) -> None:
    for relative, expected in (
        PRE_CLOSURE_HASHES.items()
    ):
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


def closure_artifact_hashes(
    root: pathlib.Path,
) -> dict[str, str]:
    result: dict[str, str] = {}

    for relative in (
        C3_VALIDATOR_REL,
        C3_TEST_REL,
    ):
        path = resolve(root, relative)

        if not path.is_file():
            raise SystemExit(
                f"Artefato de encerramento ausente: "
                f"{relative}"
            )

        result[str(relative)] = sha256_bytes(
            path.read_bytes()
        )

    return result


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

    if head != BASELINE_COMMIT:
        raise SystemExit(
            f"HEAD divergente: {head}"
        )

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

    if modified != [str(MODULE_REL)]:
        raise SystemExit(
            f"Rastreados divergentes: {modified}"
        )

    staged = [
        line
        for line in git(
            root,
            "diff",
            "--cached",
            "--name-only",
        ).splitlines()
        if line
    ]

    if staged:
        raise SystemExit(
            f"Staging inesperado: {staged}"
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

    expected_untracked = sorted(
        relative
        for relative in CANDIDATE_FILES
        if relative != str(MODULE_REL)
    )

    if untracked != expected_untracked:
        raise SystemExit(
            "Conjunto não rastreado divergente.\n"
            f"Esperado: {expected_untracked}\n"
            f"Encontrado: {untracked}"
        )

    git(
        root,
        "diff",
        "--check",
    )


def build_manifest(
    root: pathlib.Path,
) -> dict[str, Any]:
    verify_hashes(root)

    c2 = load_c2_manifest(root)

    if c2["symbol_contract"] != {
        "canonical_captures": 4,
        "canonical_consumers": 6,
        "legacy_aliases_preserved": 4,
        "legacy_consumers_remaining": 0,
        "new_public_exports": 0,
    }:
        raise SystemExit(
            "Contrato estrutural AP-005C.2 divergente."
        )

    if c2["productive_diff"] != {
        "deletions": 10,
        "files": 1,
        "insertions": 14,
        "path": str(MODULE_REL),
    }:
        raise SystemExit(
            "Diff produtivo AP-005C.2 divergente."
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": BASELINE_COMMIT,
        "branch": TARGET_BRANCH,
        "source_stabilization_manifest": str(
            C2_MANIFEST_REL
        ),
        "source_stabilization_fingerprint": (
            C2_FINGERPRINT
        ),
        "pre_closure_file_hashes": dict(
            sorted(PRE_CLOSURE_HASHES.items())
        ),
        "closure_artifact_hashes": dict(
            sorted(
                closure_artifact_hashes(
                    root
                ).items()
            )
        ),
        "symbol_contract": c2["symbol_contract"],
        "productive_diff": c2["productive_diff"],
        "candidate_files": CANDIDATE_FILES,
        "candidate_file_count": len(
            CANDIDATE_FILES
        ),
        "test_gates": {
            "ap005c_closure_contracts": 5,
            "ap005c_consolidated": 29,
            "legacy_related": 106,
            "focused_regression": 58,
            "canonical_suite_passed": 537,
            "canonical_suite_xfailed": 3,
        },
        "closure_decision": (
            "ready_for_explicit_commit_and_"
            "publication_approval"
        ),
        "staging": 0,
        "commit": 0,
        "push": 0,
        "next_step": (
            "solicitar aprovação explícita para "
            "staging dos 16 arquivos, commit isolado "
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
        "# AP-005C — Relatório de encerramento",
        "",
        "## Identificação",
        "",
        f"- Baseline: `{manifest['baseline_commit']}`",
        f"- Branch: `{manifest['branch']}`",
        (
            "- Fingerprint de estabilização: "
            f"`{manifest['source_stabilization_fingerprint']}`"
        ),
        (
            "- Fingerprint de encerramento: "
            f"`{manifest['contract_fingerprint']}`"
        ),
        "",
        "## Resultado funcional",
        "",
        "- Capturas canônicas: 4/4",
        "- Aliases históricos preservados: 4/4",
        "- Consumidores internos migrados: 6/6",
        "- Consumidores legados restantes: 0",
        "- Novos exports públicos: 0",
        "",
        "## Escopo produtivo",
        "",
        "- Arquivos produtivos alterados: 1",
        "- Inserções: 14",
        "- Remoções: 10",
        f"- Módulo: `{manifest['productive_diff']['path']}`",
        "",
        "## Validação final",
        "",
        (
            "- Contratos de encerramento: "
            f"{manifest['test_gates']['ap005c_closure_contracts']} passed"
        ),
        (
            "- Testes AP-005C consolidados: "
            f"{manifest['test_gates']['ap005c_consolidated']} passed"
        ),
        (
            "- Testes legados relacionados: "
            f"{manifest['test_gates']['legacy_related']} passed"
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
        "## Manifesto final",
        "",
        (
            f"Arquivos candidatos ao commit isolado: "
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
                "A AP-005C está funcionalmente concluída, "
                "estabilizada e pronta para consolidação."
            ),
            "",
            (
                "Nenhum arquivo foi adicionado ao staging, "
                "nenhum commit foi criado e nenhuma publicação "
                "foi realizada durante o encerramento."
            ),
            "",
            (
                "O próximo passo exige autorização explícita "
                "para staging dos 16 arquivos, commit isolado "
                "e publicação da branch."
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
            C3_MANIFEST_REL,
        ): manifest_text.encode("utf-8"),
        resolve(
            root,
            C3_REPORT_REL,
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
            "Valida o encerramento formal "
            "da AP-005C."
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
        f"arquivos candidatos="
        f"{manifest['candidate_file_count']}"
    )
    print(
        f"decisão="
        f"{manifest['closure_decision']}"
    )
    print(
        f"staging={manifest['staging']}"
    )
    print(
        f"commit={manifest['commit']}"
    )
    print(
        f"push={manifest['push']}"
    )
    print(f"arquivos {action}=2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

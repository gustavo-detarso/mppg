#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import tempfile
from collections.abc import Sequence
from typing import Any


BASE_COMMIT = (
    "6ef568b250390e12dc2e86b86a8c530188604a28"
)

AP005B_JSON_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005b_consumer_canonicalization_plan.json"
)

JSON_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005b2_prisma_adapter_batches.json"
)

MARKDOWN_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005B2_PRISMA_ADAPTER_BATCHES.md"
)

TOOL_REL = pathlib.PurePosixPath(
    "tools/refactor/"
    "ap005b2_plan_prisma_adapter_batches.py"
)

CONTRACT_TEST_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "tests/characterization/"
    "test_ap005b2_prisma_adapter_batches_contract.py"
)

EQUIVALENCE_TEST_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "tests/characterization/"
    "test_ap005b2_prisma_adapter_equivalence_characterization.py"
)

PRISMA_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "academic_pipeline/prisma_generic_orchestration.py"
)

RC10_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)

BATCHES = {
    "AP-005B2.1": {
        "title": "Núcleo e utilitários PRISMA",
        "order": 1,
        "wrappers": (
            "stage_impl_001",
            "_json_or_none_impl_001",
            "make_client_impl_001",
            "_section_impl_001",
            "research_output_paths_impl_001",
            "render_external_prisma_outputs_impl_001",
        ),
    },
    "AP-005B2.2": {
        "title": "Configuração e argumentos da curadoria",
        "order": 2,
        "wrappers": (
            "_prisma_curadoria_default_config_impl_001",
            "_prisma_curadoria_default_out_dir_impl_001",
            "_prisma_curadoria_default_prompt_impl_001",
            "_prisma_curadoria_script_path_impl_001",
            "_prisma_curadoria_arg_impl_001",
            "_prisma_curadoria_config_from_args_impl_001",
            "_prisma_curadoria_out_from_args_impl_001",
            "_prisma_curadoria_prompt_from_args_impl_001",
            "_prisma_curadoria_input_from_args_impl_001",
            "_prisma_curadoria_run_command_impl_001",
        ),
    },
    "AP-005B2.3": {
        "title": "Execução da curadoria",
        "order": 3,
        "wrappers": (
            "_prisma_curadoria_build_cmd_impl_001",
            "_prisma_curadoria_run_ia_impl_001",
            "_prisma_curadoria_reexportar_xlsx_impl_001",
            "_prisma_curadoria_pipeline_supports_flag_impl_001",
            "_prisma_curadoria_importar_no_pipeline_impl_001",
            "_prisma_curadoria_fluxo_completo_impl_001",
            "_prisma_curadoria_mostrar_caminhos_impl_001",
            "_prisma_curadoria_menu_impl_001",
            "_prisma_curadoria_dispatch_impl_001",
        ),
    },
    "AP-005B2.4": {
        "title": "Artigo genérico e entrypoint",
        "order": 4,
        "wrappers": (
            "_prisma_artigo_generico_get_arg_impl_001",
            "_prisma_artigo_generico_strip_impl_001",
            "_prisma_artigo_generico_out_dir_impl_001",
            "_prisma_artigo_generico_run_export_impl_001",
            "_prisma_artigo_generico_run_freeze_impl_001",
            "run_prisma_generic_entrypoint",
        ),
    },
}

EXPECTED_BATCH_SIZES = {
    "AP-005B2.1": 6,
    "AP-005B2.2": 10,
    "AP-005B2.3": 9,
    "AP-005B2.4": 6,
}


def repository_root() -> pathlib.Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if completed.returncode != 0:
        raise SystemExit(completed.stderr)

    return pathlib.Path(
        completed.stdout.strip()
    ).resolve()


class Repository:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root

    def baseline_bytes(
        self,
        path: pathlib.PurePosixPath,
    ) -> bytes:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "show",
                f"{BASE_COMMIT}:{path}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if completed.returncode != 0:
            raise SystemExit(
                completed.stderr.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        return completed.stdout

    def baseline_text(
        self,
        path: pathlib.PurePosixPath,
    ) -> str:
        return self.baseline_bytes(path).decode(
            "utf-8"
        )


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    copy = dict(payload)
    copy.pop("contract_fingerprint", None)

    return json.dumps(
        copy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def contract_fingerprint(
    payload: dict[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()


def top_level_functions(
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


def literal_assignment(
    tree: ast.Module,
    name: str,
) -> Any:
    matches: list[ast.AST] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == name
                for target in node.targets
            ):
                matches.append(node.value)

        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            matches.append(node.value)

    if len(matches) != 1:
        raise SystemExit(
            f"{name}: assignments={len(matches)}; "
            "esperado=1"
        )

    value = matches[0]

    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "frozenset"
        and len(value.args) == 1
    ):
        value = value.args[0]

    return ast.literal_eval(value)


def batch_for_wrapper(wrapper_name: str) -> str:
    matches = [
        batch_name
        for batch_name, batch in BATCHES.items()
        if wrapper_name in batch["wrappers"]
    ]

    if len(matches) != 1:
        raise SystemExit(
            f"{wrapper_name}: lotes={matches}; esperado=1"
        )

    return matches[0]


def rc10_consumers(
    tree: ast.Module,
) -> dict[str, tuple[str, str, int]]:
    result: dict[str, tuple[str, str, int]] = {}

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
                if alias.name in result:
                    raise SystemExit(
                        f"Import duplicado no rc10: "
                        f"{alias.name}"
                    )

                result[alias.name] = (
                    function.name,
                    alias.asname or alias.name,
                    node.lineno,
                )

    return result


def build_payload(repo: Repository) -> dict[str, Any]:
    inherited = json.loads(
        (
            repo.root / AP005B_JSON_REL
        ).read_text(encoding="utf-8")
    )

    if inherited["schema_version"] != (
        "ap005b.consumer-canonicalization-plan.v2"
    ):
        raise SystemExit(
            "Schema AP-005B inesperado."
        )

    if inherited["contract_fingerprint"] != (
        "e659a91460dd5058ba6e49942454c26650eb4455e42f1d6e2ce450125f6284c8"
    ):
        raise SystemExit(
            "Fingerprint AP-005B inesperado."
        )

    inherited_items = [
        item
        for item in inherited["items"]
        if item["application_batch"] == "AP-005B2"
    ]

    if len(inherited_items) != 31:
        raise SystemExit(
            f"Adapters herdados={len(inherited_items)}; "
            "esperado=31"
        )

    prisma_bytes = repo.baseline_bytes(PRISMA_REL)
    rc10_bytes = repo.baseline_bytes(RC10_REL)

    prisma_tree = ast.parse(
        prisma_bytes.decode("utf-8"),
        filename=str(PRISMA_REL),
    )
    rc10_tree = ast.parse(
        rc10_bytes.decode("utf-8"),
        filename=str(RC10_REL),
    )

    functions = top_level_functions(prisma_tree)
    exports = set(
        literal_assignment(prisma_tree, "__all__")
    )
    protected = set(
        literal_assignment(
            prisma_tree,
            "_PROTECTED_RUNTIME_NAMES",
        )
    )
    consumers = rc10_consumers(rc10_tree)

    entries: list[dict[str, Any]] = []

    for item in inherited_items:
        wrapper_name = item["current_name"]
        target = item["canonical_target"]
        candidate_name = target["candidate_name"]
        body_name = target["body_function"]
        batch_name = batch_for_wrapper(wrapper_name)

        wrapper = functions.get(wrapper_name)
        body = functions.get(body_name)

        if wrapper is None:
            raise SystemExit(
                f"Wrapper ausente: {wrapper_name}"
            )

        if body is None:
            raise SystemExit(
                f"Body ausente: {body_name}"
            )

        if wrapper_name not in exports:
            raise SystemExit(
                f"Wrapper fora de __all__: {wrapper_name}"
            )

        if wrapper_name not in protected:
            raise SystemExit(
                f"Wrapper desprotegido: {wrapper_name}"
            )

        if body_name not in protected:
            raise SystemExit(
                f"Body desprotegido: {body_name}"
            )

        consumer = consumers.get(wrapper_name)

        if consumer is None:
            raise SystemExit(
                f"Consumidor ausente: {wrapper_name}"
            )

        consumer_function, local_alias, import_line = (
            consumer
        )

        entries.append(
            {
                "source_candidate_id": (
                    item["source_candidate_id"]
                ),
                "batch": batch_name,
                "batch_order": BATCHES[
                    batch_name
                ]["order"],
                "wrapper_name": wrapper_name,
                "candidate_name": candidate_name,
                "body_function": body_name,
                "signature": ast.unparse(
                    wrapper.args
                ),
                "wrapper_line_baseline": wrapper.lineno,
                "body_line_baseline": body.lineno,
                "rc10_consumer_function": (
                    consumer_function
                ),
                "rc10_local_alias": local_alias,
                "rc10_import_line_baseline": import_line,
                "wrapper_exported_baseline": True,
                "wrapper_protected_baseline": True,
                "body_protected_baseline": True,
                "candidate_exists_baseline": False,
                "required_candidate_export": True,
                "required_candidate_protection": True,
                "wrapper_removal_allowed": False,
                "body_removal_allowed": False,
                "local_alias_change_allowed": False,
                "rollout_state_baseline": "not_applied",
            }
        )

    entries.sort(
        key=lambda entry: (
            entry["batch_order"],
            entry["wrapper_line_baseline"],
        )
    )

    observed_sizes: dict[str, int] = {
        batch_name: sum(
            entry["batch"] == batch_name
            for entry in entries
        )
        for batch_name in BATCHES
    }

    if observed_sizes != EXPECTED_BATCH_SIZES:
        raise SystemExit(
            f"Tamanhos de lote inesperados: "
            f"{observed_sizes}"
        )

    candidate_names = [
        entry["candidate_name"]
        for entry in entries
    ]

    if len(set(candidate_names)) != 31:
        raise SystemExit(
            "Nomes candidatos não são únicos."
        )

    payload: dict[str, Any] = {
        "schema_version": (
            "ap005b2.prisma-adapter-batches.v1"
        ),
        "phase": "AP-005B2",
        "baseline": {
            "commit": BASE_COMMIT,
            "ap005b_schema": inherited[
                "schema_version"
            ],
            "ap005b_contract_fingerprint": inherited[
                "contract_fingerprint"
            ],
            "prisma_sha256": hashlib.sha256(
                prisma_bytes
            ).hexdigest(),
            "rc10_sha256": hashlib.sha256(
                rc10_bytes
            ).hexdigest(),
        },
        "gate": {
            "productive_changes_allowed": False,
            "productive_applicator_allowed": False,
            "staging_allowed": False,
            "commit_allowed": False,
            "push_allowed": False,
            "wrapper_removal_allowed": False,
            "body_removal_allowed": False,
            "partial_batch_rollout_allowed": False,
            "message": (
                "[BLOQUEIO] Este artefato apenas divide "
                "e especifica a AP-005B2."
            ),
        },
        "scope": {
            "adapter_count": 31,
            "batch_count": 4,
            "productive_files_expected": [
                str(PRISMA_REL),
                str(RC10_REL),
            ],
            "historical_contracts_expected_to_change": [
                (
                    "software/"
                    "academic_pipeline_rc10_7_conformidade/"
                    "tests/characterization/"
                    "test_ap003e_prisma_generic_contract.py"
                ),
                (
                    "software/"
                    "academic_pipeline_rc10_7_conformidade/"
                    "tests/characterization/"
                    "test_ap003g_stabilization_contract.py"
                ),
            ],
            "allowed_outputs": [
                str(TOOL_REL),
                str(JSON_REL),
                str(MARKDOWN_REL),
                str(CONTRACT_TEST_REL),
                str(EQUIVALENCE_TEST_REL),
            ],
        },
        "batches": [
            {
                "batch": batch_name,
                "order": data["order"],
                "title": data["title"],
                "adapter_count": observed_sizes[
                    batch_name
                ],
                "wrappers": list(data["wrappers"]),
                "application_rule": (
                    "Todos os adapters do lote devem ser "
                    "aplicados e migrados juntos."
                ),
                "rollback_boundary": (
                    "O lote inteiro deve ser restaurado se "
                    "qualquer teste falhar."
                ),
                "status": "planned",
            }
            for batch_name, data in sorted(
                BATCHES.items(),
                key=lambda pair: pair[1]["order"],
            )
        ],
        "entries": entries,
        "summary": {
            "batch_sizes": observed_sizes,
            "candidate_names_unique": True,
            "candidate_collisions_baseline": 0,
            "baseline_wrappers_exported": 31,
            "baseline_wrappers_protected": 31,
            "baseline_bodies_protected": 31,
            "baseline_internal_consumers": 31,
            "baseline_candidate_test_references": 0,
            "wrappers_to_preserve": 31,
            "bodies_to_preserve": 31,
            "rc10_aliases_to_preserve": 31,
            "candidate_exports_to_add": 31,
            "candidate_protected_names_to_add": 31,
            "productive_files_changed": 0,
        },
    }

    payload["contract_fingerprint"] = (
        contract_fingerprint(payload)
    )

    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# AP-005B2 — Lotes de adapters PRISMA",
        "",
        "> Plano preparatório. Nenhum código produtivo é "
        "alterado nesta etapa.",
        "",
        "## Baseline",
        "",
        (
            f"- Commit: "
            f"`{payload['baseline']['commit']}`"
        ),
        (
            "- Fingerprint: "
            f"`{payload['contract_fingerprint']}`"
        ),
        (
            "- Hash PRISMA: "
            f"`{payload['baseline']['prisma_sha256']}`"
        ),
        (
            "- Hash RC10: "
            f"`{payload['baseline']['rc10_sha256']}`"
        ),
        "",
        "## Partição",
        "",
        "| Lote | Adapters | Escopo |",
        "|---|---:|---|",
    ]

    for batch in payload["batches"]:
        lines.append(
            f"| {batch['batch']} | "
            f"{batch['adapter_count']} | "
            f"{batch['title']} |"
        )

    lines.extend(
        [
            "",
            "## Regras obrigatórias",
            "",
            (
                "1. Um lote é atômico: não pode permanecer "
                "parcialmente aplicado."
            ),
            (
                "2. Cada adapter mantém exatamente a "
                "assinatura do wrapper correspondente."
            ),
            (
                "3. O adapter assume a chamada direta a "
                "`_invoke_with_runtime`."
            ),
            (
                "4. O wrapper histórico permanece e passa "
                "a delegar ao adapter."
            ),
            (
                "5. Adapter e wrapper permanecem em "
                "`_PROTECTED_RUNTIME_NAMES`."
            ),
            (
                "6. O adapter é acrescentado a `__all__`."
            ),
            (
                "7. O `rc10` migra o nome importado, mas "
                "preserva o alias local."
            ),
            (
                "8. Bodies, wrappers e aliases locais não "
                "podem ser removidos."
            ),
            "",
            "## Matriz nominal",
            "",
            (
                "| Lote | Wrapper | Adapter | Body | "
                "Consumidor RC10 | Alias local |"
            ),
            "|---|---|---|---|---|---|",
        ]
    )

    for entry in payload["entries"]:
        lines.append(
            f"| {entry['batch']} | "
            f"`{entry['wrapper_name']}` | "
            f"`{entry['candidate_name']}` | "
            f"`{entry['body_function']}` | "
            f"`{entry['rc10_consumer_function']}` | "
            f"`{entry['rc10_local_alias']}` |"
        )

    lines.extend(
        [
            "",
            "## Contratos históricos a atualizar",
            "",
        ]
    )

    for path in payload["scope"][
        "historical_contracts_expected_to_change"
    ]:
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "## Bloqueios atuais",
            "",
            "```text",
            "alteração produtiva = bloqueada",
            "aplicador produtivo = bloqueado",
            "rollout parcial de lote = bloqueado",
            "remoção de wrappers = bloqueada",
            "remoção de bodies = bloqueada",
            "staging = bloqueado",
            "commit = bloqueado",
            "push = bloqueado",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def generated_outputs(
    repo: Repository,
) -> dict[pathlib.PurePosixPath, bytes]:
    payload = build_payload(repo)

    return {
        JSON_REL: (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
        MARKDOWN_REL: render_markdown(
            payload
        ).encode("utf-8"),
    }


def atomic_write(
    path: pathlib.Path,
    data: bytes,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(data)

    temporary.replace(path)


def write_outputs(
    root: pathlib.Path,
    outputs: dict[pathlib.PurePosixPath, bytes],
) -> None:
    for relative, data in outputs.items():
        atomic_write(root / relative, data)


def check_outputs(
    root: pathlib.Path,
    outputs: dict[pathlib.PurePosixPath, bytes],
) -> None:
    problems: list[str] = []

    for relative, expected in outputs.items():
        path = root / relative

        if not path.is_file():
            problems.append(f"ausente: {relative}")
            continue

        if path.read_bytes() != expected:
            problems.append(f"divergente: {relative}")

    if problems:
        raise SystemExit(
            "Plano AP-005B2 não reproduzido:\n"
            + "\n".join(
                f"- {problem}"
                for problem in problems
            )
        )


def parse_arguments(
    arguments: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
    root = repository_root()
    repo = Repository(root)
    outputs = generated_outputs(repo)

    if args.write:
        write_outputs(root, outputs)
        state = "gerado"
    else:
        check_outputs(root, outputs)
        state = "reproduzido sem divergências"

    payload = json.loads(
        outputs[JSON_REL].decode("utf-8")
    )

    print(f"Plano AP-005B2 {state}.")
    print(
        f"adapters={payload['scope']['adapter_count']} "
        f"lotes={payload['scope']['batch_count']} "
        f"distribuição="
        f"{payload['summary']['batch_sizes']}"
    )
    print(
        f"fingerprint="
        f"{payload['contract_fingerprint']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

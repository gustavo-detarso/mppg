#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from typing import Any


BASE_COMMIT = (
    "6ef568b250390e12dc2e86b86a8c530188604a28"
)

AP005A_JSON_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005a_consumer_dependency_inventory.json"
)

JSON_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005b_consumer_canonicalization_plan.json"
)

PLAN_MD_REL = pathlib.PurePosixPath(
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005B_CONSUMER_CANONICALIZATION_PLAN.md"
)

TOOL_REL = pathlib.PurePosixPath(
    "tools/refactor/"
    "ap005b_plan_consumer_canonicalization.py"
)

TEST_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "tests/characterization/"
    "test_ap005b_consumer_canonicalization_plan_contract.py"
)

CLI_ID = "AP004E-e72e9bb23f1e"

DOCUMENT_IDS = {
    "AP004E-40f450199df1",
    "AP004E-a1f507c8ca1e",
}

ALIAS_IDS = {
    "AP004E-054764be4586",
    "AP004E-5fa6e68ff3fc",
    "AP004E-936e788786e4",
    "AP004E-c3f6df07093a",
}

EXPECTED_CLUSTER_COUNTS = {
    "cli_entrypoints": 1,
    "document_orchestration": 2,
    "prisma_runtime_adapters": 31,
    "toml_assignment_aliases": 4,
}

EXPECTED_EVIDENCE_COUNTS = {
    "AST-IMPORT-NAME": 35,
    "AST-NAME-IMPORTED": 35,
    "AST-NAME-LOCAL": 6,
}

PRISMA_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "academic_pipeline/prisma_generic_orchestration.py"
)

RC10_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)

TOML_GENERATOR_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "app_bundle/scripts/pipeline/"
    "academic_pipeline_toml_generator_interativo.py"
)

LEGACY_REL = pathlib.PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade/"
    "academic_pipeline/legacy.py"
)


def repository_root() -> pathlib.Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise SystemExit(completed.stderr)

    return pathlib.Path(
        completed.stdout.strip()
    ).resolve()


class Repository:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root

    def blob(
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

    def text(
        self,
        path: pathlib.PurePosixPath,
    ) -> str:
        return self.blob(path).decode("utf-8")


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    copy = dict(payload)
    copy.pop("contract_fingerprint", None)

    return json.dumps(
        copy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()


def function_nodes(
    tree: ast.AST,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    }


def prisma_body_target(
    functions: dict[
        str,
        ast.FunctionDef | ast.AsyncFunctionDef,
    ],
    wrapper_name: str,
) -> str:
    function = functions.get(wrapper_name)

    if function is None:
        raise SystemExit(
            f"Wrapper PRISMA ausente: {wrapper_name}"
        )

    matches: list[str] = []

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        if node.func.id != "_invoke_with_runtime":
            continue

        if not node.args:
            continue

        body_argument = node.args[0]

        if isinstance(body_argument, ast.Name):
            matches.append(body_argument.id)

    if len(matches) != 1:
        raise SystemExit(
            f"{wrapper_name}: esperada uma chamada a "
            f"_invoke_with_runtime; encontrado={matches}"
        )

    return matches[0]


def assignment_capture(
    tree: ast.AST,
    alias_name: str,
) -> str:
    values: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        target_names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]

        if alias_name not in target_names:
            continue

        try:
            values.append(ast.unparse(node.value))
        except AttributeError:
            values.append(type(node.value).__name__)

    if len(values) != 1:
        raise SystemExit(
            f"{alias_name}: captura encontrada "
            f"{len(values)} vezes."
        )

    return values[0]


def adapter_candidate(wrapper_name: str) -> str:
    if wrapper_name == "run_prisma_generic_entrypoint":
        return "run_prisma_generic_with_runtime"

    base = wrapper_name

    if base.endswith("_impl_001"):
        base = base[: -len("_impl_001")]

    base = base.lstrip("_")

    return f"{base}_with_runtime"


def classify(candidate_id: str) -> str:
    if candidate_id == CLI_ID:
        return "cli_entrypoints"

    if candidate_id in DOCUMENT_IDS:
        return "document_orchestration"

    if candidate_id in ALIAS_IDS:
        return "toml_assignment_aliases"

    return "prisma_runtime_adapters"


def document_target(candidate_id: str) -> dict[str, Any]:
    targets = {
        "AP004E-40f450199df1": (
            "academic_pipeline.document_orchestration."
            "load_config_impl"
        ),
        "AP004E-a1f507c8ca1e": (
            "academic_pipeline.document_orchestration."
            "load_existing_document_json_impl"
        ),
    }

    try:
        qualified_name = targets[candidate_id]
    except KeyError as error:
        raise AssertionError(candidate_id) from error

    return {
        "kind": "existing_compatibility_wrapper",
        "qualified_name": qualified_name,
        "requires_new_export": False,
    }


def build_plan(repo: Repository) -> dict[str, Any]:
    inherited = json.loads(
        repo.text(AP005A_JSON_REL)
    )

    if inherited["schema_version"] != (
        "ap005a.consumer-dependency-inventory.v3"
    ):
        raise SystemExit(
            "Schema AP-005A inesperado: "
            f"{inherited['schema_version']}"
        )

    if inherited["contract_fingerprint"] != (
        "ad563ca0c5c46d99b5d17e966e9db7eabfeb96bfdef3c09c4a5ffe39d7141309"
    ):
        raise SystemExit(
            "Fingerprint AP-005A inesperado."
        )

    migration_items = [
        item
        for item in inherited["items"]
        if item["application_wave"] == "migração prévia"
    ]

    if len(migration_items) != 38:
        raise SystemExit(
            f"Esperadas 38 superfícies; "
            f"encontradas {len(migration_items)}."
        )

    prisma_tree = ast.parse(
        repo.text(PRISMA_REL),
        filename=str(PRISMA_REL),
    )
    prisma_functions = function_nodes(prisma_tree)

    toml_tree = ast.parse(
        repo.text(TOML_GENERATOR_REL),
        filename=str(TOML_GENERATOR_REL),
    )

    legacy_tree = ast.parse(
        repo.text(LEGACY_REL),
        filename=str(LEGACY_REL),
    )

    legacy_functions = function_nodes(legacy_tree)

    if "run_legacy" not in legacy_functions:
        raise SystemExit(
            "academic_pipeline.legacy.run_legacy ausente."
        )

    evidence_counts: Counter[str] = Counter()
    consumer_file_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()

    entries: list[dict[str, Any]] = []

    for item in migration_items:
        candidate_id = item["source_candidate_id"]
        cluster = classify(candidate_id)

        cluster_counts[cluster] += 1

        internal = item["internal_consumers"]

        if not internal:
            raise SystemExit(
                f"{candidate_id} sem consumidor interno."
            )

        for evidence in internal:
            evidence_counts[evidence["kind"]] += 1
            consumer_file_counts[
                evidence["path"]
            ] += 1

            if evidence["confidence"] == "baixa":
                raise SystemExit(
                    f"{candidate_id} possui evidência interna "
                    "de baixa confiança."
                )

        entry: dict[str, Any] = {
            "source_candidate_id": candidate_id,
            "current_name": item["current_name"],
            "source_path": item["path"],
            "source_line": item["line"],
            "cluster": cluster,
            "risk": item["risk"],
            "internal_evidence_count": len(internal),
            "consumer_files": sorted(
                {
                    evidence["path"]
                    for evidence in internal
                }
            ),
            "consumer_evidence": internal,
            "removal_allowed": False,
            "wrapper_preserved_during_ap005b": True,
        }

        if cluster == "cli_entrypoints":
            entry.update(
                {
                    "ap005b_disposition": (
                        "preserved_existing_contract"
                    ),
                    "canonical_target": {
                        "kind": "existing_public_facade",
                        "qualified_name": (
                            "academic_pipeline.cli.main"
                        ),
                        "requires_new_export": False,
                    },
                    "application_batch": "PRESERVAÇÃO",
                    "reclassification_reason": (
                        "A suíte canônica exige que "
                        "academic_pipeline.main continue "
                        "delegando à fachada cli.main e que "
                        "python -m academic_pipeline continue "
                        "chamando uma função denominada main."
                    ),
                    "characterization_required": [
                        (
                            "academic_pipeline.main delega "
                            "nominalmente a cli.main"
                        ),
                        (
                            "python -m academic_pipeline "
                            "preserva main e SystemExit"
                        ),
                    ],
                }
            )

        elif cluster == "document_orchestration":
            entry.update(
                {
                    "ap005b_disposition": (
                        "preserved_existing_contract"
                    ),
                    "canonical_target": document_target(
                        candidate_id
                    ),
                    "application_batch": "PRESERVAÇÃO",
                    "reclassification_reason": (
                        "Os contratos AP-003D e AP-004C "
                        "exigem que as funções do rc10 "
                        "permaneçam wrappers finos dos "
                        "módulos extraídos."
                    ),
                    "characterization_required": [
                        (
                            "o wrapper rc10 mantém a chamada "
                            "ao símbolo extraído"
                        ),
                        (
                            "o símbolo extraído permanece "
                            "exportado e executável"
                        ),
                    ],
                }
            )

        elif cluster == "prisma_runtime_adapters":
            body_target = prisma_body_target(
                prisma_functions,
                item["current_name"],
            )

            entry.update(
                {
                    "ap005b_disposition": (
                        "requires_named_canonical_adapter"
                    ),
                    "canonical_target": {
                        "kind": "adapter_candidate",
                        "candidate_name": adapter_candidate(
                            item["current_name"]
                        ),
                        "body_function": body_target,
                        "runtime_invoker": (
                            "_invoke_with_runtime"
                        ),
                        "requires_new_export": True,
                    },
                    "application_batch": "AP-005B2",
                    "characterization_required": [
                        (
                            "adapter nomeado e wrapper legado "
                            "produzem o mesmo resultado"
                        ),
                        (
                            "injeção e restauração do namespace "
                            "global permanecem equivalentes"
                        ),
                        (
                            "assinatura pública do wrapper legado "
                            "permanece preservada"
                        ),
                    ],
                }
            )

        elif cluster == "toml_assignment_aliases":
            captured_symbol = assignment_capture(
                toml_tree,
                item["current_name"],
            )

            entry.update(
                {
                    "ap005b_disposition": (
                        "deferred_to_ap005c"
                    ),
                    "canonical_target": {
                        "kind": "captured_previous_binding",
                        "captured_expression": captured_symbol,
                        "requires_new_export": False,
                    },
                    "application_batch": "AP-005C",
                    "deferral_reason": (
                        "O alias captura a implementação anterior "
                        "antes da redefinição. Substituí-lo pelo "
                        "nome corrente produziria recursão ou "
                        "alteraria a cadeia de patches."
                    ),
                    "characterization_required": [
                        (
                            "ordem de captura e redefinição "
                            "permanece congelada"
                        ),
                        (
                            "chamada ao alias não recursa para "
                            "a implementação redefinida"
                        ),
                    ],
                }
            )

        else:
            raise AssertionError(cluster)

        entries.append(entry)

    if dict(cluster_counts) != EXPECTED_CLUSTER_COUNTS:
        raise SystemExit(
            "Clusters inesperados: "
            f"{dict(cluster_counts)}"
        )

    if dict(evidence_counts) != EXPECTED_EVIDENCE_COUNTS:
        raise SystemExit(
            "Contagens AST inesperadas: "
            f"{dict(evidence_counts)}"
        )

    if sum(evidence_counts.values()) != 76:
        raise SystemExit(
            "O total de evidências internas deveria ser 76."
        )

    expected_consumer_files = {
        (
            "software/academic_pipeline_rc10_7_conformidade/"
            "academic_pipeline/__init__.py"
        ),
        (
            "software/academic_pipeline_rc10_7_conformidade/"
            "academic_pipeline/__main__.py"
        ),
        str(RC10_REL),
        str(TOML_GENERATOR_REL),
    }

    if set(consumer_file_counts) != expected_consumer_files:
        raise SystemExit(
            "Arquivos consumidores inesperados: "
            f"{sorted(consumer_file_counts)}"
        )

    entries.sort(
        key=lambda entry: entry["source_candidate_id"]
    )

    preserved_contracts = [
        entry
        for entry in entries
        if entry["application_batch"] == "PRESERVAÇÃO"
    ]

    executable_ap005b = [
        entry
        for entry in entries
        if entry["application_batch"] == "AP-005B2"
    ]

    deferred_ap005c = [
        entry
        for entry in entries
        if entry["application_batch"] == "AP-005C"
    ]

    payload: dict[str, Any] = {
        "schema_version": (
            "ap005b.consumer-canonicalization-plan.v2"
        ),
        "phase": "AP-005B",
        "baseline": {
            "source_commit": BASE_COMMIT,
            "ap005a_schema": inherited["schema_version"],
            "ap005a_contract_fingerprint": inherited[
                "contract_fingerprint"
            ],
        },
        "gate": {
            "productive_changes_allowed": False,
            "productive_applicator_allowed": False,
            "staging_allowed": False,
            "commit_allowed": False,
            "push_allowed": False,
            "removal_allowed": False,
            "message": (
                "[BLOQUEIO] Este artefato apenas planeja "
                "a canonicalização da AP-005B."
            ),
        },
        "scope": {
            "allowed_outputs": [
                str(TOOL_REL),
                str(JSON_REL),
                str(PLAN_MD_REL),
                str(TEST_REL),
            ],
            "inherited_migration_surfaces": 38,
            "ap005b_executable_surfaces": len(
                executable_ap005b
            ),
            "preserved_existing_contracts": len(
                preserved_contracts
            ),
            "ap005c_deferred_aliases": len(
                deferred_ap005c
            ),
            "distinct_consumer_files": len(
                consumer_file_counts
            ),
            "internal_evidence_count": sum(
                evidence_counts.values()
            ),
        },
        "summary": {
            "cluster_counts": dict(
                sorted(cluster_counts.items())
            ),
            "evidence_kind_counts": dict(
                sorted(evidence_counts.items())
            ),
            "consumer_file_evidence_counts": dict(
                sorted(consumer_file_counts.items())
            ),
            "preserved_contract_surfaces": sum(
                entry["application_batch"] == "PRESERVAÇÃO"
                for entry in entries
            ),
            "ap005b2_adapter_surfaces": sum(
                entry["application_batch"] == "AP-005B2"
                for entry in entries
            ),
            "ap005c_deferred_surfaces": sum(
                entry["application_batch"] == "AP-005C"
                for entry in entries
            ),
            "low_confidence_internal_evidence": 0,
            "dynamic_consumers_in_scope": 0,
            "cyclic_components_in_scope": 0,
            "removal_candidates": 0,
            "productive_files_changed": 0,
        },
        "application_batches": [
            {
                "batch": "PRESERVAÇÃO",
                "title": (
                    "Contratos públicos e wrappers extraídos"
                ),
                "surface_count": 3,
                "requires_new_canonical_exports": False,
                "productive_files_expected": [],
                "status": (
                    "preservados após rejeição controlada "
                    "da aplicação AP-005B1"
                ),
            },
            {
                "batch": "AP-005B2",
                "title": (
                    "Adapters canônicos da orquestração PRISMA"
                ),
                "surface_count": 31,
                "requires_new_canonical_exports": True,
                "productive_files_expected": [
                    str(PRISMA_REL),
                    str(RC10_REL),
                ],
                "status": (
                    "aguardando desenho nominal dos adapters "
                    "e testes de equivalência"
                ),
            },
            {
                "batch": "AP-005C",
                "title": (
                    "Aliases de captura do gerador TOML"
                ),
                "surface_count": 4,
                "requires_new_canonical_exports": False,
                "productive_files_expected": [
                    str(TOML_GENERATOR_REL),
                ],
                "status": (
                    "adiado; substituição direta é proibida"
                ),
            },
        ],
        "items": entries,
    }

    payload["contract_fingerprint"] = fingerprint(
        payload
    )

    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    scope = payload["scope"]

    lines = [
        "# AP-005B — Plano de canonicalização de consumidores",
        "",
        "> Plano preparatório e reproduzível. Nenhum código "
        "produtivo é alterado nesta etapa.",
        "",
        "## Baseline",
        "",
        f"- Commit-base: `{BASE_COMMIT}`",
        (
            "- Fingerprint do contrato: "
            f"`{payload['contract_fingerprint']}`"
        ),
        "",
        "## Conclusão de escopo",
        "",
        (
            "- Superfícies herdadas da onda de migração: "
            f"**{scope['inherited_migration_surfaces']}**"
        ),
        (
            "- Superfícies executáveis na AP-005B: "
            f"**{scope['ap005b_executable_surfaces']}**"
        ),
        (
            "- Contratos reclassificados para preservação: "
            f"**{scope['preserved_existing_contracts']}**"
        ),
        (
            "- Aliases adiados para a AP-005C: "
            f"**{scope['ap005c_deferred_aliases']}**"
        ),
        (
            "- Arquivos consumidores distintos: "
            f"**{scope['distinct_consumer_files']}**"
        ),
        (
            "- Evidências AST internas: "
            f"**{scope['internal_evidence_count']}**"
        ),
        "",
        "## Decisão arquitetural",
        "",
        (
            "A tentativa controlada da AP-005B1 foi rejeitada "
            "pela suíte canônica e revertida. A fachada "
            "`academic_pipeline.cli.main` e os wrappers "
            "documentais extraídos constituem contratos "
            "vigentes, não consumidores a serem eliminados."
        ),
        "",
        (
            "Os quatro aliases `_original` do gerador TOML "
            "capturam bindings anteriores às redefinições. "
            "Eles não são imports comuns e não podem ser "
            "substituídos pelos nomes correntes sem risco de "
            "recursão ou alteração da ordem dos patches."
        ),
        "",
        (
            "Os 31 wrappers PRISMA delegam simultaneamente a "
            "`_invoke_with_runtime` e a uma função-corpo "
            "`_ap003e_body_*`. O helper isolado não constitui "
            "destino canônico suficiente. A AP-005B2 deverá "
            "introduzir adapters nomeados antes de migrar os "
            "consumidores do `academic_pipeline_rc10.py`."
        ),
        "",
        "## Lotes",
        "",
        "| Lote | Superfícies | Situação |",
        "|---|---:|---|",
    ]

    for batch in payload["application_batches"]:
        lines.append(
            f"| {batch['batch']} | "
            f"{batch['surface_count']} | "
            f"{batch['status']} |"
        )

    lines.extend(
        [
            "",
            "## Contagens estruturais",
            "",
            (
                "- Contratos preservados: "
                f"**{summary['preserved_contract_surfaces']}**"
            ),
            (
                "- AP-005B2 dependentes de adapters: "
                f"**{summary['ap005b2_adapter_surfaces']}**"
            ),
            (
                "- AP-005C adiadas: "
                f"**{summary['ap005c_deferred_surfaces']}**"
            ),
            "- Evidências internas de baixa confiança: **0**",
            "- Consumidores dinâmicos no escopo: **0**",
            "- Ciclos no escopo: **0**",
            "- Candidatos à remoção: **0**",
            "",
            "## Matriz nominal",
            "",
            (
                "| ID | Superfície | Cluster | Lote | "
                "Destino ou disposição |"
            ),
            "|---|---|---|---|---|",
        ]
    )

    for item in payload["items"]:
        target = item["canonical_target"]

        destination = (
            target.get("qualified_name")
            or target.get("expression")
            or target.get("candidate_name")
            or target.get("captured_expression")
            or target["kind"]
        )

        lines.append(
            f"| `{item['source_candidate_id']}` | "
            f"`{item['current_name']}` | "
            f"{item['cluster']} | "
            f"{item['application_batch']} | "
            f"`{destination}` |"
        )

    lines.extend(
        [
            "",
            "## Gates seguintes",
            "",
            (
                "1. Manter os três contratos reclassificados "
                "sem alteração produtiva."
            ),
            (
                "2. Auditar nominalmente os 31 adapters "
                "propostos para a AP-005B2."
            ),
            (
                "3. Criar testes de equivalência entre "
                "adapters canônicos e wrappers legados."
            ),
            (
                "4. Aplicar a AP-005B2 em lotes pequenos "
                "com rollback transacional."
            ),
            (
                "5. Manter os quatro aliases do gerador TOML "
                "fora da AP-005B."
            ),
            "",
            "## Bloqueios",
            "",
            "```text",
            "alteração produtiva = bloqueada",
            "aplicador produtivo = bloqueado",
            "staging = bloqueado",
            "commit = bloqueado",
            "push = bloqueado",
            "remoção = bloqueada",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def generated_outputs(
    repo: Repository,
) -> dict[pathlib.PurePosixPath, bytes]:
    payload = build_plan(repo)

    json_content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    markdown_content = render_markdown(
        payload
    ).encode("utf-8")

    return {
        JSON_REL: json_content,
        PLAN_MD_REL: markdown_content,
    }


def atomic_write(path: pathlib.Path, data: bytes) -> None:
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
    divergences: list[str] = []

    for relative, expected in outputs.items():
        path = root / relative

        if not path.is_file():
            divergences.append(
                f"ausente: {relative}"
            )
            continue

        actual = path.read_bytes()

        if actual != expected:
            divergences.append(
                f"divergente: {relative}"
            )

    if divergences:
        raise SystemExit(
            "Plano AP-005B não reproduzido:\n"
            + "\n".join(
                f"- {item}"
                for item in divergences
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
        verb = "gerado"
    else:
        check_outputs(root, outputs)
        verb = "reproduzido sem divergências"

    payload = json.loads(
        outputs[JSON_REL].decode("utf-8")
    )

    print(
        f"Plano AP-005B {verb}."
    )
    print(
        "herdadas="
        f"{payload['scope']['inherited_migration_surfaces']} "
        "ap005b="
        f"{payload['scope']['ap005b_executable_surfaces']} "
        "ap005c="
        f"{payload['scope']['ap005c_deferred_aliases']} "
        "evidências="
        f"{payload['scope']['internal_evidence_count']}"
    )
    print(
        "fingerprint="
        f"{payload['contract_fingerprint']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

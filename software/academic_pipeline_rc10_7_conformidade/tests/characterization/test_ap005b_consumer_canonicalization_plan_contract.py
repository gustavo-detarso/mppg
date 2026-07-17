from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]

PLAN = (
    REPOSITORY_ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005b_consumer_canonicalization_plan.json"
)

TOOL = (
    REPOSITORY_ROOT
    / "tools/refactor/"
    "ap005b_plan_consumer_canonicalization.py"
)

EXPECTED_BASE = (
    "6ef568b250390e12dc2e86b86a8c530188604a28"
)

EXPECTED_OUTPUTS = {
    (
        "tools/refactor/"
        "ap005b_plan_consumer_canonicalization.py"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "ap005b_consumer_canonicalization_plan.json"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "AP-005B_CONSUMER_CANONICALIZATION_PLAN.md"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005b_consumer_canonicalization_plan_contract.py"
    ),
}


def _load() -> dict:
    return json.loads(
        PLAN.read_text(encoding="utf-8")
    )


def _fingerprint(payload: dict) -> str:
    copy = dict(payload)
    copy.pop("contract_fingerprint", None)

    canonical = json.dumps(
        copy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def test_ap005b_plan_contract() -> None:
    payload = _load()

    assert payload["schema_version"] == (
        "ap005b.consumer-canonicalization-plan.v2"
    )
    assert payload["phase"] == "AP-005B"
    assert payload["baseline"]["source_commit"] == (
        EXPECTED_BASE
    )
    assert payload["contract_fingerprint"] == (
        _fingerprint(payload)
    )

    assert set(payload["scope"]["allowed_outputs"]) == (
        EXPECTED_OUTPUTS
    )

    assert payload["scope"][
        "inherited_migration_surfaces"
    ] == 38
    assert payload["scope"][
        "ap005b_executable_surfaces"
    ] == 31
    assert payload["scope"][
        "preserved_existing_contracts"
    ] == 3
    assert payload["scope"][
        "ap005c_deferred_aliases"
    ] == 4
    assert payload["scope"][
        "distinct_consumer_files"
    ] == 4
    assert payload["scope"][
        "internal_evidence_count"
    ] == 76

    assert payload["summary"]["cluster_counts"] == {
        "cli_entrypoints": 1,
        "document_orchestration": 2,
        "prisma_runtime_adapters": 31,
        "toml_assignment_aliases": 4,
    }

    assert payload["summary"][
        "evidence_kind_counts"
    ] == {
        "AST-IMPORT-NAME": 35,
        "AST-NAME-IMPORTED": 35,
        "AST-NAME-LOCAL": 6,
    }

    assert payload["summary"][
        "preserved_contract_surfaces"
    ] == 3
    assert payload["summary"][
        "ap005b2_adapter_surfaces"
    ] == 31
    assert payload["summary"][
        "ap005c_deferred_surfaces"
    ] == 4
    assert payload["summary"][
        "low_confidence_internal_evidence"
    ] == 0
    assert payload["summary"][
        "dynamic_consumers_in_scope"
    ] == 0
    assert payload["summary"][
        "cyclic_components_in_scope"
    ] == 0
    assert payload["summary"][
        "removal_candidates"
    ] == 0
    assert payload["summary"][
        "productive_files_changed"
    ] == 0

    assert payload["gate"] == {
        "commit_allowed": False,
        "message": (
            "[BLOQUEIO] Este artefato apenas planeja "
            "a canonicalização da AP-005B."
        ),
        "productive_applicator_allowed": False,
        "productive_changes_allowed": False,
        "push_allowed": False,
        "removal_allowed": False,
        "staging_allowed": False,
    }

    items = payload["items"]

    assert len(items) == 38

    cli_items = [
        item
        for item in items
        if item["cluster"] == "cli_entrypoints"
    ]
    document_items = [
        item
        for item in items
        if item["cluster"]
        == "document_orchestration"
    ]
    prisma_items = [
        item
        for item in items
        if item["cluster"]
        == "prisma_runtime_adapters"
    ]
    alias_items = [
        item
        for item in items
        if item["cluster"]
        == "toml_assignment_aliases"
    ]

    assert len(cli_items) == 1
    assert len(document_items) == 2
    assert len(prisma_items) == 31
    assert len(alias_items) == 4

    assert cli_items[0]["canonical_target"] == {
        "kind": "existing_public_facade",
        "qualified_name": (
            "academic_pipeline.cli.main"
        ),
        "requires_new_export": False,
    }

    preserved = [*cli_items, *document_items]

    assert all(
        item["application_batch"] == "PRESERVAÇÃO"
        and item["ap005b_disposition"]
        == "preserved_existing_contract"
        and item["reclassification_reason"]
        for item in preserved
    )

    assert {
        item["canonical_target"]["qualified_name"]
        for item in document_items
    } == {
        (
            "academic_pipeline.document_orchestration."
            "load_config_impl"
        ),
        (
            "academic_pipeline.document_orchestration."
            "load_existing_document_json_impl"
        ),
    }

    assert all(
        item["application_batch"] == "AP-005B2"
        and item["ap005b_disposition"]
        == "requires_named_canonical_adapter"
        and item["canonical_target"][
            "body_function"
        ].startswith("_ap003e_body_")
        and item["canonical_target"][
            "runtime_invoker"
        ] == "_invoke_with_runtime"
        and item["canonical_target"][
            "requires_new_export"
        ]
        for item in prisma_items
    )

    assert all(
        item["application_batch"] == "AP-005C"
        and item["ap005b_disposition"]
        == "deferred_to_ap005c"
        and item["canonical_target"]["kind"]
        == "captured_previous_binding"
        for item in alias_items
    )

    assert all(
        not item["removal_allowed"]
        and item["wrapper_preserved_during_ap005b"]
        for item in items
    )


def test_ap005b_plan_is_reproducible() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--check"],
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )

    assert completed.returncode == 0, (
        completed.stdout + completed.stderr
    )

    assert (
        "reproduzido sem divergências"
        in completed.stdout
    )

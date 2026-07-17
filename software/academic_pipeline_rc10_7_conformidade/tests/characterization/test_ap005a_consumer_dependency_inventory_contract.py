from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys


SOFTWARE_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]

INVENTORY = (
    REPOSITORY_ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005a_consumer_dependency_inventory.json"
)
AP004E = (
    REPOSITORY_ROOT
    / "docs/refactor/academic-pipeline/AP-004/"
    "ap004e_compatibility_inventory.json"
)
TOOL = (
    REPOSITORY_ROOT
    / "tools/refactor/ap005a_inventory_consumers.py"
)

EXPECTED_BASE = (
    "f45c123bc692b80f4796b701fe71019630dba2f5"
)

EXPECTED_OUTPUTS = {
    "tools/refactor/ap005a_inventory_consumers.py",
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "ap005a_consumer_dependency_inventory.json"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "AP-005A_CONSUMER_DEPENDENCY_INVENTORY.md"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "AP-005A_CONSUMER_MIGRATION_STRATEGY.md"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005a_consumer_dependency_inventory_contract.py"
    ),
}


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_ap005a_inventory_contract() -> None:
    payload = _load(INVENTORY)
    inherited = _load(AP004E)

    assert payload["schema_version"] == (
        "ap005a.consumer-dependency-inventory.v3"
    )
    assert payload["phase"] == "AP-005A"
    assert payload["baseline"]["source_commit"] == EXPECTED_BASE
    assert payload["summary"]["surface_count"] == 64
    assert payload["summary"]["migration_wave_count"] == 38
    assert payload["summary"]["frozen_missing_count"] == 2
    assert payload["summary"]["syntax_errors"] == 0
    assert payload["summary"]["decode_errors"] == 0
    assert payload["summary"]["removal_candidates"] == 0
    assert payload["summary"]["productive_files_changed"] == 0

    assert payload["gate"] == {
        "commit_allowed": False,
        "integration_allowed": False,
        "inventory_approval_required": True,
        "message": (
            "[BLOQUEIO] A AP-005A é exclusivamente "
            "preparatória."
        ),
        "productive_applicator_allowed": False,
        "productive_changes_allowed": False,
        "push_allowed": False,
        "removal_allowed": False,
    }

    assert set(payload["scope"]["allowed_outputs"]) == (
        EXPECTED_OUTPUTS
    )
    assert payload["contract_fingerprint"] == _fingerprint(
        payload
    )

    assert payload["scope"]["python_files_analyzed"] >= 83
    assert payload["import_graph"]["module_count"] > 0
    assert payload["import_graph"]["edge_count"] > 0

    assert (
        payload["summary"]["items_with_ambiguities"]
        == 0
    )

    by_id = {
        item["source_candidate_id"]: item
        for item in payload["items"]
    }

    assert by_id["AP004E-2d5ff25925a0"][
        "dynamic_consumers"
    ]

    reclassified_contract_ids = {
        "AP004E-15bc59c372e4",
        "AP004E-80cb0eef7050",
        "AP004E-9bb395bcbaa2",
        "AP004E-ab00608841d3",
    }

    for candidate_id in reclassified_contract_ids:
        assert not by_id[candidate_id]["ambiguities"]
        assert by_id[candidate_id]["contractual_notes"]

    assert not by_id["AP004E-15bc59c372e4"][
        "dynamic_consumers"
    ]
    assert not by_id["AP004E-80cb0eef7050"][
        "dynamic_consumers"
    ]

    migration_items = [
        item
        for item in payload["items"]
        if item["application_wave"] == "migração prévia"
    ]

    assert len(migration_items) == 38

    migration_without_internal = [
        item["source_candidate_id"]
        for item in migration_items
        if item["consumer_counts"]["internal"] == 0
    ]

    assert not migration_without_internal, (
        migration_without_internal
    )

    generic_entrypoint_contexts = [
        evidence["context"]
        for evidence in by_id["AP004E-9bb395bcbaa2"][
            "internal_consumers"
        ]
    ]

    assert not any(
        ".academic_pipeline/work" in context
        or ".academic_pipeline/cache" in context
        for context in generic_entrypoint_contexts
    )

    inherited_ids = {
        item["candidate_id"]
        for item in inherited["items"]
    }
    current_ids = {
        item["source_candidate_id"]
        for item in payload["items"]
    }

    assert len(inherited_ids) == 64
    assert current_ids == inherited_ids

    assert all(
        item["removal_eligibility"]
        == "bloqueada na AP-005A"
        for item in payload["items"]
    )


def test_ap005a_inventory_is_reproducible() -> None:
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

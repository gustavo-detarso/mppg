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
    "ap005b2_prisma_adapter_batches.json"
)

TOOL = (
    REPOSITORY_ROOT
    / "tools/refactor/"
    "ap005b2_plan_prisma_adapter_batches.py"
)

EXPECTED_OUTPUTS = {
    (
        "tools/refactor/"
        "ap005b2_plan_prisma_adapter_batches.py"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "ap005b2_prisma_adapter_batches.json"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "AP-005B2_PRISMA_ADAPTER_BATCHES.md"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005b2_prisma_adapter_batches_contract.py"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005b2_prisma_adapter_equivalence_characterization.py"
    ),
}


def _load() -> dict:
    return json.loads(
        PLAN.read_text(encoding="utf-8")
    )


def _fingerprint(payload: dict) -> str:
    copy = dict(payload)
    copy.pop("contract_fingerprint", None)

    encoded = json.dumps(
        copy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def test_ap005b2_batch_plan_contract() -> None:
    payload = _load()

    assert payload["schema_version"] == (
        "ap005b2.prisma-adapter-batches.v1"
    )
    assert payload["phase"] == "AP-005B2"
    assert payload["baseline"]["commit"] == (
        "6ef568b250390e12dc2e86b86a8c530188604a28"
    )
    assert payload["baseline"]["prisma_sha256"] == (
        "f250487a7787c967a0bad0ac38d5dbe210ff63981078d3c65e1d77655ff5f072"
    )
    assert payload["baseline"]["rc10_sha256"] == (
        "b7d2e0c8039e0a35ef1ffde343fa315dd15670728fe099fb1dd2c5c7b3fe517d"
    )
    assert payload["contract_fingerprint"] == (
        _fingerprint(payload)
    )

    assert payload["scope"]["adapter_count"] == 31
    assert payload["scope"]["batch_count"] == 4
    assert set(payload["scope"]["allowed_outputs"]) == (
        EXPECTED_OUTPUTS
    )

    assert payload["summary"]["batch_sizes"] == {
        "AP-005B2.1": 6,
        "AP-005B2.2": 10,
        "AP-005B2.3": 9,
        "AP-005B2.4": 6,
    }
    assert payload["summary"][
        "candidate_names_unique"
    ]
    assert payload["summary"][
        "candidate_collisions_baseline"
    ] == 0
    assert payload["summary"][
        "baseline_wrappers_exported"
    ] == 31
    assert payload["summary"][
        "baseline_wrappers_protected"
    ] == 31
    assert payload["summary"][
        "baseline_bodies_protected"
    ] == 31
    assert payload["summary"][
        "baseline_internal_consumers"
    ] == 31
    assert payload["summary"][
        "wrappers_to_preserve"
    ] == 31
    assert payload["summary"][
        "bodies_to_preserve"
    ] == 31
    assert payload["summary"][
        "rc10_aliases_to_preserve"
    ] == 31
    assert payload["summary"][
        "candidate_exports_to_add"
    ] == 31
    assert payload["summary"][
        "candidate_protected_names_to_add"
    ] == 31
    assert payload["summary"][
        "productive_files_changed"
    ] == 0

    assert len(payload["entries"]) == 31

    candidates = {
        entry["candidate_name"]
        for entry in payload["entries"]
    }
    wrappers = {
        entry["wrapper_name"]
        for entry in payload["entries"]
    }
    bodies = {
        entry["body_function"]
        for entry in payload["entries"]
    }
    aliases = {
        entry["rc10_local_alias"]
        for entry in payload["entries"]
    }

    assert len(candidates) == 31
    assert len(wrappers) == 31
    assert len(bodies) == 31
    assert len(aliases) == 31

    assert all(
        entry["required_candidate_export"]
        and entry["required_candidate_protection"]
        and not entry["wrapper_removal_allowed"]
        and not entry["body_removal_allowed"]
        and not entry["local_alias_change_allowed"]
        and entry["rollout_state_baseline"]
        == "not_applied"
        for entry in payload["entries"]
    )

    assert payload["gate"][
        "partial_batch_rollout_allowed"
    ] is False
    assert payload["gate"][
        "productive_changes_allowed"
    ] is False
    assert payload["gate"][
        "wrapper_removal_allowed"
    ] is False
    assert payload["gate"][
        "body_removal_allowed"
    ] is False


def test_ap005b2_batch_plan_is_reproducible() -> None:
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

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[4]

MANIFEST = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005c3_closure_manifest.json"
)

VALIDATOR = (
    ROOT
    / "tools/refactor/"
    "ap005c3_validate_closure.py"
)


def load_manifest() -> dict[str, Any]:
    return json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_closure_schema_and_baseline() -> None:
    payload = load_manifest()

    assert payload["schema_version"] == (
        "ap005c3.closure-manifest.v1"
    )

    assert payload["baseline_commit"] == (
        "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
    )

    assert payload["branch"] == (
        "ap-refactor/04-consumer-canonicalization"
    )

    assert payload[
        "source_stabilization_fingerprint"
    ] == (
        "9cfc858992cdb30343d02d6526eb36ae"
        "6e8f2cc82fecf762f6849673022528f1"
    )


def test_closure_fingerprint_is_reproducible() -> None:
    payload = load_manifest()
    expected = payload.pop(
        "contract_fingerprint"
    )

    actual = hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()

    assert actual == expected


def test_closure_manifest_is_exact() -> None:
    payload = load_manifest()

    assert payload["candidate_file_count"] == 16
    assert len(payload["candidate_files"]) == 16
    assert len(set(payload["candidate_files"])) == 16

    assert payload["symbol_contract"] == {
        "canonical_captures": 4,
        "canonical_consumers": 6,
        "legacy_aliases_preserved": 4,
        "legacy_consumers_remaining": 0,
        "new_public_exports": 0,
    }

    assert payload["productive_diff"] == {
        "deletions": 10,
        "files": 1,
        "insertions": 14,
        "path": (
            "software/academic_pipeline_rc10_7_conformidade/"
            "app_bundle/scripts/pipeline/"
            "academic_pipeline_toml_generator_interativo.py"
        ),
    }


def test_closure_gates_and_decision() -> None:
    payload = load_manifest()

    assert payload["test_gates"] == {
        "ap005c_closure_contracts": 5,
        "ap005c_consolidated": 29,
        "legacy_related": 106,
        "focused_regression": 58,
        "canonical_suite_passed": 537,
        "canonical_suite_xfailed": 3,
    }

    assert payload["closure_decision"] == (
        "ready_for_explicit_commit_and_"
        "publication_approval"
    )

    assert payload["staging"] == 0
    assert payload["commit"] == 0
    assert payload["push"] == 0


def test_closure_validator_check_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
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

    assert "arquivos candidatos=16" in result.stdout

    assert (
        "decisão=ready_for_explicit_commit_"
        "and_publication_approval"
    ) in result.stdout

    assert "staging=0" in result.stdout
    assert "commit=0" in result.stdout
    assert "push=0" in result.stdout

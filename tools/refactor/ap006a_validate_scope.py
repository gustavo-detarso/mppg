#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_HEAD = "a9d0fa1e100af966329d48629ec234e32da6ded7"
EXPECTED_FINGERPRINT = "fe2e4262338815154f42b8eaca7d33c8d3d87bb9e1302c9bcc08fc2b5326a179"
MANIFEST_REL = "docs/refactor/academic-pipeline/AP-006/ap006a_scope_manifest.json"
CLASSIFICATION_REL = "docs/refactor/academic-pipeline/AP-006/ap006a_dependency_classification.json"
AUDIT_REL = "docs/refactor/academic-pipeline/AP-006/AP-006A_PHYSICAL_NAMING_IMPACT_AUDIT.md"
DECISION_REL = "docs/refactor/academic-pipeline/AP-006/AP-006A_ARCHITECTURAL_DECISION.md"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(repo: Path) -> None:
    manifest = json.loads((repo / MANIFEST_REL).read_text(encoding="utf-8"))
    assert manifest["phase"] == "AP-006A"
    assert manifest["baseline"]["commit"] == EXPECTED_HEAD
    assert manifest["source_evidence"]["summary_fingerprint_sha256"] == EXPECTED_FINGERPRINT

    assert manifest["frozen_findings"] == {
        "active_layer_count": 6,
        "contract_file_count": 19,
        "exact_reference_file_count": 289,
        "external_active_file_count": 38,
        "internal_productive_file_count": 96,
        "package_surface_file_count": 91,
        "runtime_blocking_file_count": 153,
        "unique_match_line_count": 39141,
    }

    decision = manifest["decision"]
    assert decision["preferred_strategy"] == "encapsulate_then_migrate_with_compatibility"
    assert decision["direct_rename_decision"] == "reject_without_compatibility_layer"
    assert decision["recommended_subphases"] == 6
    assert decision["excluded_direct_destination"] == "software/academic_pipeline"
    assert manifest["subphases"] == [
        "AP-006A", "AP-006B", "AP-006C", "AP-006D", "AP-006E", "AP-006F"
    ]

    evidence = json.loads((repo / CLASSIFICATION_REL).read_text(encoding="utf-8"))
    assert evidence["provenance"]["summary_fingerprint_sha256"] == EXPECTED_FINGERPRINT
    assert evidence["summary"]["runtime_blocking_file_count"] == 153
    assert evidence["summary"]["external_active_file_count"] == 38
    assert evidence["summary"]["active_layer_count"] == 6
    assert all(evidence["layer_model"].values())

    destinations = {
        item["candidate"]: item for item in evidence["candidate_destinations"]
    }
    assert destinations["software/academic_pipeline"]["collision"] is True
    assert destinations["software/academic_pipeline_mppg"]["collision"] is False
    assert destinations["software/academic-pipeline"]["collision"] is False

    for rel, expected in manifest["contract_file_sha256"].items():
        assert _sha(repo / rel) == expected

    assert "seis subfases" in (repo / AUDIT_REL).read_text(encoding="utf-8").lower()
    assert "renomeação direta" in (repo / DECISION_REL).read_text(encoding="utf-8").lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    validate(args.repo_root.resolve())
    print("AP-006A scope contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

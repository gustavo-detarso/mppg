#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

DOC_DIR_REL = Path("docs/refactor/academic-pipeline/AP-007")
MANIFEST_REL = DOC_DIR_REL / "ap007a_scope_manifest.json"
CANONICAL_JSON_REL = DOC_DIR_REL / "ap007a_runtime_legado_inventory_canonical.json"
CANONICAL_TSV_REL = DOC_DIR_REL / "ap007a_runtime_legado_references_canonical.tsv"
RAW_JSON_REL = DOC_DIR_REL / "ap007a_runtime_legado_inventory_raw.json"
RAW_TSV_REL = DOC_DIR_REL / "ap007a_runtime_legado_references_raw.tsv"
RAW_MD_REL = DOC_DIR_REL / "AP-007A_RUNTIME_LEGACY_RAW_EVIDENCE.md"
CANONICAL_MD_REL = DOC_DIR_REL / "AP-007A_RUNTIME_LEGACY_CANONICAL_AUDIT.md"
DECISION_MD_REL = DOC_DIR_REL / "AP-007A_ARCHITECTURAL_DECISION.md"

EXPECTED_RAW_HASHES = {
    "json": "b2612edfad7bf38be62d498e942350c17a71a2f7cfe023503c2db320d927536e",
    "tsv": "3d58b0fdcdec0435b5bcef703eb50a8bde0d85a8a06e425160b4066ff94e5c30",
    "markdown": "7878a953bf1c6e6b2d4b169d0ba25da70b763c8391458afd55f5d79268164923",
}
EXPECTED_CATEGORY_COUNTS = {
    "active_compatibility_bridge": 7,
    "documentation_provenance": 5,
    "generated_legacy_command": 2,
    "generated_legacy_usage": 1,
    "internal_dynamic_reentry": 4,
    "internal_sys_argv_reentry": 4,
    "legacy_alias_consumer": 1,
    "provenance_comment": 20,
    "public_entrypoint_bridge": 2,
    "unrelated_dependency_probe": 1,
}
EXPECTED_FIRST_WAVE = [
    "--help",
    "--list-toml-profiles",
    "--list-institutions",
    "--list-layouts",
    "--explain-profile",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON não é objeto: {path}")
    return payload


def validate(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    manifest = _load_json(repo / MANIFEST_REL)
    canonical = _load_json(repo / CANONICAL_JSON_REL)

    assert manifest["schema"] == "ap007a-scope-manifest-v1"
    assert manifest["phase"] == "AP-007A"
    assert canonical["schema"] == "ap007a-runtime-legacy-canonical-inventory-v1"
    assert canonical["phase"] == "AP-007A"

    assert _sha256(repo / RAW_JSON_REL) == EXPECTED_RAW_HASHES["json"]
    assert _sha256(repo / RAW_TSV_REL) == EXPECTED_RAW_HASHES["tsv"]
    assert _sha256(repo / RAW_MD_REL) == EXPECTED_RAW_HASHES["markdown"]
    assert manifest["raw_evidence_sha256"] == EXPECTED_RAW_HASHES

    partition = canonical["semantic_partition"]
    assert partition["raw_active_high_risk_occurrences"] == 94
    assert partition["excluded_backup_or_patch_backup_occurrences"] == 37
    assert partition["excluded_old_parallel_tree_occurrences"] == 9
    assert partition["canonical_tree_raw_occurrences"] == 48
    assert partition["canonical_tree_unique_pattern_lines"] == 47
    assert partition["operational_coupling_unique_lines"] == 20
    assert partition["migration_relevant_unique_lines"] == 21
    assert partition["category_unique_line_counts"] == EXPECTED_CATEGORY_COUNTS

    public = canonical["public_cli_contract"]
    assert public["all_options_discovered_across_parsers"] == 138
    assert public["top_level_explicit_options"] == 62
    assert public["implicit_argparse_help_options"] == 1
    assert public["top_level_public_options_total"] == 63
    assert public["auxiliary_or_internal_options"] == 75
    assert public["first_native_migration_wave"] == EXPECTED_FIRST_WAVE
    assert public["excluded_false_public_candidate"] == "--list-profiles"

    implicit = canonical["implicit_context_evidence"]
    assert implicit["globals_calls_total"] == 84
    assert implicit["locals_calls_total"] == 84
    assert implicit["globals_calls_in_pipeline_core"] == 39
    assert implicit["locals_calls_in_pipeline_core"] == 39

    assert len(canonical["productive_surfaces"]) == 7
    assert len(canonical["canonical_references"]) == 47
    assert sum(bool(row["operational_coupling"]) for row in canonical["canonical_references"]) == 20
    assert sum(bool(row["migration_relevant"]) for row in canonical["canonical_references"]) == 21

    with (repo / CANONICAL_TSV_REL).open("r", encoding="utf-8", newline="") as handle:
        tsv_rows = list(csv.DictReader(handle, delimiter="	"))
    assert len(tsv_rows) == 47
    assert Counter(row["reviewed_category"] for row in tsv_rows) == Counter(EXPECTED_CATEGORY_COUNTS)

    decision = canonical["architectural_decision"]
    assert decision["strategy"] == "native_runtime_with_explicit_argv_and_enumerated_legacy_fallback"
    assert decision["run_legacy_removal"] == "deferred_until_command_family_equivalence"
    assert len(decision["subphases"]) == 6
    assert len(decision["closure_criteria"]) == 11

    assert manifest["runtime_productive_files_modified"] == []
    materialized = set(manifest["materialized_paths"])
    assert not any(
        path.startswith("software/academic_pipeline_mppg/academic_pipeline/")
        or path.startswith("software/academic_pipeline_mppg/app_bundle/scripts/pipeline/")
        for path in materialized
    )

    for relative, expected in manifest["primary_artifact_sha256"].items():
        assert _sha256(repo / relative) == expected

    canonical_md = (repo / CANONICAL_MD_REL).read_text(encoding="utf-8")
    decision_md = (repo / DECISION_MD_REL).read_text(encoding="utf-8")
    assert "47 linhas canônicas" in canonical_md
    assert "20" in canonical_md and "acoplamento operacional" in canonical_md
    assert "native_runtime_with_explicit_argv_and_enumerated_legacy_fallback" in decision_md
    assert "não será uma cópia tipada" in decision_md

    return {
        "ok": True,
        "phase": "AP-007A",
        "raw_references": 7506,
        "canonical_lines": 47,
        "operational_lines": 20,
        "productive_surfaces": 7,
        "public_options": 63,
        "runtime_productive_files_modified": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida a materialização documental da AP-007A.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = validate(args.repo)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print("AP-007A runtime legacy inventory contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

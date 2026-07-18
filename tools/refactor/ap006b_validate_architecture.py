#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_HEAD = "9745f3727403c7ed7637faa12141cd16310f18c1"
EXPECTED_FINGERPRINT = "248523bda24d985e1b1d7a0aa7de3c4cafade6bd640860c93da79d79b08ffbc7"

TARGET_REL = "docs/refactor/academic-pipeline/AP-006/AP-006B_TARGET_ARCHITECTURE.md"
CONTRACT_REL = "docs/refactor/academic-pipeline/AP-006/AP-006B_COMPATIBILITY_CONTRACT.md"
DECISION_REL = "docs/refactor/academic-pipeline/AP-006/ap006b_architecture_decision.json"
WAVES_REL = "docs/refactor/academic-pipeline/AP-006/ap006b_consumer_wave_manifest.json"


def validate(repo: Path) -> None:
    decision = json.loads((repo / DECISION_REL).read_text(encoding="utf-8"))
    waves = json.loads((repo / WAVES_REL).read_text(encoding="utf-8"))

    assert decision["phase"] == "AP-006B"
    assert decision["baseline"]["commit"] == EXPECTED_HEAD
    assert decision["source_evidence"]["summary_fingerprint_sha256"] == EXPECTED_FINGERPRINT

    architecture = decision["target_architecture"]
    assert architecture["selected_physical_target"] == "software/academic_pipeline_mppg"
    assert architecture["compatibility_bridge"] == (
        "tracked_relative_symlink_plus_canonical_resource_resolver"
    )
    assert architecture["symlink_supported"] is True
    assert architecture["duplicate_tree"] == "rejected"
    assert architecture["direct_move_without_bridge"] == "rejected"

    public = decision["public_contract"]
    assert public["distribution_name"] == "academic-pipeline-mppg"
    assert public["console_script"] == {
        "name": "academic-pipeline",
        "target": "academic_pipeline.cli:main",
    }
    assert public["python_import_surfaces"] == ["academic_pipeline", "app_bundle"]

    resolution = decision["resource_resolution_contract"]
    assert resolution["packaged_resources"] == "importlib.resources"
    assert resolution["repository_resources"] == "single_canonical_root_resolver"
    assert "infer_repository_root_from_versioned_directory_name" in resolution["forbidden"]

    assert waves["phase"] == "AP-006B"
    assert waves["status"] == "upper_bound_inventory"
    assert waves["inventory_summary"]["active_reference_file_count"] == 133
    assert waves["inventory_summary"]["resolver_signal_line_count"] == 494
    assert waves["interpretation"]["counts_are_upper_bounds"] is True
    assert waves["interpretation"]["not_equal_to_required_edits"] is True

    wave_map = {item["wave"]: item for item in waves["waves"]}
    assert wave_map["wave_1_internal_runtime"]["owner_phase"] == "AP-006C"
    assert wave_map["wave_1_packaging_and_entrypoints"]["owner_phase"] == "AP-006C"
    assert wave_map["wave_2_contracts_and_validators"]["owner_phase"] == "AP-006D"
    assert wave_map["wave_3_external_operational"]["active_file_upper_bound"] == 0
    assert wave_map["wave_4_sources_and_regeneration"]["owner_phase"] == "AP-006D"

    target_text = (repo / TARGET_REL).read_text(encoding="utf-8")
    contract_text = (repo / CONTRACT_REL).read_text(encoding="utf-8")
    assert "software/academic_pipeline_mppg" in target_text
    assert "AP-006F" in target_text
    assert "importlib.resources" in contract_text
    assert "árvore duplicada" in contract_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    validate(args.repo_root.resolve())
    print("AP-006B architecture contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

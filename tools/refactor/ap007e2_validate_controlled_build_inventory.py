from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JSON_PATH = REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e2_controlled_build_inventory.json'
EXPECTED_SCHEMA = "ap007e2_controlled_build_inventory.v1"
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007E2_CONTROLLED_BUILD_INVENTORY.md', 'docs/refactor/academic-pipeline/AP-007/ap007e2_controlled_build_inventory.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e2_controlled_build_inventory_contract.py', 'tools/refactor/ap007e2_validate_controlled_build_inventory.py']


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks = [
        (data.get("schema") == EXPECTED_SCHEMA, "schema"),
        (data.get("phase") == "AP-007E.2", "phase"),
        (data.get("status") in {"build_approved", "build_approved_with_classified_findings"}, "status"),
        (data.get("scope", {}).get("materialized_paths") == EXPECTED_PATHS, "materialized paths"),
        (data.get("scope", {}).get("build_executed") is True, "build executed"),
        (data.get("scope", {}).get("installation_executed") is False, "no installation"),
        (data.get("scope", {}).get("dependency_installation_executed") is False, "no dependency installation"),
        (data.get("scope", {}).get("git_write_executed") is False, "no git write"),
        (len(data.get("builds", [])) == 2, "two builds"),
        (len(data.get("snapshots", [])) == 2, "two snapshots"),
        (all(s.get("residual_filter", {}).get("applied_before_extraction") is True for s in data.get("snapshots", [])), "pre-extraction residual filter"),
        (all(s.get("residual_filter", {}).get("destination_path_constructed_before_classification") is False for s in data.get("snapshots", [])), "classification before destination paths"),
        (all(s.get("archive_member_count_total", 0) == s.get("member_count_extracted", 0) + s.get("residual_member_count_excluded", 0) for s in data.get("snapshots", [])), "snapshot member accounting"),
        (all(b.get("wheel", {}).get("member_count", 0) > 0 for b in data.get("builds", [])), "wheel members"),
        (all(b.get("sdist", {}).get("member_count", 0) > 0 for b in data.get("builds", [])), "sdist members"),
        (data.get("reproducibility", {}).get("normalized_reproducible") is True, "normalized reproducibility"),
        (data.get("summary", {}).get("entrypoint_metadata_validated") is True, "entrypoint"),
        (data.get("summary", {}).get("required_runtime_files_present") is True, "runtime files"),
        (data.get("summary", {}).get("residual_paths_packaged") == 0, "no residual paths"),
        (data.get("summary", {}).get("absolute_worktree_path_leaks") == 0, "no absolute leaks"),
    ]
    for passed, label in checks:
        if not passed:
            failures.append(label)
    for rel in EXPECTED_PATHS:
        if not (REPO / rel).is_file():
            failures.append(f"missing {rel}")
    print(f"AP007E2_VALIDATION_FAILURES={len(failures)}")
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        return 1
    print("[GATE] AP-007E.2: SDIST E WHEEL CONSTRUÍDOS EM DUAS SANDBOXES, INVENTARIADOS E COM REPRODUTIBILIDADE NORMALIZADA VALIDADA; WORKTREE PRESERVADO, SEM INSTALAÇÃO NO AMBIENTE CANÔNICO, STAGING, COMMIT, TAG OU PUSH.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

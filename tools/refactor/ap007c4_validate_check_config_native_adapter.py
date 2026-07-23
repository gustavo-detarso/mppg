#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

EXPECTED_HEAD = (
    "8f30abdcb6bf811f869e09c1fb49ec2d15e0579b"
)
EXPECTED_PATHS = {
    "software/academic_pipeline_mppg/academic_pipeline/runtime.py",
    "software/academic_pipeline_mppg/academic_pipeline/doctor_runtime.py",
    "docs/refactor/academic-pipeline/AP-007/AP-007C2_DOCTOR_NATIVE_ADAPTER.md",
    "docs/refactor/academic-pipeline/AP-007/ap007c2_doctor_native_adapter.json",
    "software/academic_pipeline_mppg/tests/characterization/"
    "test_ap007c2_doctor_native_adapter_contract.py",
    "tools/refactor/ap007c2_validate_doctor_native_adapter.py",
    "docs/refactor/academic-pipeline/AP-007/AP-007C3_DOCTOR_PUBLIC_INTEGRATION.md",
    "docs/refactor/academic-pipeline/AP-007/ap007c3_doctor_public_integration.json",
    "software/academic_pipeline_mppg/tests/characterization/"
    "test_ap007c3_doctor_public_integration_contract.py",
    "tools/refactor/ap007c3_validate_doctor_public_integration.py",
    "software/academic_pipeline_mppg/academic_pipeline/check_config_runtime.py",
    "docs/refactor/academic-pipeline/AP-007/AP-007C4_CHECK_CONFIG_NATIVE_ADAPTER.md",
    "docs/refactor/academic-pipeline/AP-007/ap007c4_check_config_native_adapter.json",
    "software/academic_pipeline_mppg/tests/characterization/"
    "test_ap007c4_check_config_native_adapter_contract.py",
    "tools/refactor/ap007c4_validate_check_config_native_adapter.py",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status_paths(repo: Path) -> set[str]:
    raw = subprocess.check_output(
        [
            "git", "-C", str(repo), "status",
            "--porcelain=v1", "--untracked-files=all",
        ],
        text=True,
    )
    return {
        line[3:].strip().strip('"')
        for line in raw.splitlines()
        if line
    }


def validate(repo: Path) -> dict[str, object]:
    assert git(repo, "rev-parse", "HEAD") == EXPECTED_HEAD
    assert status_paths(repo) == EXPECTED_PATHS
    assert not git(repo, "diff", "--cached", "--name-only")

    payload = json.loads(
        (
            repo
            / "docs/refactor/academic-pipeline/AP-007/"
            "ap007c4_check_config_native_adapter.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["phase"] == "AP-007C.4"
    assert payload["status"] == (
        "check_config_adapter_materialized_not_integrated"
    )
    assert set(payload["candidate_paths"]) == EXPECTED_PATHS
    assert payload["semantic_exit_codes"] == [0, 2]
    assert payload["public_route"] == {
        "doctor": "native_doctor",
        "check_config": "legacy_fallback",
        "integration_phase": "AP-007C.5",
    }

    for relative, expected in payload[
        "artifact_sha256"
    ].items():
        assert sha(repo / relative) == expected

    software = repo / "software/academic_pipeline_mppg"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))

    from academic_pipeline import (
        check_config_runtime,
        runtime,
    )

    assert dataclasses.is_dataclass(
        check_config_runtime.CheckConfigRuntimeContext
    )
    assert {
        field.name
        for field in dataclasses.fields(
            check_config_runtime.CheckConfigRuntimeContext
        )
    } == {
        "load_config",
        "apply_cli_path_overrides",
        "output_paths",
        "research_output_paths",
        "external_search_enabled",
        "check_config",
        "print_check_config_report",
        "write_json",
    }
    assert runtime.select_runtime_route(
        ("--doctor",)
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(
        ("--check-config",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK

    return {
        "ok": True,
        "phase": "AP-007C.4",
        "path_count": 15,
        "public_integration": False,
        "semantic_exit_codes": [0, 2],
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    repo = (
        Path(args[0]).resolve()
        if args
        else Path.cwd().resolve()
    )
    print(json.dumps(
        validate(repo),
        ensure_ascii=False,
        sort_keys=True,
    ))
    print("AP-007C.4 check-config adapter contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

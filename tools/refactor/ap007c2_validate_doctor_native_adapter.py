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
    "software/academic_pipeline_mppg/"
    "academic_pipeline/doctor_runtime.py",
    "docs/refactor/academic-pipeline/AP-007/"
    "AP-007C2_DOCTOR_NATIVE_ADAPTER.md",
    "docs/refactor/academic-pipeline/AP-007/"
    "ap007c2_doctor_native_adapter.json",
    "software/academic_pipeline_mppg/tests/characterization/"
    "test_ap007c2_doctor_native_adapter_contract.py",
    "tools/refactor/"
    "ap007c2_validate_doctor_native_adapter.py",
}
PRESERVED = {
    "software/academic_pipeline_mppg/academic_pipeline/cli.py":
        "79b8b7f58397645b6378bbe29566180850da41a4bd5e1beabcdbcf498c196b19",
    "software/academic_pipeline_mppg/academic_pipeline/runtime.py":
        "2c83a7628160b6287e48c97e836f71f5d609cb2402ce1f86d6f5ee181ec6c4f2",
    "software/academic_pipeline_mppg/academic_pipeline/cli_parser.py":
        "f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8",
    "software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py":
        "9255c4b924fd61b7120b8c5e02684d338f6788de42ae7c352b049a488a308afe",
    "software/academic_pipeline_mppg/academic_pipeline/legacy.py":
        "f11ddffc30f60ac0c5e0856e8bf00ffaae866a8df806fd3c2b99f1afaa09e6b9",
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
    result = set()
    for line in raw.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result.add(path.strip().strip('"'))
    return result


def validate(repo: Path) -> dict[str, object]:
    assert git(repo, "rev-parse", "HEAD") == EXPECTED_HEAD
    assert status_paths(repo) == EXPECTED_PATHS
    assert not git(repo, "diff", "--cached", "--name-only")
    for relative, expected in PRESERVED.items():
        assert sha(repo / relative) == expected

    manifest = json.loads(
        (
            repo
            / "docs/refactor/academic-pipeline/AP-007/"
            "ap007c2_doctor_native_adapter.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["phase"] == "AP-007C.2"
    assert manifest["status"] == (
        "doctor_adapter_materialized_not_integrated"
    )
    assert set(manifest["candidate_paths"]) == EXPECTED_PATHS
    assert manifest["semantic_exit_codes"] == [0, 2]
    assert manifest["public_route"]["doctor"] == (
        "legacy_fallback"
    )
    for relative, expected in manifest[
        "artifact_sha256"
    ].items():
        assert sha(repo / relative) == expected

    software = repo / "software/academic_pipeline_mppg"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from academic_pipeline import doctor_runtime, runtime

    assert dataclasses.is_dataclass(
        doctor_runtime.DoctorRuntimeContext
    )
    assert {
        field.name
        for field in dataclasses.fields(
            doctor_runtime.DoctorRuntimeContext
        )
    } == {
        "load_config",
        "apply_cli_path_overrides",
        "output_paths",
        "research_output_paths",
        "external_search_enabled",
        "run_doctor",
        "print_doctor_report",
        "write_json",
    }
    assert runtime.select_runtime_route(
        ("--doctor",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK

    return {
        "ok": True,
        "phase": "AP-007C.2",
        "path_count": 5,
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
    print("AP-007C.2 doctor adapter contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

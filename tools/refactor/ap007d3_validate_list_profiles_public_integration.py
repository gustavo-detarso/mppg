#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def status_paths(repo: Path) -> set[str]:
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    )
    return {
        line[3:].strip().strip('"')
        for line in raw.splitlines()
        if line
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    repo = Path(args[0]).resolve() if args else Path.cwd().resolve()
    manifest = (
        repo
        / "docs/refactor/academic-pipeline/AP-007/"
        "ap007d3_list_profiles_public_integration.json"
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d3-list-profiles-public-integration/v1"
    assert data["status"] == "list_profiles_publicly_integrated"
    assert data["candidate_path_count"] == 18
    assert status_paths(repo) == set(data["candidate_paths"])
    assert not git(repo, "diff", "--cached", "--name-only")
    for relative, expected in data["artifact_sha256"].items():
        assert hashlib.sha256(
            (repo / relative).read_bytes()
        ).hexdigest() == expected

    software = repo / "software/academic_pipeline_mppg"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from academic_pipeline import runtime
    parser = runtime._build_parser()
    parsed = parser.parse_args(["--list-profiles"])
    assert parsed.list_profiles is True
    matches = [
        action
        for action in parser._actions
        if "--list-profiles" in action.option_strings
    ]
    assert len(matches) == 1
    assert runtime.select_runtime_route(
        ("--list-profiles",)
    ) is runtime.RuntimeRoute.NATIVE_LIST_PROFILES
    assert runtime.select_runtime_route(
        ("--help", "--list-profiles")
    ) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE
    assert runtime.select_runtime_route(
        ("--doctor", "--list-profiles")
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(
        ("--check-config", "--list-profiles")
    ) is runtime.RuntimeRoute.NATIVE_CHECK_CONFIG
    assert runtime.select_runtime_route(
        ("--tui", "--list-profiles")
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK
    print("[OK] AP-007D.3 validator")
    print("route=native_list_profiles")
    print("parser_option=--list-profiles")
    print("candidate_path_count=18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

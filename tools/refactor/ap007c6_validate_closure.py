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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def validate(repo: Path) -> dict[str, object]:
    manifest_path = (
        repo
        / "docs/refactor/academic-pipeline/AP-007/"
        "ap007c6_closure_manifest.json"
    )
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert payload["phase"] == "AP-007C.6"
    assert payload["status"] == (
        "ready_for_isolated_commit_decision"
    )
    assert payload["candidate_path_count"] == 23
    assert len(payload["candidate_paths"]) == 23
    assert len(set(payload["candidate_paths"])) == 23
    assert status_paths(repo) == set(
        payload["candidate_paths"]
    )
    assert not git(repo, "diff", "--cached", "--name-only")
    assert git(repo, "rev-parse", "HEAD") == (
        "8f30abdcb6bf811f869e09c1fb49ec2d15e0579b"
    )

    for relative, expected in payload[
        "artifact_sha256"
    ].items():
        assert sha(repo / relative) == expected

    software = repo / "software/academic_pipeline_mppg"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from academic_pipeline import runtime

    assert runtime.select_runtime_route(
        ("--doctor",)
    ) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(
        ("--check-config",)
    ) is runtime.RuntimeRoute.NATIVE_CHECK_CONFIG
    assert runtime.select_runtime_route(
        ("--tui",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK

    return {
        "ok": True,
        "phase": "AP-007C.6",
        "candidate_path_count": 23,
        "status": payload["status"],
        "commit_authorized": False,
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
    print("AP-007C.6 closure contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def status_paths(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line[3:] for line in result.stdout.splitlines() if len(line) >= 4}


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    manifest = repo / "docs/refactor/academic-pipeline/AP-007/ap007d6_closure_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d6-closure/v1"
    assert data["phase"] == "AP-007D.6"
    assert data["status"] == "ready_for_isolated_commit_decision"
    assert data["candidate_path_count"] == 48
    assert status_paths(repo) == set(data["candidate_paths"])
    assert data["runtime"]["sha256"] == "b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c"
    assert hashlib.sha256((repo / data["runtime"]["path"]).read_bytes()).hexdigest() == data["runtime"]["sha256"]
    for rel, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((repo / rel).read_bytes()).hexdigest() == expected
    assert data["commit_authorized"] is False
    assert data["tag_authorized"] is False
    assert data["push_authorized"] is False
    print("[OK] AP-007D.6 closure validator")
    print("status=ready_for_isolated_commit_decision")
    print("candidate_path_count=48")
    print("runtime_sha256=" + data["runtime"]["sha256"])
    print("native_public_command_count=" + str(len(data["native_public_commands"])))
    print("error_resolution_count=" + str(len(data["errors_and_resolutions"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

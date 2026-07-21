#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

SOURCE_HEAD = "aed79d72f6c26fabcdda00f25d058b32fdc3fd75"
SOURCE_TREE = "f4004337607e21e1ba89330928b4473ffd739dcd"
BRANCH = "ap-refactor/04-consumer-canonicalization"
UPSTREAM = "origin/ap-refactor/04-consumer-canonicalization"
OLD = "academic_pipeline_rc10_7_conformidade"
NEW = "academic_pipeline_mppg"
SETUP = Path("software/academic_pipeline_mppg/app_bundle/docs/SETUP_PIPENV.md")
CONTRACT = Path("docs/refactor/academic-pipeline/AP-006/ap006e3_consumer_stabilization.json")

def run(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()

def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )

def status_lines(repo: Path) -> list[str]:
    raw = run(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return sorted(line for line in raw.splitlines() if line)

def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return run_git(repo, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0

def introduction_commit(repo: Path, owned_paths: tuple[str, ...]) -> str:
    commits = set()
    for rel in owned_paths:
        assert run(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", rel) == rel
        commit = run(repo, "log", "--format=%H", "--diff-filter=A", "-1", "--", rel)
        assert commit, rel
        commits.add(commit)
    assert len(commits) == 1, commits
    commit = commits.pop()
    assert is_ancestor(repo, SOURCE_HEAD, commit)
    assert is_ancestor(repo, commit, "HEAD")
    return commit

def detect_mode(
    repo: Path,
    requested: str,
    owned_paths: tuple[str, ...],
) -> tuple[str, str | None]:
    observed = status_lines(repo)
    head = run(repo, "rev-parse", "HEAD")
    owned_untracked = {f"?? {path}" for path in owned_paths}

    if head == SOURCE_HEAD and owned_untracked.issubset(set(observed)):
        mode, commit = "precommit", None
    elif observed == [] and head != SOURCE_HEAD and is_ancestor(repo, SOURCE_HEAD, head):
        mode, commit = "postcommit", introduction_commit(repo, owned_paths)
    else:
        raise AssertionError({
            "head": head,
            "status": observed,
            "owned_paths": owned_paths,
        })

    if requested != "auto":
        assert mode == requested, (mode, requested)
    return mode, commit

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "precommit", "postcommit"), default="auto")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    data = json.loads((repo / CONTRACT).read_text())
    owned_paths = tuple(data["contract_owned_paths"])

    assert run(repo, "branch", "--show-current") == BRANCH
    assert run(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") == UPSTREAM
    assert run(repo, "diff", "--name-only") == ""
    assert run(repo, "diff", "--cached", "--name-only") == ""

    mode, publication_commit = detect_mode(repo, args.mode, owned_paths)
    ahead, behind = map(
        int,
        run(repo, "rev-list", "--left-right", "--count", "HEAD...@{upstream}").split(),
    )
    assert behind == 0
    if mode == "precommit":
        assert ahead == 0
        assert run(repo, "rev-parse", "HEAD") == SOURCE_HEAD
        assert run(repo, "rev-parse", "HEAD^{tree}") == SOURCE_TREE

    assert data["phase"] == "AP-006E.3"
    assert data["materialized_paths"] == []
    assert data["productive_code_change_count"] == 0
    assert data["setup_documentation_decision"] == (
        "preserve_ap006d3_nonoperational_compatibility_reference"
    )

    bridge = repo / "software/academic_pipeline_rc10_7_conformidade"
    canonical = repo / "software/academic_pipeline_mppg"
    assert bridge.is_symlink()
    assert os.readlink(bridge) == NEW
    assert bridge.resolve() == canonical.resolve()

    setup = (repo / SETUP).read_bytes()
    source_setup = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{SOURCE_HEAD}:{SETUP}"]
    )
    assert setup == source_setup
    text = setup.decode("utf-8")
    assert text.count(OLD) == 1
    assert text.count(NEW) == 1

    for item in data["historical_refactor_tools"]:
        rel = item["path"]
        current = (repo / rel).read_bytes()
        baseline = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{SOURCE_HEAD}:{rel}"]
        )
        assert current == baseline
        assert hashlib.sha256(current).hexdigest() == item["sha256"]

    for rel in data["generated_manual_review_paths"]:
        assert (repo / rel).read_bytes() == subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{SOURCE_HEAD}:{rel}"]
        )

    print(json.dumps({
        "phase": "AP-006E.3",
        "status": "ok",
        "mode": mode,
        "publication_commit": publication_commit,
        "ahead_of_upstream": ahead,
        "behind_upstream": behind,
        "materialized_path_count": 0,
        "historical_refactor_tool_count": len(data["historical_refactor_tools"]),
        "generated_manual_review_path_count": len(data["generated_manual_review_paths"]),
        "productive_code_change_count": 0,
        "setup_decision": data["setup_documentation_decision"],
        "bridge_decision": data["compatibility_bridge"]["decision"],
    }, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

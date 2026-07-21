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
CONTRACT = Path("docs/refactor/academic-pipeline/AP-006/ap006e5_closure.json")

EXPECTED_PATHS = (
    "docs/refactor/academic-pipeline/AP-006/AP-006E1_DISTRIBUTION_COMPATIBILITY_BASELINE.md",
    "docs/refactor/academic-pipeline/AP-006/AP-006E3_CONSUMER_STABILIZATION.md",
    "docs/refactor/academic-pipeline/AP-006/AP-006E5_CLOSURE_REPORT.md",
    "docs/refactor/academic-pipeline/AP-006/ap006e1_distribution_compatibility_baseline.json",
    "docs/refactor/academic-pipeline/AP-006/ap006e3_consumer_stabilization.json",
    "docs/refactor/academic-pipeline/AP-006/ap006e5_closure.json",
    "software/academic_pipeline_mppg/tests/characterization/test_ap006e1_distribution_compatibility_baseline.py",
    "software/academic_pipeline_mppg/tests/characterization/test_ap006e3_consumer_stabilization.py",
    "software/academic_pipeline_mppg/tests/characterization/test_ap006e5_closure_contract.py",
    "tools/refactor/ap006e1_validate_distribution_compatibility_baseline.py",
    "tools/refactor/ap006e3_validate_consumer_stabilization.py",
    "tools/refactor/ap006e5_validate_closure.py",
)

PREDECESSOR_HASHES = {
    "docs/refactor/academic-pipeline/AP-006/AP-006E1_DISTRIBUTION_COMPATIBILITY_BASELINE.md":
        "26c17256639c6c3d3034eee6c86410b10b25522baa6d12bb39be63a4c556d19e",
    "docs/refactor/academic-pipeline/AP-006/ap006e1_distribution_compatibility_baseline.json":
        "597266b971d148c9518c034881b3263a36b89ca45e8138155d162de318e256fa",
    "software/academic_pipeline_mppg/tests/characterization/test_ap006e1_distribution_compatibility_baseline.py":
        "be10cb9f5b119b54855466d3d47c1bcf71f41edef3a7dea18057a2de3f2db77b",
    "tools/refactor/ap006e1_validate_distribution_compatibility_baseline.py":
        "a8bac300c432362aaece805e95e5f1f98d1db42bfdb02550979040733235afab",
    "docs/refactor/academic-pipeline/AP-006/AP-006E3_CONSUMER_STABILIZATION.md":
        "26c81dad2bfd5e0bc71f1cb005143e27ae04fda00559fef8bae24ffd0a503db4",
    "docs/refactor/academic-pipeline/AP-006/ap006e3_consumer_stabilization.json":
        "dd159063696e8b619fd5a74bc4c9794c85469f42aee9e92856add703938ef080",
    "software/academic_pipeline_mppg/tests/characterization/test_ap006e3_consumer_stabilization.py":
        "942640741f37c447f94c4c60aad33d5a007f5493cfaf581ce9a46aef5ceb0ecf",
    "tools/refactor/ap006e3_validate_consumer_stabilization.py":
        "444b2d8fa24340dc61e60410f2cf66193e343e37cf79905f1398f53549da2f31",
}

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )

def status_lines(repo: Path) -> list[str]:
    raw = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return sorted(line for line in raw.splitlines() if line)

def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return run_git(repo, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0

def introduction_commit(repo: Path) -> str:
    commits = set()
    for rel in EXPECTED_PATHS:
        assert git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", rel) == rel
        commit = git(repo, "log", "--format=%H", "--diff-filter=A", "-1", "--", rel)
        assert commit, rel
        commits.add(commit)
    assert len(commits) == 1, commits
    commit = commits.pop()
    changed = sorted(
        line for line in git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "--root",
            commit,
        ).splitlines()
        if line
    )
    assert changed == sorted(EXPECTED_PATHS), (commit, changed)
    assert is_ancestor(repo, SOURCE_HEAD, commit)
    assert is_ancestor(repo, commit, "HEAD")
    return commit

def detect_mode(repo: Path, requested: str) -> tuple[str, str | None]:
    observed = status_lines(repo)
    precommit = sorted(f"?? {path}" for path in EXPECTED_PATHS)
    head = git(repo, "rev-parse", "HEAD")

    if observed == precommit and head == SOURCE_HEAD:
        mode, commit = "precommit", None
    elif observed == [] and head != SOURCE_HEAD and is_ancestor(repo, SOURCE_HEAD, head):
        mode, commit = "postcommit", introduction_commit(repo)
    else:
        raise AssertionError({
            "head": head,
            "status": observed,
            "expected_precommit": precommit,
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

    assert git(repo, "branch", "--show-current") == BRANCH
    assert git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") == UPSTREAM
    assert git(repo, "diff", "--name-only") == ""
    assert git(repo, "diff", "--cached", "--name-only") == ""

    mode, commit = detect_mode(repo, args.mode)
    divergence = git(repo, "rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    ahead, behind = map(int, divergence.split())
    assert behind == 0
    if mode == "precommit":
        assert ahead == 0
        assert git(repo, "rev-parse", "HEAD^{tree}") == SOURCE_TREE

    assert data["phase"] == "AP-006E.5"
    assert data["status"] == "closed"
    assert data["source_commit"] == SOURCE_HEAD
    assert data["source_tree"] == SOURCE_TREE
    assert data["candidate_path_count"] == 12
    assert tuple(data["candidate_paths"]) == EXPECTED_PATHS
    assert data["productive_code_change_count"] == 0
    assert data["validation"]["baseline_passed"] == 626
    assert data["validation"]["candidate_total_passed"] == 631
    assert data["validation"]["delta_passed"] == 5
    assert data["validation"]["regression_count"] == 0
    assert data["validation"]["wheel_member_count"] == 110
    assert data["validation"]["non_record_changed_member_count"] == 0
    assert data["validation"]["legacy_physical_path_count"] == 0
    assert data["validation"]["console_help_rc"] == 0
    assert data["validation"]["module_help_rc"] == 0
    assert data["compatibility_bridge"]["decision"] == "preserve_until_ap006f"
    assert data["fallback_decision"] == "preserve_until_ap006f"
    assert data["publication_policy"]["explicit_authorization_required"] is True
    assert data["publication_policy"]["validator_modes"] == ["auto", "precommit", "postcommit"]

    bridge = repo / "software/academic_pipeline_rc10_7_conformidade"
    canonical = repo / "software/academic_pipeline_mppg"
    assert bridge.is_symlink()
    assert os.readlink(bridge) == "academic_pipeline_mppg"
    assert bridge.resolve() == canonical.resolve()

    for rel, expected in PREDECESSOR_HASHES.items():
        actual = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
        assert actual == expected, (rel, actual, expected)

    print(json.dumps({
        "phase": data["phase"],
        "status": "ok",
        "mode": mode,
        "publication_commit": commit,
        "candidate_path_count": data["candidate_path_count"],
        "productive_code_change_count": data["productive_code_change_count"],
        "baseline_passed": data["validation"]["baseline_passed"],
        "candidate_total_passed": data["validation"]["candidate_total_passed"],
        "regression_count": data["validation"]["regression_count"],
        "ahead_of_upstream": ahead,
        "behind_upstream": behind,
        "bridge_decision": data["compatibility_bridge"]["decision"],
    }, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def run(repo: Path, args: list[str]) -> str:
    return subprocess.run(
        args,
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def commit_paths(repo: Path, commit: str) -> dict[str, str]:
    raw = run(
        repo,
        ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", commit],
    ).splitlines()
    result = {}
    for line in raw:
        status, path = line.split("\t", 1)
        result[path] = status
    return result


def validate(repo: Path, manifest_path: Path) -> dict[str, object]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["phase"] == "AP-006D.4"
    assert data["status"] == "closure_materialized"
    assert data["decision"] == "close_ap006d4_after_five_verified_waves"
    assert len(data["waves"]) == 5
    assert len(data["validators"]) == 4
    assert data["focused_contract_tests"] == {"failed": 0, "passed": 9}
    assert data["integrated_suite"] == {
        "errors": 0,
        "failed": 0,
        "passed": 624,
        "xfailed": 3,
        "xpassed": 0,
    }

    for wave in data["waves"]:
        commit = wave["commit"]
        run(repo, ["git", "cat-file", "-e", f"{commit}^{{commit}}"])
        assert run(repo, ["git", "rev-parse", f"{commit}^"]) == wave["parent"]
        assert run(repo, ["git", "show", "-s", "--format=%T", commit]) == wave["tree"]
        assert run(repo, ["git", "show", "-s", "--format=%s", commit]) == wave["subject"]
        assert commit_paths(repo, commit) == wave["paths"]

    for item in data["validators"]:
        path = repo / item["path"]
        assert path.is_file(), item["path"]
        assert item["status"] == "ok"

    head = run(repo, ["git", "rev-parse", "HEAD"])
    baseline_head = data["baseline"]["refactor_head"]
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline_head, head],
        cwd=repo,
    )
    assert ancestor_check.returncode == 0, (baseline_head, head)

    bridge = repo / "software/academic_pipeline_rc10_7_conformidade"
    assert bridge.is_symlink()
    assert os.readlink(bridge) == "academic_pipeline_mppg"

    constraints = data["constraints"]
    for key in (
        "closure_is_declarative_only",
        "compatibility_bridge_preserved",
        "five_waves_required",
        "all_wave_commits_published",
        "all_contract_validators_required",
        "integrated_suite_required",
    ):
        assert constraints[key] is True, key
    assert constraints["productive_code_modification_performed"] is False

    fingerprint_payload = {
        "decision": data["decision"],
        "baseline": data["baseline"],
        "audit_evidence": data["audit_evidence"],
        "waves": data["waves"],
        "validators": data["validators"],
        "focused_contract_tests": data["focused_contract_tests"],
        "integrated_suite": data["integrated_suite"],
        "constraints": data["constraints"],
        "next_gate": data["next_gate"],
    }
    observed_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert observed_fingerprint == data["fingerprint_sha256"]

    return {
        "status": "ok",
        "wave_count": len(data["waves"]),
        "validator_count": len(data["validators"]),
        "integrated_passed": data["integrated_suite"]["passed"],
        "integrated_xfailed": data["integrated_suite"]["xfailed"],
        "fingerprint_sha256": observed_fingerprint,
        "head": head,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.repo.resolve(), args.manifest.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

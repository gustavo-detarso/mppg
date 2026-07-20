#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


SEGMENT_PATTERN = re.compile(
    r"(^|[._-])"
    r"(backup|backups|bkp|bak|archive|archives|archived|attic|"
    r"historico|historica|historical|history|legacy|legado|legada|"
    r"obsolete|deprecated|old|previous|snapshot|snapshots|saved|copia|copy)"
    r"([._-]|$)",
    re.IGNORECASE,
)
BACKUP_SUFFIX_PATTERN = re.compile(
    r"(\.bak|\.backup|\.bkp|\.old|\.orig|\.save|\.saved|~)$",
    re.IGNORECASE,
)
LEGACY_BRIDGE = "software/academic_pipeline_rc10_7_conformidade"

CONTRACT_OWNED_PATHS = {
    "docs/refactor/academic-pipeline/AP-006/"
    "AP-006D4E_BACKUP_EVIDENCE_PRESERVATION_CONTRACT.md",
    "docs/refactor/academic-pipeline/AP-006/"
    "ap006d4e_backup_evidence_preservation_contract.json",
    "software/academic_pipeline_mppg/tests/characterization/"
    "test_ap006d4e_backup_evidence_preservation_contract.py",
    "tools/refactor/ap006d4e_validate_backup_evidence_preservation.py",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_tree(repo: Path) -> dict[str, dict[str, str]]:
    raw = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    result = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, encoded_path = item.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        path = encoded_path.decode("utf-8", errors="surrogateescape")
        result[path] = {
            "mode": mode,
            "kind": kind,
            "oid": oid,
        }
    return result


def is_candidate_path(path: str) -> bool:
    path_obj = Path(path)
    if any(SEGMENT_PATTERN.search(segment) for segment in path_obj.parts):
        return True
    if BACKUP_SUFFIX_PATTERN.search(path_obj.name):
        return True
    return path.startswith(LEGACY_BRIDGE + "/")


def blob(repo: Path, oid: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "-p", oid],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def validate(repo: Path, manifest_path: Path) -> dict[str, object]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4E"
    assert data["status"] == "backup_evidence_preservation_materialized"
    assert data["decision"] == (
        "preserve_all_backup_and_historical_evidence_in_place"
    )
    assert data["summary"]["candidate_count"] == 177
    assert data["summary"]["productive_reference_count"] == 0
    assert data["summary"]["preserve_in_place_count"] == 177

    constraints = data["constraints"]
    for key in (
        "preserve_all_candidates_in_place",
        "delete_forbidden",
        "move_forbidden",
        "rename_forbidden",
        "content_modification_forbidden",
        "filesystem_recursive_scan_forbidden",
        "git_object_validation_required",
        "compatibility_bridge_preserved",
    ):
        assert constraints[key] is True, key

    tree = parse_tree(repo)
    assert CONTRACT_OWNED_PATHS <= set(tree), (
        CONTRACT_OWNED_PATHS - set(tree)
    )
    observed_candidate_paths = {
        path
        for path, entry in tree.items()
        if (
            entry["kind"] == "blob"
            and is_candidate_path(path)
            and path not in CONTRACT_OWNED_PATHS
        )
    }
    recorded_paths = {item["path"] for item in data["candidates"]}
    assert CONTRACT_OWNED_PATHS.isdisjoint(recorded_paths)
    assert observed_candidate_paths == recorded_paths, (
        observed_candidate_paths ^ recorded_paths
    )

    classification_counts = Counter()
    for record in data["candidates"]:
        path = record["path"]
        entry = tree[path]
        assert entry["mode"] == record["mode"], path
        assert entry["oid"] == record["oid"], path
        payload = blob(repo, entry["oid"])
        assert len(payload) == record["size_bytes"], path
        assert sha256_bytes(payload) == record["sha256"], path
        assert record["productive_reference_count"] == 0, path
        assert record["disposition"] == (
            "preserve_in_place_as_historical_evidence"
        ), path
        classification_counts[record["classification"]] += 1

    assert dict(classification_counts) == data["summary"][
        "classification_counts"
    ]

    bridge = repo / data["bridge"]["path"]
    assert bridge.is_symlink()
    assert os.readlink(bridge) == data["bridge"]["expected_symlink_target"]

    fingerprint_payload = {
        "baseline_commit": data["baseline_commit"],
        "records": data["candidates"],
        "audit_evidence": data["audit_evidence"],
        "constraints": data["constraints"],
    }
    observed_fingerprint = sha256_bytes(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert observed_fingerprint == data["fingerprint_sha256"]

    return {
        "status": "ok",
        "candidate_count": len(recorded_paths),
        "productive_reference_count": 0,
        "classification_counts": dict(classification_counts),
        "fingerprint_sha256": observed_fingerprint,
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

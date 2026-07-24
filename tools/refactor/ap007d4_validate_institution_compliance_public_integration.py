#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    manifest = repo / "docs/refactor/academic-pipeline/AP-007/ap007d4_institution_compliance_public_integration.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d4-institution-compliance-public-integration/v1"
    assert data["status"] == "institution_compliance_publicly_integrated"
    assert data["candidate_path_count"] == 31
    assert data["integration_origin"] in {"fresh_transactional_patch", "reconciled_preexisting_exact_write"}
    raw = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"], text=True)
    observed = {line[3:].strip().strip('"') for line in raw.splitlines() if line}
    assert observed == set(data["candidate_paths"])
    assert not subprocess.check_output(["git", "-C", str(repo), "diff", "--cached", "--name-only"], text=True).strip()
    for relative, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected
    software = repo / "software/academic_pipeline_mppg"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from academic_pipeline import runtime
    assert runtime.select_runtime_route(("--config", "x.toml", "--check-institution-compliance")) is runtime.RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE
    assert runtime.select_runtime_route(("--doctor", "--check-institution-compliance")) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(("--list-profiles",)) is runtime.RuntimeRoute.NATIVE_LIST_PROFILES
    print("[OK] AP-007D.4 institution compliance public integration validator")
    print("route=native_institution_compliance")
    print("candidate_path_count=31")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    manifest = repo / "docs/refactor/academic-pipeline/AP-007/ap007d5_doi_manifest_public_integration.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d5-doi-manifest-public-integration/v1"
    assert data["status"] == "doi_manifest_publicly_integrated"
    assert data["public_route"] == "native_doi_manifest"
    assert data["candidate_path_count"] == 44
    assert data["adapter"]["sha256"] == "583f6dfa1f8a4fff84408f6f96b7f53fbd382b6072932849c69adda753c98aef"
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
    assert runtime.select_runtime_route(("--make-doi-manifest", "--input-dir", "in", "--output", "out.csv")) is runtime.RuntimeRoute.NATIVE_DOI_MANIFEST
    assert runtime.select_runtime_route(("--doctor", "--make-doi-manifest")) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(("--config", "x.toml", "--check-institution-compliance")) is runtime.RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE
    print("[OK] AP-007D.5 DOI manifest public integration validator")
    print("route=native_doi_manifest")
    print("candidate_path_count=44")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

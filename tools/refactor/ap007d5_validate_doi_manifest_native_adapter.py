#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    manifest_path = repo / "docs/refactor/academic-pipeline/AP-007/ap007d5_doi_manifest_native_adapter.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d5-doi-manifest-native-adapter/v1"
    assert data["status"] == "materialized_route_still_legacy"
    assert data["command"] == "--make-doi-manifest"
    assert data["public_route_changed"] is False
    assert data["legacy_defect"] == "missing_legacy_bootstrap_dependency:dotenv"
    assert data["characterization_payload_sha256"] == "bb7cf63340657a4bf0c0f0d25b4bf9c239780ebdd107029069b5bd95fece48e4"
    adapter = repo / data["adapter"]["path"]
    source = adapter.read_text(encoding="utf-8")
    ast.parse(source)
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == data["adapter"]["sha256"]
    assert data["adapter"]["closure_strategy"] == "minimal_same_module_ast_closure"
    assert data["adapter"]["entrypoint"] == "run_make_doi_manifest_command"
    for forbidden in (
        "project_tools", "bibliography_manager", "academic_pipeline_rc10",
        "dotenv", "pydantic", "run_legacy", "importlib", "globals(", "locals(",
    ):
        assert forbidden not in source
    software = repo / "software/academic_pipeline_mppg"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from academic_pipeline import doi_manifest_runtime, runtime
    assert doi_manifest_runtime.DOI_MANIFEST_OPTION == "--make-doi-manifest"
    route = runtime.select_runtime_route((
        "--make-doi-manifest", "--input-dir", "/tmp/input", "--output", "/tmp/output.csv",
    ))
    assert route is runtime.RuntimeRoute.LEGACY_FALLBACK
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    observed = {line[3:].strip().strip('"') for line in raw.splitlines() if line}
    assert observed == set(data["candidate_paths"])
    assert data["candidate_path_count"] == 40
    for relative, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--cached", "--name-only"],
        text=True,
    ).strip()
    print("[OK] AP-007D.5 DOI manifest adapter validator")
    print("status=materialized_route_still_legacy")
    print("legacy_defect=missing_legacy_bootstrap_dependency:dotenv")
    print(f"candidate_path_count={data['candidate_path_count']}")
    print(f"adapter_sha256={data['adapter']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

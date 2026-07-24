#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    manifest = repo / "docs/refactor/academic-pipeline/AP-007/ap007d5_higher_risk_characterization.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d5-higher-risk-characterization/v3"
    assert data["status"] == "selected_for_isolated_dependency_decoupling_adapter"
    assert data["command"] == "--make-doi-manifest"
    resolution = data["canonical_surface"]["canonical_resolution"]
    assert resolution["strategy"] == "minimal_same_module_ast_closure"
    assert resolution["relative_imports"] == []
    assert resolution["unresolved_names"] == []
    assert data["sandbox_contract"]["network_attempts"] == 0
    assert data["sandbox_contract"]["unexpected_generated_files"] == []
    assert data["public_state"]["route"] == "legacy_fallback"
    assert data["public_state"]["executable_under_canonical_python"] is False
    defect = data["public_state"]["baseline_defect"]
    assert defect["class"] == "unrelated_legacy_bootstrap_dependency_coupling"
    missing = defect["first_missing_module"]
    assert isinstance(missing, str) and missing
    assert missing.split(".", 1)[0] not in defect["closure_import_roots"]
    assert {item["missing_module"] for item in defect["scenarios"].values()} == {missing}
    assert data["decision"]["adapter_may_be_materialized"] is True
    assert data["candidate_path_count"] == 35
    copy = json.loads(json.dumps(data))
    observed_payload = copy["integrity"]["payload_sha256"]
    copy["integrity"]["payload_sha256"] = ""
    canonical = json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == observed_payload
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    observed_paths = {line[3:].strip().strip('"') for line in raw.splitlines() if line}
    assert observed_paths == set(data["candidate_paths"])
    for relative, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected
    print("[OK] AP-007D.5 higher-risk characterization validator v3")
    print("command=--make-doi-manifest")
    print("status=selected_for_isolated_dependency_decoupling_adapter")
    print("legacy_defect=missing_legacy_bootstrap_dependency:" + missing)
    print("candidate_path_count=35")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

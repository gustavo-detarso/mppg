#!/usr/bin/env python3
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    manifest_path = repo / "docs/refactor/academic-pipeline/AP-007/ap007d4_institution_compliance_native_adapter.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d4-institution-compliance-native-adapter/v1"
    assert data["status"] == "materialized_route_still_legacy"
    assert data["command"] == "--check-institution-compliance"
    assert data["public_route_changed"] is False
    assert data["characterization_payload_sha256"] == "a077f9a82c3200562138db0596118a2b8ac20bcf4b04eebc6dd7d4ffc2ef62b4"
    adapter = repo / data["adapter"]["path"]
    source = adapter.read_text(encoding="utf-8")
    ast.parse(source)
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == data["adapter"]["sha256"]
    for forbidden in (
        "globals(", "locals(", "sys.path", "importlib",
        "academic_pipeline_rc10", "run_legacy", "LEGACY_MODULE_NAME",
    ):
        assert forbidden not in source

    software = repo / "software/academic_pipeline_mppg"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from academic_pipeline import institution_compliance_runtime, runtime
    cls = institution_compliance_runtime.InstitutionComplianceRuntimeContext
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen
    assert hasattr(cls, "__slots__")
    assert runtime.select_runtime_route(
        ("--check-institution-compliance",)
    ) is runtime.RuntimeRoute.LEGACY_FALLBACK

    raw = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    observed = {line[3:].strip().strip('"') for line in raw.splitlines() if line}
    assert observed == set(data["candidate_paths"])
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--cached", "--name-only"],
        text=True,
    ).strip()
    print("[OK] AP-007D.4 institution compliance adapter validator")
    print("status=materialized_route_still_legacy")
    print(f"candidate_path_count={len(data['candidate_paths'])}")
    print(f"adapter_sha256={data['adapter']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

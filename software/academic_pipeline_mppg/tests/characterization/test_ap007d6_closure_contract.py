from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from academic_pipeline import runtime

TEST_FILE = Path(__file__).resolve()
REPO = TEST_FILE.parents[4]
MANIFEST = REPO / "docs/refactor/academic-pipeline/AP-007/ap007d6_closure_manifest.json"


def payload() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_identity_and_scope() -> None:
    data = payload()
    assert data["schema"] == "ap007d6-closure/v1"
    assert data["phase"] == "AP-007D.6"
    assert data["status"] == "ready_for_isolated_commit_decision"
    assert data["candidate_path_count"] == 48
    assert len(data["candidate_paths"]) == 48
    assert len(set(data["candidate_paths"])) == 48


def test_runtime_and_artifact_hashes_match() -> None:
    data = payload()
    runtime_path = REPO / data["runtime"]["path"]
    assert hashlib.sha256(runtime_path.read_bytes()).hexdigest() == data["runtime"]["sha256"]
    for rel, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == expected


def test_list_profiles_route_is_native() -> None:
    assert runtime.select_runtime_route(("--list-profiles",)) is runtime.RuntimeRoute.NATIVE_LIST_PROFILES


def test_institution_compliance_route_is_native() -> None:
    argv = ("--config", "x.toml", "--check-institution-compliance")
    assert runtime.select_runtime_route(argv) is runtime.RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE


def test_doi_manifest_routes_are_native() -> None:
    by_dir = ("--make-doi-manifest", "--input-dir", "in", "--output", "out.csv")
    by_zip = ("--make-doi-manifest", "--input-zip", "in.zip", "--output", "out.csv")
    assert runtime.select_runtime_route(by_dir) is runtime.RuntimeRoute.NATIVE_DOI_MANIFEST
    assert runtime.select_runtime_route(by_zip) is runtime.RuntimeRoute.NATIVE_DOI_MANIFEST


def test_help_keeps_precedence() -> None:
    assert runtime.select_runtime_route(("--help", "--make-doi-manifest")) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE


def test_competing_operational_commands_remain_legacy() -> None:
    assert runtime.select_runtime_route(("--doctor", "--make-doi-manifest")) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(("--list-profiles", "--make-doi-manifest")) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(("--check-institution-compliance", "--make-doi-manifest")) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_exit_contracts_are_documented() -> None:
    data = payload()
    assert data["exit_contracts"]["check_institution_compliance"] == {"ok": 0, "usage_error": 1, "not_ok": 2}
    assert data["exit_contracts"]["make_doi_manifest"] == {"success": 0, "usage_error": 1}


def test_adapters_do_not_import_legacy_bootstrap() -> None:
    data = payload()
    forbidden = {"academic_pipeline_rc10", "dotenv", "pydantic", "run_legacy"}
    for rel in data["adapter_paths"]:
        source = (REPO / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any(any(token in name for token in forbidden) for name in imported)


def test_no_globals_or_locals_bridge_in_runtime() -> None:
    data = payload()
    source = (REPO / data["runtime"]["path"]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "globals" not in calls
    assert "locals" not in calls


def test_errors_and_resolutions_are_recorded() -> None:
    data = payload()
    assert len(data["errors_and_resolutions"]) >= 20
    assert all(item["error"] and item["resolution"] for item in data["errors_and_resolutions"])


def test_commit_is_not_authorized() -> None:
    data = payload()
    assert data["commit_authorized"] is False
    assert data["tag_authorized"] is False
    assert data["push_authorized"] is False

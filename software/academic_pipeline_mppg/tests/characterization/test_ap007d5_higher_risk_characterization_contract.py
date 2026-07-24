from __future__ import annotations
import hashlib
import json
from pathlib import Path

TEST_FILE = Path(__file__).resolve()
REPO = TEST_FILE.parents[4]
MANIFEST = REPO / "docs/refactor/academic-pipeline/AP-007/ap007d5_higher_risk_characterization.json"

def payload() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))

def test_ap007d5_characterization_contract() -> None:
    data = payload()
    assert data["schema"] == "ap007d5-higher-risk-characterization/v3"
    assert data["status"] == "selected_for_isolated_dependency_decoupling_adapter"
    assert data["command"] == "--make-doi-manifest"
    assert data["historical_characterization"]["score"] == 26
    assert data["historical_characterization"]["risk"] == "moderate"
    resolution = data["canonical_surface"]["canonical_resolution"]
    assert resolution["strategy"] == "minimal_same_module_ast_closure"
    assert resolution["file"].endswith("/project_tools.py")
    assert resolution["function"] == "make_doi_manifest"
    assert resolution["tracked"] is True
    assert resolution["ast_confirmed"] is True
    assert resolution["relative_imports"] == []
    assert resolution["unresolved_names"] == []
    assert "make_doi_manifest" in resolution["closure_names"]
    assert data["sandbox_contract"]["timeout_seconds"] == 30
    assert data["sandbox_contract"]["network_attempts"] == 0
    assert data["sandbox_contract"]["unexpected_generated_files"] == []
    assert data["public_state"]["route"] == "legacy_fallback"
    assert data["public_state"]["public_route_changed"] is False
    assert data["public_state"]["executable_under_canonical_python"] is False
    assert data["public_state"]["direct_and_public_outputs_equivalent"] is False
    defect = data["public_state"]["baseline_defect"]
    assert defect["class"] == "unrelated_legacy_bootstrap_dependency_coupling"
    missing = defect["first_missing_module"]
    assert isinstance(missing, str) and missing
    assert missing.split(".", 1)[0] not in defect["closure_import_roots"]
    assert defect["consistent_across_scenarios"] is True
    assert defect["legacy_loader_observed"] is True
    assert defect["legacy_bootstrap_observed"] is True
    scenario_modules = {
        item["missing_module"]
        for item in defect["scenarios"].values()
    }
    assert scenario_modules == {missing}
    assert data["decision"]["adapter_may_be_materialized"] is True
    assert data["decision"]["public_route_may_change"] is False
    assert data["candidate_path_count"] == 35
    copy = json.loads(json.dumps(data))
    observed = copy["integrity"]["payload_sha256"]
    copy["integrity"]["payload_sha256"] = ""
    canonical = json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == observed

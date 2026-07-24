from __future__ import annotations
import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[4] / "docs/refactor/academic-pipeline/AP-007/ap007d4_second_wave_characterization.json"


def test_ap007d4_second_wave_characterization_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d4-second-wave-characterization/v3"
    assert data["command"] == "--check-institution-compliance"
    assert data["status"] == "selected_for_isolated_adapter"
    assert data["selected"] is True
    assert data["decision"]["adapter_may_be_materialized"] is True
    assert data["decision"]["public_route_may_change"] is False
    assert data["blocking_effects"] == {}
    assert data["unresolved_project_calls_raw"] == [
        "re.sub",
        "resolve",
        "unicodedata.combining",
    ]
    assert data["unresolved_project_calls"] == []
    assert {item["classification"] for item in data["non_project_call_evidence"]} == {
        "stdlib_module_attribute",
        "non_mutating_path_resolution_method",
    }
    assert {item["call"] for item in data["non_project_call_evidence"]} == set(
        data["unresolved_project_calls_raw"]
    )
    assert data["call_graph_truncated"] is False
    assert data["score"]["risk"] == "low"
    assert data["score"]["value"] >= 27
    assert data["handlers"]
    assert data["resolved_seed_handlers"]
    assert data["transitive_functions"]
    assert data["tests"]
    clone = json.loads(json.dumps(data))
    observed = clone["integrity"]["payload_sha256"]
    clone["integrity"]["payload_sha256"] = ""
    canonical = json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == observed

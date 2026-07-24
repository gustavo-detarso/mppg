#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    path = repo / "docs/refactor/academic-pipeline/AP-007/ap007d4_second_wave_characterization.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d4-second-wave-characterization/v3"
    assert data["command"] == "--check-institution-compliance"
    assert data["status"] == "selected_for_isolated_adapter"
    assert data["selected"] is True
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
    assert data["handlers"] and data["transitive_functions"] and data["tests"]
    clone = json.loads(json.dumps(data))
    observed = clone["integrity"]["payload_sha256"]
    clone["integrity"]["payload_sha256"] = ""
    canonical = json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == observed
    print("[OK] AP-007D.4 validator v3")
    print("command=--check-institution-compliance")
    print("status=selected_for_isolated_adapter")
    print(f"score={data['score']['value']}/{data['score']['maximum']}")
    print(f"handlers={len(data['handlers'])}")
    print(f"transitive_functions={len(data['transitive_functions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

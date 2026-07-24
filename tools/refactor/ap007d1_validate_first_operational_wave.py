from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED_HEAD = 'ab066e68947ac5f33f1c12c9a7db5086d0f93790'
EXPECTED_TREE = 'e8bebd0b88a719a765ba6cc78150b01e024de9dd'
EXPECTED_RUNTIME_SHA256 = '7a2fd63ae060c74fe3f06e2eaf7457f176dc5059c0ac7aa95fc23805e810b1e6'
EXPECTED_CANDIDATES = ['--check-institution-compliance', '--list-profiles', '--make-doi-manifest']
DEFAULT_REL = 'docs/refactor/academic-pipeline/AP-007/ap007d1_first_operational_wave_decision.json'
RUNTIME_REL = 'software/academic_pipeline_mppg/academic_pipeline/runtime.py'


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--decision")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    decision = Path(args.decision).resolve() if args.decision else repo / DEFAULT_REL
    payload = json.loads(decision.read_text(encoding="utf-8"))
    assert git(repo, "rev-parse", "HEAD") == EXPECTED_HEAD
    assert git(repo, "show", "-s", "--format=%T", "HEAD") == EXPECTED_TREE
    assert sha256_file(repo / RUNTIME_REL) == EXPECTED_RUNTIME_SHA256
    assert payload["schema"] == 'ap007d1-first-operational-wave-decision/v2'
    assert [item["flag"] for item in payload["candidates_by_input_order"]] == EXPECTED_CANDIDATES
    selected = payload["decision"]["selected_flags"]
    assert len(selected) <= 1
    by_flag = {item["flag"]: item for item in payload["candidates"]}
    for item in by_flag.values():
        assert item["resolved_seed_handlers"]
        assert "handler_not_resolved" not in item["hard_exclusions"]
        assert all(record["method"] != "unresolved" for record in item["seed_resolution_evidence"])
    for flag in selected:
        assert by_flag[flag]["first_wave_eligible"] is True
        assert by_flag[flag]["hard_exclusions"] == []
    assert "--make-doi-manifest" not in selected
    clone = dict(payload)
    clone["integrity"] = {"decision_payload_sha256": ""}
    observed = hashlib.sha256(json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert payload["integrity"]["decision_payload_sha256"] == observed
    print("[OK] AP-007D.1 validator")
    print(f"status={payload['decision']['status']}")
    print(f"selected={','.join(selected) or '<nenhum>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED_HEAD = 'ab066e68947ac5f33f1c12c9a7db5086d0f93790'
EXPECTED_TREE = 'e8bebd0b88a719a765ba6cc78150b01e024de9dd'
EXPECTED_RUNTIME_SHA256 = '7a2fd63ae060c74fe3f06e2eaf7457f176dc5059c0ac7aa95fc23805e810b1e6'
EXPECTED_NATIVE = {'--help': 'native_first_wave', '--list-toml-profiles': 'native_first_wave', '--list-institutions': 'native_first_wave', '--list-layouts': 'native_first_wave', '--explain-profile': 'native_first_wave', '--doctor': 'native_doctor', '--check-config': 'native_check_config'}
RUNTIME_REL = 'software/academic_pipeline_mppg/academic_pipeline/runtime.py'
DEFAULT_INVENTORY_REL = "docs/refactor/academic-pipeline/AP-007/ap007d0_operational_command_inventory.json"


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
    parser.add_argument("--inventory")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    inventory = Path(args.inventory).resolve() if args.inventory else repo / DEFAULT_INVENTORY_REL
    payload = json.loads(inventory.read_text(encoding="utf-8"))

    assert git(repo, "rev-parse", "HEAD") == EXPECTED_HEAD
    assert git(repo, "show", "-s", "--format=%T", "HEAD") == EXPECTED_TREE
    assert sha256_file(repo / RUNTIME_REL) == EXPECTED_RUNTIME_SHA256
    assert payload["schema"] == "ap007d0-operational-command-inventory/v2"
    assert payload["runtime_dispatch"]["expected_native_routes"] == EXPECTED_NATIVE
    assert payload["runtime_dispatch"]["observed_native_routes"] == EXPECTED_NATIVE
    legacy = {item["flag"] for item in payload["legacy_commands"]}
    recommended = set(payload["first_wave_recommendation"]["flags"])
    assert legacy.isdisjoint(EXPECTED_NATIVE)
    assert recommended.isdisjoint(EXPECTED_NATIVE)

    by_flag = {item["flag"]: item for item in payload["legacy_commands"]}
    for flag in recommended:
        command = by_flag[flag]
        assert command["first_wave_eligible"] is True
        assert command["hard_exclusions"] == []
        assert command["score_total"] >= 24

    clone = dict(payload)
    clone["integrity"] = {"inventory_payload_sha256": ""}
    observed = hashlib.sha256(json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert payload["integrity"]["inventory_payload_sha256"] == observed
    print("[OK] AP-007D.0 validator v2")
    print(f"legacy_commands={len(payload['legacy_commands'])}")
    print(f"recommended={','.join(payload['first_wave_recommendation']['flags']) or '<nenhuma>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

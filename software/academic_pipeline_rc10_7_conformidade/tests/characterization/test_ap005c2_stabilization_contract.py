from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[4]

MANIFEST = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005c2_stabilization_manifest.json"
)

VALIDATOR = (
    ROOT
    / "tools/refactor/"
    "ap005c2_validate_stabilization.py"
)

EXPECTED_HASHES = {
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "app_bundle/scripts/pipeline/"
        "academic_pipeline_toml_generator_interativo.py"
    ): (
        "9d627348fcdc3b9ec727abb3c2862eb26"
        "b11bbd1d1bc744958d892f9f4afa7f9"
    ),
    (
        "tools/refactor/"
        "ap005c_inventory_toml_capture_aliases.py"
    ): (
        'aed2b3859c124052b0ffa2d0b6a309f6485af3af18added6504c1d65c7fb8137'
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "ap005c_toml_capture_alias_inventory.json"
    ): (
        "f97714602a8c0d076d54819ec429ad8c"
        "492768e1754ea2a326e4e3f71dfc5f63"
    ),
    (
        "docs/refactor/academic-pipeline/AP-005/"
        "AP-005C_TOML_CAPTURE_ALIAS_STRATEGY.md"
    ): (
        "4e01b0596b7f55033074f775ababaff48"
        "f84fbef46312d490c4a05e5c7792e6c"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c_toml_capture_alias_inventory_contract.py"
    ): (
        "afbd6003479a49b9633f54f41032ddf2"
        "906a82147073728b675642e7c695c170"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c_toml_capture_alias_"
        "semantics_characterization.py"
    ): (
        "031a7f56feab2fca7d6729bb5ed117f9"
        "2abe26a05f80866ca9218d4b539f4795"
    ),
    (
        "software/academic_pipeline_rc10_7_conformidade/"
        "tests/characterization/"
        "test_ap005c1_toml_capture_alias_application_contract.py"
    ): (
        "a17498273c482fa6e5855eb78cce2ed1"
        "adc595f7718299e6d2cb3419fca2c7e3"
    ),
    (
        "tools/refactor/"
        "ap005c1_apply_toml_capture_aliases.py"
    ): (
        "4be0bc2bc9de73513f7743be8489a750"
        "006dc58c835c21e74cf0f231f07f4a68"
    ),
}


def load_manifest() -> dict[str, Any]:
    return json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_stabilization_schema_and_baseline() -> None:
    payload = load_manifest()

    assert payload["schema_version"] == (
        "ap005c2.stabilization-manifest.v1"
    )

    assert payload["baseline_commit"] == (
        "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
    )

    assert payload["branch"] == (
        "ap-refactor/04-consumer-canonicalization"
    )


def test_stabilization_fingerprint_is_reproducible() -> None:
    payload = load_manifest()
    expected = payload.pop(
        "contract_fingerprint"
    )

    actual = hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()

    assert actual == expected


def test_stabilization_core_hashes_match() -> None:
    payload = load_manifest()

    assert payload["core_file_hashes"] == (
        EXPECTED_HASHES
    )

    for relative, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()

        assert actual == expected


def test_stabilization_symbol_contract_is_exact() -> None:
    payload = load_manifest()

    assert payload["symbol_contract"] == {
        "canonical_captures": 4,
        "legacy_aliases_preserved": 4,
        "canonical_consumers": 6,
        "legacy_consumers_remaining": 0,
        "new_public_exports": 0,
    }

    assert len(payload["entries"]) == 4

    assert sum(
        entry["productive_consumer_count"]
        for entry in payload["entries"]
    ) == 6

    assert all(
        not entry["legacy_load_lines"]
        for entry in payload["entries"]
    )


def test_stabilization_candidate_manifest_and_diff() -> None:
    payload = load_manifest()

    assert payload["candidate_file_count"] == 12

    assert len(payload["candidate_files"]) == 12

    assert len(
        set(payload["candidate_files"])
    ) == 12

    assert payload["productive_diff"] == {
        "files": 1,
        "insertions": 14,
        "deletions": 10,
        "path": (
            "software/academic_pipeline_rc10_7_conformidade/"
            "app_bundle/scripts/pipeline/"
            "academic_pipeline_toml_generator_interativo.py"
        ),
    }

    assert payload["test_gates"] == {
        "legacy_related_tests": 106,
        "ap005c_tests": 24,
        "focused_regression": 53,
        "canonical_suite_passed": 532,
        "canonical_suite_xfailed": 3,
    }


def test_stabilization_validator_check_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--root",
            str(ROOT),
            "--check",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, (
        result.stdout,
        result.stderr,
    )

    assert "capturas canônicas=4" in result.stdout
    assert "aliases preservados=4" in result.stdout
    assert "consumidores canônicos=6" in result.stdout
    assert "consumidores legados=0" in result.stdout
    assert "arquivos candidatos=12" in result.stdout

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[4]

INVENTORY = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005c_toml_capture_alias_inventory.json"
)

TOOL = (
    ROOT
    / "tools/refactor/"
    "ap005c_inventory_toml_capture_aliases.py"
)


def load_inventory() -> dict[str, Any]:
    return json.loads(
        INVENTORY.read_text(encoding="utf-8")
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_inventory_schema_and_baseline() -> None:
    payload = load_inventory()

    assert payload["schema_version"] == (
        "ap005c.toml-capture-alias-inventory.v1"
    )

    assert payload["baseline_commit"] == (
        "9372de8f621c9012a28d4c4a9a64e252a398bdf3"
    )

    assert payload["productive_module_sha256"] == (
        "7b3ff44794275df2a3470796e78a25c3c"
        "87ca2c44f93fac6ec18eee397c89beb"
    )


def test_inventory_fingerprint_is_reproducible() -> None:
    payload = load_inventory()
    expected = payload.pop(
        "contract_fingerprint"
    )

    actual = hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()

    assert actual == expected


def test_inventory_contains_exact_four_aliases() -> None:
    payload = load_inventory()

    mapping = {
        entry["legacy_alias"]: (
            entry["canonical_capture_name"]
        )
        for entry in payload["entries"]
    }

    assert mapping == {
        "_original_ensure_reference_policy": (
            "_captured_wiz_input_"
            "ensure_reference_policy"
        ),
        "_wiz_disable_references_original": (
            "_captured_wiz_disable_references"
        ),
        "_render_toml_original": (
            "_captured_render_toml"
        ),
        "_collect_outputs_and_options_original": (
            "_captured_collect_outputs_and_options"
        ),
    }

    assert payload["summary"] == {
        "aliases": 4,
        "previous_bindings_confirmed": 4,
        "later_redefinitions_confirmed": 4,
        "productive_references": 6,
        "removal_allowed": 0,
        "direct_substitution_allowed": 0,
    }


def test_inventory_generator_check_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
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

    assert "aliases=4" in result.stdout
    assert "referências produtivas=6" in result.stdout

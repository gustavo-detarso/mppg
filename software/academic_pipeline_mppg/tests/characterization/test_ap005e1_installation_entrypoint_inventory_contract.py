from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
INVENTORY = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005e1_installation_entrypoint_inventory.json"
)
INVENTORY_MD = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "AP-005E1_INSTALLATION_ENTRYPOINT_INVENTORY.md"
)
STRATEGY_MD = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "AP-005E1_INSTALLATION_ENTRYPOINT_STRATEGY.md"
)
TOOL = ROOT / "tools/refactor/ap005e1_inventory_installation_entrypoints.py"

EXPECTED_COMMIT = "ba28822c826c37022581bf88c6a1b488e2c618de"
EXPECTED_PACKAGES = [
    "academic_pipeline",
    "app_bundle",
    "app_bundle.scripts",
    "app_bundle.scripts.pipeline",
    "app_bundle.scripts.pipeline.article_workflow",
]


def _payload() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_ap005e1_schema_baseline_and_fingerprint() -> None:
    payload = _payload()
    assert payload["schema_version"] == (
        "ap005e1.installation-entrypoint-inventory.v1"
    )
    assert payload["phase"] == "AP-005E.1"
    assert payload["baseline"] == {
        "branch": "ap-refactor/04-consumer-canonicalization",
        "commit": EXPECTED_COMMIT,
        "upstream": "origin/ap-refactor/04-consumer-canonicalization",
    }

    fingerprint = payload.pop("fingerprint")
    assert fingerprint == hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def test_ap005e1_public_metadata_is_exact() -> None:
    metadata = _payload()["metadata"]
    assert metadata["build_system"] == {
        "build-backend": "setuptools.build_meta",
        "requires": ["setuptools>=68", "wheel"],
    }
    assert metadata["project"] == {
        "dependencies": [],
        "description": (
            "Pipeline acadêmica modular para geração, validação "
            "e renderização de documentos."
        ),
        "name": "academic-pipeline-mppg",
        "requires_python": ">=3.11",
        "scripts": {
            "academic-pipeline": "academic_pipeline.cli:main"
        },
        "version": "0.1.0",
    }


def test_ap005e1_package_discovery_and_census_are_exact() -> None:
    payload = _payload()
    assert payload["package_discovery"]["selected_packages"] == EXPECTED_PACKAGES
    assert payload["source_root_census"] == {
        "excluded_test_python_files": 23,
        "init_files": 5,
        "non_python_files": 184,
        "other_python_files": 2,
        "python_files": 90,
        "selected_package_non_python_files": 3,
        "selected_package_python_files": 65,
        "tracked_total": 274,
    }


def test_ap005e1_entrypoint_chain_is_preserved() -> None:
    entrypoints = _payload()["entrypoints"]
    assert entrypoints["console_script"] == {
        "name": "academic-pipeline",
        "target": "academic_pipeline.cli:main",
    }
    assert entrypoints["module_entrypoint"]["target"] == (
        "academic_pipeline.cli:main"
    )
    assert entrypoints["public_package_function"] == {
        "all": ["main"],
        "symbol": "academic_pipeline.main",
        "target": "academic_pipeline.cli:main",
    }
    assert entrypoints["compatibility_chain"][-1].endswith(
        "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    )


def test_ap005e1_distinguishes_evidence_from_gates() -> None:
    observations = _payload()["audit_observations"]
    assert observations["current_pipenv_distribution"][
        "classification"
    ] == "environment_observation_not_gate"
    assert observations["raw_layout_reference_counts"][
        "classification"
    ] == "algorithmic_evidence_not_gate"
    assert observations["historical_python_universe"] == {
        "auditable": 146,
        "top_level_backups_relative_to_software": 119,
        "tracked_total": 265,
    }


def test_ap005e1_decision_is_non_productive_and_defers_corrections() -> None:
    decisions = _payload()["decisions"]
    assert decisions["productive_change_required_in_ap005e1"] is False
    assert decisions["preserve_console_script"] is True
    assert decisions["preserve_module_entrypoint"] is True
    assert decisions["preserve_public_main"] is True
    assert decisions["preserve_legacy_entrypoint"] is True
    assert decisions["preserve_legacy_path_bridge"] is True
    assert decisions["wheel_contents_are_proven"] is False
    assert decisions["isolated_installation_is_proven"] is False
    assert decisions["defer_corrections_until_ap005e2"] is True
    assert decisions["broad_package_reorganization_allowed"] is False


def test_ap005e1_records_required_ap005e2_gates() -> None:
    gates = _payload()["ap005e2_gates"]
    assert len(gates) == 11
    assert any("build wheel and sdist" in gate for gate in gates)
    assert any("fresh temporary virtual environment" in gate for gate in gates)
    assert any("outside the checkout" in gate for gate in gates)
    assert any("academic_pipeline.__file__" in gate for gate in gates)
    assert any("package data" in gate for gate in gates)


def test_ap005e1_strategy_and_inventory_record_scope() -> None:
    inventory = INVENTORY_MD.read_text(encoding="utf-8")
    strategy = STRATEGY_MD.read_text(encoding="utf-8")

    assert inventory.endswith("\n")
    assert not inventory.endswith("\n\n")
    assert strategy.endswith("\n")
    assert not strategy.endswith("\n\n")

    assert "Nenhuma alteração produtiva" in inventory
    assert "manifesto exato do wheel e do sdist" in inventory
    assert "não renomear pacotes ou módulos" in strategy
    assert "A AP-005E.3 somente poderá alterar" in strategy
    assert "formalmente `no-op`" in strategy


def test_ap005e1_tool_remains_bound_to_its_historical_snapshot() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert '"merge-base",' in source
    assert '"--is-ancestor",' in source
    assert "if head != EXPECTED_BASELINE_COMMIT:" not in source
    assert "if remote_head != EXPECTED_BASELINE_COMMIT:" not in source
    assert "check_files(files)" in source

"""Contrato congelado do inventário AP-004E.

Gerado por tools/refactor/ap004e_inventory_compatibility.py.
Não editar manualmente: regenere o inventário após decisão explícita.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_HEAD = '389f0ae526d12327a58ce23937225cf05b032566'
EXPECTED_SCHEMA = 'ap004e.compatibility-inventory.v6'
EXPECTED_FINGERPRINT = 'cee4120c2602bb12e78fe7d41cf22fc261b8a64647c2c2b9d6e256903d5574e3'
EXPECTED_ITEM_COUNT = 64
EXPECTED_CLASSIFICATION_COUNTS = {'alias canônico necessário': 4, 'bridge de importação necessária': 2, 'compatibilidade interna necessária': 40, 'compatibilidade ligada aos três xfail': 4, 'compatibilidade protegida por decisão da AP-004B': 5, 'compatibilidade pública durável': 6, 'entrypoint público preservado': 2, 'reexport necessário': 6, 'wrapper histórico congelado': 2}
EXPECTED_PROTECTED_SYMBOLS = ['_refs_v6_strip_org', '_ap003d_impl__refs_v6_strip_org', 'WorkflowState._normalize', 'extract_org_abstracts', '_ap003f_pipeline_core']
EXPECTED_FROZEN_FILES = ['software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_13.py', 'software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_14.py']


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        marker = parent / "docs/refactor/academic-pipeline/AP-004/ap004e_compatibility_inventory.json"
        if marker.is_file():
            return parent
    raise AssertionError("não foi possível localizar a raiz do repositório")


def _load_inventory() -> dict:
    path = _repo_root() / "docs/refactor/academic-pipeline/AP-004/ap004e_compatibility_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_basis(data: dict) -> str:
    basis = {
        "schema_version": data["schema_version"],
        "baseline": data["baseline"],
        "scope": data["scope"],
        "classification_model": data["classification_model"],
        "summary": data["summary"],
        "entrypoints": data["entrypoints"],
        "items": data["items"],
        "gate": data["gate"],
    }
    encoded = json.dumps(
        basis,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_ap004e_inventory_contract_is_frozen() -> None:
    data = _load_inventory()
    assert data["schema_version"] == EXPECTED_SCHEMA
    assert data["baseline"]["head"] == EXPECTED_HEAD
    assert data["baseline"]["remote_head"] == EXPECTED_HEAD
    assert data["baseline"]["divergence"] == [0, 0]
    assert data["summary"]["item_count"] == EXPECTED_ITEM_COUNT
    assert data["summary"]["classification_counts"] == EXPECTED_CLASSIFICATION_COUNTS
    assert data["contract_fingerprint"] == EXPECTED_FINGERPRINT
    assert _fingerprint_basis(data) == EXPECTED_FINGERPRINT


def test_ap004e_protected_and_frozen_surfaces_are_present() -> None:
    data = _load_inventory()
    assert data["scope"]["protected_symbols"] == EXPECTED_PROTECTED_SYMBOLS
    assert data["scope"]["frozen_files"] == EXPECTED_FROZEN_FILES
    current_names = {item["current_name"] for item in data["items"]}
    for symbol in EXPECTED_PROTECTED_SYMBOLS:
        assert symbol in current_names
    for path in EXPECTED_FROZEN_FILES:
        assert Path(path).stem in current_names


def test_ap004e_gate_blocks_productive_actions() -> None:
    data = _load_inventory()
    assert data["generation"]["read_only_product_scan"] is True
    assert data["generation"]["productive_code_changed"] is False
    assert data["generation"]["applicator_created"] is False
    assert data["generation"]["commit_created"] is False
    assert data["generation"]["push_performed"] is False
    assert data["generation"]["integration_performed"] is False
    assert data["gate"]["inventory_approval_required"] is True
    assert data["gate"]["productive_applicator_allowed"] is False
    assert data["gate"]["productive_changes_allowed"] is False
    assert data["gate"]["commit_allowed"] is False
    assert data["gate"]["push_allowed"] is False
    assert data["gate"]["integration_allowed"] is False

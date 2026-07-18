"""Contrato de caracterização do inventário preparatório AP-004D."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_SCHEMA = 'ap004d-version-marker-inventory/2'
EXPECTED_DIGEST = '2059d15dceb68a105e6b03b4fa15e900730ab398e1dc1eb03dd13143578571b1'
EXPECTED_CLASSIFICATIONS = set(['colisao_destino', 'marcador_ambiguo_decisao_manual', 'marcador_caminho_fisico_fora_escopo', 'marcador_comentario_historico', 'marcador_interno_removivel', 'marcador_necessario_compatibilidade', 'marcador_preso_contrato_historico', 'marcador_privado_renomeavel_ast', 'marcador_protegido_xfail', 'marcador_string_operacional', 'ocorrencia_apenas_documental', 'ocorrencia_snapshot_fixture_manifesto'])
REQUIRED_RECORD_KEYS = {
    "id",
    "current",
    "proposed",
    "path",
    "line",
    "column",
    "occurrence_type",
    "ast_scope",
    "consumers",
    "consumer_count",
    "contracts",
    "risk",
    "decision",
    "reason",
    "wave",
    "compatibility_required",
    "classification",
    "marker_kind",
    "context",
    "collision",
}


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / ".git").exists():
            return parent
    raise AssertionError("Raiz Git não encontrada a partir do teste AP-004D")


def _load_inventory() -> tuple[Path, dict]:
    repo_root = _find_repo_root()
    path = repo_root / "docs/refactor/academic-pipeline/AP-004/ap004d_version_marker_inventory.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _logical_digest(payload: dict) -> str:
    normalized = dict(payload)
    normalized.pop("inventory_sha256", None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_ap004d_inventory_schema_and_digest() -> None:
    path, payload = _load_inventory()
    assert path.is_file()
    assert payload["schema_version"] == EXPECTED_SCHEMA
    assert payload["inventory_sha256"] == EXPECTED_DIGEST
    assert _logical_digest(payload) == EXPECTED_DIGEST
    assert payload["application_gate"]["productive_applicator_allowed"] is False


def test_ap004d_inventory_records_are_complete_and_unique() -> None:
    _, payload = _load_inventory()
    records = payload["records"]
    assert payload["summary"]["record_count"] == len(records)
    assert len({record["id"] for record in records}) == len(records)
    assert all(REQUIRED_RECORD_KEYS <= set(record) for record in records)
    assert all(record["classification"] in EXPECTED_CLASSIFICATIONS for record in records)
    assert all(not (record["decision"] == "candidato" and record["collision"]) for record in records)


def test_ap004d_protected_and_frozen_contracts_are_not_candidates() -> None:
    _, payload = _load_inventory()
    protected = set(payload["scope"]["protected_symbols"])
    frozen = set(payload["scope"]["frozen_fulltext_files"])
    record_names = {record["current"] for record in payload["records"]}
    assert protected <= record_names
    assert any(
        record["current"] == "_normalize"
        and "WorkflowState._normalize" in record["context"]
        for record in payload["records"]
    )
    for record in payload["records"]:
        if record["current"] in protected or Path(record["path"]).name in frozen:
            assert record["decision"] != "candidato"
        if record["classification"] == "marcador_protegido_xfail":
            assert record["decision"] == "preservar"
            assert record["wave"] == "onda_0_preservacao"


def test_ap004d_markdown_artifacts_reference_same_digest() -> None:
    repo_root = _find_repo_root()
    for relative in (
        "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_INVENTORY.md",
        "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_STRATEGY.md",
    ):
        text = (repo_root / relative).read_text(encoding="utf-8")
        assert EXPECTED_DIGEST in text

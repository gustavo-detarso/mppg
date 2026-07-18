from __future__ import annotations

import pytest
from pydantic import ValidationError

from document_model import (
    AcademicDocument,
    BibliographyInfo,
    Block,
    Citation,
    DocumentMetadata,
    Section,
    TextSpan,
    slugify,
)


def _minimal_document() -> AcademicDocument:
    return AcademicDocument(
        metadata=DocumentMetadata(titulo="Documento de teste"),
        sections=[
            Section(
                title="Introdução",
                blocks=[
                    Block(
                        type="paragraph",
                        content=[TextSpan(text="Texto acadêmico suficientemente claro.")],
                    )
                ],
            )
        ],
    )


def test_citation_normalizes_deduplicates_and_removes_at_prefix() -> None:
    citation = Citation(keys=[" @silva2020 ", "silva2020", "", "@souza2021"])

    assert citation.keys == ["silva2020", "souza2021"]


def test_paragraph_block_promotes_text_to_inline_content() -> None:
    block = Block(type="paragraph", text="Texto legado do parágrafo.")

    assert block.content == [TextSpan(text="Texto legado do parágrafo.")]


@pytest.mark.parametrize(
    ("raw_level", "expected"),
    [(0, 1), (-10, 1), (3, 3), (99, 6)],
)
def test_heading_block_clamps_level(raw_level: int, expected: int) -> None:
    block = Block(type="heading", text="Título", level=raw_level)

    assert block.level == expected


def test_section_generates_slug_id_and_clamps_level() -> None:
    section = Section(title="Análise de Políticas Públicas", level=20)

    assert section.id == "analise_de_politicas_publicas"
    assert section.level == 6


def test_slugify_has_stable_fallback_for_empty_text() -> None:
    assert slugify("") == "secao"
    assert slugify("  Ação & Evidência  ") == "acao_evidencia"


def test_academic_document_requires_at_least_one_section() -> None:
    with pytest.raises(ValidationError, match="ao menos uma seção"):
        AcademicDocument(
            metadata=DocumentMetadata(titulo="Documento vazio"),
            sections=[],
        )


def test_strict_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        DocumentMetadata(titulo="Teste", campo_inexistente="valor")


def test_document_round_trip_preserves_canonical_structure() -> None:
    original = _minimal_document()
    restored = AcademicDocument.model_validate_json(original.model_dump_json())

    assert restored == original


def test_bibliography_defaults_are_stable() -> None:
    bibliography = BibliographyInfo()

    assert bibliography.bib_path == ""
    assert bibliography.style == "apa"
    assert bibliography.entries_used == []

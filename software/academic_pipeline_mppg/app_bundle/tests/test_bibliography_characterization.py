from __future__ import annotations

from bibliography_manager import (
    BibMetadata,
    bib_entry_key,
    bibtex_escape,
    crossref_item_to_meta,
    deduplicate_entries,
    entry_identity,
    expand_metadata_provider_selection,
    extract_doi_from_text,
    extract_field,
    make_bib_key,
    normalize_doi,
    render_bib_entry,
    split_bib_entries,
)


def test_normalize_doi_accepts_url_prefix_and_trailing_period() -> None:
    assert normalize_doi(" HTTPS://doi.org/10.1234/ABC.9. ") == "10.1234/abc.9"


def test_extract_doi_from_prose_removes_terminal_punctuation() -> None:
    text = "Disponível em doi: 10.5555/Example-2024. Consulte o artigo."

    assert extract_doi_from_text(text) == "10.5555/example-2024"


def test_metadata_provider_selection_expands_master_and_aliases() -> None:
    assert expand_metadata_provider_selection(["todas"]) == [
        "crossref",
        "openalex",
        "semantic_scholar",
        "scopus",
    ]
    assert expand_metadata_provider_selection(
        ["semanticscholar", "crossref", "crossref", "desconhecido"]
    ) == ["semantic_scholar", "crossref"]


def test_make_bib_key_is_deterministic_and_adds_collision_suffix() -> None:
    meta = BibMetadata(
        title="Política pública baseada em evidências",
        authors=["Maria da Silva"],
        year="2020",
    )
    used: set[str] = set()

    first = make_bib_key(meta, used)
    second = make_bib_key(meta, used)

    assert first == "silva2020_politica_publica"
    assert second == "silva2020_politica_publica_2"


def test_bibtex_escape_handles_reserved_characters() -> None:
    assert bibtex_escape("A&B 50% valor_1 #teste") == (
        r"A\&B 50\% valor\_1 \#teste"
    )


def test_render_bib_entry_falls_back_to_misc_and_escapes_fields() -> None:
    meta = BibMetadata(
        entry_type="tipo-inexistente",
        title="Política & Gestão",
        authors=["Silva, Maria"],
        year="2024",
        doi="10.1/teste",
    )

    entry = render_bib_entry("silva2024", meta)

    assert entry.startswith("@misc{silva2024,")
    assert r"title = {Política \& Gestão}" in entry
    assert "author = {Silva, Maria}" in entry
    assert "doi = {10.1/teste}" in entry


def test_split_bib_entries_handles_nested_braces() -> None:
    text = """
    comentário inicial
    @article{a,
      title = {Título com {subtítulo}},
      year = {2020}
    }

    @book{b,
      title = {Outro título},
      year = {2021}
    }
    """

    entries = split_bib_entries(text)

    assert len(entries) == 2
    assert bib_entry_key(entries[0]) == "a"
    assert bib_entry_key(entries[1]) == "b"
    assert extract_field(entries[0], "title") == "Título com {subtítulo}"


def test_entry_identity_prefers_normalized_doi() -> None:
    entry = """
    @article{x,
      title = {Título},
      doi = {https://doi.org/10.1000/ABC}
    }
    """

    assert entry_identity(entry) == "doi:10.1000/abc"


def test_deduplicate_entries_keeps_highest_quality_entry() -> None:
    sparse = """
    @article{a,
      title = {Título},
      doi = {10.1000/teste}
    }
    """
    rich = """
    @article{b,
      title = {Título completo},
      author = {Silva, Maria},
      year = {2024},
      journaltitle = {Revista},
      pages = {1--20},
      doi = {10.1000/teste}
    }
    """

    deduplicated = deduplicate_entries([sparse, rich])

    assert len(deduplicated) == 1
    assert bib_entry_key(deduplicated[0]) == "b"


def test_crossref_conversion_maps_article_metadata() -> None:
    item = {
        "type": "journal-article",
        "title": ["Título do artigo"],
        "container-title": ["Revista de Teste"],
        "issued": {"date-parts": [[2023, 5, 1]]},
        "author": [{"family": "Silva", "given": "Maria"}],
        "DOI": "10.1000/TESTE",
        "page": "10-20",
        "volume": "4",
        "issue": "2",
        "URL": "https://example.test/artigo",
    }

    meta = crossref_item_to_meta(item)

    assert meta.entry_type == "article"
    assert meta.title == "Título do artigo"
    assert meta.authors == ["Maria Silva"]
    assert meta.year == "2023"
    assert meta.journaltitle == "Revista de Teste"
    assert meta.doi == "10.1000/teste"

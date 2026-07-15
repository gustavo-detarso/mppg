from __future__ import annotations

from pathlib import Path

import pytest

from citation_renderer import (
    extract_cited_keys_from_model_inline,
    extract_latex_cited_keys,
    render_latex_citation,
    render_latex_inlines,
)
from document_builder import (
    _meta_from_cfg,
    _safe_int,
    repair_section_blocks,
)
from document_model import (
    AcademicDocument,
    BibliographyInfo,
    Block,
    Citation,
    DiagnosticsInfo,
    DocumentMetadata,
    Section,
    TableData,
    TextSpan,
)
from document_validator import (
    find_technical_leaks_in_text,
    raise_if_errors,
    sanitize_document_model_raw_bibkeys,
    sanitize_document_model_technical_leaks,
    validate_document_model,
    validate_org_text,
)
from render_org_latex import (
    clean_heading_title,
    enforce_abnt_biblatex_options,
    normalize_biblatex_style,
    render_block_org,
    render_document_body,
    render_table_block_org,
    sanitize_numeric_citation_markers_in_org,
    sanitize_raw_bibkeys_in_org,
)


def _document(
    *,
    text: str = "Texto acadêmico de teste.",
    content: list[TextSpan | Citation] | None = None,
    entries: list[str] | None = None,
    programa: str = "",
    curso: str = "Mestrado em Políticas Públicas",
) -> AcademicDocument:
    block = Block(
        type="paragraph",
        text=text if content is not None else "",
        content=content if content is not None else [TextSpan(text=text)],
    )
    return AcademicDocument(
        metadata=DocumentMetadata(
            titulo="Documento de teste",
            programa=programa,
            curso=curso,
        ),
        sections=[Section(title="Introdução", blocks=[block])],
        bibliography=BibliographyInfo(entries_used=entries or []),
    )


@pytest.mark.parametrize(
    ("value", "default", "minimum", "maximum", "expected"),
    [
        ("12", 3, 0, None, 12),
        ("inválido", 7, 0, None, 7),
        ("-5", 7, 2, None, 2),
        ("999", 7, 0, 20, 20),
    ],
)
def test_safe_int_characterizes_bounds(
    value: object,
    default: int,
    minimum: int,
    maximum: int | None,
    expected: int,
) -> None:
    assert _safe_int(value, default, minimum, maximum) == expected


def test_metadata_precedence_uses_document_then_activity_then_defaults() -> None:
    cfg = {
        "documento": {
            "tipo_documento": "atividade",
            "titulo_trabalho": "Título documental",
            "autor": "Autora documental",
            "course_name": "Curso documental",
        },
        "atividade": {
            "titulo_trabalho": "Título da atividade",
            "aluno": "Aluno da atividade",
            "curso": "Curso da atividade",
            "disciplina": "Disciplina de teste",
        },
    }

    metadata = _meta_from_cfg(cfg)

    assert metadata["tipo_documento"] == "atividade"
    assert metadata["titulo"] == "Título documental"
    assert metadata["autor"] == "Autora documental"
    assert metadata["curso"] == "Curso documental"
    assert metadata["disciplina"] == "Disciplina de teste"


def test_repair_section_blocks_recovers_text_when_content_has_only_citation() -> None:
    section = Section(
        title="Resultados",
        blocks=[
            Block(
                type="paragraph",
                text=(
                    "Este parágrafo contém uma explicação substantiva que deve "
                    "permanecer visível no documento final [1]."
                ),
                content=[Citation(keys=["silva2020"])],
            )
        ],
    )

    repair_section_blocks(section)
    block = section.blocks[0]

    assert block.text == ""
    assert isinstance(block.content[0], TextSpan)
    assert "[1]" not in block.content[0].text
    assert isinstance(block.content[1], Citation)
    assert block.content[1].keys == ["silva2020"]


def test_repair_section_blocks_removes_numeric_markers_from_lists_and_tables() -> None:
    section = Section(
        title="Síntese",
        blocks=[
            Block(type="bullet_list", items=["Primeiro achado [1]", "Segundo [1, 2]"]),
            Block(
                type="table",
                table=TableData(
                    headers=["Dimensão [1]", "Resultado"],
                    rows=[["Política [2]", "Efeito observado [1]"]],
                ),
            ),
        ],
    )

    repair_section_blocks(section)

    assert section.blocks[0].items == ["Primeiro achado", "Segundo"]
    assert section.blocks[1].table.headers == ["Dimensão", "Resultado"]
    assert section.blocks[1].table.rows == [["Política", "Efeito observado"]]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("parenthetical", r"\parencite{silva2020,souza2021}"),
        ("narrative", r"\textcite{silva2020} e \textcite{souza2021}"),
        ("author", r"\citeauthor{silva2020,souza2021}"),
        ("year", r"\citeyear{silva2020,souza2021}"),
    ],
)
def test_render_latex_citation_modes(mode: str, expected: str) -> None:
    citation = Citation(mode=mode, keys=["@silva2020", "souza2021"])

    assert render_latex_citation(citation) == expected


def test_render_latex_inlines_preserves_emphasis_and_citation_order() -> None:
    content = [
        TextSpan(text="Análise ", bold=True),
        Citation(keys=["silva2020"]),
        TextSpan(text=" complementar", italic=True),
    ]

    assert render_latex_inlines(content) == (
        r"\textbf{Análise }\parencite{silva2020}\textit{ complementar}"
    )


def test_citation_extractors_deduplicate_keys_in_first_seen_order() -> None:
    content = [
        Citation(keys=["a", "b"]),
        Citation(keys=["b", "@c"]),
    ]
    latex = r"\parencite{a,b} e \textcite{b,c}"

    assert extract_cited_keys_from_model_inline(content) == ["a", "b", "c"]
    assert extract_latex_cited_keys(latex) == ["a", "b", "c"]


def test_technical_leak_detection_avoids_ocr_inside_democracia() -> None:
    assert find_technical_leaks_in_text("A democracia exige participação.") == []
    assert "OCR" in find_technical_leaks_in_text("O texto foi obtido por OCR.")


def test_technical_leak_sanitizer_changes_visible_text_but_not_diagnostics() -> None:
    doc = _document(text="O pipeline usou OCR e fulltext_cache.")
    doc.diagnostics = DiagnosticsInfo(
        prompts_json='{"pipeline": "OCR", "path": "fulltext_cache"}'
    )

    sanitized, changed = sanitize_document_model_technical_leaks(doc)

    visible = sanitized.sections[0].blocks[0].content[0].text
    assert "pipeline" not in visible.lower()
    assert "OCR" not in visible
    assert "fulltext_cache" not in visible
    assert sanitized.diagnostics.prompts_json == (
        '{"pipeline": "OCR", "path": "fulltext_cache"}'
    )
    assert changed


def test_raw_bibkey_sanitizer_converts_visible_key_to_latex_citation() -> None:
    doc = _document(text="O argumento de silva2020 sustenta a análise.")

    sanitized, changed = sanitize_document_model_raw_bibkeys(doc, ["silva2020"])

    text = sanitized.sections[0].blocks[0].content[0].text
    assert text == r"O argumento de \parencite{silva2020} sustenta a análise."
    assert changed == ["sections[0].blocks[0].content[0].text"]


def test_document_validation_reports_missing_key_and_duplicate_cover_fields() -> None:
    doc = _document(
        content=[
            TextSpan(text="Texto com evidência."),
            Citation(keys=["ausente2024"]),
        ],
        programa="Programa X",
        curso="Programa X",
    )

    errors = validate_document_model(doc, ["presente2024"])

    assert any("ausente2024" in error for error in errors)
    assert any("metadata.programa" in error for error in errors)


def test_org_validation_ignores_key_in_header_but_rejects_visible_raw_key() -> None:
    key = "silva2020"
    header_only = (
        f"#+LATEX_HEADER: \\addbibresource{{{key}.bib}}\n"
        "* Introdução\nTexto válido com \\parencite{silva2020}.\n"
    )
    visible_raw = header_only + "A chave silva2020 apareceu em prosa.\n"

    assert validate_org_text(header_only, [key]) == []
    assert any("chaves BibTeX cruas" in error for error in validate_org_text(visible_raw, [key]))


def test_org_validation_rejects_numeric_citation_markers() -> None:
    errors = validate_org_text("* Resultados\nO estudo demonstrou o efeito [1, 2].\n", [])

    assert any("citações numéricas" in error for error in errors)


def test_raise_if_errors_formats_all_messages() -> None:
    with pytest.raises(RuntimeError, match=r"Falha de teste:\n- erro A\n- erro B"):
        raise_if_errors(["erro A", "erro B"], "Falha de teste")


def test_heading_cleaner_removes_manual_numbering() -> None:
    assert clean_heading_title("2.1 Introdução") == "Introdução"
    assert clean_heading_title("3) Resultados") == "Resultados"


def test_small_table_uses_native_org_pipe_format() -> None:
    block = Block(
        type="table",
        id="tab_teste",
        table=TableData(
            caption="Tabela simples",
            headers=["Item", "Valor"],
            rows=[["A", "1"]],
        ),
    )

    rendered = render_table_block_org(block)

    assert "#+CAPTION: Tabela simples" in rendered
    assert "| Item | Valor |" in rendered
    assert "longtblr" not in rendered


def test_wide_table_uses_responsive_latex_and_landscape() -> None:
    block = Block(
        type="table",
        id="tab_larga",
        table=TableData(
            caption="Tabela larga",
            headers=["A", "B", "C", "D", "E"],
            rows=[["1", "2", "3", "4", "5"]],
        ),
    )

    rendered = render_table_block_org(block)

    assert r"\begin{longtblr}" in rendered
    assert r"\begin{landscape}" in rendered
    assert "label={tab_larga}" in rendered


def test_render_block_uses_raw_text_when_inline_content_is_citation_only() -> None:
    block = Block(
        type="paragraph",
        text=(
            "Texto substantivo que não pode desaparecer quando o conteúdo "
            "estruturado contém apenas uma citação."
        ),
        content=[Citation(keys=["silva2020"])],
    )

    rendered = render_block_org(block)

    assert rendered.startswith("Texto substantivo")
    assert rendered.endswith(r"\parencite{silva2020}")


def test_document_body_skips_ai_generated_reference_section() -> None:
    doc = AcademicDocument(
        metadata=DocumentMetadata(titulo="Teste"),
        sections=[
            Section(
                title="Introdução",
                blocks=[Block(type="paragraph", text="Texto principal.")],
            ),
            Section(
                title="Referências",
                blocks=[Block(type="paragraph", text="silva2020")],
            ),
        ],
        bibliography=BibliographyInfo(entries_used=["silva2020"]),
    )

    body = render_document_body(doc)

    assert "* Introdução" in body
    assert "Texto principal." in body
    assert "Referências" not in body
    assert "silva2020" not in body


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ABNT autor data", "abnt"),
        ("apa7", "apa"),
        ("Vancouver", "numeric"),
        ("authoryear", "authoryear"),
    ],
)
def test_bibliography_style_aliases(raw: str, expected: str) -> None:
    assert normalize_biblatex_style(raw) == expected


def test_abnt_options_are_enforced_without_duplicates() -> None:
    options = enforce_abnt_biblatex_options(
        "backend=biber,style=apa,sorting=nty",
        "abnt",
    )

    assert options.split(",") == [
        "style=abnt",
        "backend=biber",
        "sorting=nty",
        "giveninits=true",
    ]


def test_raw_bibkey_org_sanitizer_changes_only_visible_lines() -> None:
    doc = _document(entries=["silva2020"])
    org = (
        "#+LATEX_HEADER: \\addbibresource{silva2020.bib}\n"
        "#+begin_src text\n"
        "silva2020\n"
        "#+end_src\n"
        "Texto visível silva2020.\n"
    )

    sanitized = sanitize_raw_bibkeys_in_org(org, doc)

    assert r"\addbibresource{silva2020.bib}" in sanitized
    assert "#+begin_src text\nsilva2020\n#+end_src" in sanitized
    assert r"Texto visível \parencite{silva2020}." in sanitized


def test_numeric_org_sanitizer_maps_numbers_to_bibliography_order() -> None:
    org = (
        "#+LATEX_HEADER: definição [1]\n"
        "Resultado [1, 2] e comando existente \\parencite{c}.\n"
    )

    sanitized = sanitize_numeric_citation_markers_in_org(org, ["a", "b", "c"])

    assert "#+LATEX_HEADER: definição [1]" in sanitized
    assert r"Resultado \parencite{a,b}" in sanitized
    assert r"\parencite{c}" in sanitized

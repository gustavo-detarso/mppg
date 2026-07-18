from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import prisma_busca_externa as external
import prisma_pipeline


def test_expand_provider_selection_expands_all_in_declared_order() -> None:
    assert external.expand_provider_selection(["todas"]) == external.PROVIDER_ORDER


def test_expand_provider_selection_normalizes_aliases_and_deduplicates() -> None:
    assert external.expand_provider_selection(
        ["open alex", "cross-ref", "semantic", "openalex", "desconhecida"]
    ) == ["openalex", "crossref", "semantic_scholar"]


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"busca_prisma": {"ativo": True, "modo": "busca_externa"}}, True),
        ({"busca_prisma": {"ativo": True, "modo": "manual"}}, False),
        ({"busca_prisma": {"ativo": False, "modo": "busca_externa"}}, False),
        ({"busca_prisma": "inválido"}, False),
    ],
)
def test_external_search_enabled_requires_exact_mode(
    cfg: dict,
    expected: bool,
) -> None:
    assert external.external_search_enabled(cfg) is expected


def test_value_helpers_characterize_current_normalization() -> None:
    assert external._list(" A; B,\nC ") == ["A", "B", "C"]
    assert external._norm("  Decisão baseada em Evidência ") == (
        "decisao baseada em evidencia"
    )
    assert external._doi("https://doi.org/10.1000/ABC.)") == "10.1000/abc"
    assert external._title_key("Ação & Evidência!") == "acaoevidencia"
    assert external._int("12") == 12
    assert external._int("x") is None


def test_authors_and_openalex_abstract_are_materialized() -> None:
    authors = external._authors(
        [{"display_name": "Ana Silva"}, {"name": "Bruno Souza"}, ""]
    )
    abstract = external._openalex_abstract(
        {"política": [1], "evidência": [0], "pública": [2]}
    )

    assert authors == "Ana Silva; Bruno Souza"
    assert abstract == "evidência política pública"


def test_safe_url_redacts_credentials_and_email() -> None:
    url = (
        "https://example.test?q=x&api_key=segredo"
        "&email=pessoa@example.test&token=abc"
    )

    sanitized = external._safe_url(url)

    assert "segredo" not in sanitized
    assert "pessoa@example.test" not in sanitized
    assert "token=REDACTED" in sanitized


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", None),
        ("5", 5.0),
        ("-2", 0.0),
        ("inválido", None),
    ],
)
def test_retry_after_seconds_numeric_and_invalid(
    value: str,
    expected: float | None,
) -> None:
    assert external._retry_after_seconds(value) == expected


def test_record_normalizes_metadata_and_removes_html() -> None:
    record = external._record(
        "crossref",
        "123",
        title="  Título   do estudo ",
        authors=[{"family": "Silva"}],
        year=2024,
        venue="  Revista   X ",
        doi="DOI: 10.1000/XYZ.",
        abstract="<p>Resumo <b>útil</b>.</p>",
        citations="7",
    )

    assert record["id_registro"] == "crossref:123"
    assert record["titulo"] == "Título do estudo"
    assert record["autores"] == "Silva"
    assert record["periodico"] == "Revista X"
    assert record["doi"] == "10.1000/xyz"
    assert record["resumo"] == "Resumo útil ."
    assert record["citacoes"] == 7
    assert record["fontes"] == ["crossref"]


def test_deduplicate_merges_doi_records_and_keeps_best_metadata() -> None:
    first = {
        "titulo": "Estudo",
        "doi": "10.1000/x",
        "fontes": ["crossref"],
        "consultas_busca": ["A"],
        "citacoes": 2,
        "resumo": "",
    }
    second = {
        "titulo": "Estudo mais completo",
        "doi": "https://doi.org/10.1000/X",
        "fontes": ["openalex"],
        "consultas_busca": ["B"],
        "citacoes": 9,
        "resumo": "Resumo",
    }

    rows, removed = external._deduplicate([first, second])

    assert removed == 1
    assert rows == [first]
    assert first["resumo"] == "Resumo"
    assert first["fontes"] == ["crossref", "openalex"]
    assert first["consultas_busca"] == ["A", "B"]
    assert first["citacoes"] == 9


def test_score_counts_each_matching_keyword_once() -> None:
    record = {
        "titulo": "Política pública baseada em evidências",
        "resumo": "Avaliação de política.",
        "periodico": "",
    }

    assert external._score(record, ["política", "evidência", "ausente"]) == 2.0


def test_parse_ai_json_accepts_markdown_fence() -> None:
    payload = external._parse_ai_json(
        '```json\n{"registros": [{"id_registro": "x"}]}\n```'
    )

    assert payload["registros"][0]["id_registro"] == "x"


def test_parse_ai_json_rejects_non_object() -> None:
    with pytest.raises(RuntimeError, match="objeto JSON"):
        external._parse_ai_json('["x"]')


@pytest.mark.parametrize(
    ("raw", "score", "poor", "expected"),
    [
        ("alta aderência", 10, False, "PRIORIDADE_ALTA"),
        ("", 85, False, "PRIORIDADE_ALTA"),
        ("", 50, False, "REVISAR_HUMANO"),
        ("", 10, False, "PROVAVEL_EXCLUSAO"),
        ("", 90, True, "INCERTO_METADADOS"),
    ],
)
def test_normalize_ai_recommendation_uses_aliases_and_fallbacks(
    raw: str,
    score: float,
    poor: bool,
    expected: str,
) -> None:
    assert external._normalize_ai_recommendation(
        raw,
        score,
        metadata_poor=poor,
    ) == expected


def test_apply_ai_pretriage_clamps_scores_and_sanitizes_text() -> None:
    record = {"titulo": "Título", "resumo": "Resumo"}
    external._apply_ai_pretriage(
        record,
        {
            "escore": 150,
            "confianca": -10,
            "recomendacao": "revisar",
            "justificativa": "A\x00" * 600,
        },
    )

    assert record["status_pre_triagem_ia"] == "CONCLUIDA"
    assert record["escore_aderencia_ia"] == 100.0
    assert record["confianca_ia"] == 0.0
    assert record["recomendacao_ia"] == "REVISAR_HUMANO"
    assert "\x00" not in record["justificativa_ia"]
    assert len(record["justificativa_ia"]) <= 900


def test_disabled_ai_pretriage_marks_every_record_without_client() -> None:
    records = [{"id_registro": "a"}, {"id_registro": "b"}]
    audit = external._pretriage_with_ai(
        records,
        {"ai_screen_enabled": False},
        client=None,
        model=None,
    )

    assert audit["enabled"] is False
    assert [row["status_pre_triagem_ia"] for row in records] == [
        "DESATIVADA",
        "DESATIVADA",
    ]


def test_enabled_ai_pretriage_requires_client() -> None:
    with pytest.raises(RuntimeError, match="cliente OpenAI"):
        external._pretriage_with_ai(
            [{"id_registro": "a"}],
            {"ai_screen_enabled": True},
            client=None,
            model="modelo",
        )


def test_select_triage_records_preserves_uncertainty_reserve() -> None:
    records = [
        {
            "id_registro": "alta",
            "titulo": "A",
            "status_pre_triagem_ia": "CONCLUIDA",
            "recomendacao_ia": "PRIORIDADE_ALTA",
            "escore_aderencia_ia": 95,
            "confianca_ia": 99,
            "pontuacao_relevancia": 2,
        },
        {
            "id_registro": "incerta",
            "titulo": "B",
            "status_pre_triagem_ia": "FALHA_LOTE",
            "recomendacao_ia": "INCERTO_METADADOS",
            "escore_aderencia_ia": 0,
            "confianca_ia": 0,
            "pontuacao_relevancia": 0,
        },
        {
            "id_registro": "baixa",
            "titulo": "C",
            "status_pre_triagem_ia": "CONCLUIDA",
            "recomendacao_ia": "PROVAVEL_EXCLUSAO",
            "escore_aderencia_ia": 5,
            "confianca_ia": 90,
            "pontuacao_relevancia": 0,
        },
    ]
    config = {
        "initial_limit": 2,
        "ai_screen_enabled": True,
        "ai_screen_review_reserve": 1,
        "ai_screen_min_confidence": 40,
    }

    selected = external._select_triage_records(records, config)

    assert {row["id_registro"] for row in selected} == {"alta", "incerta"}
    assert selected[0]["id_registro"] == "alta"


def test_request_json_rejects_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(external, "_request_text", lambda *_a, **_k: "não-json")

    with pytest.raises(RuntimeError, match="JSON válido"):
        external._request_json("https://example.test")


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"relatorio_pesquisa": {"ativo": True}}, True),
        ({"relatorio_pesquisa": {"ativo": False}}, False),
        ({"relatorio_pesquisa": "x"}, False),
    ],
)
def test_prisma_enabled_reads_boolean_flag(cfg: dict, expected: bool) -> None:
    assert prisma_pipeline.prisma_enabled(cfg) is expected


def test_prisma_output_paths_respects_explicit_research_paths(tmp_path: Path) -> None:
    cfg = {
        "__config_dir__": str(tmp_path),
        "paths": {
            "research_output_dir": "pesquisa",
            "research_prefix": "matriz",
            "create_research_subdir": True,
        },
    }

    out_dir, prefix = prisma_pipeline.prisma_output_paths(
        cfg,
        tmp_path / "documento",
        "paper",
    )

    assert prefix == "matriz"
    assert out_dir == (tmp_path / "pesquisa" / "matriz").resolve()
    assert out_dir.is_dir()


def test_prisma_report_outputs_short_circuits_when_disabled(tmp_path: Path) -> None:
    assert prisma_pipeline.run_prisma_report_outputs(
        {},
        [],
        [],
        SimpleNamespace(),
        tmp_path,
        "paper",
    ) is None


def test_prisma_report_outputs_orchestrates_selected_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "prisma"
    report = SimpleNamespace(
        model_dump=lambda: {"ok": True},
        diagnostics=SimpleNamespace(avisos=[]),
    )
    bib = SimpleNamespace(
        bib_path=tmp_path / "source.bib",
        keys=["x"],
    )
    bib.bib_path.write_text("@article{x,title={X}}", encoding="utf-8")
    written: list[Path] = []

    def fake_output_paths(*_a):
        out.mkdir(parents=True, exist_ok=True)
        return out, "rel"

    monkeypatch.setattr(prisma_pipeline, "prisma_output_paths", fake_output_paths)
    monkeypatch.setattr(prisma_pipeline, "build_prisma_report", lambda *_a, **_k: report)
    monkeypatch.setattr(prisma_pipeline, "validate_prisma_report", lambda *_a, **_k: [])
    monkeypatch.setattr(prisma_pipeline, "raise_if_prisma_errors", lambda _m: None)
    monkeypatch.setattr(
        prisma_pipeline,
        "render_prisma_flow_svg",
        lambda _r, path: path,
    )
    monkeypatch.setattr(
        prisma_pipeline,
        "render_prisma_org",
        lambda _r, path, *_a, **_k: path.write_text("* PRISMA\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        prisma_pipeline,
        "render_prisma_xlsx",
        lambda _r, path: path,
    )
    monkeypatch.setattr(
        prisma_pipeline,
        "write_json",
        lambda path, _value: written.append(path),
    )
    cfg = {
        "__config_dir__": str(tmp_path),
        "relatorio_pesquisa": {
            "ativo": True,
            "exportar_json": True,
            "exportar_fluxograma": True,
            "exportar_org": True,
            "exportar_pdf": False,
            "exportar_docx": False,
            "exportar_xlsx": True,
        },
    }

    paths = prisma_pipeline.run_prisma_report_outputs(
        cfg,
        [],
        [],
        bib,
        tmp_path / "documento",
        "paper",
    )

    assert paths["output_dir"] == str(out)
    assert paths["json"].endswith("rel.prisma_report.json")
    assert paths["fluxograma_svg"].endswith("rel_fluxo_prisma.svg")
    assert paths["org"].endswith("rel.org")
    assert paths["xlsx"].endswith("rel.xlsx")
    assert paths["summary"].endswith("rel.prisma_outputs.json")
    assert len(written) == 2

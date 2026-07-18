from __future__ import annotations

import argparse

import pytest

from pathlib import Path

from academic_pipeline_rc10 import (
    _refs_apply_runtime_policy,
    _refs_disabled,
    _refs_v6_strip_org,
    apply_cli_path_overrides,
    output_paths,
    research_output_paths,
    resolve_bib_for_existing_document,
    work_cache_paths,
)
from document_model import (
    AcademicDocument,
    BibliographyInfo,
    Block,
    DocumentMetadata,
    Section,
)


def _document(bib_path: str = "", entries: list[str] | None = None) -> AcademicDocument:
    return AcademicDocument(
        metadata=DocumentMetadata(titulo="Documento"),
        sections=[
            Section(
                title="Introdução",
                blocks=[Block(type="paragraph", text="Texto.")],
            )
        ],
        bibliography=BibliographyInfo(
            bib_path=bib_path,
            entries_used=entries or [],
        ),
    )


def test_output_paths_uses_project_name_and_creates_subdirectory(tmp_path: Path) -> None:
    cfg = {
        "__config_dir__": str(tmp_path),
        "projeto": {"nome": "atividade_teste"},
        "paths": {"document_output_dir": "saida"},
    }

    out_dir, prefix = output_paths(cfg)

    assert prefix == "atividade_teste"
    assert out_dir == (tmp_path / "saida" / "atividade_teste").resolve()
    assert out_dir.is_dir()


def test_output_paths_can_disable_document_subdirectory(tmp_path: Path) -> None:
    cfg = {
        "__config_dir__": str(tmp_path),
        "paths": {
            "document_output_dir": "saida",
            "document_prefix": "prefixo",
            "create_document_subdir": False,
        },
    }

    out_dir, prefix = output_paths(cfg)

    assert prefix == "prefixo"
    assert out_dir == (tmp_path / "saida").resolve()


def test_research_output_paths_uses_independent_default_location(tmp_path: Path) -> None:
    cfg = {
        "__config_dir__": str(tmp_path),
        "projeto": {"nome": "meu_projeto"},
    }

    out_dir, prefix = research_output_paths(cfg)

    assert prefix == "relatorio_prisma_meu_projeto"
    assert out_dir == (
        tmp_path / "output_pesquisa" / "relatorio_prisma_meu_projeto"
    ).resolve()
    assert out_dir.is_dir()


def test_work_cache_paths_create_isolated_prefix_directories(tmp_path: Path) -> None:
    cfg = {
        "__config_dir__": str(tmp_path),
        "paths": {
            "work_dir": "operacao",
            "cache_dir": "cache",
        },
    }

    work_dir, cache_dir = work_cache_paths(cfg, "documento")

    assert work_dir == (tmp_path / "operacao" / "documento").resolve()
    assert cache_dir == (tmp_path / "cache" / "documento").resolve()
    assert work_dir.is_dir()
    assert cache_dir.is_dir()


def test_cli_path_overrides_have_priority_and_preserve_other_values() -> None:
    cfg = {
        "paths": {
            "document_output_dir": "toml-output",
            "research_prefix": "existente",
        },
        "documento": {"tipo_documento": "paper"},
    }
    args = argparse.Namespace(
        output_dir="cli-output",
        work_dir="cli-work",
        cache_dir="",
        research_output_dir="cli-research",
        output_prefix="cli-prefix",
        no_output_subdir=True,
        layout="atividade_fgv",
        tipo_conteudo="matriz",
        genero_academico="atividade",
    )

    result = apply_cli_path_overrides(cfg, args)

    assert result is cfg
    assert result["paths"] == {
        "document_output_dir": "cli-output",
        "research_prefix": "existente",
        "work_dir": "cli-work",
        "research_output_dir": "cli-research",
        "document_prefix": "cli-prefix",
        "create_document_subdir": False,
    }
    assert result["documento"] == {
        "tipo_documento": "paper",
        "layout": "atividade_fgv",
        "tipo_conteudo": "matriz",
        "genero_academico": "atividade",
    }


def test_reference_policy_explicit_bibliography_flag_has_priority() -> None:
    assert _refs_disabled({"bibliografia": {"ativo": False}})
    assert not _refs_disabled(
        {
            "bibliografia": {"ativo": True},
            "documento": {"referencias_formais": False},
        }
    )


def test_reference_policy_supports_legacy_all_flags_disabled() -> None:
    cfg = {
        "documentos_locais": {
            "auto_detect_bib": False,
            "gerar_bib_revisado_ia": False,
        },
        "documento": {"usar_citacoes_latex_diretas": False},
    }

    assert _refs_disabled(cfg)


def test_runtime_reference_policy_is_idempotent_and_disables_all_sources() -> None:
    cfg = {
        "bibliografia": {"ativo": False},
        "documento": {},
        "documentos_locais": {},
        "orientacoes": {"inline": "Orientação original."},
    }

    first = _refs_apply_runtime_policy(cfg)
    second = _refs_apply_runtime_policy(first)

    assert first is cfg
    assert second is cfg
    assert cfg["bibliografia"]["gerar_arquivo_bib"] is False
    assert cfg["bibliografia"]["buscar_metadados_por_doi"] is False
    assert cfg["documento"]["referencias_formais"] is False
    assert cfg["documento"]["usar_citacoes_latex_diretas"] is False
    assert cfg["documentos_locais"]["extrair_doi_dos_pdfs"] is False
    assert cfg["orientacoes"]["inline"].count(
        "Não inclua citações no corpo do texto"
    ) == 1


def test_reference_strip_removes_directives_and_reference_section() -> None:
    org = (
        "#+TITLE: Teste\n"
        "#+LATEX_HEADER: \\addbibresource{teste.bib}\n"
        "* Introdução\n"
        "Texto principal.\n"
        "* Referências\n"
        "Entrada visível.\n"
        "* Conclusão\n"
        "Texto final.\n"
        "\\printbibliography\n"
    )

    cleaned = _refs_v6_strip_org(org)

    assert "addbibresource" not in cleaned
    assert "Referências" not in cleaned
    assert "Entrada visível" not in cleaned
    assert "printbibliography" not in cleaned
    assert "* Introdução" in cleaned
    assert "* Conclusão" in cleaned
    assert "Texto final." in cleaned


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Defeito legado catalogado: _refs_v6_strip_org usa o padrão 'para' "
        "em vez de 'paren' e ainda não remove \\parencite."
    ),
)
def test_reference_strip_should_remove_parenthetical_citations() -> None:
    cleaned = _refs_v6_strip_org(
        "* Introdução\nTexto com \\parencite{silva2020}.\n"
    )

    assert "parencite" not in cleaned


def test_existing_document_bibliography_is_copied_and_keys_are_inferred(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "fonte"
    out_dir = tmp_path / "saida"
    source_dir.mkdir()
    out_dir.mkdir()

    document_json = source_dir / "atividade.document.json"
    document_json.write_text("{}", encoding="utf-8")
    source_bib = source_dir / "referencias.bib"
    source_bib.write_text(
        "@article{silva2020, title={A}}\n"
        "@book{souza2021, title={B}}\n",
        encoding="utf-8",
    )
    document = _document(bib_path="referencias.bib")

    resolved, keys = resolve_bib_for_existing_document(
        document,
        document_json,
        out_dir,
        "atividade",
    )

    assert resolved == out_dir / "referencias.bib"
    assert resolved.read_text(encoding="utf-8") == source_bib.read_text(
        encoding="utf-8"
    )
    assert keys == ["silva2020", "souza2021"]

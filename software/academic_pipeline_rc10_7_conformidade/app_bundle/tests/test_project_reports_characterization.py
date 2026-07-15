from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import institution_compliance
import project_tools
import prompt_lock
import quality_report
from document_model import (
    AcademicDocument,
    BibliographyInfo,
    Block,
    Citation,
    DocumentMetadata,
    Section,
    TextSpan,
)


def _write_example_templates(app_bundle: Path) -> None:
    examples = app_bundle / "config" / "examples"
    examples.mkdir(parents=True)
    base = (
        '[projeto]\n'
        'nome = "paper_nome_do_tema"\n'
        '[instituicao]\n'
        'perfil = "antigo"\n'
        '[paths]\n'
        'document_prefix = "paper_nome_do_tema"\n'
        '[documentos_locais]\n'
        'input_zip = "../../projetos/paper_nome_do_tema/documentos-base.zip"\n'
    )
    (examples / "paper_rc10_exemplo.toml").write_text(base, encoding="utf-8")
    (examples / "atividade_rc10_exemplo.toml").write_text(
        base.replace("paper_nome_do_tema", "atividade_aula_2"),
        encoding="utf-8",
    )
    (examples / "relatorio_prisma_rc10_exemplo.toml").write_text(
        base.replace("paper_nome_do_tema", "relatorio_prisma_atividade_aula_2"),
        encoding="utf-8",
    )


def test_make_doi_manifest_from_directory_filters_supported_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fontes"
    source.mkdir()
    (source / "a.pdf").write_bytes(b"%PDF")
    (source / "b.txt").write_text("texto", encoding="utf-8")
    (source / "c.exe").write_text("ignorar", encoding="utf-8")
    output = tmp_path / "doi_manifest.csv"

    report = project_tools.make_doi_manifest(None, source, output)

    assert report["total_files"] == 2
    assert report["files"] == ["a.pdf", "b.txt"]
    assert output.read_text(encoding="utf-8").splitlines() == [
        "arquivo,doi",
        "a.pdf,",
        "b.txt,",
    ]


def test_make_doi_manifest_from_zip_deduplicates_names(tmp_path: Path) -> None:
    archive = tmp_path / "fontes.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pasta/a.pdf", "x")
        zf.writestr("pasta/b.docx", "x")
        zf.writestr("pasta/c.exe", "x")
    output = tmp_path / "manifest.csv"

    report = project_tools.make_doi_manifest(archive, None, output)

    assert report["files"] == ["pasta/a.pdf", "pasta/b.docx"]


def test_make_doi_manifest_requires_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input-zip ou --input-dir"):
        project_tools.make_doi_manifest(None, None, tmp_path / "out.csv")


def test_init_project_creates_expected_safe_structure(tmp_path: Path) -> None:
    app_bundle = tmp_path / "app_bundle"
    _write_example_templates(app_bundle)

    result = project_tools.init_project(
        "Meu Projeto",
        project_type="paper",
        base_dir=tmp_path,
        institution="fgv",
    )

    assert result.project_dir == app_bundle / "projetos" / "meu_projeto"
    assert result.config_path.exists()
    assert result.doi_manifest_path.read_text(encoding="utf-8") == "arquivo,doi\n"
    assert result.readme_path.exists()
    with zipfile.ZipFile(result.documentos_zip_path) as zf:
        assert zf.namelist() == ["README.txt"]
    config = result.config_path.read_text(encoding="utf-8")
    assert 'perfil = "fgv"' in config
    assert "documentos-base.zip" in config
    assert "../../projetos/paper_nome_do_tema" not in config


def test_init_project_refuses_nonempty_existing_directory(tmp_path: Path) -> None:
    app_bundle = tmp_path / "app_bundle"
    _write_example_templates(app_bundle)
    target = app_bundle / "projetos" / "existente"
    target.mkdir(parents=True)
    (target / "manual.txt").write_text("não sobrescrever", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Projeto já existe"):
        project_tools.init_project("existente", base_dir=tmp_path)


def test_template_config_selects_prisma_template_and_rewrites_profile(
    tmp_path: Path,
) -> None:
    app_bundle = tmp_path / "app_bundle"
    _write_example_templates(app_bundle)

    rendered = project_tools._template_config(
        app_bundle,
        "analise",
        "paper_prisma",
        institution="fgv",
    )

    assert 'perfil = "fgv"' in rendered
    assert "analise" in rendered
    assert "relatorio_prisma_atividade_aula_2" not in rendered


def test_render_bib_inspection_markdown_groups_duplicates_and_issues() -> None:
    report = {
        "bib_path": "refs.bib",
        "entries_total": 2,
        "keys_total": 2,
        "duplicate_groups_total": 1,
        "duplicate_groups": {"doi:10.1/x": ["a", "b"]},
        "issues_total": 1,
        "issues": [
            {
                "key": "a",
                "title": "Título",
                "issues": ["paginas_ausentes"],
            }
        ],
        "ok": False,
    }

    text = project_tools.render_bib_inspection_markdown(report)

    assert "# Inspeção bibliográfica" in text
    assert "REVISAR" in text
    assert "Duplicatas prováveis" in text
    assert "paginas_ausentes" in text


def test_prompt_lock_builds_reproducible_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompt_lock, "now_iso", lambda: "2026-07-15T10:00:00-03:00")
    monkeypatch.setattr(prompt_lock, "PIPELINE_VERSION", "rc-test")
    monkeypatch.setattr(
        prompt_lock,
        "prompt_report_for_cfg",
        lambda _cfg: {
            "document": {
                "total_chars": 12,
                "sources": [],
            }
        },
    )
    cfg = {
        "__config_path__": "/tmp/projeto.toml",
        "__institution_profile_name__": "fgv",
        "__institution_profile_path__": "/tmp/fgv.toml",
    }

    lock = prompt_lock.build_prompt_lock(cfg)

    assert lock == {
        "version": "rc-test",
        "generated_at": "2026-07-15T10:00:00-03:00",
        "config_path": "/tmp/projeto.toml",
        "institution_profile": "fgv",
        "institution_profile_path": "/tmp/fgv.toml",
        "prompts": {
            "document": {
                "total_chars": 12,
                "sources": [],
            }
        },
    }


def test_prompt_lock_markdown_reports_empty_and_sanitized_sources() -> None:
    lock = {
        "version": "v",
        "generated_at": "agora",
        "config_path": "cfg.toml",
        "institution_profile": None,
        "prompts": {
            "empty": {"total_chars": 0, "sources": []},
            "document": {
                "total_chars": 20,
                "sources": [
                    {
                        "category": "global",
                        "path": "prompt.md",
                        "sha256": "abc",
                        "chars": 20,
                        "sanitized": True,
                    }
                ],
            },
        },
    }

    text = prompt_lock.render_prompt_lock_markdown(lock)

    assert "Perfil institucional: `nenhum`" in text
    assert "Nenhum prompt carregado" in text
    assert "saneado:  sim" in text


def _document(
    *,
    conclusion: bool = False,
    citation_key: str = "silva2020",
) -> AcademicDocument:
    sections = [
        Section(
            title="Introdução",
            blocks=[
                Block(
                    type="paragraph",
                    content=[
                        TextSpan(text="Texto acadêmico de teste. " * 50),
                        Citation(keys=[citation_key]),
                    ],
                )
            ],
        )
    ]
    if conclusion:
        sections.append(
            Section(
                title="Conclusão",
                blocks=[
                    Block(
                        type="paragraph",
                        text="Conclusão substantiva. " * 50,
                    )
                ],
            )
        )
    return AcademicDocument(
        metadata=DocumentMetadata(
            titulo="Documento",
            tipo_documento="paper",
        ),
        sections=sections,
        bibliography=BibliographyInfo(entries_used=["silva2020", "souza2021"]),
    )


def test_quality_report_collects_citations_and_not_cited_entries() -> None:
    report = quality_report.build_quality_report(
        _document(conclusion=True),
        bib_keys=["silva2020"],
    )

    assert report["cited_keys"] == ["silva2020"]
    assert report["citations_total"] == 1
    assert report["bibliography_entries_not_cited"] == ["souza2021"]
    assert report["missing_in_bib"] == []


def test_quality_report_reports_missing_bib_key_and_conclusion() -> None:
    report = quality_report.build_quality_report(
        _document(conclusion=False, citation_key="ausente"),
        bib_keys=["silva2020"],
    )

    assert report["missing_in_bib"] == ["ausente"]
    assert any("citações sem chave" in warning for warning in report["warnings"])
    assert any("conclusão" in warning for warning in report["warnings"])


def test_quality_org_scan_ignores_nonvisible_technical_terms(
    tmp_path: Path,
) -> None:
    org = tmp_path / "doc.org"
    org.write_text(
        "#+BEGIN_COMMENT\npipeline fulltext_cache\n#+END_COMMENT\n"
        "#+LATEX_HEADER: pipeline\n"
        "* Introdução\nA democracia exige participação.\n",
        encoding="utf-8",
    )

    scan = quality_report._org_scan(org)

    assert scan["technical_terms_found"] == []


def test_quality_org_scan_detects_visible_pipeline_and_org_cite(
    tmp_path: Path,
) -> None:
    org = tmp_path / "doc.org"
    org.write_text(
        "* Introdução\nO pipeline gerou [cite:@silva2020].\n",
        encoding="utf-8",
    )

    scan = quality_report._org_scan(org)

    assert scan["contains_org_cite"] is True
    assert scan["technical_terms_found"] == ["pipeline"]


def test_quality_markdown_contains_sections_and_warnings() -> None:
    report = {
        "title": "Teste",
        "type": "paper",
        "total_words": 10,
        "citations_total": 1,
        "ok": False,
        "sections": [{"title": "Introdução", "words": 10, "blocks": 1}],
        "warnings": ["Alerta"],
        "bibliography_entries_not_cited": ["x"],
    }

    text = quality_report.render_quality_markdown(report)

    assert "| Introdução | 10 | 1 |" in text
    assert "- Alerta" in text
    assert "`x`" in text


def test_visible_technical_term_detection_avoids_democracia_false_positive() -> None:
    org = "* Texto\nA democracia é importante.\n"

    assert institution_compliance._find_technical_terms_visible(org) == []


def test_visible_technical_term_detection_ignores_org_headers() -> None:
    org = (
        "#+LATEX_HEADER: academic_pipeline fulltext_cache\n"
        "* Texto\nConteúdo válido.\n"
    )

    assert institution_compliance._find_technical_terms_visible(org) == []


def test_institution_rules_merge_general_and_document_specific(tmp_path: Path) -> None:
    profile = tmp_path / "institutions" / "fgv" / "profile.toml"
    validators = profile.parent / "validators"
    validators.mkdir(parents=True)
    profile.write_text("[instituicao]\nnome='FGV'\n", encoding="utf-8")
    (validators / "fgv_rules.toml").write_text(
        "[bibliografia]\nestilo_padrao='abnt'\nsistema_citacao='autor-data'\n",
        encoding="utf-8",
    )
    (validators / "paper_rules.toml").write_text(
        "[bibliografia]\nnotas_referencia=false\n[layout]\nfonte=12\n",
        encoding="utf-8",
    )
    cfg = {
        "__institution_profile_path__": str(profile),
        "documento": {"tipo_documento": "paper"},
    }

    rules = institution_compliance.load_institution_rules(cfg)

    assert rules["bibliografia"] == {
        "estilo_padrao": "abnt",
        "sistema_citacao": "autor-data",
        "notas_referencia": False,
    }
    assert rules["layout"]["fonte"] == 12


def test_compliance_report_marks_missing_references_as_failure(
    tmp_path: Path,
) -> None:
    org = tmp_path / "doc.org"
    org.write_text("* Introdução\nTexto sem bibliografia.\n", encoding="utf-8")
    cfg = {
        "__config_dir__": str(tmp_path),
        "__institution_profile_name__": "fgv",
        "documento": {
            "tipo_documento": "paper",
            "program_name": "",
            "course_name": "Curso",
        },
        "bibliografia": {"latex_style": "abnt"},
    }

    report = institution_compliance.run_institution_compliance(
        cfg,
        org_path=org,
    )

    assert report["ok"] is False
    assert any(item["id"] == "org.references" for item in report["errors"])


def test_compliance_markdown_groups_statuses() -> None:
    report = {
        "ok": False,
        "version": "v",
        "generated_at": "agora",
        "institution_profile": "fgv",
        "document_type": "paper",
        "artifacts": {"org": "doc.org"},
        "items": [
            {"id": "a", "status": "fail", "message": "Falhou", "detail": ""},
            {"id": "b", "status": "pass", "message": "Aprovou", "detail": "ok"},
        ],
    }

    text = institution_compliance.render_compliance_markdown(report)

    assert "Status geral: **ATENÇÃO**" in text
    assert "## Pendências críticas" in text
    assert "`a`: Falhou" in text
    assert "`b`: Aprovou — ok" in text


def test_write_compliance_reports_emits_json_and_markdown(tmp_path: Path) -> None:
    report = {
        "ok": True,
        "version": "v",
        "generated_at": "agora",
        "document_type": "paper",
        "artifacts": {},
        "items": [],
    }

    md, js = institution_compliance.write_compliance_reports(
        report,
        tmp_path / "resultado",
    )

    assert md.exists()
    assert js.exists()
    assert json.loads(js.read_text(encoding="utf-8"))["ok"] is True

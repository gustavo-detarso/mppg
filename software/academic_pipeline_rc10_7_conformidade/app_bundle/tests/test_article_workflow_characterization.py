from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from article_workflow import ArticleWorkflow, WorkflowState
from article_workflow.state import (
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_PENDING,
)


def _write_cfg(path: Path, *, research_dir: str = "output_pesquisa") -> None:
    path.write_text(
        '[projeto]\nnome="teste"\n'
        '[documento]\ntipo_documento="paper"\n'
        '[paths]\n'
        f'research_output_dir="{research_dir}"\n'
        'research_prefix="relatorio_prisma_teste"\n'
        'create_research_subdir=true\n',
        encoding="utf-8",
    )


def test_workflow_state_defaults_unlock_only_briefing(tmp_path: Path) -> None:
    state = WorkflowState(tmp_path / "artigo_state.json")

    assert state.status("briefing") == STATUS_PENDING
    assert state.status("toml_prisma") == STATUS_BLOCKED
    assert state.data["current_stage"] == "briefing"


def test_workflow_mark_persists_history_and_unlocks_next_stage(tmp_path: Path) -> None:
    path = tmp_path / "artigo_state.json"
    state = WorkflowState(path)

    state.mark("briefing", STATUS_OK, evidence=["briefing.txt"], message="Concluído")

    restored = WorkflowState(path)
    assert restored.status("briefing") == STATUS_OK
    assert restored.status("toml_prisma") == STATUS_PENDING
    assert restored.data["history"][-1]["stage"] == "briefing"
    assert restored.data["history"][-1]["evidence"] == ["briefing.txt"]


def test_workflow_corrupted_state_falls_back_to_default(tmp_path: Path) -> None:
    path = tmp_path / "artigo_state.json"
    path.write_text("{inválido", encoding="utf-8")

    state = WorkflowState(path)

    assert state.status("briefing") == STATUS_PENDING
    assert state.data["schema_version"] == 1


def test_workflow_can_run_reports_missing_dependency_label(tmp_path: Path) -> None:
    state = WorkflowState(tmp_path / "state.json")

    allowed, message = state.can_run("toml_prisma")

    assert not allowed
    assert "Briefing do artigo" in message


def test_workflow_rejects_unknown_stage(tmp_path: Path) -> None:
    state = WorkflowState(tmp_path / "state.json")

    with pytest.raises(KeyError, match="Etapa desconhecida"):
        state.mark("inexistente", STATUS_OK)


def test_article_workflow_guesses_most_recent_toml(tmp_path: Path) -> None:
    older = tmp_path / "antigo.toml"
    newer = tmp_path / "novo.toml"
    _write_cfg(older)
    _write_cfg(newer)

    # Fixamos mtimes diferentes para não depender da resolução temporal
    # do filesystem usado pelo pytest.
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))

    workflow = ArticleWorkflow(tmp_path)

    assert workflow.cfg_art == newer.resolve()
    assert workflow.prefix == "novo"


def test_research_output_dir_resolves_relative_to_prisma_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "prisma.toml"
    _write_cfg(cfg, research_dir="dados_saida")
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    result = workflow.research_output_dir()

    assert result == (tmp_path / "dados_saida" / "relatorio_prisma_teste").resolve()


def test_briefing_requires_minimum_content_length(tmp_path: Path) -> None:
    cfg = tmp_path / "artigo.toml"
    _write_cfg(cfg)
    briefing = tmp_path / "briefing_artigo.txt"
    briefing.write_text("curto", encoding="utf-8")
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    invalid = workflow.validate_briefing()
    briefing.write_text("Tema:\n" + "x" * 100, encoding="utf-8")
    valid = workflow.validate_briefing()

    assert not invalid.ok
    assert invalid.status == STATUS_PENDING
    assert valid.ok
    assert valid.status == STATUS_OK


def test_toml_validation_distinguishes_missing_and_invalid_configs(tmp_path: Path) -> None:
    missing = ArticleWorkflow(tmp_path, cfg_art=None, prisma_cfg=None).validate_toml_prisma()
    invalid_cfg = tmp_path / "invalido.toml"
    invalid_cfg.write_text('[projeto]\nnome="x"\n', encoding="utf-8")
    invalid = ArticleWorkflow(tmp_path, cfg_art=invalid_cfg, prisma_cfg=invalid_cfg).validate_toml_prisma()

    assert missing.status == STATUS_PENDING
    assert invalid.status == STATUS_ERROR
    assert "não parece" in invalid.message


def test_prisma_artifact_search_prioritizes_research_output(tmp_path: Path) -> None:
    cfg = tmp_path / "prisma.toml"
    _write_cfg(cfg)
    research = tmp_path / "output_pesquisa" / "relatorio_prisma_teste"
    research.mkdir(parents=True)
    dados = tmp_path / "dados_prisma"
    dados.mkdir()
    in_research = research / "a.busca_prisma_log.json"
    in_dados = dados / "b.busca_prisma_log.json"
    in_research.write_text("{}", encoding="utf-8")
    in_dados.write_text("{}", encoding="utf-8")
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    files = workflow.find_prisma_artifacts(["*.busca_prisma_log.json"])

    assert set(files) == {in_research, in_dados}


def test_human_review_transitions_from_blocked_to_confirmed(tmp_path: Path) -> None:
    cfg = tmp_path / "prisma.toml"
    _write_cfg(cfg)
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    blocked = workflow.validate_revisao_humana()
    xlsx = tmp_path / "dados_prisma" / "teste.curadoria_ia_referencias.xlsx"
    xlsx.parent.mkdir()
    xlsx.write_bytes(b"xlsx")
    pending = workflow.validate_revisao_humana()
    workflow.mark_human_review(xlsx)
    confirmed = workflow.validate_revisao_humana()

    assert blocked.status == STATUS_BLOCKED
    assert pending.status == STATUS_PENDING
    assert confirmed.ok
    assert confirmed.status == STATUS_OK
    assert confirmed.evidence == [str(xlsx.resolve())]


def test_org_validation_requires_abnt_and_printbibliography(tmp_path: Path) -> None:
    cfg = tmp_path / "artigo.toml"
    _write_cfg(cfg)
    out = tmp_path / "output"
    out.mkdir()
    (out / "artigo.bib").write_text("@article{x,title={X}}", encoding="utf-8")
    org = out / "artigo.org"
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    org.write_text("* Texto\n", encoding="utf-8")
    invalid = workflow.validate_artigo_org()
    org.write_text(
        "#+LATEX_HEADER: \\usepackage[backend=biber,style=abnt]{biblatex}\n"
        "\\printbibliography\n",
        encoding="utf-8",
    )
    valid = workflow.validate_artigo_org()

    assert invalid.status == STATUS_ERROR
    assert "style=abnt" in invalid.message
    assert valid.ok


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Defeito legado catalogado: refresh_from_files marca etapas posteriores "
        "como bloqueadas, mas WorkflowState.save/_normalize volta a etapa para "
        "pendente quando a dependência imediata está OK, ignorando falha anterior."
    ),
)
def test_refresh_from_files_should_keep_downstream_stages_blocked_after_first_failure(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "artigo.toml"
    _write_cfg(cfg)
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    results = workflow.refresh_from_files()

    assert results[0].key == "briefing"
    assert workflow.state.status("briefing") == STATUS_PENDING
    assert workflow.state.status("toml_prisma") == STATUS_OK
    assert workflow.state.status("prisma_preliminar") == STATUS_BLOCKED
    assert workflow.state.status("pdf_final") == STATUS_BLOCKED


def test_format_status_reports_recommended_next_action(tmp_path: Path) -> None:
    cfg = tmp_path / "artigo.toml"
    _write_cfg(cfg)
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    text = workflow.format_status()

    assert "Status estrutural do artigo PRISMA" in text
    assert "[PENDENTE]" in text
    assert "Próxima ação recomendada: Briefing do artigo" in text
    assert str(tmp_path / "artigo_state.json") in text


def test_pdf_validation_accepts_existing_file_when_pdftotext_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "artigo.toml"
    _write_cfg(cfg)
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)
    workflow.output_dir.mkdir()
    workflow.pdf_path.write_bytes(b"%PDF-1.4\n% fake\n")
    monkeypatch.setattr("article_workflow.validators.shutil.which", lambda _name: None)

    result = workflow.validate_pdf_final()

    assert result.ok
    assert result.status == STATUS_OK


def test_mark_stage_ok_records_evidence_and_message(tmp_path: Path) -> None:
    cfg = tmp_path / "artigo.toml"
    _write_cfg(cfg)
    workflow = ArticleWorkflow(tmp_path, cfg_art=cfg, prisma_cfg=cfg)

    workflow.mark_stage_ok("briefing", evidence=["a.txt"], message="Aceito")

    record = workflow.state.record("briefing")
    assert record.status == STATUS_OK
    assert record.evidence == ["a.txt"]
    assert record.message == "Aceito"

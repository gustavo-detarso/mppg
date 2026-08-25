from __future__ import annotations

import tomllib
from pathlib import Path

from academic_pipeline.list_profiles_runtime import PRESETS

ROOT = Path(__file__).resolve().parents[2]


def test_public_fichamento_profile_is_exposed_once():
    matches = [preset for preset in PRESETS if preset.key == "fichamento_fgv"]
    assert len(matches) == 1
    assert matches[0].document_type == "atividade"
    assert matches[0].executar_documento is True


def test_institution_profile_registers_fichamento_content_type_and_layout():
    profile = tomllib.loads(
        (ROOT / "app_bundle/institutions/fgv/institution_profile.toml").read_text(encoding="utf-8")
    )
    spec = profile["document_content_types"]["fichamento"]
    assert spec["default_layout"] == "fichamento_qualitativo"
    assert spec["prompt"] == "app://prompts/document/fichamento.txt"
    assert spec["documento"]["tipo_conteudo"] == "fichamento"
    assert spec["documento"]["genero_academico"] == "atividade"
    assert spec["documento"]["layout"] == "fichamento_qualitativo"
    layout = profile["layouts"]["fichamento_qualitativo"]
    assert layout["genero_academico"] == "atividade"
    assert layout["front_matter"] == "fichamento_qualitativo"
    assert layout["classe_latex"] == "fgv-paper"


def test_generator_contains_single_toml_authority_for_corpus_mode_and_layout():
    source = (ROOT / "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py").read_text(encoding="utf-8")
    assert 'key="fichamento_fgv"' in source
    assert '"corpus_externo"' in source
    assert '"corpus_hibrido"' in source
    assert 'lines.append("[selecao_corpus]")' in source
    assert "fichamento.modo_corpus" not in source
    assert 'data["layout"] = str(data.get("layout") or "fichamento_qualitativo")' in source


def test_runtime_dispatches_document_corpus_through_generic_bridge():
    source = (ROOT / "academic_pipeline/default_runtime.py").read_text(encoding="utf-8")
    assert "from academic_pipeline.external_corpus_orchestration import resolve_document_corpus" in source
    assert "docs, source_info = resolve_document_corpus(" in source


def test_generic_external_corpus_module_does_not_own_document_generation():
    source = (ROOT / "academic_pipeline/external_corpus_orchestration.py").read_text(encoding="utf-8")
    assert "run_external_prisma_search" in source
    assert "try_download_candidate" in source
    assert "discover_local_documents" in source
    assert "build_document_model" not in source
    assert "render_docx" not in source

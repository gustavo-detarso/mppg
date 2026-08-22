from __future__ import annotations
import tomllib
from pathlib import Path
from academic_pipeline.list_profiles_runtime import PRESETS
ROOT=Path(__file__).resolve().parents[2]
def test_public_fichamento_profile_is_exposed_once():
    matches=[p for p in PRESETS if p.key=="fichamento_fgv"]; assert len(matches)==1; assert matches[0].document_type=="atividade" and matches[0].executar_documento is True
def test_institution_profile_registers_fichamento_content_type():
    profile=tomllib.loads((ROOT/"app_bundle/institutions/fgv/institution_profile.toml").read_text(encoding="utf-8")); spec=profile["document_content_types"]["fichamento"]; assert spec["default_layout"]=="atividade_fgv"; assert spec["prompt"]=="app://prompts/document/fichamento.txt"
def test_generator_contains_single_toml_authority_for_corpus_mode():
    source=(ROOT/"app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py").read_text(encoding="utf-8"); assert 'key="fichamento_fgv"' in source; assert '"corpus_externo"' in source; assert '"corpus_hibrido"' in source; assert 'lines.append("[selecao_corpus]")' in source; assert "fichamento.modo_corpus" not in source
def test_runtime_dispatches_document_corpus_through_generic_bridge():
    source=(ROOT/"academic_pipeline/default_runtime.py").read_text(encoding="utf-8"); assert "from academic_pipeline.external_corpus_orchestration import resolve_document_corpus" in source; assert "docs, source_info = resolve_document_corpus(" in source
def test_generic_external_corpus_module_does_not_own_document_generation():
    source=(ROOT/"academic_pipeline/external_corpus_orchestration.py").read_text(encoding="utf-8"); assert "run_external_prisma_search" in source; assert "try_download_candidate" in source; assert "discover_local_documents" in source; assert "build_document_model" not in source and "render_docx" not in source

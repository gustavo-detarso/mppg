#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes mínimos de fumaça para rc10.3.

Executar a partir da raiz do app_bundle:
  python -m pytest app_bundle/tests
ou diretamente:
  python app_bundle/tests/test_rc10_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "scripts" / "pipeline"
sys.path.insert(0, str(PIPE))

from diagnostics import check_config, run_doctor  # noqa: E402
from document_model import AcademicDocument, DocumentMetadata, BibliographyInfo, Section, Block, TextSpan  # noqa: E402
from render_org_latex import render_org_latex  # noqa: E402
from document_validator import validate_org_text  # noqa: E402
from institution_profiles import apply_institution_profile, available_institution_profiles  # noqa: E402


def sample_document() -> AcademicDocument:
    return AcademicDocument(
        metadata=DocumentMetadata(
            tipo_documento="paper",
            titulo="Título de Teste",
            autor="Autor de Teste",
            instituicao="Fundação Getúlio Vargas",
            curso="Mestrado Acadêmico em Políticas Públicas e Governo",
            disciplina="Disciplina de Teste",
            professor="Professor de Teste",
        ),
        sections=[Section(id="intro", level=1, title="Introdução", blocks=[Block(type="paragraph", content=[TextSpan(text="Texto sem problemas.")])])],
        bibliography=BibliographyInfo(bib_path="teste.bib", entries_used=[]),
    )


def test_doctor_returns_dict():
    report = run_doctor({"__config_dir__": str(ROOT)})
    assert "checks" in report
    assert "version" in report


def test_check_config_detects_duplicate_program_course():
    cfg = {
        "__config_dir__": str(ROOT),
        "documento": {
            "tipo_documento": "paper",
            "program_name": "Mestrado Acadêmico em Políticas Públicas e Governo",
            "course_name": "Mestrado Acadêmico em Políticas Públicas e Governo",
            "output_dir": str(ROOT / "_tmp"),
        },
        "documentos_locais": {},
    }
    report = check_config(cfg)
    assert not report["ok"]
    assert any("program_name" in e for e in report["errors"])


def test_render_org_without_empty_citation(tmp_path: Path):
    doc = sample_document()
    org_path = tmp_path / "teste.org"
    org = render_org_latex(doc, org_path, "teste.bib", cfg={"documento": {"tipo_documento": "paper"}, "bibliografia": {"latex_style": "apa"}})
    assert "<empty citation>" not in org
    assert "[cite:" not in org
    assert not validate_org_text(org, [])


if __name__ == "__main__":
    test_doctor_returns_dict()
    test_check_config_detects_duplicate_program_course()
    import tempfile
    test_render_org_without_empty_citation(Path(tempfile.mkdtemp()))
    print("OK")

# rc10.4 smoke checks (executar manualmente via python, sem pytest obrigatório)
def _rc10_4_imports():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'scripts' / 'pipeline'
    sys.path.insert(0, str(root))
    from project_tools import make_doi_manifest, inspect_bib  # noqa
    from quality_report import build_quality_report  # noqa
    return True

if __name__ == '__main__':
    try:
        assert _rc10_4_imports()
        print('rc10.6 imports: OK')
    except Exception as exc:
        print(f'rc10.4 imports: ERRO: {exc}')
        raise


def test_institution_profile_fgv_loads():
    cfg = {
        "__config_dir__": str(ROOT),
        "instituicao": {"perfil": "fgv"},
        "documento": {"tipo_documento": "paper"},
    }
    applied = apply_institution_profile(cfg)
    assert applied["__institution_profile_name__"] == "fgv"
    assert applied["latex"]["latex_extra_path"].endswith("institutions/fgv/latex")
    assert "fgv" in available_institution_profiles(ROOT)


def test_prompt_manager_loads_and_sanitizes():
    from prompt_manager import load_prompt_bundle
    cfg = {
        "__config_dir__": str(ROOT / "config" / "examples"),
        "prompts": {
            "ativos": True,
            "global_paths": ["../../prompts/global/orientacao_geral_execucao.txt"],
            "paper_paths": ["../../prompts/document/paper.txt"],
        },
        "documento": {"tipo_documento": "paper"},
    }
    bundle = load_prompt_bundle(cfg, "document", document_type="paper")
    assert bundle.sources
    assert "Chain of Thought" not in bundle.text
    assert "cadeia de pensamento" not in bundle.text.lower()


def test_rc10_7_compliance_and_prompt_lock_imports(tmp_path: Path):
    from institution_compliance import run_institution_compliance
    from institution_explainer import explain_profile
    from prompt_lock import build_prompt_lock
    cfg = {
        "__config_dir__": str(ROOT),
        "instituicao": {"perfil": "fgv"},
        "documento": {"tipo_documento": "paper", "output_dir": str(tmp_path), "prefixo": "teste"},
    }
    cfg = apply_institution_profile(cfg)
    assert "Perfil institucional: fgv" in explain_profile("fgv")
    lock = build_prompt_lock(cfg)
    assert "prompts" in lock
    report = run_institution_compliance(cfg)
    assert "items" in report

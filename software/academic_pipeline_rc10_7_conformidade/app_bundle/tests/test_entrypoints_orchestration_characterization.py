from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import academic_pipeline_rc10 as rc10
from document_model import AcademicDocument, Block, DocumentMetadata, Section


PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"


@pytest.mark.parametrize(
    ("script", "fragment"),
    [
        ("academic_pipeline_rc10.py", "document_model canônico"),
        ("academic_pipeline_toml_generator_interativo.py", "Gerador interativo completo"),
        ("academic_pipeline_tui.py", "Central operacional visual"),
        ("academic_pipeline_gui.py", "Interface gráfica FGV"),
        ("artigo_prisma_workflow.py", "Estado e validação"),
        ("gerar_artigo_final_unificado.py", "Gera artigo final"),
        ("render_docx_canonico.py", "DOCX ABNT/FGV canônico"),
    ],
)
def test_entrypoint_help_is_offline_and_successful(script: str, fragment: str) -> None:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(PIPELINE)
    proc = subprocess.run(
        [sys.executable, str(PIPELINE / script), "--help"],
        text=True,
        capture_output=True,
        env=env,
        timeout=40,
    )

    assert proc.returncode == 0
    assert fragment in (proc.stdout + proc.stderr)


def test_rc10_source_preserves_two_top_level_main_definitions() -> None:
    source = Path(rc10.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]

    assert len(mains) == 2
    assert mains[0].args.vararg is None
    assert mains[1].args.vararg is not None
    assert mains[1].args.kwarg is not None


def test_stage_prints_flushed_execution_marker(capsys: pytest.CaptureFixture[str]) -> None:
    rc10.stage("Validando contrato")

    assert capsys.readouterr().out == "[ETAPA] Validando contrato\n"


def test_make_client_requires_api_key_without_loading_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rc10, "load_dotenv", lambda **_kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=lambda **_kwargs: object()),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        rc10.make_client()


def test_prisma_generic_strip_removes_boolean_value_and_equals_flags() -> None:
    argv = [
        "--config", "cfg.toml",
        "--prisma-exportar-bib",
        "--prisma-artigo-dir", "artigo",
        "--prisma-csl-path=estilo.csl",
        "--doctor",
    ]

    assert rc10._prisma_artigo_generico_strip(argv) == [
        "--config", "cfg.toml", "--doctor"
    ]


def test_prisma_generic_out_dir_uses_config_parent(tmp_path: Path) -> None:
    cfg = tmp_path / "projeto.toml"
    cfg.write_text("[projeto]\nnome='x'\n", encoding="utf-8")

    result = rc10._prisma_artigo_generico_out_dir(["--config", str(cfg)])

    assert result == (
        cfg.resolve().parent
        / "output_pesquisa"
        / "relatorio_prisma_projeto"
    )


def test_prisma_curadoria_build_cmd_preserves_optional_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "curadoria.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(rc10, "_prisma_curadoria_script_path", lambda: script)
    monkeypatch.setattr(rc10, "_prisma_curadoria_config_from_args", lambda _a: "cfg.toml")
    monkeypatch.setattr(rc10, "_prisma_curadoria_out_from_args", lambda _a: "saida")
    monkeypatch.setattr(rc10, "_prisma_curadoria_prompt_from_args", lambda _a: "prompt.yml")
    monkeypatch.setattr(rc10, "_prisma_curadoria_input_from_args", lambda _a, **_k: "entrada.xlsx")
    args = SimpleNamespace(
        prisma_curadoria_max_incluir=20,
        prisma_curadoria_top_n_candidatos=55,
        prisma_curadoria_limiar_minimo=70,
    )

    cmd = rc10._prisma_curadoria_build_cmd(args, usar_ia=True)

    assert cmd[:5] == [
        sys.executable, str(script), "--config", "cfg.toml", "--out-dir"
    ]
    assert cmd[5] == "saida"
    assert ["--prompt-curadoria", "prompt.yml"] == cmd[6:8]
    assert "--usar-ia" in cmd
    assert cmd[-6:] == [
        "--max-incluir", "20",
        "--top-n-candidatos", "55",
        "--limiar-minimo-inclusao", "70",
    ]


@pytest.mark.parametrize(
    ("flag", "handler", "expected"),
    [
        ("prisma_curadoria_menu", "_prisma_curadoria_menu", 11),
        ("prisma_curadoria_reexportar_xlsx", "_prisma_curadoria_reexportar_xlsx", 12),
        ("prisma_curadoria_fluxo_completo", "_prisma_curadoria_fluxo_completo", 13),
        ("prisma_curadoria_importar", "_prisma_curadoria_importar_no_pipeline", 14),
    ],
)
def test_prisma_curadoria_dispatch_routes_by_priority(
    flag: str,
    handler: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        prisma_curadoria_menu=False,
        prisma_curadoria_reexportar_xlsx=False,
        prisma_curadoria_fluxo_completo=False,
        prisma_curadoria_importar=False,
        prisma_curadoria_ia=False,
        prisma_curadoria_sem_ia=False,
    )
    setattr(args, flag, True)
    monkeypatch.setattr(rc10, handler, lambda _a: expected)

    assert rc10._prisma_curadoria_dispatch(args) == expected


def test_prisma_curadoria_dispatch_passes_sem_ia_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        prisma_curadoria_menu=False,
        prisma_curadoria_reexportar_xlsx=False,
        prisma_curadoria_fluxo_completo=False,
        prisma_curadoria_importar=False,
        prisma_curadoria_ia=True,
        prisma_curadoria_sem_ia=True,
    )
    seen: list[bool] = []
    monkeypatch.setattr(
        rc10,
        "_prisma_curadoria_run_ia",
        lambda _a, *, usar_ia=True: seen.append(usar_ia) or 7,
    )

    assert rc10._prisma_curadoria_dispatch(args) == 7
    assert seen == [False]


def test_recompile_orchestrates_cleanup_compile_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org = tmp_path / "artigo.org"
    org.write_text("* Texto\n", encoding="utf-8")
    pdf = tmp_path / "artigo.pdf"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        rc10,
        "_resolve_latex_paths_for_recompile",
        lambda _args, _cfg: (None, None, "lualatex"),
    )
    monkeypatch.setattr(rc10, "clean_aux_files", lambda _org: ["antigo.aux"])
    monkeypatch.setattr(
        rc10,
        "run_compile_sequence",
        lambda *_a, **_k: pdf,
    )
    monkeypatch.setattr(
        rc10,
        "write_outputs_manifest",
        lambda path, outputs: captured.update(manifest_path=path, outputs=outputs),
    )
    monkeypatch.setattr(
        rc10,
        "make_run_report",
        lambda **kwargs: {"mode": kwargs["extra"]["mode"]},
    )
    monkeypatch.setattr(
        rc10,
        "write_json",
        lambda path, report: captured.update(report_path=path, report=report),
    )
    monkeypatch.setattr(rc10, "print_outputs", lambda *_a, **_k: None)
    args = SimpleNamespace(
        org=str(org),
        no_clean=False,
        academic_writing="",
        latex_extra_path="",
        pdf_engine="",
    )

    assert rc10.run_recompile(args, None) == 0
    assert captured["outputs"] == {
        "org": str(org.resolve()),
        "pdf": str(pdf),
        "removed_aux": ["antigo.aux"],
    }
    assert captured["report"] == {"mode": "recompile"}


def test_external_prisma_renderer_can_disable_all_outputs(tmp_path: Path) -> None:
    cfg = {"relatorio_pesquisa": {"exportar_org": False, "exportar_pdf": False}}

    assert rc10.render_external_prisma_outputs(
        cfg, tmp_path, "relatorio", {}, phase="preliminar"
    ) == (None, None)


def test_external_prisma_renderer_compiles_requested_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org = tmp_path / "relatorio.org"
    pdf = tmp_path / "relatorio.pdf"
    monkeypatch.setattr(
        rc10,
        "render_external_prisma_org_report",
        lambda *_a, **_k: org,
    )
    monkeypatch.setattr(
        rc10,
        "run_compile_sequence",
        lambda path, **kwargs: pdf if path == org and kwargs["pdf_engine"] == "xelatex" else None,
    )
    monkeypatch.setitem(
        sys.modules,
        "prisma_diagrama_fluxo",
        types.SimpleNamespace(ensure_prisma_flow_diagram=lambda **_k: None),
    )
    cfg = {
        "__config_dir__": str(tmp_path),
        "relatorio_pesquisa": {"exportar_org": True, "exportar_pdf": True},
        "latex": {"pdf_engine": "xelatex"},
    }

    assert rc10.render_external_prisma_outputs(
        cfg, tmp_path, "relatorio", {}, phase="final"
    ) == (org, pdf)


def _minimal_document() -> AcademicDocument:
    return AcademicDocument(
        metadata=DocumentMetadata(titulo="Documento"),
        sections=[Section(title="Introdução", blocks=[Block(type="paragraph", text="Texto.")])],
    )


def test_additional_language_orchestration_writes_isolated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _minimal_document()
    translated = _minimal_document()
    translated.metadata.titulo = "Document"
    bib = tmp_path / "refs.bib"
    bib.write_text("@article{x,title={X}}", encoding="utf-8")

    monkeypatch.setattr(rc10, "requested_translation_languages", lambda _cfg: [("en", "English")])
    monkeypatch.setattr(rc10, "translation_batch_size", lambda _cfg: 3333)
    monkeypatch.setattr(
        rc10,
        "translate_document_model",
        lambda *_a, **_k: (translated, {"campos_traduzidos": 2}),
    )

    def fake_org(_doc, path, _bib_name, **_kwargs):
        path.write_text("* Introduction\nText.\n", encoding="utf-8")
        return path.read_text(encoding="utf-8")

    monkeypatch.setattr(rc10, "render_org_latex", fake_org)
    monkeypatch.setattr(rc10, "validate_org_text", lambda *_a, **_k: [])
    monkeypatch.setattr(rc10, "raise_if_errors", lambda *_a, **_k: None)
    monkeypatch.setattr(rc10, "build_quality_report", lambda *_a, **_k: {"warnings": []})
    monkeypatch.setattr(
        rc10,
        "write_quality_report",
        lambda _report, path: path.write_text("# Quality\n", encoding="utf-8"),
    )

    result, warnings = rc10.render_additional_language_versions(
        client=object(),
        model="modelo",
        cfg={},
        document=document,
        bib_path=bib,
        bib_keys=["x"],
        out_dir=tmp_path,
        prefix="paper",
        doc_cfg={"exportar_pdf": False, "exportar_docx": False},
        latex_cfg={},
        config_dir=tmp_path,
    )

    language = result["en"]
    assert warnings == []
    assert Path(language["document_json"]).exists()
    assert Path(language["translation_audit"]).exists()
    assert Path(language["org"]).exists()
    assert Path(language["bib"]).read_text(encoding="utf-8") == bib.read_text(encoding="utf-8")
    assert language["pdf"] is None
    assert language["docx"] is None

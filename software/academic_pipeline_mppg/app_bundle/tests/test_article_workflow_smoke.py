from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = PROJECT_ROOT / "app_bundle" / "scripts" / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from article_workflow import ArticleWorkflow


def test_article_workflow_detects_minimal_artifacts(tmp_path):
    art = tmp_path / "artigo"
    out = art / "output"
    dados = art / "dados_prisma"
    out.mkdir(parents=True)
    dados.mkdir(parents=True)

    (art / "briefing_artigo.txt").write_text("Tema:\n" + "x" * 100, encoding="utf-8")
    cfg = art / "artigo_final.toml"
    cfg.write_text('[projeto]\nnome="teste"\n[documento]\ntipo_documento="paper"\n', encoding="utf-8")
    (dados / "teste.busca_prisma_log.json").write_text("{}", encoding="utf-8")
    (dados / "teste.curadoria_ia_referencias.xlsx").write_bytes(b"fake-xlsx")
    (dados / "teste.referencias_incluidas.bib").write_text("@article{x, title={X}}", encoding="utf-8")
    (dados / "artigo_longo_fulltext").mkdir()
    (dados / "artigo_longo_fulltext" / "corpus_fulltext_compilado.md").write_text("corpus", encoding="utf-8")
    (out / "artigo_final.org").write_text(
        "#+LATEX_HEADER: \\usepackage[backend=biber,style=abnt]{biblatex}\n\\printbibliography\n",
        encoding="utf-8",
    )
    (out / "artigo_final.bib").write_text("@article{x, title={X}}", encoding="utf-8")
    (out / "artigo_final.pdf").write_bytes(b"%PDF-1.4\n% fake\n")

    wf = ArticleWorkflow(art, cfg_art=cfg, prisma_cfg=cfg)
    wf.mark_human_review(dados / "teste.curadoria_ia_referencias.xlsx")
    results = {item.key: item for item in wf.validations()}

    assert results["briefing"].ok
    assert results["toml_prisma"].ok
    assert results["prisma_preliminar"].ok
    assert results["xlsx_cut"].ok
    assert results["revisao_humana_xlsx"].ok
    assert results["prisma_final"].ok
    assert results["fulltext"].ok
    assert results["artigo_org"].ok

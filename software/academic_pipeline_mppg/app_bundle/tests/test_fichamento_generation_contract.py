from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
from corpus_manager import SourceDoc
from document_builder import FICHAMENTO_SECTION_TITLES, _is_fichamento_profile, _validate_fichamento_corpus_contract, _validate_fichamento_document_contract
import academic_pipeline.external_corpus_orchestration as external

def _cfg(mode="documentos_locais"):
    return {"projeto":{"preset":"fichamento_fgv"},"documento":{"tipo_documento":"atividade","tipo_conteudo":"fichamento"},"pipeline":{"modo_entrada":mode},"selecao_corpus":{"quantidade_minima_textos":3,"quantidade_alvo_textos":3,"quantidade_maxima_textos":4,"revisao_humana":False,"min_caracteres_texto_substantivo":100},"pesquisa":{"tema":"Tema","recorte":"Recorte"},"busca_prisma":{}}
def _doc(label,text="conteúdo substantivo "*20): return SourceDoc(path=label,kind="documento_base",label=label,extracted_text=text)
def test_fichamento_profile_detection_and_minimum_gate():
    cfg=_cfg(); assert _is_fichamento_profile(cfg)
    with pytest.raises(RuntimeError,match="ao menos 3 textos"): _validate_fichamento_corpus_contract(cfg,[_doc("a"),_doc("b")])
    _validate_fichamento_corpus_contract(cfg,[_doc("a"),_doc("b"),_doc("c")])
def test_fichamento_requires_exact_seven_section_order():
    valid=SimpleNamespace(sections=[SimpleNamespace(title=t) for t in FICHAMENTO_SECTION_TITLES]); _validate_fichamento_document_contract(valid)
    invalid=SimpleNamespace(sections=[SimpleNamespace(title=t) for t in reversed(FICHAMENTO_SECTION_TITLES)])
    with pytest.raises(RuntimeError,match="sete seções canônicas"): _validate_fichamento_document_contract(invalid)
def test_prompt_contains_assignment_and_natural_writing_contract():
    text=(Path(__file__).resolve().parents[1]/"prompts/document/fichamento.txt").read_text(encoding="utf-8")
    for title in FICHAMENTO_SECTION_TITLES: assert title in text
    assert "prosa acadêmica natural" in text and "Não invente autores" in text and "detectores de IA" in text
class _Candidate:
    def __init__(self,idx):
        self.pool_id=idx; self.source_csv="triagem.csv"; self.source_row_number=idx+1; self.title=f"Artigo {idx}"; self.authors="Autor"; self.year="2026"; self.journal="Revista"; self.doi=f"10.1000/{idx}"; self.url=f"https://example.invalid/{idx}"; self.abstract="Resumo"; self.score_aderencia=100-idx; self.prioridade_manual=False
def test_external_corpus_counts_only_downloaded_and_validated_fulltexts(tmp_path,monkeypatch):
    cfg=_cfg("corpus_externo"); cfg["__config_dir__"]=str(tmp_path); cfg["paths"]={"research_output_dir":str(tmp_path/"research"),"research_prefix":"fichamento"}
    def fake_search(cfg,out_dir,prefix,*,progress=None,client=None,model=None):
        out_dir.mkdir(parents=True,exist_ok=True); triage=out_dir/f"{prefix}.triagem_titulo_resumo.csv"; triage.write_text("titulo,doi,incluir_final\n",encoding="utf-8"); return {"artefatos":{"planilha_triagem_csv":str(triage)}}
    monkeypatch.setattr(external,"run_external_prisma_search",fake_search); monkeypatch.setattr(external,"load_candidates",lambda *a,**k:[_Candidate(i) for i in range(1,5)])
    def fake_download(candidate,fulltext_dir,**kwargs):
        fulltext_dir.mkdir(parents=True,exist_ok=True); pdf=fulltext_dir/f"{candidate.pool_id}.pdf"; pdf.write_bytes(f"%PDF-fake-{candidate.pool_id}".encode("ascii")); return True,{"pdf_path":str(pdf),"pdf_sha256":f"sha-{candidate.pool_id}","download_status":"pdf_original_baixado","download_note":"download ok","pdf_url":f"https://example.invalid/{candidate.pool_id}.pdf","source":"fixture"}
    monkeypatch.setattr(external,"try_download_candidate",fake_download); monkeypatch.setattr(external,"read_text_file_with_diagnostics",lambda path,max_chars=60000:("texto substantivo "*100,[]))
    docs,info=external.resolve_document_corpus(cfg,tmp_path/"work",client=object(),model="fixture"); assert len(docs)==3; assert info["external"]["validated_fulltexts"]==3
def test_human_review_blocks_before_download_when_selection_is_pending(tmp_path,monkeypatch):
    cfg=_cfg("corpus_externo"); cfg["selecao_corpus"]["revisao_humana"]=True; cfg["__config_dir__"]=str(tmp_path); cfg["paths"]={"research_output_dir":str(tmp_path/"research"),"research_prefix":"fichamento"}
    def fake_search(cfg,out_dir,prefix,*,progress=None,client=None,model=None):
        out_dir.mkdir(parents=True,exist_ok=True); triage=out_dir/f"{prefix}.triagem_titulo_resumo.csv"; triage.write_text("titulo,doi,incluir_final\nArtigo 1,10.1000/1,PENDENTE\n",encoding="utf-8"); return {"artefatos":{"planilha_triagem_csv":str(triage)}}
    monkeypatch.setattr(external,"run_external_prisma_search",fake_search); monkeypatch.setattr(external,"load_candidates",lambda *a,**k:[_Candidate(1)])
    with pytest.raises(RuntimeError,match="revisão humana"): external.resolve_document_corpus(cfg,tmp_path/"work",client=object(),model="fixture")

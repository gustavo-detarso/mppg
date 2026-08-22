# Generic external full-text corpus orchestration.
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from app_bundle.scripts.pipeline.corpus_manager import SourceDoc, discover_local_documents, read_text_file_with_diagnostics
from app_bundle.scripts.pipeline.prisma_busca_externa import run_external_prisma_search
from app_bundle.scripts.pipeline.prisma_fulltext_garantido import load_candidates, try_download_candidate

Progress = Callable[[str], None] | None

def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value=cfg.get(name,{})
    return value if isinstance(value,dict) else {}

def _config_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("__config_dir__") or Path.cwd())).expanduser().resolve()

def _resolve_config_path(cfg: dict[str, Any], raw: Any, fallback: str) -> Path:
    p=Path(str(raw or fallback).strip()).expanduser()
    if not p.is_absolute(): p=_config_dir(cfg)/p
    return p.resolve()

def _positive_int(value: Any, name: str) -> int:
    try: parsed=int(value)
    except (TypeError,ValueError) as exc: raise RuntimeError(f"{name} deve ser inteiro positivo.") from exc
    if parsed<1: raise RuntimeError(f"{name} deve ser inteiro positivo.")
    return parsed

def _selection_contract(cfg: dict[str, Any]) -> tuple[int,int,int,bool,int]:
    s=_section(cfg,"selecao_corpus")
    minimum=_positive_int(s.get("quantidade_minima_textos",3),"quantidade_minima_textos")
    target=_positive_int(s.get("quantidade_alvo_textos",minimum),"quantidade_alvo_textos")
    maximum=_positive_int(s.get("quantidade_maxima_textos",target),"quantidade_maxima_textos")
    if not minimum<=target<=maximum: raise RuntimeError("[selecao_corpus] deve respeitar quantidade_minima_textos <= quantidade_alvo_textos <= quantidade_maxima_textos.")
    human=bool(s.get("revisao_humana",True)); min_chars=_positive_int(s.get("min_caracteres_texto_substantivo",800),"min_caracteres_texto_substantivo")
    return minimum,target,maximum,human,min_chars

def _norm(value: Any) -> str: return re.sub(r"\s+"," ",str(value or "").strip().casefold())
def _norm_doi(value: Any) -> str:
    text=_norm(value).removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:")
    return text.strip()

def _selected_rows(path: Path) -> list[dict[str,str]]:
    if not path.is_file(): return []
    with path.open("r",encoding="utf-8-sig",newline="") as fh: return [dict(row) for row in csv.DictReader(fh)]

def _human_selected_identity(rows: list[dict[str,str]]) -> tuple[set[str],set[str]]:
    dois=set(); titles=set()
    for row in rows:
        if _norm(row.get("incluir_final")) not in {"incluir","incluido","incluído","sim","yes","1","true"}: continue
        doi=_norm_doi(row.get("doi")); title=_norm(row.get("titulo") or row.get("título") or row.get("title"))
        if doi: dois.add(doi)
        if title: titles.add(title)
    return dois,titles

def _candidate_selected(candidate: Any, dois: set[str], titles: set[str]) -> bool:
    doi=_norm_doi(getattr(candidate,"doi","")); title=_norm(getattr(candidate,"title",""))
    return (bool(doi) and doi in dois) or (bool(title) and title in titles)

def _sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _source_doc_from_download(candidate: Any, download: dict[str,Any], *, min_chars: int) -> SourceDoc|None:
    raw=str(download.get("pdf_path") or "").strip()
    if not raw: return None
    pdf=Path(raw).expanduser().resolve()
    if not pdf.is_file(): return None
    text,warnings=read_text_file_with_diagnostics(pdf,max_chars=60000)
    if len(str(text or "").strip())<min_chars: return None
    metadata={"external_corpus":{"title":getattr(candidate,"title",""),"authors":getattr(candidate,"authors",""),"year":getattr(candidate,"year",""),"journal":getattr(candidate,"journal",""),"doi":getattr(candidate,"doi",""),"landing_url":getattr(candidate,"url",""),"score_aderencia":getattr(candidate,"score_aderencia",""),"source_csv":getattr(candidate,"source_csv",""),"source_row_number":getattr(candidate,"source_row_number",""),"download_source":download.get("source") or download.get("collector") or "","download_url":download.get("pdf_url") or "","download_status":download.get("download_status") or "","pdf_sha256":download.get("pdf_sha256") or _sha256_file(pdf),"local_path":str(pdf)}}
    if warnings: metadata.update({"warnings":list(warnings),"warning_count":len(warnings)})
    return SourceDoc(path=str(pdf),kind="documento_base",label=pdf.name,extracted_text=text,metadata=metadata)

def _doc_identity(doc: SourceDoc) -> tuple[str,str]:
    p=Path(str(doc.path)).expanduser(); h=""
    if p.is_file():
        try:h=_sha256_file(p)
        except OSError:h=""
    return h,_norm(doc.label)

def _dedupe_docs(docs: list[SourceDoc]) -> list[SourceDoc]:
    hashes=set(); labels=set(); out=[]
    for doc in docs:
        h,label=_doc_identity(doc)
        if h and h in hashes: continue
        if label and label in labels: continue
        if h: hashes.add(h)
        if label: labels.add(label)
        out.append(doc)
    return out

def _external_dirs(cfg: dict[str,Any]) -> tuple[Path,str,Path]:
    paths=_section(cfg,"paths"); research=_resolve_config_path(cfg,paths.get("research_output_dir"),"output/pesquisa"); prefix=str(paths.get("research_prefix") or "external_corpus").strip() or "external_corpus"; root=research/"external_corpus"
    return root,prefix,root/"fulltext_garantido"/"pdfs_originais"

def _ensure_search(cfg: dict[str,Any], search_dir: Path, prefix: str, *, progress: Progress, client: Any, model: str) -> Path:
    search_dir.mkdir(parents=True,exist_ok=True); triage=search_dir/f"{prefix}.triagem_titulo_resumo.csv"
    if not triage.is_file():
        result=run_external_prisma_search(cfg,search_dir,prefix,progress=progress,client=client,model=model)
        artifact=str((result.get("artefatos",{}) if isinstance(result,dict) else {}).get("planilha_triagem_csv") or "")
        if artifact: triage=Path(artifact).expanduser().resolve()
    if not triage.is_file(): raise RuntimeError("A busca externa não produziu a planilha de triagem esperada.")
    return triage

def _external_docs(cfg: dict[str,Any], *, progress: Progress, client: Any, model: str) -> tuple[list[SourceDoc],dict[str,Any]]:
    minimum,target,maximum,human_review,min_chars=_selection_contract(cfg)
    search_dir,prefix,fulltext_dir=_external_dirs(cfg); triage_path=_ensure_search(cfg,search_dir,prefix,progress=progress,client=client,model=model)
    context=json.dumps({"pesquisa":_section(cfg,"pesquisa"),"busca_prisma":_section(cfg,"busca_prisma")},ensure_ascii=False,sort_keys=True)
    candidates=load_candidates(search_dir,[],context,[])
    if not candidates: raise RuntimeError("A busca externa não produziu candidatos bibliográficos utilizáveis.")
    if human_review:
        dois,titles=_human_selected_identity(_selected_rows(triage_path))
        if not dois and not titles: raise RuntimeError(f"A revisão humana do corpus externo está pendente. Revise a planilha {triage_path}, marque incluir_final=INCLUIR para os textos escolhidos e execute novamente.")
        candidates=[c for c in candidates if _candidate_selected(c,dois,titles)]
        if len(candidates)<minimum: raise RuntimeError(f"A triagem humana selecionou {len(candidates)} registro(s), abaixo do mínimo {minimum}.")
    fulltext_dir.mkdir(parents=True,exist_ok=True); search_cfg=_section(cfg,"busca_prisma")
    email=str(search_cfg.get("email_contato") or "").strip() or os.environ.get("UNPAYWALL_EMAIL","") or os.environ.get("OPENALEX_EMAIL","") or os.environ.get("EMAIL","")
    s2_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"); core_key=os.environ.get("CORE_API_KEY"); elsevier_key=os.environ.get("ELSEVIER_API_KEY") or os.environ.get("SCOPUS_API_KEY"); elsevier_insttoken=os.environ.get("ELSEVIER_INSTTOKEN")
    selection=_section(cfg,"selecao_corpus"); min_similarity=float(selection.get("min_similaridade_titulo",0.82) or 0.82); priority_similarity=float(selection.get("min_similaridade_titulo_prioritario",0.62) or 0.62)
    docs=[]; downloads=[]; rejected=[]
    for candidate in candidates:
        if len(docs)>=target: break
        if progress: progress(f"Buscando full text: {getattr(candidate,'title','')[:90]}")
        ok,download=try_download_candidate(candidate,fulltext_dir,email=email,s2_key=s2_key,core_key=core_key,elsevier_key=elsevier_key,elsevier_insttoken=elsevier_insttoken,min_title_similarity=min_similarity,priority_min_title_similarity=priority_similarity)
        record={"title":getattr(candidate,"title",""),"doi":getattr(candidate,"doi",""),"ok":bool(ok),**dict(download)}
        if not ok: rejected.append(record); continue
        source_doc=_source_doc_from_download(candidate,download,min_chars=min_chars)
        if source_doc is None: record["validation_status"]="downloaded_but_not_substantive"; rejected.append(record); continue
        record["validation_status"]="validated_substantive_fulltext"; downloads.append(record); docs.append(source_doc)
    docs=_dedupe_docs(docs)
    if len(docs)>maximum: docs=docs[:maximum]
    if len(docs)<minimum: raise RuntimeError(f"Corpus externo insuficiente: {len(docs)} full text(s) baixado(s) e validado(s); mínimo exigido={minimum}. Resultados apenas com metadata/abstract/DOI/URL ou downloads falhos/inválidos não contam para o corpus.")
    return docs,{"mode":"corpus_externo","search_dir":str(search_dir),"triage_path":str(triage_path),"fulltext_dir":str(fulltext_dir),"minimum":minimum,"target":target,"maximum":maximum,"human_review":human_review,"validated_fulltexts":len(docs),"downloads":downloads,"rejected":rejected}

def resolve_document_corpus(cfg: dict[str,Any], work_dir: Path, *, stage: Progress=None, client: Any=None, model: str="") -> tuple[list[SourceDoc],dict[str,Any]]:
    mode=str(_section(cfg,"pipeline").get("modo_entrada") or "documentos_locais").strip()
    if mode=="documentos_locais": return discover_local_documents(cfg,work_dir)
    if mode not in {"corpus_externo","corpus_hibrido"}: raise RuntimeError(f"[pipeline].modo_entrada={mode!r} não é um modo documental suportado por este runtime.")
    local_docs=[]; local_info={}
    if mode=="corpus_hibrido":
        if stage: stage("Descobrindo corpus local da composição híbrida")
        local_docs,local_info=discover_local_documents(cfg,work_dir)
        local_docs=[d for d in local_docs if str(getattr(d,"extracted_text","") or "").strip() and not str(getattr(d,"kind","") or "").endswith("_erro")]
    if stage: stage("Resolvendo corpus externo com full text garantido")
    external_docs,external_info=_external_docs(cfg,progress=stage,client=client,model=model)
    merged=_dedupe_docs([*local_docs,*external_docs]); minimum,_,maximum,_,_=_selection_contract(cfg)
    if len(merged)>maximum: merged=merged[:maximum]
    if len(merged)<minimum: raise RuntimeError(f"Corpus {mode} insuficiente após deduplicação: {len(merged)} texto(s); mínimo={minimum}.")
    return merged,{"mode":mode,"local":local_info,"external":external_info,"admitted_documents":len(merged)}

__all__=["resolve_document_corpus"]

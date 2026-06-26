#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from bibliography_manager import BibBuildResult, bib_entry_key, split_bib_entries
from corpus_manager import SourceDoc
from prisma_model import (
    Diagnostics,
    PrismaFlow,
    PrismaMetadata,
    PrismaReport,
    QueryRecord,
    ScreeningCriteria,
    SearchStrategy,
    StudyRecord,
)
from utils import normalize_title_loose, resolve_path, shorten_text
from prompt_manager import load_prompt_bundle


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _str_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _clean_doi(value: Any) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    return doi.strip().strip(".")


def _extract_bib_field(entry: str, field: str) -> str:
    m = re.search(rf"(?is)\b{re.escape(field)}\s*=\s*\{{", entry)
    if not m:
        return ""
    i = m.end()
    depth = 1
    start = i
    for j in range(i, len(entry)):
        if entry[j] == "{":
            depth += 1
        elif entry[j] == "}":
            depth -= 1
            if depth == 0:
                return re.sub(r"\s+", " ", entry[start:j]).strip()
    return ""


def _plain_bib_value(value: str) -> str:
    text = str(value or "")
    replacements = {
        r"{\'a}": "á", r"\'a": "á", r"{\`a}": "à", r"\`a": "à", r"{\~a}": "ã", r"\~a": "ã",
        r"{\'e}": "é", r"\'e": "é", r"{\^e}": "ê", r"\^e": "ê",
        r"{\'i}": "í", r"\'i": "í", r"{\'o}": "ó", r"\'o": "ó", r"{\^o}": "ô", r"\^o": "ô", r"{\~o}": "õ", r"\~o": "õ",
        r"{\'u}": "ú", r"\'u": "ú", r"{\c{c}}": "ç", r"\c{c}": "ç",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_bib_metadata(bib_path: Path | None) -> dict[str, dict[str, Any]]:
    if not bib_path or not bib_path.exists():
        return {}
    entries = split_bib_entries(bib_path.read_text(encoding="utf-8", errors="ignore"))
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = bib_entry_key(entry)
        if not key:
            continue
        authors_raw = _extract_bib_field(entry, "author") or _extract_bib_field(entry, "editor")
        authors = [_plain_bib_value(a.strip()) for a in re.split(r"\s+and\s+", authors_raw, flags=re.I) if a.strip()]
        out[key] = {
            "bib_key": key,
            "titulo": _plain_bib_value(_extract_bib_field(entry, "title")),
            "autores": authors,
            "ano": _plain_bib_value(_extract_bib_field(entry, "year")),
            "doi": _clean_doi(_extract_bib_field(entry, "doi")),
            "url": _plain_bib_value(_extract_bib_field(entry, "url")),
            "fonte": _plain_bib_value(_extract_bib_field(entry, "journaltitle") or _extract_bib_field(entry, "booktitle") or _extract_bib_field(entry, "publisher")),
        }
    return out


def metadata_from_config(cfg: dict[str, Any]) -> PrismaMetadata:
    atividade = cfg.get("atividade", {}) if isinstance(cfg.get("atividade"), dict) else {}
    pesquisa = cfg.get("pesquisa", {}) if isinstance(cfg.get("pesquisa"), dict) else {}
    rel = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}
    return PrismaMetadata(
        titulo=str(rel.get("titulo") or "Relatório de Pesquisa PRISMA"),
        tema=str(pesquisa.get("tema") or rel.get("tema") or ""),
        recorte=str(pesquisa.get("recorte") or rel.get("recorte") or ""),
        objetivo=str(pesquisa.get("objetivo") or rel.get("objetivo") or ""),
        pergunta_pesquisa=str(pesquisa.get("pergunta_pesquisa") or rel.get("pergunta_pesquisa") or ""),
        responsavel=str(atividade.get("aluno") or rel.get("responsavel") or "Gustavo M. Mendes de Tarso"),
        instituicao=str(rel.get("institution_name") or "Fundação Getúlio Vargas"),
        curso=str(atividade.get("curso") or rel.get("course_name") or "Mestrado Acadêmico em Políticas Públicas e Governo"),
        disciplina=str(atividade.get("disciplina") or rel.get("discipline_name") or ""),
        professor=str(atividade.get("professor") or rel.get("professor_name") or ""),
        cidade=str(atividade.get("polo") or rel.get("city_name") or "Brasília"),
        data_execucao=str(rel.get("data_execucao") or datetime.now().strftime("%Y-%m-%d")),
        tipo_relatorio=str(rel.get("tipo") or "prisma"),
    )


def search_strategy_from_config(cfg: dict[str, Any]) -> SearchStrategy:
    busca = cfg.get("busca", {}) if isinstance(cfg.get("busca"), dict) else {}
    queries = cfg.get("queries", {}) if isinstance(cfg.get("queries"), dict) else {}
    local = cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}
    pesquisa = cfg.get("pesquisa", {}) if isinstance(cfg.get("pesquisa"), dict) else {}
    rel = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}

    bases = _str_list(rel.get("bases")) or _str_list(pesquisa.get("bases")) or _str_list(local.get("fontes_metadados"))
    if not bases:
        for key, label in (
            ("usar_semantic_scholar", "Semantic Scholar"),
            ("usar_crossref", "Crossref"),
            ("usar_openalex", "OpenAlex"),
            ("usar_scopus", "Scopus"),
            ("usar_pubmed", "PubMed"),
            ("usar_core", "CORE"),
        ):
            if busca.get(key):
                bases.append(label)
    if not bases and local:
        bases = ["Corpus local", "Crossref", "OpenAlex"]

    idiomas = _str_list(pesquisa.get("idiomas")) or _str_list(rel.get("idiomas")) or ["português"]
    periodo = str(pesquisa.get("periodo") or rel.get("periodo") or "")

    query_records: list[QueryRecord] = []
    for key, value in queries.items():
        if not value or key.startswith("__"):
            continue
        if isinstance(value, list):
            for q in value:
                if str(q).strip():
                    query_records.append(QueryRecord(base=key, query=str(q).strip()))
        elif str(value).strip():
            query_records.append(QueryRecord(base=key, query=str(value).strip()))
    if not query_records and rel.get("queries"):
        for item in _as_list(rel.get("queries")):
            if isinstance(item, dict):
                query_records.append(QueryRecord.model_validate(item))
            elif str(item).strip():
                query_records.append(QueryRecord(base="manual", query=str(item).strip()))
    if not query_records and local:
        query_records.append(QueryRecord(base="corpus_local", query="Documentos-base locais informados no TOML", resultados_brutos=0))

    return SearchStrategy(bases=bases, idiomas=idiomas, periodo=periodo, queries=query_records)


def criteria_from_config(cfg: dict[str, Any]) -> ScreeningCriteria:
    rel = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}
    triagem = cfg.get("triagem", {}) if isinstance(cfg.get("triagem"), dict) else {}
    inclusao = _str_list(rel.get("criterios_inclusao")) or _str_list(triagem.get("criterios_inclusao")) or [
        "Aderência substantiva ao tema, recorte e objetivo da pesquisa.",
        "Disponibilidade de metadados mínimos para identificação bibliográfica.",
        "Texto acadêmico, técnico-científico ou documento-base definido pelo usuário.",
    ]
    exclusao = _str_list(rel.get("criterios_exclusao")) or _str_list(triagem.get("criterios_exclusao")) or [
        "Fora do tema ou do recorte empírico/conceitual.",
        "Registro duplicado.",
        "Ausência de relação substantiva com a pergunta de pesquisa.",
    ]
    return ScreeningCriteria(inclusao=inclusao, exclusao=exclusao)


def study_from_doc(doc: SourceDoc, bib_meta: dict[str, dict[str, Any]], key_by_doc_path: dict[str, str]) -> StudyRecord:
    key = doc.bib_key or key_by_doc_path.get(str(Path(doc.path).resolve()), "")
    meta = bib_meta.get(key, {}) if key else {}
    return StudyRecord(
        bib_key=key,
        titulo=str(meta.get("titulo") or Path(doc.path).stem.replace("_", " ").replace("-", " ").strip()),
        autores=list(meta.get("autores") or []),
        ano=str(meta.get("ano") or ""),
        doi=str(meta.get("doi") or ""),
        url=str(meta.get("url") or ""),
        base="corpus_local",
        fonte=str(meta.get("fonte") or "documento local"),
        arquivo_local=str(doc.metadata.get("fulltext_cache_path") or doc.path),
        decisao="incluido",
        justificativa="Documento-base local incorporado ao corpus e considerado na elaboração do relatório.",
        resumo=shorten_text(doc.extracted_text, 900),
        metadados={"kind": doc.kind, **(doc.metadata or {})},
    )


def _record_from_any(raw: dict[str, Any], *, default_decision: str = "identificado") -> StudyRecord | None:
    title = raw.get("titulo") or raw.get("title") or raw.get("nome") or raw.get("paper_title")
    if not title:
        return None
    authors_raw = raw.get("autores") or raw.get("authors") or []
    if isinstance(authors_raw, str):
        authors = [a.strip() for a in re.split(r";|\s+and\s+", authors_raw) if a.strip()]
    elif isinstance(authors_raw, list):
        authors = [str(a.get("name") if isinstance(a, dict) else a).strip() for a in authors_raw if str(a).strip()]
    else:
        authors = []
    decision = str(raw.get("decisao") or raw.get("decision") or raw.get("status") or default_decision).strip().lower()
    if "incl" in decision or "include" in decision or "selecion" in decision:
        decision_norm = "incluido"
    elif "duplic" in decision:
        decision_norm = "duplicado"
    elif "excl" in decision or "exclude" in decision:
        decision_norm = "excluido"
    elif "eleg" in decision:
        decision_norm = "elegivel"
    elif "tri" in decision:
        decision_norm = "triado"
    else:
        decision_norm = default_decision
    score = raw.get("score_aderencia") or raw.get("score") or raw.get("similarity")
    try:
        score_float = float(score) if score is not None and str(score).strip() else None
    except Exception:
        score_float = None
    return StudyRecord(
        bib_key=str(raw.get("bib_key") or raw.get("key") or "").strip().lstrip("@"),
        titulo=str(title),
        autores=authors,
        ano=str(raw.get("ano") or raw.get("year") or ""),
        doi=_clean_doi(raw.get("doi") or raw.get("DOI")),
        url=str(raw.get("url") or raw.get("pdf_url") or raw.get("landing_page_url") or ""),
        base=str(raw.get("base") or raw.get("source") or raw.get("fonte") or ""),
        fonte=str(raw.get("venue") or raw.get("journal") or raw.get("journaltitle") or raw.get("fonte") or ""),
        arquivo_local=str(raw.get("downloaded_pdf_path") or raw.get("local_file_path") or raw.get("arquivo_local") or ""),
        score_aderencia=score_float,
        decisao=decision_norm,  # type: ignore[arg-type]
        motivo=str(raw.get("motivo") or raw.get("reason") or raw.get("exclusion_reason") or ""),
        justificativa=str(raw.get("justificativa") or raw.get("justification") or raw.get("rationale") or ""),
        resumo=shorten_text(str(raw.get("abstract") or raw.get("resumo") or raw.get("summary") or ""), 900),
        metadados={k: v for k, v in raw.items() if k not in {"abstract", "resumo", "summary"}},
    )


def _collect_records_from_json(value: Any, bucket_name: str = "") -> list[StudyRecord]:
    records: list[StudyRecord] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                rec = _record_from_any(item, default_decision="incluido" if "incl" in bucket_name or "selected" in bucket_name else "identificado")
                if rec:
                    records.append(rec)
            elif isinstance(item, (list, tuple)):
                records.extend(_collect_records_from_json(item, bucket_name))
    elif isinstance(value, dict):
        for key, val in value.items():
            key_norm = normalize_title_loose(str(key))
            if key_norm in {"selected_all", "selecionados", "incluidos", "included", "included_studies", "elegiveis", "eligible", "excluidos", "excluded", "triagem", "results", "items", "records"}:
                records.extend(_collect_records_from_json(val, key_norm))
            elif isinstance(val, (dict, list)):
                records.extend(_collect_records_from_json(val, key_norm))
    return records


def load_records_from_research_dir(path: Path | None) -> tuple[list[StudyRecord], list[str]]:
    if not path or not path.exists() or not path.is_dir():
        return [], []
    records: list[StudyRecord] = []
    sources: list[str] = []
    for json_path in sorted(path.rglob("*.json")):
        # evita reprocessar relatórios canônicos já gerados
        if "prisma_report" in json_path.name or "rc10_report" in json_path.name:
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        found = _collect_records_from_json(data)
        if found:
            records.extend(found)
            sources.append(str(json_path))
    # Dedup por DOI/título
    seen: set[str] = set()
    unique: list[StudyRecord] = []
    for rec in records:
        ident = ("doi:" + rec.doi.lower()) if rec.doi else ("title:" + normalize_title_loose(rec.titulo))
        if ident not in seen:
            seen.add(ident)
            unique.append(rec)
    return unique, sources


def classify_records(records: list[StudyRecord]) -> tuple[list[StudyRecord], list[StudyRecord], list[StudyRecord], list[StudyRecord]]:
    included: list[StudyRecord] = []
    excluded: list[StudyRecord] = []
    duplicates: list[StudyRecord] = []
    all_records: list[StudyRecord] = []
    for rec in records:
        all_records.append(rec)
        if rec.decisao == "incluido":
            included.append(rec)
        elif rec.decisao == "duplicado":
            duplicates.append(rec)
        elif rec.decisao == "excluido":
            excluded.append(rec)
    return included, excluded, duplicates, all_records


def build_prisma_report(
    cfg: dict[str, Any],
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    bib_result: BibBuildResult,
    output_dir: Path,
    prefix: str,
) -> PrismaReport:
    rel = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()

    external_json = resolve_path(rel.get("prisma_json_path"), config_dir) if rel.get("prisma_json_path") else None
    if external_json and external_json.exists():
        return PrismaReport.model_validate_json(external_json.read_text(encoding="utf-8"))

    bib_meta = parse_bib_metadata(bib_result.bib_path)
    pesquisa_dir = resolve_path(rel.get("pesquisa_dir_existente") or (cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline"), dict) else {}).get("pesquisa_dir_existente"), config_dir)
    artifact_records, artifact_sources = load_records_from_research_dir(pesquisa_dir)

    if artifact_records:
        included, excluded, duplicates, all_records = classify_records(artifact_records)
        if not included:
            # fallback: registros elegíveis/triados viram incluídos quando não há decisão explícita.
            included = [r for r in artifact_records if r.decisao in {"elegivel", "triado", "identificado"}]
            for r in included:
                r.decisao = "incluido"
    else:
        included = [study_from_doc(doc, bib_meta, bib_result.key_by_doc_path) for doc in docs]
        excluded = []
        duplicates = []
        all_records = included.copy()
        artifact_sources = [str(Path(d.path)) for d in docs]

    # Enriquece registros incluídos com .bib quando houver chave.
    for rec in included + excluded + duplicates:
        if rec.bib_key and rec.bib_key in bib_meta:
            meta = bib_meta[rec.bib_key]
            rec.titulo = rec.titulo or str(meta.get("titulo") or "")
            rec.autores = rec.autores or list(meta.get("autores") or [])
            rec.ano = rec.ano or str(meta.get("ano") or "")
            rec.doi = rec.doi or str(meta.get("doi") or "")
            rec.url = rec.url or str(meta.get("url") or "")
            rec.fonte = rec.fonte or str(meta.get("fonte") or "")

    identified = int(rel.get("total_identificados") or len(all_records) or len(included) + len(excluded) + len(duplicates))
    dup_count = int(rel.get("total_duplicados") or len(duplicates))
    after_dedup = int(rel.get("total_apos_deduplicacao") or max(0, identified - dup_count))
    fulltext_eval = int(rel.get("total_avaliados_texto_completo") or len(included) + len(excluded))
    flow = PrismaFlow(
        identificados=identified,
        duplicados_removidos=dup_count,
        apos_deduplicacao=after_dedup,
        triados_titulo_resumo=int(rel.get("total_triados") or after_dedup),
        excluidos_titulo_resumo=int(rel.get("total_excluidos_triagem") or max(0, after_dedup - fulltext_eval)),
        avaliados_texto_completo=fulltext_eval,
        excluidos_texto_completo=int(rel.get("total_excluidos_texto_completo") or len(excluded)),
        incluidos=int(rel.get("total_incluidos") or len(included)),
    )

    diagnostics = Diagnostics(
        fontes_artefatos=artifact_sources,
        parametros={
            "output_dir": str(output_dir),
            "prefix": prefix,
            "orientacoes": [o.label for o in orientations],
            "bib_path": str(bib_result.bib_path),
            "prompts": load_prompt_bundle(cfg, "prisma").report(),
        },
    )
    if artifact_records and not pesquisa_dir:
        diagnostics.avisos.append("Registros de pesquisa encontrados por heurística, mas pesquisa_dir_existente não foi informado explicitamente.")
    if not artifact_records:
        diagnostics.avisos.append("Relatório PRISMA construído a partir do corpus local; não houve etapa externa de busca/triagem vinculada.")

    return PrismaReport(
        metadata=metadata_from_config(cfg),
        search_strategy=search_strategy_from_config(cfg),
        criteria=criteria_from_config(cfg),
        flow=flow,
        included_studies=included,
        excluded_studies=excluded,
        duplicate_studies=duplicates,
        all_records=all_records,
        diagnostics=diagnostics,
    )

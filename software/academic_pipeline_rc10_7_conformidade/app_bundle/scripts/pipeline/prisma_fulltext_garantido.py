#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    import tomllib
except Exception:
    tomllib = None


def norm_ascii(v: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(v or ""))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_col(v: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm_ascii(str(v or "").lower())).strip("_")


def slug(v: str, max_len: int = 120) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", norm_ascii(v or "").lower()).strip("_")
    return re.sub(r"_+", "_", s)[:max_len].strip("_") or "artigo"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff_csv(path: Path):
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:30000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        return csv.excel


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, dialect=sniff_csv(path)))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pick(row: dict[str, Any], *names: str) -> str:
    d = {norm_col(k): v for k, v in row.items()}
    for name in names:
        v = d.get(norm_col(name))
        if v not in (None, ""):
            return str(v).strip()
    return ""


def doi_from_text(text: str) -> str:
    m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", str(text or ""), flags=re.I)
    if not m:
        return ""
    return m.group(0).strip().rstrip(".,;)").replace("\\_", "_").replace("%5C_", "_")


def normalize_doi(v: str) -> str:
    v = str(v or "").strip()
    v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v, flags=re.I)
    v = v.replace("\\_", "_").replace("%5C_", "_")
    return doi_from_text(v) or v


def read_env(root: Path) -> dict[str, str]:
    env = {}
    p = root / ".env"
    if not p.exists():
        return env
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def read_toml(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists() or tomllib is None:
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_relative(base_file: Path, maybe_path: str) -> Path:
    p = Path(str(maybe_path)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_file.parent / p).resolve()


def dados_pesquisa_path_from_config(config: Path | None) -> Path | None:
    if not config or not config.exists():
        return None
    d = read_toml(config)
    for candidate in [
        d.get("pesquisa", {}).get("dados_pesquisa_path"),
        d.get("busca_prisma", {}).get("estrategia_busca_path"),
        d.get("busca_prisma", {}).get("criterios_path"),
    ]:
        if candidate:
            return resolve_relative(config, str(candidate))
    return None


def flatten_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return " ".join(flatten_text(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flatten_text(x) for x in obj)
    return str(obj)


def research_context(prisma_config: Path | None, dados_pesquisa_path: Path | None) -> str:
    meta_path = dados_pesquisa_path or dados_pesquisa_path_from_config(prisma_config)
    pieces = []
    if prisma_config and prisma_config.exists():
        cfg = read_toml(prisma_config)
        pieces.append(flatten_text(cfg.get("pesquisa", {})))
        pieces.append(flatten_text(cfg.get("busca_prisma", {})))
    if meta_path and meta_path.exists():
        meta = read_toml(meta_path)
        pieces.append(flatten_text(meta))
    return " ".join(pieces)


def find_pool_csvs(prisma_out_dir: Path, extra_csvs: list[Path]) -> list[Path]:
    patterns = [
        "*.curadoria_ia_referencias.csv",
        "*curadoria*referencias*.csv",
        "*.triagem_humana.csv",
        "*.triagem_titulo_resumo.csv",
        "*triagem*.csv",
        "*referencias_incluidas_seminario.csv",
        "*referencias_incluidas*.csv",
    ]
    found: list[Path] = []
    for p in extra_csvs:
        if p.exists() and p.resolve() not in found:
            found.append(p.resolve())
    for pat in patterns:
        for p in sorted(prisma_out_dir.glob(pat)):
            if p.exists() and p.resolve() not in found:
                found.append(p.resolve())
    return found


def text_tokens(text: str) -> set[str]:
    text = norm_ascii(text or "").lower()
    words = re.findall(r"[a-z0-9]{4,}", text)
    stop = {
        "with", "from", "that", "this", "what", "study", "using", "used", "based", "paper",
        "article", "medical", "health", "assessment", "evaluation", "analysis", "review",
        "benefit", "benefits", "method", "methods", "result", "results", "conclusion",
        "objective", "objectives", "background", "purpose", "systematic",
    }
    return {w for w in words if w not in stop}


def parse_numeric_score(v: str) -> float | None:
    if v in (None, ""):
        return None
    s = str(v).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    val = float(m.group(0))
    if val > 1:
        val = val / 100.0 if val <= 100 else 1.0
    return max(0.0, min(1.0, val))


def status_boost(row: dict[str, Any]) -> float:
    blob = " ".join(str(v) for v in row.values()).lower()
    if any(x in blob for x in ["incluido", "incluída", "include", "included", "selecionado", "aceito"]):
        return 0.18
    if any(x in blob for x in ["revisar", "duvida", "dúvida", "maybe", "borderline"]):
        return 0.08
    if any(x in blob for x in ["excluir", "excluido", "excluded", "reject"]):
        return -0.10
    return 0.0


def is_priority_title(title: str, regexes: list[str]) -> bool:
    blob = norm_ascii(title or "").lower()
    for pat in regexes:
        if not pat:
            continue
        try:
            if re.search(pat, blob, flags=re.I):
                return True
        except re.error:
            if pat.lower() in blob:
                return True
    return False


def candidate_score(row: dict[str, Any], title: str, abstract: str, research_ctx: str, priority: bool) -> float:
    score_cols = [
        "score_aderencia", "aderencia", "score_curadoria", "score_ia", "score",
        "relevance_score", "similaridade", "nota", "pontuacao",
    ]
    scores = [parse_numeric_score(pick(row, c)) for c in score_cols]
    scores = [s for s in scores if s is not None]
    explicit = max(scores) if scores else None

    qtokens = text_tokens(research_ctx)
    ctokens = text_tokens(title + " " + abstract + " " + flatten_text(row))
    overlap = len(qtokens & ctokens) / max(1, len(qtokens))
    lexical = min(1.0, overlap * 2.5)

    final = 0.50 * lexical + 0.50 * (explicit if explicit is not None else lexical)
    final += status_boost(row)
    if priority:
        final = max(final + 0.40, 0.995)
    return round(max(0.0, min(1.0, final)), 4)


def dedupe_key(title: str, doi: str) -> str:
    if doi:
        return "doi:" + normalize_doi(doi).lower()
    title_norm = re.sub(r"\W+", "", norm_ascii(title).lower())
    return "title:" + title_norm[:160]


@dataclass
class Candidate:
    pool_id: int
    source_csv: str
    source_row_number: int
    title: str
    authors: str
    year: str
    journal: str
    doi: str
    url: str
    abstract: str
    score_aderencia: float
    prioridade_manual: bool
    row_raw_json: str


def load_candidates(prisma_out_dir: Path, extra_csvs: list[Path], research_ctx: str, priority_regexes: list[str]) -> list[Candidate]:
    csvs = find_pool_csvs(prisma_out_dir, extra_csvs)
    rows_all: list[Candidate] = []
    seen: set[str] = set()

    for csv_path in csvs:
        try:
            rows = read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Falha lendo CSV {csv_path}: {e}")
            continue

        for i, row in enumerate(rows, start=2):
            title = pick(row, "titulo", "título", "title", "article_title", "nome", "paper_title")
            if not title or len(title) < 8:
                continue
            authors = pick(row, "autores", "authors", "author", "criadores", "creators")
            year = pick(row, "ano", "year", "publication_year", "published_year", "published", "data_publicacao")
            journal = pick(row, "periodico", "periódico", "journal", "journaltitle", "venue", "source", "container_title", "publication")
            doi = normalize_doi(pick(row, "doi") or doi_from_text(" ".join(str(v) for v in row.values())))
            url = pick(row, "url", "link", "source_url", "landing_page", "doi_url")
            abstract = pick(row, "resumo", "abstract", "summary")
            key = dedupe_key(title, doi)
            if key in seen:
                continue
            seen.add(key)

            priority = is_priority_title(title, priority_regexes)
            score = candidate_score(row, title, abstract, research_ctx, priority)
            rows_all.append(Candidate(
                pool_id=0,
                source_csv=str(csv_path),
                source_row_number=i,
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                url=url,
                abstract=abstract,
                score_aderencia=score,
                prioridade_manual=priority,
                row_raw_json=json.dumps(row, ensure_ascii=False),
            ))

    rows_all.sort(key=lambda c: (not c.prioridade_manual, -c.score_aderencia, c.title.lower()))
    for idx, c in enumerate(rows_all, start=1):
        c.pool_id = idx
    return rows_all


def fetch_bytes(url: str, headers: dict[str, str] | None = None, timeout: int = 45) -> tuple[bytes, str, str]:
    h = {"User-Agent": "academic_pipeline_fulltext_garantido/1.10"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        content_type = resp.headers.get("Content-Type", "")
        final_url = resp.geturl()
    return data, content_type, final_url


def fetch_text(url: str, headers: dict[str, str] | None = None, timeout: int = 45) -> tuple[str, str]:
    data, _, final_url = fetch_bytes(url, headers=headers, timeout=timeout)
    return data.decode("utf-8", errors="replace"), final_url


def fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 45) -> dict[str, Any]:
    text, _ = fetch_text(url, headers=headers, timeout=timeout)
    return json.loads(text)


def save_pdf_data(data: bytes, dest: Path) -> tuple[bool, str]:
    if not data[:4096].lstrip().startswith(b"%PDF"):
        return False, "resposta não parece PDF"
    if len(data) < 10_000:
        return False, "PDF pequeno demais; provável página de erro"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True, "ok"


def download_pdf(url: str, dest: Path, headers: dict[str, str] | None = None, timeout: int = 80) -> tuple[bool, str]:
    try:
        h = {
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
            "User-Agent": "academic_pipeline_fulltext_garantido/1.10",
        }
        if headers:
            h.update(headers)
        data, _, _ = fetch_bytes(url, headers=h, timeout=timeout)
        return save_pdf_data(data, dest)
    except Exception as e:
        return False, str(e)


def html_pdf_links(url: str, max_links: int = 12) -> list[str]:
    try:
        html, final_url = fetch_text(url, headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.1"})
    except Exception:
        return []
    links = []
    for href in re.findall(r"href\s*=\s*[\"']([^\"']+)[\"']", html, flags=re.I):
        h = href.strip()
        if not h or h.startswith("#") or h.lower().startswith(("javascript:", "mailto:")):
            continue
        u = urllib.parse.urljoin(final_url, h)
        ul = u.lower()
        if ".pdf" in ul or "/pdf" in ul or "download" in ul:
            links.append(u)
    for u in re.findall(r"https?://[^\"'\s<>]+", html, flags=re.I):
        ul = u.lower()
        if ".pdf" in ul or "/pdf" in ul:
            links.append(u.rstrip("\\,.;)"))
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_links:
            break
    return out


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_ascii(a).lower(), norm_ascii(b).lower()).ratio()


def openalex_abstract(w: dict[str, Any]) -> str:
    inv = w.get("abstract_inverted_index") or {}
    if not inv:
        return ""
    pos = []
    for word, indexes in inv.items():
        for i in indexes:
            pos.append((i, word))
    return " ".join(word for _, word in sorted(pos))


def candidate_item(pdf_url: str, source: str, **kwargs) -> dict[str, Any]:
    d = {"pdf_url": pdf_url, "source": source}
    d.update(kwargs)
    return d


def collect_direct_url_and_landing(c: Candidate) -> list[dict[str, Any]]:
    out = []
    urls = []
    if c.url:
        urls.append(c.url)
    if c.doi:
        urls.append("https://doi.org/" + c.doi)
    for u in urls:
        if ".pdf" in u.lower():
            out.append(candidate_item(u, "url_direta_pdf", title_found=c.title, doi_found=c.doi, year_found=c.year))
        for pdf in html_pdf_links(u):
            out.append(candidate_item(pdf, "html_landing_pdf_discovery", title_found=c.title, doi_found=c.doi, year_found=c.year, landing_url=u))
    return out


def collect_unpaywall(c: Candidate, email: str) -> list[dict[str, Any]]:
    if not c.doi or not email:
        return []
    url = "https://api.unpaywall.org/v2/" + urllib.parse.quote(c.doi, safe="") + "?email=" + urllib.parse.quote(email)
    try:
        data = fetch_json(url)
    except Exception as e:
        return [candidate_item("", "unpaywall", note=f"falha Unpaywall: {e}")]
    out = []
    locs = []
    if data.get("best_oa_location"):
        locs.append(data.get("best_oa_location"))
    locs.extend(data.get("oa_locations") or [])
    for loc in locs:
        for key in ("url_for_pdf", "url"):
            u = loc.get(key)
            if u and (key == "url_for_pdf" or ".pdf" in u.lower() or "/pdf" in u.lower()):
                out.append(candidate_item(
                    u, "unpaywall",
                    title_found=data.get("title") or "", doi_found=c.doi,
                    year_found=data.get("year") or "", landing_url=loc.get("url") or ""
                ))
    return out


def candidates_from_openalex_work(w: dict[str, Any], source: str) -> list[dict[str, Any]]:
    out = []
    locs = []
    if w.get("best_oa_location"):
        locs.append(w.get("best_oa_location"))
    locs.extend(w.get("locations") or [])
    for loc in locs:
        u = loc.get("pdf_url") or ""
        if u:
            out.append(candidate_item(
                u, source,
                title_found=w.get("title") or "",
                doi_found=(w.get("doi") or "").replace("https://doi.org/", ""),
                year_found=w.get("publication_year") or "",
                landing_url=loc.get("landing_page_url") or "",
                abstract_found=openalex_abstract(w),
            ))
    return out


def collect_openalex(c: Candidate, email: str, min_title_similarity: float, deep: bool) -> list[dict[str, Any]]:
    out = []
    mailto = ("&mailto=" + urllib.parse.quote(email)) if email else ""
    if c.doi:
        url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(c.doi, safe="") + ("?mailto=" + urllib.parse.quote(email) if email else "")
        try:
            out += candidates_from_openalex_work(fetch_json(url), "openalex_doi")
        except Exception:
            pass
    if c.title:
        per_page = 25 if deep else 10
        url = "https://api.openalex.org/works?search=" + urllib.parse.quote(c.title) + f"&filter=is_oa:true&per-page={per_page}" + mailto
        try:
            data = fetch_json(url)
            for w in data.get("results") or []:
                sim = title_similarity(c.title, w.get("title") or "")
                if sim >= min_title_similarity:
                    for item in candidates_from_openalex_work(w, "openalex_title"):
                        item["title_similarity"] = round(sim, 4)
                        out.append(item)
        except Exception:
            pass
    return out


def collect_crossref(c: Candidate, min_title_similarity: float, deep: bool) -> list[dict[str, Any]]:
    out = []
    urls = []
    if c.doi:
        urls.append("https://api.crossref.org/works/" + urllib.parse.quote(c.doi, safe=""))
    if c.title:
        rows = 20 if deep else 8
        urls.append("https://api.crossref.org/works?query.bibliographic=" + urllib.parse.quote(c.title) + f"&rows={rows}")
    for url in urls:
        try:
            data = fetch_json(url)
        except Exception:
            continue
        messages = []
        msg = data.get("message") or {}
        if isinstance(msg.get("items"), list):
            messages = msg.get("items") or []
        elif msg:
            messages = [msg]
        for m in messages:
            title_found = (m.get("title") or [""])[0]
            if title_found and c.title and title_similarity(c.title, title_found) < min_title_similarity and not c.doi:
                continue
            for link in m.get("link") or []:
                u = link.get("URL") or ""
                ct = link.get("content-type") or ""
                if u and ("pdf" in ct.lower() or ".pdf" in u.lower() or "/pdf" in u.lower()):
                    out.append(candidate_item(
                        u, "crossref_link",
                        title_found=title_found, doi_found=m.get("DOI") or c.doi,
                        year_found="", landing_url=m.get("URL") or ""
                    ))
            if m.get("URL"):
                for pdf in html_pdf_links(m.get("URL")):
                    out.append(candidate_item(pdf, "crossref_landing_pdf_discovery", title_found=title_found, doi_found=m.get("DOI") or c.doi, landing_url=m.get("URL") or ""))
    return out


def collect_semantic_scholar(c: Candidate, api_key: str | None, min_title_similarity: float, deep: bool) -> list[dict[str, Any]]:
    headers = {"x-api-key": api_key} if api_key else {}
    out = []
    if c.doi:
        url = "https://api.semanticscholar.org/graph/v1/paper/" + urllib.parse.quote("DOI:" + c.doi, safe=":") + "?fields=title,year,venue,externalIds,openAccessPdf,abstract,url"
        try:
            d = fetch_json(url, headers=headers)
            pdf = (d.get("openAccessPdf") or {}).get("url")
            if pdf:
                out.append(candidate_item(pdf, "semantic_scholar_doi", title_found=d.get("title") or "", doi_found=(d.get("externalIds") or {}).get("DOI") or c.doi, year_found=d.get("year") or "", landing_url=d.get("url") or "", abstract_found=d.get("abstract") or ""))
        except Exception:
            pass
    if c.title:
        limit = 20 if deep else 8
        url = "https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote(c.title) + f"&limit={limit}&fields=title,year,venue,externalIds,openAccessPdf,abstract,url"
        try:
            data = fetch_json(url, headers=headers)
            for d in data.get("data") or []:
                sim = title_similarity(c.title, d.get("title") or "")
                pdf = (d.get("openAccessPdf") or {}).get("url")
                if pdf and sim >= min_title_similarity:
                    out.append(candidate_item(pdf, "semantic_scholar_title", title_found=d.get("title") or "", doi_found=(d.get("externalIds") or {}).get("DOI") or "", year_found=d.get("year") or "", landing_url=d.get("url") or "", abstract_found=d.get("abstract") or "", title_similarity=round(sim, 4)))
        except Exception:
            pass
    return out


def collect_europepmc(c: Candidate, min_title_similarity: float) -> list[dict[str, Any]]:
    queries = []
    if c.doi:
        queries.append(f'DOI:"{c.doi}"')
    if c.title:
        queries.append(f'TITLE:"{c.title}"')
    out = []
    for q in queries:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&pageSize=15&query=" + urllib.parse.quote(q)
        try:
            data = fetch_json(url)
        except Exception:
            continue
        for r in (data.get("resultList") or {}).get("result") or []:
            found_title = r.get("title") or ""
            sim = title_similarity(c.title, found_title) if c.title and found_title else 0
            if c.title and sim < min_title_similarity and not c.doi:
                continue
            urls = (((r.get("fullTextUrlList") or {}).get("fullTextUrl")) or [])
            for u in urls:
                pdf = u.get("url") or ""
                if pdf and (u.get("documentStyle", "").lower() == "pdf" or ".pdf" in pdf.lower() or "/pdf" in pdf.lower()):
                    out.append(candidate_item(pdf, "europepmc", title_found=found_title, doi_found=r.get("doi") or c.doi, year_found=r.get("pubYear") or "", abstract_found=r.get("abstractText") or "", title_similarity=round(sim, 4)))
    return out


def collect_core(c: Candidate, core_key: str | None, min_title_similarity: float, deep: bool) -> list[dict[str, Any]]:
    if not core_key or not c.title:
        return []
    try:
        q = urllib.parse.quote(c.title)
        limit = 20 if deep else 10
        url = f"https://api.core.ac.uk/v3/search/works?q={q}&limit={limit}"
        data = fetch_json(url, headers={"Authorization": "Bearer " + core_key})
    except Exception:
        return []
    out = []
    results = data.get("results") or data.get("data") or []
    for r in results:
        title_found = r.get("title") or ""
        sim = title_similarity(c.title, title_found) if title_found else 0
        if sim < min_title_similarity:
            continue
        pdfs = []
        for key in ("downloadUrl", "download_url", "fullTextLink", "fulltext_url", "pdfUrl"):
            if r.get(key):
                pdfs.append(r.get(key))
        for item in (r.get("links") or []):
            if isinstance(item, dict):
                u = item.get("url") or item.get("href")
                if u:
                    pdfs.append(u)
        for u in pdfs:
            if u:
                out.append(candidate_item(u, "core", title_found=title_found, doi_found=r.get("doi") or "", year_found=r.get("yearPublished") or r.get("publishedDate") or "", landing_url=str(r.get("sourceFulltextUrls") or ""), title_similarity=round(sim, 4)))
    return out


def elsevier_headers(api_key: str, insttoken: str | None = None, accept: str = "application/json") -> dict[str, str]:
    h = {
        "X-ELS-APIKey": api_key,
        "Accept": accept,
        "User-Agent": "academic_pipeline_fulltext_garantido/1.10",
    }
    if insttoken:
        h["X-ELS-Insttoken"] = insttoken
    return h


def collect_elsevier_article_retrieval(c: Candidate, api_key: str | None, insttoken: str | None) -> list[dict[str, Any]]:
    if not api_key:
        return []
    out = []
    ids = []
    if c.doi:
        ids.append(("doi", c.doi))
    raw = c.row_raw_json or ""
    for pii in re.findall(r"\bS\d{16,18}\b", raw):
        ids.append(("pii", pii))
    for scopus_id in re.findall(r"\b2-s2\.0-\d+\b", raw):
        ids.append(("eid", scopus_id))

    seen = set()
    for id_type, value in ids:
        key = (id_type, value)
        if key in seen:
            continue
        seen.add(key)
        url = f"https://api.elsevier.com/content/article/{id_type}/" + urllib.parse.quote(value, safe="") + "?httpAccept=application/pdf"
        out.append(candidate_item(url, f"elsevier_article_retrieval_{id_type}", title_found=c.title, doi_found=c.doi, year_found=c.year, elsevier_api="article_retrieval", elsevier_id_type=id_type))
    return out


def collect_elsevier_abstract_links(c: Candidate, api_key: str | None, insttoken: str | None) -> list[dict[str, Any]]:
    if not api_key:
        return []
    out = []
    urls = []
    if c.doi:
        urls.append(("doi", "https://api.elsevier.com/content/abstract/doi/" + urllib.parse.quote(c.doi, safe="") + "?httpAccept=application/json"))
    raw = c.row_raw_json or ""
    for eid in re.findall(r"\b2-s2\.0-\d+\b", raw):
        urls.append(("eid", "https://api.elsevier.com/content/abstract/eid/" + urllib.parse.quote(eid, safe="") + "?httpAccept=application/json"))

    for id_type, url in urls:
        try:
            data = fetch_json(url, headers=elsevier_headers(api_key, insttoken))
        except Exception:
            continue
        blob = json.dumps(data, ensure_ascii=False)
        for pdf in re.findall(r"https?://[^\"'\s<>]+(?:pdf|PDF)[^\"'\s<>]*", blob):
            out.append(candidate_item(pdf.rstrip("\\,.;)"), f"elsevier_abstract_{id_type}_pdf_link", title_found=c.title, doi_found=c.doi, elsevier_api="abstract_retrieval"))
        for pii in sorted(set(re.findall(r"\bS\d{16,18}\b", blob))):
            art_url = "https://api.elsevier.com/content/article/pii/" + urllib.parse.quote(pii, safe="") + "?httpAccept=application/pdf"
            out.append(candidate_item(art_url, "elsevier_article_retrieval_pii_from_abstract", title_found=c.title, doi_found=c.doi, elsevier_api="article_retrieval", elsevier_id_type="pii"))
        for link in re.findall(r"https?://[^\"'\s<>]+", blob):
            if "sciencedirect.com" in link.lower():
                for pdf in html_pdf_links(link):
                    out.append(candidate_item(pdf, "elsevier_sciencedirect_landing_pdf_discovery", title_found=c.title, doi_found=c.doi, landing_url=link, elsevier_api="sciencedirect_landing"))
    return out


def collect_scopus_search(c: Candidate, api_key: str | None, insttoken: str | None, min_title_similarity: float, deep: bool) -> list[dict[str, Any]]:
    if not api_key:
        return []
    out = []
    queries = []
    if c.doi:
        queries.append("DOI(" + c.doi + ")")
    if c.title:
        safe_title = c.title.replace('"', " ")
        queries.append('TITLE("' + safe_title + '")')
    for q in queries:
        count = 20 if deep else 8
        url = "https://api.elsevier.com/content/search/scopus?query=" + urllib.parse.quote(q) + f"&count={count}&field=dc:title,prism:doi,eid,prism:pii,prism:url,prism:coverDate,dc:creator"
        try:
            data = fetch_json(url, headers=elsevier_headers(api_key, insttoken))
        except Exception:
            continue
        entries = (((data.get("search-results") or {}).get("entry")) or [])
        for e in entries:
            title_found = e.get("dc:title") or ""
            sim = title_similarity(c.title, title_found) if title_found and c.title else 0
            if c.title and sim < min_title_similarity and not c.doi:
                continue
            doi = e.get("prism:doi") or c.doi
            pii = e.get("prism:pii") or ""
            eid = e.get("eid") or ""
            landing = e.get("prism:url") or ""
            if doi:
                art_url = "https://api.elsevier.com/content/article/doi/" + urllib.parse.quote(doi, safe="") + "?httpAccept=application/pdf"
                out.append(candidate_item(art_url, "scopus_search_elsevier_article_doi", title_found=title_found, doi_found=doi, year_found=e.get("prism:coverDate") or "", landing_url=landing, title_similarity=round(sim, 4), elsevier_api="article_retrieval"))
            if pii:
                art_url = "https://api.elsevier.com/content/article/pii/" + urllib.parse.quote(pii, safe="") + "?httpAccept=application/pdf"
                out.append(candidate_item(art_url, "scopus_search_elsevier_article_pii", title_found=title_found, doi_found=doi, year_found=e.get("prism:coverDate") or "", landing_url=landing, title_similarity=round(sim, 4), elsevier_api="article_retrieval"))
            if eid:
                art_url = "https://api.elsevier.com/content/article/eid/" + urllib.parse.quote(eid, safe="") + "?httpAccept=application/pdf"
                out.append(candidate_item(art_url, "scopus_search_elsevier_article_eid", title_found=title_found, doi_found=doi, year_found=e.get("prism:coverDate") or "", landing_url=landing, title_similarity=round(sim, 4), elsevier_api="article_retrieval"))
            if landing:
                for pdf in html_pdf_links(landing):
                    out.append(candidate_item(pdf, "scopus_landing_pdf_discovery", title_found=title_found, doi_found=doi, year_found=e.get("prism:coverDate") or "", landing_url=landing, title_similarity=round(sim, 4)))
    return out


def collect_all_pdf_candidates(
    c: Candidate,
    *,
    email: str,
    s2_key: str | None,
    core_key: str | None,
    elsevier_key: str | None,
    elsevier_insttoken: str | None,
    min_title_similarity: float,
    priority_min_title_similarity: float,
) -> list[dict[str, Any]]:
    deep = c.prioridade_manual
    min_sim = priority_min_title_similarity if deep else min_title_similarity
    collectors = [
        ("direct_url_and_landing", lambda: collect_direct_url_and_landing(c)),
        ("unpaywall", lambda: collect_unpaywall(c, email)),
        ("openalex", lambda: collect_openalex(c, email, min_sim, deep)),
        ("crossref", lambda: collect_crossref(c, min_sim, deep)),
        ("semantic_scholar", lambda: collect_semantic_scholar(c, s2_key, min_sim, deep)),
        ("europepmc", lambda: collect_europepmc(c, min_sim)),
        ("core", lambda: collect_core(c, core_key, min_sim, deep)),
        ("elsevier_article_retrieval", lambda: collect_elsevier_article_retrieval(c, elsevier_key, elsevier_insttoken)),
        ("elsevier_abstract_links", lambda: collect_elsevier_abstract_links(c, elsevier_key, elsevier_insttoken)),
        ("scopus_search", lambda: collect_scopus_search(c, elsevier_key, elsevier_insttoken, min_sim, deep)),
    ]

    out: list[dict[str, Any]] = []
    for name, fn in collectors:
        try:
            items = fn()
            for it in items:
                if it.get("pdf_url"):
                    it["collector"] = name
                    out.append(it)
        except Exception as e:
            out.append(candidate_item("", name, note=f"erro coletor {name}: {e}"))

    seen = set()
    uniq = []
    for item in out:
        u = item.get("pdf_url") or ""
        if not u or u in seen:
            continue
        seen.add(u)
        if item.get("title_found"):
            item["title_similarity"] = item.get("title_similarity") or round(title_similarity(c.title, item.get("title_found") or ""), 4)
        uniq.append(item)

    return uniq[:60] if deep else uniq[:25]


def try_download_candidate(
    c: Candidate,
    fulltext_dir: Path,
    *,
    email: str,
    s2_key: str | None,
    core_key: str | None,
    elsevier_key: str | None,
    elsevier_insttoken: str | None,
    min_title_similarity: float,
    priority_min_title_similarity: float,
) -> tuple[bool, dict[str, Any]]:
    candidates = collect_all_pdf_candidates(
        c,
        email=email,
        s2_key=s2_key,
        core_key=core_key,
        elsevier_key=elsevier_key,
        elsevier_insttoken=elsevier_insttoken,
        min_title_similarity=min_title_similarity,
        priority_min_title_similarity=priority_min_title_similarity,
    )

    base = fulltext_dir / f"{c.pool_id:04d}_{slug(c.year + '_' + c.title)}.pdf"
    errors = []
    for i, cand in enumerate(candidates, start=1):
        url = cand.get("pdf_url") or ""
        dest = base.with_name(base.stem + f"_{i:02d}" + base.suffix)
        headers = None
        if cand.get("elsevier_api") and elsevier_key:
            headers = elsevier_headers(elsevier_key, elsevier_insttoken, accept="application/pdf")
        ok, msg = download_pdf(url, dest, headers=headers)
        if ok:
            cand = dict(cand)
            cand.update({
                "pdf_path": str(dest),
                "pdf_sha256": sha256_file(dest),
                "download_status": "pdf_original_baixado",
                "download_note": "download ok",
                "tentativas_pdf": len(candidates),
            })
            return True, cand
        errors.append(f"{cand.get('source') or cand.get('collector')}: {msg}")

    return False, {
        "download_status": "texto_completo_nao_recuperado",
        "download_note": " | ".join(errors[:12]) if errors else "sem URL de PDF encontrada nas fontes consultadas",
        "tentativas_pdf": len(candidates),
    }


def bib_escape(v: str) -> str:
    return str(v or "").replace("{", "\\{").replace("}", "\\}").replace("\n", " ").strip()


def bib_key(row: dict[str, Any]) -> str:
    author = str(row.get("authors") or "ref").split(";")[0].split(",")[0].strip().split(" ")[-1]
    year = str(row.get("year") or "sdata")
    title = str(row.get("title") or "")
    key = slug(f"{author}_{year}_{title}", 70).replace("_", "")
    return key or f"ref{row.get('ordem_inclusao_final', '')}"


def write_bib(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    used = set()
    for r in rows:
        key = bib_key(r)
        base = key
        k = 2
        while key in used:
            key = f"{base}{k}"
            k += 1
        used.add(key)
        fields = {
            "title": r.get("title", ""),
            "author": r.get("authors", ""),
            "year": r.get("year", ""),
            "journal": r.get("journal", ""),
            "doi": r.get("doi", ""),
            "url": r.get("url", "") or r.get("pdf_url", ""),
        }
        body = []
        for fk, fv in fields.items():
            fv = bib_escape(str(fv or ""))
            if fv:
                body.append(f"  {fk} = {{{fv}}}")
        entries.append("@article{" + key + ",\n" + ",\n".join(body) + "\n}")
    path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    b = path.with_suffix(path.suffix + ".bak_fulltext_garantido_v1_10_" + time.strftime("%Y%m%d_%H%M%S"))
    b.write_bytes(path.read_bytes())
    return b


def activate_selection(dados_dir: Path, output_csv: Path, output_bib: Path, prefix: str | None) -> None:
    prefix = prefix or "relatorio_prisma_prisma_fluxo_pmf"
    targets = [
        dados_dir / f"{prefix}.referencias_incluidas_seminario.csv",
        dados_dir / f"{prefix}.referencias_incluidas.csv",
    ]
    for t in targets:
        backup(t)
        t.write_bytes(output_csv.read_bytes())
        print(f"[OK] Seleção final ativada em: {t}")

    bib_target = dados_dir / f"{prefix}.referencias_incluidas.bib"
    backup(bib_target)
    bib_target.write_bytes(output_bib.read_bytes())
    print(f"[OK] BibTeX final ativado em: {bib_target}")


def md_escape(v: Any) -> str:
    return str(v or "").replace("|", "/").replace("\n", " ").strip()


def write_manual_pending(path: Path, priority_missing: list[dict[str, Any]], all_missing: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Pendências de texto completo\n\n")
        f.write("Este arquivo lista registros sem PDF recuperado automaticamente. Registros prioritários devem ser resolvidos antes de gerar a versão final do artigo.\n\n")
        if priority_missing:
            f.write("## Prioritários sem full text\n\n")
            for r in priority_missing:
                q = urllib.parse.quote(str(r.get("title") or ""))
                f.write(f"### {md_escape(r.get('title'))}\n\n")
                f.write(f"- Pool ID: {r.get('pool_id')}\n")
                f.write(f"- Score: {r.get('score_aderencia')}\n")
                f.write(f"- DOI: {md_escape(r.get('doi'))}\n")
                f.write(f"- URL: {md_escape(r.get('url'))}\n")
                f.write(f"- Tentativas de PDF: {r.get('tentativas_pdf')}\n")
                f.write(f"- Nota: {md_escape(r.get('download_note'))}\n")
                f.write(f"- Busca OpenAlex: https://openalex.org/search?filter=title.search:{q}\n")
                f.write(f"- Busca Semantic Scholar: https://www.semanticscholar.org/search?q={q}\n")
                f.write(f"- Busca Crossref: https://search.crossref.org/?q={q}\n\n")
        f.write("## Todos os registros sem full text\n\n")
        for r in all_missing:
            f.write(f"- [{r.get('pool_id')}] score={r.get('score_aderencia')} prioridade={r.get('prioridade_manual')} — {md_escape(r.get('title'))}\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Curadoria PRISMA com garantia de full text original, incluindo Elsevier/Scopus.")
    p.add_argument("--prisma-out-dir", required=True)
    p.add_argument("--art-dir", required=True)
    p.add_argument("--prisma-config", default=None)
    p.add_argument("--dados-pesquisa-path", default=None)
    p.add_argument("--extra-csv", action="append", default=[])
    p.add_argument("--target-n", type=int, default=14)
    p.add_argument("--max-candidatos", type=int, default=300)
    p.add_argument("--email", default=None)
    p.add_argument("--semantic-scholar-api-key", default=None)
    p.add_argument("--core-api-key", default=None)
    p.add_argument("--elsevier-api-key", default=None)
    p.add_argument("--elsevier-insttoken", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--min-title-similarity", type=float, default=0.82)
    p.add_argument("--priority-min-title-similarity", type=float, default=0.62)
    p.add_argument("--titulo-prioritario-regex", action="append", default=[
        r"atestmed",
        r"atestado por incapacidade",
        r"sistematica do atestmed",
        r"sistemática do atestmed",
    ])
    p.add_argument("--exigir-prioritarios", action="store_true", help="Falha se algum título prioritário não tiver full text recuperado.")
    p.add_argument("--prefix", default=None)
    p.add_argument("--ativar-selecao-final", action="store_true")
    p.add_argument("--fail-if-insufficient", action="store_true")
    args = p.parse_args(argv)

    prisma_out = Path(args.prisma_out_dir).resolve()
    art_dir = Path(args.art_dir).resolve()
    dados_dir = art_dir / "dados_prisma"
    root = Path(args.project_root).resolve()
    prisma_config = Path(args.prisma_config).resolve() if args.prisma_config else None
    dados_pesquisa_path = Path(args.dados_pesquisa_path).resolve() if args.dados_pesquisa_path else None
    extra_csvs = [Path(x).resolve() for x in args.extra_csv]

    env = read_env(root)
    email = args.email or os.environ.get("UNPAYWALL_EMAIL") or env.get("UNPAYWALL_EMAIL") or os.environ.get("OPENALEX_EMAIL") or env.get("OPENALEX_EMAIL") or env.get("EMAIL") or "gustavo.detarso@gmail.com"
    s2_key = args.semantic_scholar_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or env.get("SEMANTIC_SCHOLAR_API_KEY")
    core_key = args.core_api_key or os.environ.get("CORE_API_KEY") or env.get("CORE_API_KEY")
    elsevier_key = args.elsevier_api_key or os.environ.get("ELSEVIER_API_KEY") or env.get("ELSEVIER_API_KEY") or os.environ.get("SCOPUS_API_KEY") or env.get("SCOPUS_API_KEY")
    elsevier_insttoken = args.elsevier_insttoken or os.environ.get("ELSEVIER_INSTTOKEN") or env.get("ELSEVIER_INSTTOKEN")

    if not prisma_out.exists():
        raise SystemExit(f"ERRO: prisma-out-dir não existe: {prisma_out}")
    dados_dir.mkdir(parents=True, exist_ok=True)

    research_ctx = research_context(prisma_config, dados_pesquisa_path)
    print("[ETAPA] Carregando universo PRISMA preliminar/curadoria")
    candidates = load_candidates(prisma_out, extra_csvs, research_ctx, args.titulo_prioritario_regex)
    if not candidates:
        raise SystemExit("ERRO: nenhum candidato com título encontrado no universo PRISMA.")
    candidates = candidates[: max(args.max_candidatos, args.target_n)]
    print(f"[OK] Candidatos carregados: {len(candidates)}")
    print(f"[OK] Títulos prioritários no universo: {sum(1 for c in candidates if c.prioridade_manual)}")
    if not elsevier_key:
        print("[WARN] ELSEVIER_API_KEY/SCOPUS_API_KEY não configurado. As tentativas Scopus/Elsevier serão ignoradas.")

    out_dir = dados_dir / "fulltext_garantido"
    fulltext_dir = out_dir / "pdfs_originais"
    fulltext_dir.mkdir(parents=True, exist_ok=True)

    included: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    priority_missing: list[dict[str, Any]] = []

    print("[ETAPA] Baixando texto completo original com fontes abertas + Scopus/Elsevier")
    for c in candidates:
        if len(included) >= args.target_n and not c.prioridade_manual:
            matrix.append({
                "pool_id": c.pool_id,
                "status_prisma_fulltext": "nao_avaliado_apos_atingir_alvo",
                "motivo": "alvo de estudos com full text já atingido",
                **asdict(c),
            })
            continue

        ok, dl = try_download_candidate(
            c,
            fulltext_dir,
            email=email,
            s2_key=s2_key,
            core_key=core_key,
            elsevier_key=elsevier_key,
            elsevier_insttoken=elsevier_insttoken,
            min_title_similarity=args.min_title_similarity,
            priority_min_title_similarity=args.priority_min_title_similarity,
        )

        row_base = {
            "pool_id": c.pool_id,
            "source_csv": c.source_csv,
            "source_row_number": c.source_row_number,
            "score_aderencia": c.score_aderencia,
            "prioridade_manual": c.prioridade_manual,
            "title": c.title,
            "authors": c.authors,
            "year": c.year,
            "journal": c.journal,
            "doi": c.doi,
            "url": c.url,
            "abstract": c.abstract,
            "row_raw_json": c.row_raw_json,
            "download_status": dl.get("download_status", ""),
            "download_note": dl.get("download_note", ""),
            "tentativas_pdf": dl.get("tentativas_pdf", ""),
            "pdf_path": dl.get("pdf_path", ""),
            "pdf_sha256": dl.get("pdf_sha256", ""),
            "pdf_url": dl.get("pdf_url", ""),
            "download_source": dl.get("source", ""),
            "collector": dl.get("collector", ""),
            "title_found": dl.get("title_found", ""),
            "title_similarity": dl.get("title_similarity", ""),
            "doi_found": dl.get("doi_found", ""),
            "elsevier_api": dl.get("elsevier_api", ""),
        }

        if ok:
            row = dict(row_base)
            row["status_prisma_fulltext"] = "incluido_sintese_final"
            row["motivo"] = "texto completo original recuperado"
            row["ordem_inclusao_final"] = len(included) + 1
            included.append(row)
            matrix.append(row)
            tag = "PRIORITÁRIO " if c.prioridade_manual else ""
            print(f"[{c.pool_id:03d}] {tag}INCLUÍDO {len(included):02d}/{args.target_n}: {c.title[:100]}")
        else:
            row = dict(row_base)
            row["status_prisma_fulltext"] = "excluido_texto_completo_nao_recuperado"
            row["motivo"] = "texto completo original não recuperado após fontes abertas, landing pages e Scopus/Elsevier"
            row["ordem_inclusao_final"] = ""
            exclusions.append(row)
            matrix.append(row)
            if c.prioridade_manual:
                priority_missing.append(row)
            tag = "PRIORITÁRIO " if c.prioridade_manual else ""
            print(f"[{c.pool_id:03d}] {tag}EXCLUÍDO sem full text: {c.title[:100]}")

        time.sleep(args.sleep)

    included.sort(key=lambda r: (not bool(r.get("prioridade_manual")), -float(r.get("score_aderencia") or 0), int(r.get("pool_id") or 999999)))
    if len(included) > args.target_n:
        overflow = included[args.target_n:]
        included = included[:args.target_n]
        for r in overflow:
            r2 = dict(r)
            r2["status_prisma_fulltext"] = "excluido_por_limite_apos_priorizacao"
            r2["motivo"] = "texto completo recuperado, mas ficou fora do alvo após priorização de títulos essenciais"
            exclusions.append(r2)

    for idx, r in enumerate(included, start=1):
        r["ordem_inclusao_final"] = idx

    prefix = args.prefix
    if not prefix:
        bibs = sorted(dados_dir.glob("*.referencias_incluidas.bib"))
        prefix = bibs[0].name.replace(".referencias_incluidas.bib", "") if bibs else "relatorio_prisma_prisma_fluxo_pmf"

    included_csv = out_dir / f"{prefix}.referencias_incluidas_fulltext_garantido.csv"
    matrix_csv = out_dir / f"{prefix}.matriz_prisma_fulltext_garantido.csv"
    exclusions_csv = out_dir / f"{prefix}.exclusoes_fulltext_nao_recuperado.csv"
    included_bib = out_dir / f"{prefix}.referencias_incluidas_fulltext_garantido.bib"
    counts_json = out_dir / f"{prefix}.contagens_prisma_fulltext_garantido.json"
    md_path = out_dir / f"{prefix}.relatorio_fulltext_garantido.md"
    pending_md = out_dir / f"{prefix}.pendencias_download_manual.md"

    write_csv(included_csv, included)
    write_csv(matrix_csv, matrix)
    write_csv(exclusions_csv, exclusions)
    write_bib(included_bib, included)

    counts = {
        "registros_pool_lidos": len(candidates),
        "alvo_estudos_com_fulltext": args.target_n,
        "estudos_incluidos_com_fulltext_original": len(included),
        "excluidos_por_texto_completo_nao_recuperado": len([r for r in exclusions if r.get("status_prisma_fulltext") == "excluido_texto_completo_nao_recuperado"]),
        "prioritarios_sem_texto_completo": len(priority_missing),
        "nao_avaliados_apos_atingir_alvo": sum(1 for r in matrix if r.get("status_prisma_fulltext") == "nao_avaliado_apos_atingir_alvo"),
        "atingiu_alvo": len(included) >= args.target_n,
        "elsevier_scopus_habilitado": bool(elsevier_key),
        "core_habilitado": bool(core_key),
        "regra": "somente estudos do universo PRISMA preliminar/curadoria com PDF original recuperado entram na síntese final; títulos prioritários são tentados com busca profunda",
    }
    counts_json.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manual_pending(pending_md, priority_missing, [r for r in exclusions if r.get("status_prisma_fulltext") == "excluido_texto_completo_nao_recuperado"])

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Curadoria final com garantia de full text — v1.10\n\n")
        f.write("## Regra metodológica\n\n")
        f.write("A seleção final foi formada exclusivamente por registros do universo PRISMA preliminar/curadoria cujo texto completo original foi recuperado por fontes automatizadas legítimas. Registros aderentes sem texto completo foram documentados como excluídos por `texto completo não recuperado` e substituídos pelo próximo candidato ranqueado.\n\n")
        f.write("A versão v1.10 acrescenta tentativas Scopus/Elsevier quando `ELSEVIER_API_KEY` ou `SCOPUS_API_KEY` estiver configurado, além de descoberta por landing pages HTML, Crossref por título, OpenAlex por título, Semantic Scholar, Europe PMC e CORE quando disponível.\n\n")
        f.write("## Contagens\n\n")
        for k, v in counts.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Estudos incluídos com full text original\n\n")
        f.write("| Ordem | Prioritário | Score | Ano | Título | Fonte PDF | PDF |\n")
        f.write("|---:|---|---:|---:|---|---|---|\n")
        for r in included:
            f.write(f"| {r.get('ordem_inclusao_final')} | {r.get('prioridade_manual')} | {r.get('score_aderencia')} | {md_escape(r.get('year'))} | {md_escape(r.get('title'))} | {md_escape(r.get('download_source'))} | {md_escape(r.get('pdf_path'))} |\n")
        f.write("\n## Exclusões por texto completo não recuperado\n\n")
        f.write("| Pool ID | Prioritário | Score | Tentativas | Título | Motivo |\n")
        f.write("|---:|---|---:|---:|---|---|\n")
        for r in exclusions:
            if r.get("status_prisma_fulltext") == "excluido_texto_completo_nao_recuperado":
                f.write(f"| {r.get('pool_id')} | {r.get('prioridade_manual')} | {r.get('score_aderencia')} | {r.get('tentativas_pdf')} | {md_escape(r.get('title'))} | {md_escape(r.get('motivo'))} |\n")

    orient = dados_dir / "orientacoes_fulltext_garantido_artigo.md"
    with orient.open("w", encoding="utf-8") as f:
        f.write("# Orientação obrigatória — artigo com full text garantido\n\n")
        f.write("Use o relatório e a matriz de full text garantido em `dados_prisma/fulltext_garantido/`.\n\n")
        f.write("Regras:\n")
        f.write("1. O artigo final deve mobilizar somente como estudos incluídos aqueles marcados como `incluido_sintese_final`.\n")
        f.write("2. Cada estudo incluído deve ter aderência explícita ao tema, recorte, objetivo e pergunta de pesquisa.\n")
        f.write("3. A matriz de evidências deve conter: estudo, base de leitura, achado principal, aderência ao ATESTMED, implicação para análise documental, teleperícia ou perícia presencial, risco/limitação e uso na proposta operacional.\n")
        f.write("4. Registros excluídos por `texto completo não recuperado` devem aparecer apenas na metodologia/PRISMA, não como evidência final.\n")
        f.write("5. A indisponibilidade de full text é motivo documentado de exclusão na etapa de elegibilidade.\n")
        f.write("6. O artigo 'A sistemática do ATESTMED...' ou qualquer estudo prioritário recuperado deve receber tratamento analítico destacado, pela aderência direta ao objeto do artigo.\n")

    if args.ativar_selecao_final:
        activate_selection(dados_dir, included_csv, included_bib, prefix)

    print("[OK] Saídas geradas:")
    for pth in [included_csv, included_bib, matrix_csv, exclusions_csv, counts_json, md_path, pending_md, orient]:
        print(f"- {pth}")

    exit_code = 0
    if priority_missing:
        print(f"[ERRO] Há {len(priority_missing)} título(s) prioritário(s) sem full text. Veja: {pending_md}")
        if args.exigir_prioritarios:
            exit_code = 3

    if len(included) < args.target_n:
        print(f"[ERRO] Alvo não atingido: {len(included)} de {args.target_n} estudos com full text original.")
        if args.fail_if_insufficient:
            exit_code = 2 if exit_code == 0 else exit_code

    if exit_code == 0:
        print("[OK] Curadoria full text garantido concluída.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

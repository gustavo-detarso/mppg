#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

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

def slug(v: str, max_len: int = 110) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", norm_ascii(v or "").lower()).strip("_")
    return re.sub(r"_+", "_", s)[:max_len].strip("_") or "artigo"

def sniff_csv(path: Path):
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:10000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        return csv.excel

def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, dialect=sniff_csv(path)))

def pick(row: dict, *names: str) -> str:
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

def read_env(root: Path) -> dict:
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

def read_toml(path: Path | None) -> dict:
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

def flatten_text(obj) -> str:
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

def fetch_json(url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    h = {"User-Agent": "academic_pipeline_fulltext_downloader/1.1"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))

def download_pdf(url: str, dest: Path, timeout: int = 60) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "academic_pipeline_fulltext_downloader/1.1",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data[:2048].lstrip().startswith(b"%PDF"):
            return False, "resposta não parece PDF"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True, "ok"
    except Exception as e:
        return False, str(e)

def collect_pdf_urls_from_unpaywall(doi: str, email: str) -> list[dict]:
    if not doi or not email:
        return []
    url = "https://api.unpaywall.org/v2/" + urllib.parse.quote(doi, safe="") + "?email=" + urllib.parse.quote(email)
    try:
        data = fetch_json(url)
    except Exception as e:
        return [{"pdf_url": "", "source": "unpaywall", "note": f"falha Unpaywall: {e}"}]
    out = []
    locs = []
    if data.get("best_oa_location"):
        locs.append(data.get("best_oa_location"))
    locs.extend(data.get("oa_locations") or [])
    for loc in locs:
        for key in ("url_for_pdf", "url"):
            u = loc.get(key)
            if u and (key == "url_for_pdf" or ".pdf" in u.lower()):
                out.append({
                    "pdf_url": u,
                    "source": "unpaywall",
                    "title": data.get("title") or "",
                    "doi": doi,
                    "year": data.get("year") or "",
                    "landing_url": loc.get("url") or "",
                })
    return out

def collect_pdf_urls_from_openalex_doi(doi: str, email: str) -> list[dict]:
    if not doi:
        return []
    mailto = "&mailto=" + urllib.parse.quote(email) if email else ""
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="") + "?" + mailto.lstrip("&")
    try:
        data = fetch_json(url)
    except Exception:
        return []
    return candidates_from_openalex_work(data, source="openalex_doi")

def candidates_from_openalex_work(w: dict, source: str = "openalex") -> list[dict]:
    out = []
    locs = []
    if w.get("best_oa_location"):
        locs.append(w.get("best_oa_location"))
    locs.extend(w.get("locations") or [])
    for loc in locs:
        u = loc.get("pdf_url") or loc.get("landing_page_url") or ""
        if u and (loc.get("pdf_url") or ".pdf" in u.lower()):
            out.append({
                "pdf_url": u,
                "source": source,
                "title": w.get("title") or "",
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "year": w.get("publication_year") or "",
                "landing_url": loc.get("landing_page_url") or "",
                "abstract": openalex_abstract(w),
            })
    return out

def openalex_abstract(w: dict) -> str:
    inv = w.get("abstract_inverted_index") or {}
    if not inv:
        return ""
    pos = []
    for word, indexes in inv.items():
        for i in indexes:
            pos.append((i, word))
    return " ".join(word for _, word in sorted(pos))

def collect_pdf_urls_from_crossref(doi: str) -> list[dict]:
    if not doi:
        return []
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        data = fetch_json(url)
    except Exception:
        return []
    msg = data.get("message") or {}
    out = []
    for link in msg.get("link") or []:
        u = link.get("URL") or ""
        ct = link.get("content-type") or ""
        if u and ("pdf" in ct.lower() or ".pdf" in u.lower()):
            out.append({
                "pdf_url": u,
                "source": "crossref_link",
                "title": (msg.get("title") or [""])[0],
                "doi": doi,
                "year": "",
                "landing_url": "",
            })
    return out

def collect_pdf_urls_from_semantic_scholar(doi: str, title: str, api_key: str | None = None) -> list[dict]:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    identifiers = []
    if doi:
        identifiers.append("DOI:" + doi)
    if title:
        identifiers.append(None)
    out = []
    if doi:
        url = "https://api.semanticscholar.org/graph/v1/paper/" + urllib.parse.quote("DOI:" + doi, safe=":") + "?fields=title,year,venue,externalIds,openAccessPdf,abstract,url"
        try:
            d = fetch_json(url, headers=headers)
            pdf = (d.get("openAccessPdf") or {}).get("url")
            if pdf:
                out.append({
                    "pdf_url": pdf,
                    "source": "semantic_scholar_doi",
                    "title": d.get("title") or "",
                    "doi": doi,
                    "year": d.get("year") or "",
                    "landing_url": d.get("url") or "",
                    "abstract": d.get("abstract") or "",
                })
        except Exception:
            pass
    if title:
        q = urllib.parse.quote(title)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=5&fields=title,year,venue,externalIds,openAccessPdf,abstract,url"
        try:
            data = fetch_json(url, headers=headers)
            for d in data.get("data") or []:
                pdf = (d.get("openAccessPdf") or {}).get("url")
                if pdf:
                    out.append({
                        "pdf_url": pdf,
                        "source": "semantic_scholar_search",
                        "title": d.get("title") or "",
                        "doi": (d.get("externalIds") or {}).get("DOI") or "",
                        "year": d.get("year") or "",
                        "landing_url": d.get("url") or "",
                        "abstract": d.get("abstract") or "",
                    })
        except Exception:
            pass
    return out

def collect_pdf_urls_from_europepmc(doi: str, title: str) -> list[dict]:
    q = ""
    if doi:
        q = f'DOI:"{doi}"'
    elif title:
        q = f'TITLE:"{title}"'
    if not q:
        return []
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&pageSize=5&query=" + urllib.parse.quote(q)
    try:
        data = fetch_json(url)
    except Exception:
        return []
    out = []
    for r in (data.get("resultList") or {}).get("result") or []:
        urls = (((r.get("fullTextUrlList") or {}).get("fullTextUrl")) or [])
        for u in urls:
            pdf = u.get("url") or ""
            if pdf and (u.get("documentStyle", "").lower() == "pdf" or ".pdf" in pdf.lower()):
                out.append({
                    "pdf_url": pdf,
                    "source": "europepmc",
                    "title": r.get("title") or "",
                    "doi": r.get("doi") or doi,
                    "year": r.get("pubYear") or "",
                    "landing_url": "",
                    "abstract": r.get("abstractText") or "",
                })
    return out

def tokens(text: str) -> set[str]:
    text = norm_ascii(text or "").lower()
    words = re.findall(r"[a-z0-9]{4,}", text)
    stop = {"with", "from", "that", "this", "what", "study", "using", "medical", "health", "paper", "article", "assessment", "evaluation"}
    return {w for w in words if w not in stop}

def score_candidate(original_title: str, research_ctx: str, cand: dict) -> float:
    ctitle = cand.get("title") or ""
    cabstract = cand.get("abstract") or ""
    s_title = SequenceMatcher(None, norm_ascii(original_title).lower(), norm_ascii(ctitle).lower()).ratio() if original_title and ctitle else 0
    qtokens = tokens(original_title + " " + research_ctx)
    ctokens = tokens(ctitle + " " + cabstract)
    overlap = len(qtokens & ctokens) / max(1, len(qtokens))
    return round(0.65 * s_title + 0.35 * overlap, 4)

def search_openalex_proxy(title: str, research_ctx: str, email: str, per_page: int = 25) -> list[dict]:
    queries = []
    if title:
        queries.append(title)
    # A second query broadens the search if exact title has no OA PDF.
    core_words = " ".join(list(tokens(research_ctx))[:10])
    if core_words:
        queries.append((title + " " + core_words)[:240])
    mailto = "&mailto=" + urllib.parse.quote(email) if email else ""
    out = []
    seen = set()
    for query in queries:
        url = "https://api.openalex.org/works?search=" + urllib.parse.quote(query) + f"&filter=is_oa:true&per-page={per_page}" + mailto
        try:
            data = fetch_json(url)
        except Exception:
            continue
        for w in data.get("results") or []:
            for c in candidates_from_openalex_work(w, source="openalex_proxy"):
                key = c.get("pdf_url") or c.get("doi") or c.get("title")
                if key in seen:
                    continue
                seen.add(key)
                c["proxy_score"] = score_candidate(title, research_ctx, c)
                out.append(c)
        time.sleep(0.25)
    return sorted(out, key=lambda x: x.get("proxy_score", 0), reverse=True)

def try_download_candidates(candidates: list[dict], dest_base: Path, status_ok: str, min_score: float | None = None) -> tuple[str, dict, str]:
    for i, c in enumerate(candidates, start=1):
        if min_score is not None and c.get("proxy_score", 0) < min_score:
            continue
        pdf_url = c.get("pdf_url") or ""
        if not pdf_url:
            continue
        dest = dest_base.with_name(dest_base.stem + f"_{i:02d}" + dest_base.suffix)
        ok, msg = download_pdf(pdf_url, dest)
        if ok:
            c = dict(c)
            c["pdf_path"] = str(dest)
            return status_ok, c, "download ok"
        else:
            c["last_error"] = msg
    return "", {}, "nenhum candidato baixável"

def find_input_csv(dados_dir: Path) -> Path:
    candidates = sorted(dados_dir.glob("*referencias_incluidas_seminario.csv"))
    if not candidates:
        candidates = sorted(dados_dir.glob("*referencias_incluidas*.csv"))
    if not candidates:
        candidates = sorted(dados_dir.glob("*triagem_humana.csv"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum CSV de referências encontrado em {dados_dir}")
    return candidates[0]

def md_escape(v: str) -> str:
    return str(v or "").replace("|", "/").replace("\n", " ").strip()

def main(argv=None):
    p = argparse.ArgumentParser(description="Baixa PDFs OA das referências PRISMA e, opcionalmente, PDFs proxy OA tematicamente próximos.")
    p.add_argument("--dados-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--prisma-config", default=None)
    p.add_argument("--dados-pesquisa-path", default=None)
    p.add_argument("--email", default=None)
    p.add_argument("--semantic-scholar-api-key", default=None)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--fallback-proxy-oa", action="store_true", help="Baixa artigo OA proxy quando o PDF da referência original não for encontrado.")
    p.add_argument("--min-proxy-score", type=float, default=0.24)
    args = p.parse_args(argv)

    dados_dir = Path(args.dados_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else dados_dir / "fulltext_artigos"
    root = Path(args.project_root).resolve()
    env = read_env(root)

    email = args.email or os.environ.get("UNPAYWALL_EMAIL") or env.get("UNPAYWALL_EMAIL") or os.environ.get("OPENALEX_EMAIL") or env.get("OPENALEX_EMAIL") or env.get("EMAIL") or os.environ.get("OPENAI_EMAIL")
    s2_key = args.semantic_scholar_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or env.get("SEMANTIC_SCHOLAR_API_KEY")

    prisma_config = Path(args.prisma_config).resolve() if args.prisma_config else None
    dados_pesquisa_path = Path(args.dados_pesquisa_path).resolve() if args.dados_pesquisa_path else None
    ctx = research_context(prisma_config, dados_pesquisa_path)

    if not email:
        print("[WARN] Nenhum e-mail encontrado para Unpaywall/OpenAlex. Configure UNPAYWALL_EMAIL ou OPENALEX_EMAIL.")

    input_csv = find_input_csv(dados_dir)
    rows = read_csv(input_csv)
    if args.max and args.max > 0:
        rows = rows[:args.max]

    manifest = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows, start=1):
        titulo = pick(row, "titulo", "título", "title", "article_title", "nome")
        autores = pick(row, "autores", "authors", "author", "criadores", "creators")
        ano = pick(row, "ano", "year", "publication_year", "published_year", "published")
        doi = normalize_doi(pick(row, "doi") or doi_from_text(" ".join(str(v) for v in row.values())))
        url = pick(row, "url", "link", "source_url", "landing_page")
        base = out_dir / f"{slug(f'{idx:02d}_{ano}_{titulo}')}.pdf"

        original_candidates = []
        original_candidates += collect_pdf_urls_from_unpaywall(doi, email or "")
        original_candidates += collect_pdf_urls_from_openalex_doi(doi, email or "")
        original_candidates += collect_pdf_urls_from_crossref(doi)
        original_candidates += collect_pdf_urls_from_semantic_scholar(doi, titulo, s2_key)
        original_candidates += collect_pdf_urls_from_europepmc(doi, titulo)
        if url and ".pdf" in url.lower():
            original_candidates.append({"pdf_url": url, "source": "url_original_pdf", "title": titulo, "doi": doi, "year": ano, "landing_url": url})

        # de-duplicate
        seen = set()
        original_candidates = [c for c in original_candidates if c.get("pdf_url") and not (c.get("pdf_url") in seen or seen.add(c.get("pdf_url")))]

        status, chosen, note = try_download_candidates(original_candidates, base, "pdf_original_baixado")
        is_proxy = False

        if not status and args.fallback_proxy_oa:
            proxies = search_openalex_proxy(titulo, ctx, email or "")
            status, chosen, note = try_download_candidates(proxies, base.with_name(base.stem + "_PROXY.pdf"), "pdf_proxy_oa_baixado", min_score=args.min_proxy_score)
            is_proxy = bool(status)

        if not status:
            status = "sem_pdf_baixado"
            chosen = {}
            note = note or "sem PDF OA localizado"

        manifest.append({
            "id": idx,
            "status": status,
            "tipo_texto_fisico": "referencia_original" if status == "pdf_original_baixado" else ("proxy_oa_tematico" if status == "pdf_proxy_oa_baixado" else "ausente"),
            "proxy": "sim" if is_proxy else "nao",
            "proxy_score": chosen.get("proxy_score", ""),
            "ano_original": ano,
            "autores_original": autores,
            "titulo_original": titulo,
            "doi_original": doi,
            "url_original": url,
            "fonte_pdf": chosen.get("source", ""),
            "pdf_url": chosen.get("pdf_url", ""),
            "pdf_path": chosen.get("pdf_path", ""),
            "titulo_pdf_baixado": chosen.get("title", ""),
            "doi_pdf_baixado": chosen.get("doi", ""),
            "ano_pdf_baixado": chosen.get("year", ""),
            "nota": note,
        })
        print(f"[{idx:02d}/{len(rows):02d}] {status}: {titulo[:90]}")
        time.sleep(args.sleep)

    manifest_csv = dados_dir / "manifesto_fulltext_artigos.csv"
    fields = [
        "id", "status", "tipo_texto_fisico", "proxy", "proxy_score",
        "ano_original", "autores_original", "titulo_original", "doi_original", "url_original",
        "fonte_pdf", "pdf_url", "pdf_path", "titulo_pdf_baixado", "doi_pdf_baixado", "ano_pdf_baixado", "nota",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)

    original_count = sum(1 for r in manifest if r["status"] == "pdf_original_baixado")
    proxy_count = sum(1 for r in manifest if r["status"] == "pdf_proxy_oa_baixado")
    missing_count = sum(1 for r in manifest if r["status"] == "sem_pdf_baixado")

    md = dados_dir / "orientacoes_fulltext_artigos.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Full text e textos proxy dos artigos incluídos no PRISMA\n\n")
        f.write(f"CSV de entrada: `{input_csv}`\n\n")
        f.write(f"Referências avaliadas: {len(manifest)}\n\n")
        f.write(f"PDFs da referência original baixados: {original_count}\n\n")
        f.write(f"PDFs proxy open-access tematicamente próximos baixados: {proxy_count}\n\n")
        f.write(f"Referências ainda sem texto físico: {missing_count}\n\n")
        f.write("## Regras metodológicas obrigatórias\n\n")
        f.write("- Um PDF proxy **não substitui** a referência original no PRISMA.\n")
        f.write("- Use PDF original para discutir diretamente a referência selecionada.\n")
        f.write("- Use PDF proxy apenas como apoio contextual/temático, identificando que é texto proxy.\n")
        f.write("- Se a referência original não tiver full text baixado, não atribua a ela achados específicos não presentes em resumo/metadados.\n")
        f.write("- A matriz de evidências deve distinguir: `referencia_original`, `proxy_oa_tematico` e `ausente`.\n")
        f.write("- Se um proxy for usado como fonte substantiva, ele deve ser citado como fonte adicional, não como se fosse o artigo original.\n\n")
        f.write("## Manifesto\n\n")
        f.write("| ID | Status | Proxy | Score | Original | PDF baixado |\n")
        f.write("|---:|---|---|---:|---|---|\n")
        for r in manifest:
            f.write(f"| {r['id']} | {md_escape(r['status'])} | {r['proxy']} | {r['proxy_score']} | {md_escape(r['titulo_original'])} | {md_escape(r['titulo_pdf_baixado'] or r['pdf_path'])} |\n")

    print(f"[OK] Manifesto CSV: {manifest_csv}")
    print(f"[OK] Orientação full text/proxy: {md}")
    print(f"[OK] PDFs originais: {original_count} de {len(manifest)}")
    print(f"[OK] PDFs proxy OA: {proxy_count} de {len(manifest)}")
    print(f"[OK] Sem texto físico: {missing_count} de {len(manifest)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .corpus_manager import SourceDoc
else:
    from corpus_manager import SourceDoc
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .prompt_manager import load_prompt_bundle
else:
    from prompt_manager import load_prompt_bundle
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import (
        normalize_title_loose,
        shorten_text,
        write_text,
        write_json,
        resolve_path,
    )
else:
    from utils import (
        normalize_title_loose,
        shorten_text,
        write_text,
        write_json,
        resolve_path,
    )


# Fontes efetivamente implementadas para enriquecimento de metadados de
# documentos locais por DOI. A lista é distinta dos provedores de descoberta
# do fluxo PRISMA: uma base só aparece aqui quando possui adaptador de
# metadados neste módulo.
METADATA_SOURCE_ORDER = ("crossref", "openalex", "semantic_scholar", "scopus")

# A chave aprovada do Semantic Scholar possui cota cumulativa de uma requisição
# por segundo. O limitador também cobre o enriquecimento de DOI em papers locais.
SEMANTIC_SCHOLAR_MIN_INTERVAL = 1.05
_SEMANTIC_SCHOLAR_LAST_REQUEST_AT = 0.0


def _metadata_env(*names: str) -> str:
    """Lê a primeira variável de ambiente não vazia, sem expor seu valor."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def metadata_provider_statuses() -> dict[str, dict[str, Any]]:
    """Informa capacidade de consulta por fonte, sem revelar credenciais.

    O wizard usa somente os rótulos e o estado; chaves e e-mails permanecem
    exclusivamente no ambiente/.env. Crossref e OpenAlex aceitam consulta
    pública, mas o e-mail melhora a identificação do cliente. Semantic Scholar
    também mantém rota pública; Scopus exige chave.
    """
    crossref_email = _metadata_env("CROSSREF_EMAIL", "CROSSREF_MAILTO", "OPENALEX_EMAIL", "OPENALEX_MAILTO")
    openalex_email = _metadata_env("OPENALEX_EMAIL", "OPENALEX_MAILTO", "CROSSREF_EMAIL", "CROSSREF_MAILTO")
    semantic_key = _metadata_env("SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY")
    scopus_key = _metadata_env("SCOPUS_API_KEY", "ELSEVIER_API_KEY")
    return {
        "crossref": {
            "label": "Crossref",
            "available": True,
            "status": "disponível — e-mail de contato detectado" if crossref_email else "disponível — sem e-mail de contato no .env",
        },
        "openalex": {
            "label": "OpenAlex",
            "available": True,
            "status": "disponível — e-mail de contato detectado" if openalex_email else "disponível — sem e-mail de contato no .env",
        },
        "semantic_scholar": {
            "label": "Semantic Scholar",
            "available": True,
            "status": "chave detectada" if semantic_key else "disponível sem chave — sujeito a limite público",
        },
        "scopus": {
            "label": "Scopus",
            "available": bool(scopus_key),
            "status": "chave detectada" if scopus_key else "chave não detectada — consulta será ignorada",
        },
    }


def metadata_provider_selection_choices() -> list[tuple[str, str]]:
    """Retorna opções visuais para a tela de fontes de metadados."""
    statuses = metadata_provider_statuses()
    choices = [("todas", "[Todas] — selecionar todas as fontes de metadados compatíveis")]
    for source in METADATA_SOURCE_ORDER:
        item = statuses[source]
        choices.append((source, f"{item['label']} — {item['status']}"))
    return choices


def expand_metadata_provider_selection(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    """Expande a opção mestre ``todas`` para as fontes suportadas."""
    raw = [str(value).strip().lower().replace("-", "_") for value in values if str(value).strip()]
    if "todas" in raw:
        return list(METADATA_SOURCE_ORDER)
    out: list[str] = []
    for source in raw:
        if source == "semanticscholar":
            source = "semantic_scholar"
        if source in METADATA_SOURCE_ORDER and source not in out:
            out.append(source)
    return out


class BibMetadata(BaseModel):
    entry_type: str = "article"
    title: str
    authors: list[str] = Field(default_factory=list)
    editors: list[str] = Field(default_factory=list)
    year: str | None = None
    booktitle: str | None = None
    journaltitle: str | None = None
    publisher: str | None = None
    location: str | None = None
    pages: str | None = None
    volume: str | None = None
    number: str | None = None
    doi: str | None = None
    isbn: str | None = None
    url: str | None = None
    note: str | None = None


@dataclass
class BibBuildResult:
    bib_path: Path
    keys: list[str]
    diagnostics_path: Path | None = None
    key_by_doc_path: dict[str, str] = field(default_factory=dict)
    entries: list[str] = field(default_factory=list)


def normalize_doi(value: str | None) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip().strip(".")


def extract_doi_from_text(text: str) -> str | None:
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text or "")
    if not m:
        return None
    return normalize_doi(m.group(0).rstrip(".,);]"))


def load_doi_manifest(path: Path | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path or not path.exists():
        return result
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = str(row.get("arquivo") or row.get("filename") or row.get("file") or row.get("path") or "").strip()
            doi = normalize_doi(row.get("doi") or row.get("DOI"))
            if not filename or not doi:
                continue
            p = Path(filename)
            for key in {filename, filename.lower(), p.name, p.name.lower(), p.stem, p.stem.lower(), normalize_title_loose(filename), normalize_title_loose(p.name), normalize_title_loose(p.stem)}:
                if key:
                    result[key] = doi
    return result


def doi_for_doc(doc: SourceDoc, manifest: dict[str, str]) -> tuple[str | None, str | None]:
    p = Path(doc.path)
    keys = (str(p), str(p).lower(), p.name, p.name.lower(), p.stem, p.stem.lower(), normalize_title_loose(p.name), normalize_title_loose(p.stem))
    for key in keys:
        if key in manifest:
            return manifest[key], "manifest"
    doi = extract_doi_from_text(doc.extracted_text)
    if doi:
        return doi, "extracted"
    return None, None


def http_get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 25) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec academic metadata lookup
            raw = resp.read().decode(resp.headers.get_content_charset() or "utf-8", errors="replace")
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def _people_crossref(raw: list[dict[str, Any]] | None) -> list[str]:
    out = []
    for a in raw or []:
        name = " ".join(x for x in [str(a.get("given") or "").strip(), str(a.get("family") or "").strip()] if x).strip()
        if name:
            out.append(name)
    return out


def crossref_item_to_meta(item: dict[str, Any]) -> BibMetadata:
    typ = str(item.get("type") or "journal-article")
    entry_type = "article" if "journal" in typ else ("book" if typ == "book" else ("incollection" if "chapter" in typ else "misc"))
    title = " ".join(item.get("title") or []).strip() or "Sem título"
    container = " ".join(item.get("container-title") or []).strip()
    issued = item.get("issued") or item.get("published-print") or item.get("published-online") or {}
    year = None
    try:
        year = str(issued.get("date-parts", [[None]])[0][0])
    except Exception:
        pass
    return BibMetadata(
        entry_type=entry_type,
        title=title,
        authors=_people_crossref(item.get("author")),
        editors=_people_crossref(item.get("editor")),
        year=year,
        journaltitle=container if entry_type == "article" else None,
        booktitle=container if entry_type in {"incollection", "inbook"} else None,
        publisher=item.get("publisher"),
        pages=item.get("page"),
        volume=item.get("volume"),
        number=item.get("issue"),
        doi=normalize_doi(item.get("DOI")),
        isbn=(item.get("ISBN") or [None])[0] if isinstance(item.get("ISBN"), list) else item.get("ISBN"),
        url=item.get("URL"),
    )


def crossref_by_doi(doi: str) -> BibMetadata | None:
    mailto = _metadata_env("CROSSREF_EMAIL", "CROSSREF_MAILTO", "OPENALEX_EMAIL", "OPENALEX_MAILTO")
    headers = {"User-Agent": f"academic-pipeline-rc10 ({mailto or 'local'})"}
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    data = http_get_json(url, headers=headers)
    msg = data.get("message") if isinstance(data, dict) else None
    return crossref_item_to_meta(msg) if isinstance(msg, dict) else None


def openalex_by_doi(doi: str) -> BibMetadata | None:
    mailto = _metadata_env("OPENALEX_EMAIL", "OPENALEX_MAILTO", "CROSSREF_EMAIL", "CROSSREF_MAILTO")
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="")
    if mailto:
        url += "?" + urllib.parse.urlencode({"mailto": mailto})
    data = http_get_json(url, headers={"User-Agent": f"academic-pipeline-rc10 ({mailto or 'local'})"})
    if not isinstance(data, dict) or not data.get("display_name"):
        return None
    biblio = data.get("biblio") or {}
    source = ((data.get("primary_location") or {}).get("source") or {})
    authors = []
    for item in data.get("authorships") or []:
        name = ((item or {}).get("author") or {}).get("display_name")
        if name:
            authors.append(str(name))
    first_page, last_page = biblio.get("first_page"), biblio.get("last_page")
    pages = f"{first_page}--{last_page}" if first_page and last_page else None
    typ = str(data.get("type") or "article")
    entry_type = "article" if typ in {"article", "journal-article"} else ("book" if typ == "book" else ("incollection" if "chapter" in typ else "misc"))
    return BibMetadata(
        entry_type=entry_type,
        title=str(data.get("display_name") or ""),
        authors=authors,
        year=str(data.get("publication_year") or "") or None,
        journaltitle=source.get("display_name") if entry_type == "article" else None,
        publisher=source.get("host_organization_name"),
        pages=pages,
        volume=biblio.get("volume"),
        number=biblio.get("issue"),
        doi=normalize_doi(data.get("doi")),
        url=data.get("landing_page_url") or data.get("id"),
    )


def _wait_for_semantic_scholar() -> None:
    """Respeita a cota cumulativa da API Semantic Scholar sem expor a chave."""
    global _SEMANTIC_SCHOLAR_LAST_REQUEST_AT
    elapsed = time.monotonic() - _SEMANTIC_SCHOLAR_LAST_REQUEST_AT
    if _SEMANTIC_SCHOLAR_LAST_REQUEST_AT and elapsed < SEMANTIC_SCHOLAR_MIN_INTERVAL:
        time.sleep(SEMANTIC_SCHOLAR_MIN_INTERVAL - elapsed)
    _SEMANTIC_SCHOLAR_LAST_REQUEST_AT = time.monotonic()


def semantic_scholar_by_doi(doi: str) -> BibMetadata | None:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY") or ""
    headers = {"User-Agent": "academic-pipeline-rc10"}
    if key:
        headers["x-api-key"] = key
    _wait_for_semantic_scholar()
    fields = "title,authors,year,venue,externalIds,url,publicationVenue"
    url = "https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(doi, safe="") + "?" + urllib.parse.urlencode({"fields": fields})
    data = http_get_json(url, headers=headers)
    if not isinstance(data, dict) or not data.get("title"):
        return None
    venue = data.get("venue") or ((data.get("publicationVenue") or {}).get("name") if isinstance(data.get("publicationVenue"), dict) else None)
    doi_found = None
    if isinstance(data.get("externalIds"), dict):
        doi_found = data["externalIds"].get("DOI")
    authors = [str(a.get("name")) for a in data.get("authors") or [] if isinstance(a, dict) and a.get("name")]
    return BibMetadata(entry_type="article", title=str(data.get("title")), authors=authors, year=str(data.get("year") or "") or None, journaltitle=venue, doi=normalize_doi(doi_found or doi), url=data.get("url"))


def scopus_by_doi(doi: str) -> BibMetadata | None:
    key = os.getenv("SCOPUS_API_KEY") or os.getenv("ELSEVIER_API_KEY") or ""
    if not key:
        return None
    headers = {"X-ELS-APIKey": key, "Accept": "application/json", "User-Agent": "academic-pipeline-rc10"}
    query = f"DOI({doi})"
    url = "https://api.elsevier.com/content/search/scopus?" + urllib.parse.urlencode({"query": query, "count": "1"})
    data = http_get_json(url, headers=headers)
    entries = (((data or {}).get("search-results") or {}).get("entry") or []) if isinstance(data, dict) else []
    if not entries:
        return None
    item = entries[0]
    title = item.get("dc:title") or "Sem título"
    authors = [str(item.get("dc:creator"))] if item.get("dc:creator") else []
    year = None
    m = re.search(r"(18|19|20)\d{2}", str(item.get("prism:coverDate") or item.get("prism:coverDisplayDate") or ""))
    if m:
        year = m.group(0)
    return BibMetadata(
        entry_type="article",
        title=str(title),
        authors=authors,
        year=year,
        journaltitle=item.get("prism:publicationName"),
        volume=item.get("prism:volume"),
        number=item.get("prism:issueIdentifier"),
        doi=normalize_doi(item.get("prism:doi") or doi),
        url=item.get("prism:url"),
    )


def _metadata_sources(cfg: dict[str, Any]) -> list[str]:
    local = cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}
    raw = local.get("fontes_metadados") or list(METADATA_SOURCE_ORDER)
    if isinstance(raw, str):
        raw = [raw]
    out = expand_metadata_provider_selection(list(raw or []))
    return out or ["crossref", "openalex"]


def lookup_by_doi(doi: str, cfg: dict[str, Any]) -> tuple[BibMetadata | None, str]:
    for source in _metadata_sources(cfg):
        try:
            if source == "crossref":
                meta = crossref_by_doi(doi)
            elif source == "openalex":
                meta = openalex_by_doi(doi)
            elif source == "semantic_scholar":
                meta = semantic_scholar_by_doi(doi)
            elif source == "scopus":
                meta = scopus_by_doi(doi)
            else:
                meta = None
            if meta:
                return meta, source
        except Exception:
            continue
    return None, "doi_lookup_failed"


def infer_bib_metadata_ai(client: Any, model: str, doc: SourceDoc, doi: str | None = None, cfg: dict[str, Any] | None = None) -> BibMetadata:
    prompt_bundle = load_prompt_bundle(cfg or {}, "bibliography") if cfg else None
    prompt_extras = (prompt_bundle.text if prompt_bundle else "") or "Nenhuma diretiva complementar carregada."
    prompt = f"""
Extraia metadados bibliográficos do documento abaixo e retorne JSON no schema solicitado.
Regras: não invente DOI, páginas, periódico, editora ou autores. Se souber que é livro, use entry_type=book; se capítulo, incollection; se artigo, article.
Não inclua observações técnicas.

Diretivas complementares carregadas pelo prompt bank:
{prompt_extras}

DOI conhecido, se houver: {doi or ''}
Arquivo: {doc.label}
Texto extraído:
{shorten_text(doc.extracted_text, 25000)}
""".strip()
    resp = client.responses.parse(model=model, input=[{"role": "user", "content": prompt}], text_format=BibMetadata)
    if resp.output_parsed is None:
        raise RuntimeError("IA não retornou metadados bibliográficos estruturados.")
    meta = resp.output_parsed
    if doi:
        meta.doi = doi
    return meta


def make_bib_key(meta: BibMetadata, used: set[str]) -> str:
    author = meta.authors[0] if meta.authors else (meta.editors[0] if meta.editors else "anon")
    parts = normalize_title_loose(author).split()
    surname = parts[-1] if parts else "anon"
    year = re.search(r"(18|19|20)\d{2}", str(meta.year or ""))
    yr = year.group(0) if year else "sd"
    words = [w for w in normalize_title_loose(meta.title).split() if len(w) >= 3][:2]
    base = f"{surname}{yr}_{'_'.join(words) if words else 'trabalho'}"
    key = base
    i = 2
    while key in used:
        key = f"{base}_{i}"
        i += 1
    used.add(key)
    return key


def bibtex_escape(text: str) -> str:
    text = str(text or "").replace("\n", " ").strip()
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}
    return "".join(repl.get(ch, ch) for ch in text)


def render_bib_entry(key: str, meta: BibMetadata) -> str:
    entry_type = meta.entry_type.lower().strip() or "misc"
    allowed = {"article", "book", "incollection", "inbook", "report", "thesis", "misc"}
    if entry_type not in allowed:
        entry_type = "misc"
    fields = []
    if meta.authors:
        fields.append(("author", " and ".join(meta.authors)))
    if meta.editors:
        fields.append(("editor", " and ".join(meta.editors)))
    for name in ["title", "booktitle", "journaltitle", "publisher", "location", "year", "volume", "number", "pages", "doi", "isbn", "url", "note"]:
        value = getattr(meta, name, None)
        if value:
            fields.append((name, str(value)))
    body = ",\n  ".join(f"{n} = {{{bibtex_escape(v)}}}" for n, v in fields)
    return f"@{entry_type}{{{key},\n  {body}\n}}"


def split_bib_entries(text: str) -> list[str]:
    entries = []
    i = 0
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        brace = text.find("{", at)
        if brace < 0:
            break
        depth = 0
        for j in range(brace, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    entries.append(text[at:j + 1].strip())
                    i = j + 1
                    break
        else:
            break
    return entries


def bib_entry_key(entry: str) -> str | None:
    m = re.match(r"\s*@[^{]+\{\s*([^,]+)\s*,", entry, re.S)
    return m.group(1).strip() if m else None


def extract_field(entry: str, name: str) -> str:
    m = re.search(rf"(?is)\b{re.escape(name)}\s*=\s*", entry)
    if not m:
        return ""
    i = m.end()
    while i < len(entry) and entry[i].isspace():
        i += 1
    if i >= len(entry):
        return ""
    if entry[i] == "{":
        depth, start = 0, i + 1
        for j in range(i, len(entry)):
            if entry[j] == "{":
                depth += 1
            elif entry[j] == "}":
                depth -= 1
                if depth == 0:
                    return re.sub(r"\s+", " ", entry[start:j]).strip()
    return ""


def entry_identity(entry: str) -> str:
    doi = normalize_doi(extract_field(entry, "doi"))
    if doi:
        return "doi:" + doi
    title = normalize_title_loose(extract_field(entry, "title"))
    year = extract_field(entry, "year")
    author = normalize_title_loose(extract_field(entry, "author") or extract_field(entry, "editor"))
    if title:
        return "title:" + "|".join([title, year, author.split()[0] if author.split() else ""])
    return "key:" + str(bib_entry_key(entry))


def entry_quality(entry: str) -> tuple[int, int]:
    score = 0
    for field, pts in [("doi", 50), ("author", 15), ("editor", 10), ("year", 10), ("journaltitle", 10), ("booktitle", 8), ("publisher", 8), ("pages", 5), ("volume", 3), ("number", 3)]:
        if extract_field(entry, field):
            score += pts
    low = (extract_field(entry, "note") + " " + extract_field(entry, "author")).lower()
    if "metadados" in low or "material fornecido" in low:
        score -= 40
    return (score, len(entry))


def deduplicate_entries(entries: list[str]) -> list[str]:
    groups: dict[str, list[str]] = {}
    for e in entries:
        if bib_entry_key(e):
            groups.setdefault(entry_identity(e), []).append(e)
    return [max(group, key=entry_quality) for group in groups.values()]


def _parse_bib_meta(entry: str) -> dict[str, str]:
    return {name: extract_field(entry, name) for name in ["title", "doi", "year", "author", "editor"]}


def _match_doc_to_bib_key(doc: SourceDoc, entries: list[str], manifest: dict[str, str]) -> str | None:
    doi, _ = doi_for_doc(doc, manifest)
    if doi:
        for e in entries:
            if normalize_doi(extract_field(e, "doi")) == doi:
                return bib_entry_key(e)
    p = Path(doc.path)
    file_norms = {normalize_title_loose(p.stem), normalize_title_loose(p.name), normalize_title_loose(doc.label)}
    best_key, best_score = None, 0.0
    for e in entries:
        key = bib_entry_key(e)
        if not key:
            continue
        title_norm = normalize_title_loose(extract_field(e, "title"))
        key_norm = normalize_title_loose(key)
        candidates = [title_norm, key_norm]
        for cand in candidates:
            if cand and any(cand in f or f in cand for f in file_norms if f):
                return key
        # token overlap fallback
        ft = set(" ".join(file_norms).split())
        tt = set(title_norm.split())
        if ft and tt:
            score = len(ft & tt) / max(1, min(len(ft), len(tt)))
            if score > best_score:
                best_score, best_key = score, key
    return best_key if best_score >= 0.45 else None


def build_bibliography(cfg: dict[str, Any], docs: list[SourceDoc], output_dir: Path, prefix: str, client: Any | None, model: str) -> BibBuildResult:
    local = cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    bib_path = output_dir / f"{prefix}.bib"
    manifest_path = resolve_path(local.get("doi_manifest_path"), config_dir) if local.get("doi_manifest_path") else None
    manifest = load_doi_manifest(manifest_path)

    existing_bib = None
    for section_name in ("bibliografia", "documento", "documentos_locais"):
        section = cfg.get(section_name, {}) if isinstance(cfg.get(section_name), dict) else {}
        raw = section.get("bib_path") or section.get("referencias_bib")
        p = resolve_path(raw, config_dir) if raw else None
        if p and p.exists():
            existing_bib = p
            break
    if existing_bib:
        entries = split_bib_entries(existing_bib.read_text(encoding="utf-8", errors="ignore"))
        entries = deduplicate_entries(entries)
        keys = [k for e in entries if (k := bib_entry_key(e))]
        key_by_doc_path: dict[str, str] = {}
        for doc in docs:
            key = _match_doc_to_bib_key(doc, entries, manifest)
            if key:
                doc.bib_key = key
                key_by_doc_path[str(Path(doc.path).resolve())] = key
        write_text(bib_path, "\n\n".join(entries).strip() + "\n")
        return BibBuildResult(bib_path=bib_path, keys=keys, entries=entries, key_by_doc_path=key_by_doc_path)

    used: set[str] = set()
    entries: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    key_by_doc_path: dict[str, str] = {}

    for doc in docs:
        doi, doi_source = doi_for_doc(doc, manifest)
        meta: BibMetadata | None = None
        source = "fallback"
        if doi and bool(local.get("buscar_metadados_por_doi", True)):
            meta, source = lookup_by_doi(doi, cfg)
        if meta is None and client is not None and bool(local.get("gerar_bib_revisado_ia", True)):
            try:
                meta = infer_bib_metadata_ai(client, model, doc, doi, cfg)
                source = "ai"
            except Exception as exc:
                diagnostics.append({"doc": doc.label, "ai_error": str(exc)})
        if meta is None:
            meta = BibMetadata(title=Path(doc.path).stem.replace("_", " ").replace("-", " ").title(), authors=[], year=str(local.get("ano_padrao") or "s.d."), doi=doi)
        if doi and not meta.doi:
            meta.doi = doi
        key = make_bib_key(meta, used)
        doc.bib_key = key
        key_by_doc_path[str(Path(doc.path).resolve())] = key
        entries.append(render_bib_entry(key, meta))
        diagnostics.append({"doc": doc.label, "bib_key": key, "source": source, "doi": doi, "doi_source": doi_source, "metadata": meta.model_dump()})

    entries = deduplicate_entries(entries)
    keys = [k for e in entries if (k := bib_entry_key(e))]

    # Após deduplicar, alguma chave atribuída a um documento pode ter sido
    # removida. Recalcula o mapeamento documento -> chave canônica para evitar
    # que a IA veja/cite chaves inexistentes no .bib final.
    key_by_doc_path = {}
    for doc in docs:
        canonical = _match_doc_to_bib_key(doc, entries, manifest)
        if canonical:
            doc.bib_key = canonical
            key_by_doc_path[str(Path(doc.path).resolve())] = canonical

    write_text(bib_path, "\n\n".join(entries).strip() + "\n")
    diag_path = output_dir / f"{prefix}_bibliografia_diagnostico.json"
    write_json(diag_path, diagnostics)
    return BibBuildResult(bib_path=bib_path, keys=keys, diagnostics_path=diag_path, key_by_doc_path=key_by_doc_path, entries=entries)

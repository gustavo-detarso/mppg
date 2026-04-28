#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline integrado de pesquisa + paper para o projeto MPPG.

Objetivo:
1. Ler um TOML unificado.
2. Extrair dele um TOML compatível com o gerador_pesquisa_rc_1.py.
3. Executar opcionalmente o gerador_pesquisa_rc_1.py (PRISMA ou empírico).
4. Montar um bundle de handoff com os artefatos gerados.
5. Gerar opcionalmente um paper em Org-mode a partir do contexto e dos artefatos da pesquisa.
6. Preservar o .org da pesquisa e o .org do paper como artefatos distintos.

Este script não tenta reimplementar o motor metodológico do gerador_pesquisa_rc_1.py.
Ele o reutiliza como etapa de pesquisa e assume a etapa redacional do paper, consumindo explicitamente os PDFs selecionados na pesquisa quando disponíveis.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

try:
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
DEFAULT_RESEARCH_SCRIPT = "./gerador_pesquisa_rc_1.py"
DEFAULT_AUTHOR = "Gustavo M. Mendes de Tarso"
DEFAULT_INSTITUTION = "Faculdade Getúlio Vargas"
DEFAULT_PAPER_TEMPLATE = "template_paper.org"
DEFAULT_FALLBACK_TEMPLATE = "template_research.org"
DEFAULT_STYLE = "apa"
DEBUG = False

RESEARCH_SECTIONS = {
    "atividade",
    "pesquisa",
    "bibliografia",
    "busca",
    "triagem",
    "queries",
    "saida",
    "latex",
    "openai",
    "controle",
}

TEXT_SUFFIXES = {".txt", ".md", ".org", ".rst", ".tex", ".json", ".csv", ".yaml", ".yml", ".xml"}
BINARY_SUFFIXES = {".pdf", ".docx"}
READABLE_SUFFIXES = TEXT_SUFFIXES | BINARY_SUFFIXES

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class ResearchPaths:
    root_dir: Path
    org_path: Path | None = None
    bib_path: Path | None = None
    debug_path: Path | None = None
    pdf_path: Path | None = None
    prisma_svg_path: Path | None = None
    prisma_pdf_path: Path | None = None
    config_path: Path | None = None
    fulltext_cache_dir: Path | None = None
    selected_entries: list[dict[str, Any]] = field(default_factory=list)
    selected_fulltext_paths: list[Path] = field(default_factory=list)
    fulltext_paths: list[Path] = field(default_factory=list)

@dataclass
class SourceDoc:
    path: str
    kind: str
    label: str
    extracted_text: str
    summary: str | None = None
    bib_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PaperContext:
    tema: str
    recorte: str
    objetivo: str
    pergunta_pesquisa: str | None = None
    hipotese: str | None = None
    palavras_chave: list[str] = field(default_factory=list)
    titulo_sugerido: str | None = None
    tipo_estudo: str | None = None
    idiomas: list[str] = field(default_factory=lambda: ["português", "inglês"])
    modo_origem: str = "revisao_sistematica"
    titulo_trabalho_base: str | None = None

class FinalFrontMatterOutput(BaseModel):
    title: str
    paper_type: str
    cover_note: str

class InferredBibMetadata(BaseModel):
    entry_type: str = "article"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    journaltitle: str | None = None
    doi: str | None = None
    url: str | None = None
    note: str | None = None

class RewrittenPaperContextOutput(BaseModel):
    tema: str
    recorte: str
    objetivo: str
    pergunta_pesquisa: str | None = None
    hipotese: str | None = None
    titulo_sugerido: str | None = None
    rationale: str | None = None

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def debug_print(*parts: object) -> None:
    if DEBUG:
        print("[DEBUG]", *parts, file=sys.stderr)

def load_env() -> None:
    load_dotenv(override=False)

def make_client(model_override: str | None = None) -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrado no .env ou no ambiente.")
    return OpenAI(api_key=api_key), (model_override or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)

def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    repl = {
        "á": "a", "à": "a", "â": "a", "ã": "a", "é": "e", "ê": "e", "í": "i",
        "ó": "o", "ô": "o", "õ": "o", "ú": "u", "ç": "c",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z0-9_\-\s]", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    return text.strip("_") or "item"

def ensure_command(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Comando não encontrado no PATH: {name}")
    return found

def shorten_text(text: str, limit: int = 8000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def read_text_file(path: Path, max_chars: int = 40000) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return shorten_text(path.read_text(encoding="utf-8", errors="ignore"), max_chars)
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        chunks: list[str] = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                chunks.append(text)
                total += len(text)
                if total >= max_chars:
                    break
        return shorten_text("\n".join(chunks), max_chars)
    if suffix == ".docx":
        if docx is not None:
            d = docx.Document(str(path))
            return shorten_text("\n".join(p.text for p in d.paragraphs if p.text), max_chars)
        raise RuntimeError("python-docx não está disponível para ler .docx")
    raise RuntimeError(f"Arquivo não suportado para leitura textual: {path}")

def add_guidance_from_value(docs: list[SourceDoc], raw_value: Any, kind: str, label: str, max_chars: int = 20000) -> None:
    if raw_value is None:
        return
    raw = str(raw_value).strip()
    if not raw:
        return

    maybe_path = Path(os.path.expanduser(raw)).resolve()
    if maybe_path.exists() and maybe_path.is_file():
        try:
            if maybe_path.suffix.lower() in READABLE_SUFFIXES:
                text = read_text_file(maybe_path, max_chars=max_chars)
            else:
                text = shorten_text(maybe_path.read_text(encoding="utf-8", errors="ignore"), max_chars)
            docs.append(SourceDoc(
                path=str(maybe_path),
                kind=kind,
                label=label or maybe_path.name,
                extracted_text=text,
            ))
            return
        except Exception as exc:
            debug_print(f"Falha ao ler orientação {maybe_path}: {exc}")

    # fallback: trata o valor como texto inline
    docs.append(SourceDoc(
        path=f"inline:{kind}",
        kind=kind,
        label=label,
        extracted_text=shorten_text(raw, max_chars),
    ))

def read_template_raw(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if raw.lstrip().startswith("# -*- mode: snippet -*-"):
        marker = "\n# --\n"
        pos = raw.find(marker)
        if pos != -1:
            return raw[pos + len(marker):]
    return raw

def replace_org_header_line(org_text: str, prefix: str, new_value: str) -> str:
    pattern = re.compile(rf'^(\s*{re.escape(prefix)}\s*).*$' , re.MULTILINE)
    replacement = rf'\1{new_value}'
    if pattern.search(org_text):
        return pattern.sub(replacement, org_text, count=1)
    return f"{prefix} {new_value}\n" + org_text

def replace_latex_header_macro(org_text: str, macro: str, new_value: str) -> str:
    pattern = re.compile(rf'^(\s*#\+LATEX_HEADER:\s*\\{re.escape(macro)}\{{).*(\}}\s*)$', re.MULTILINE)
    replacement = rf'\1{new_value}\2'
    if pattern.search(org_text):
        return pattern.sub(replacement, org_text, count=1)
    return org_text

def apply_citation_style(org_text: str, bib_filename: str, style: str) -> str:
    style = (style or DEFAULT_STYLE).strip().lower()
    cite_line = f"#+CITE_EXPORT: biblatex backend=biber,style={style},sortcites=true,sorting=nyt,giveninits=true,maxcitenames=2,maxbibnames=20,uniquelist=minyear"
    if re.search(r"(?im)^#\+CITE_EXPORT:", org_text):
        org_text = re.sub(r"(?im)^#\+CITE_EXPORT:.*$", cite_line, org_text)
    else:
        org_text = cite_line + "\n" + org_text
    if re.search(r"(?im)^#\+BIBLIOGRAPHY:\s+.*$", org_text):
        org_text = re.sub(r"(?im)^#\+BIBLIOGRAPHY:\s+.*$", f"#+BIBLIOGRAPHY: {bib_filename}", org_text)
    else:
        org_text += f"\n#+BIBLIOGRAPHY: {bib_filename}\n"
    if "#+PRINT_BIBLIOGRAPHY:" not in org_text:
        org_text += "\n* Referências\n#+PRINT_BIBLIOGRAPHY:\n"
    return org_text

def ensure_cover_command(org_text: str) -> str:
    if "\\usepapercover" not in org_text or "#+LATEX: \\makemytitle" in org_text:
        return org_text
    marker = "#+begin_abstract"
    if marker in org_text:
        return org_text.replace(marker, "#+LATEX: \\makemytitle\n\n" + marker, 1)
    return org_text + "\n#+LATEX: \\makemytitle\n"

def cleanup_generated_org(org_text: str) -> str:
    org_text = re.sub(r"\n{3,}", "\n\n", org_text)
    return org_text.strip() + "\n"

def split_bib_entries(text: str) -> list[str]:
    entries: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at == -1:
            break
        brace = text.find("{", at)
        if brace == -1:
            break
        depth = 0
        j = brace
        while j < n:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    entry = text[at:j + 1].strip()
                    if entry:
                        entries.append(entry)
                    i = j + 1
                    break
            j += 1
        else:
            entry = text[at:].strip()
            if entry:
                entries.append(entry)
            break
    return entries

def bib_entry_key(entry: str) -> str | None:
    m = re.match(r"\s*@[^{]+\{\s*([^,]+)\s*,", entry, flags=re.DOTALL)
    return m.group(1).strip() if m else None

def parse_bib_entries(path: Path | None) -> tuple[list[str], list[str]]:
    if path is None or not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8", errors="ignore")
    entries = split_bib_entries(text)
    keys = [k for e in entries if (k := bib_entry_key(e))]
    return entries, keys


def normalize_title_loose(text: str) -> str:
    text = (text or "").strip().lower()
    repl = {
        "á": "a", "à": "a", "â": "a", "ã": "a", "é": "e", "ê": "e", "í": "i",
        "ó": "o", "ô": "o", "õ": "o", "ú": "u", "ç": "c",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_doi(value: str | None) -> str:
    return (value or "").strip().lower()

def extract_bib_field(entry: str, field: str) -> str | None:
    patterns = [
        rf'(?is)\b{re.escape(field)}\s*=\s*\{{(.*?)\}}',
        rf'(?is)\b{re.escape(field)}\s*=\s*"(.*?)"',
    ]
    for pattern in patterns:
        m = re.search(pattern, entry)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return None

def parse_bib_entry_meta(entry: str) -> dict[str, Any]:
    return {
        "key": bib_entry_key(entry),
        "title": extract_bib_field(entry, "title"),
        "doi": extract_bib_field(entry, "doi"),
        "url": extract_bib_field(entry, "url"),
        "year": extract_bib_field(entry, "year"),
        "author": extract_bib_field(entry, "author"),
    }

def match_bib_key_for_selected(entry: dict[str, Any], bib_entries: list[str]) -> str | None:
    doi = normalize_doi(entry.get("doi"))
    title_norm = normalize_title_loose(str(entry.get("title") or ""))
    for bib in bib_entries:
        meta = parse_bib_entry_meta(bib)
        if not meta.get("key"):
            continue
        if doi and normalize_doi(meta.get("doi")) == doi:
            return str(meta["key"])
    for bib in bib_entries:
        meta = parse_bib_entry_meta(bib)
        if not meta.get("key"):
            continue
        bib_title_norm = normalize_title_loose(str(meta.get("title") or ""))
        if title_norm and bib_title_norm and (title_norm == bib_title_norm or title_norm in bib_title_norm or bib_title_norm in title_norm):
            return str(meta["key"])
    return None

def assign_bib_keys_to_selected_docs(base_docs: list[SourceDoc], research_paths: ResearchPaths, bib_entries: list[str]) -> None:
    by_path: dict[str, str] = {}
    by_title: dict[str, str] = {}
    for entry in research_paths.selected_entries:
        key = match_bib_key_for_selected(entry, bib_entries)
        if not key:
            continue
        title = str(entry.get("title") or "")
        if title:
            by_title[normalize_title_loose(title)] = key
        pdf_raw = entry.get("downloaded_pdf_path")
        if pdf_raw:
            p = Path(str(pdf_raw)).expanduser()
            try:
                p = p.resolve()
                by_path[str(p)] = key
                by_path[p.name] = key
            except Exception:
                by_path[Path(str(pdf_raw)).name] = key

    for doc in base_docs:
        if not doc.kind.startswith("texto_selecionado"):
            continue
        if doc.bib_key:
            continue
        key = None
        p = Path(doc.path)
        if p.exists():
            key = by_path.get(str(p.resolve())) or by_path.get(p.name)
        if key is None:
            title_norm = normalize_title_loose(doc.label)
            key = by_title.get(title_norm)
        if key:
            doc.bib_key = key

def extract_cited_keys_from_org(org_text: str) -> list[str]:
    keys = re.findall(r"@([A-Za-z0-9_:\-]+)", org_text)
    seen = []
    used = set()
    for k in keys:
        if k not in used:
            used.add(k)
            seen.append(k)
    return seen

def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or "", flags=re.UNICODE))

def count_org_words_per_top_section(org_text: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    matches = list(re.finditer(r"(?m)^\*\s+(.+)$", org_text))
    if not matches:
        return {"documento_total": count_words(org_text)}
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(org_text)
        body = org_text[start:end]
        sections[title] = count_words(body)
    sections["documento_total"] = count_words(org_text)
    return sections

def build_reference_usage_map(org_text: str, base_docs: list[SourceDoc], extra_docs: list[SourceDoc], bib_keys: list[str]) -> dict[str, Any]:
    cited_keys = extract_cited_keys_from_org(org_text)
    selected_keys = sorted({d.bib_key for d in base_docs if d.kind.startswith("texto_selecionado") and d.bib_key})
    extra_keys = sorted({d.bib_key for d in extra_docs if d.bib_key})
    selected_cited = [k for k in cited_keys if k in selected_keys]
    extra_cited = [k for k in cited_keys if k in extra_keys]
    cited_known = set(selected_keys) | set(extra_keys)
    other_cited = [k for k in cited_keys if k not in cited_known]
    bib_not_cited = [k for k in bib_keys if k not in cited_keys]

    return {
        "cited_keys": cited_keys,
        "selected_keys": selected_keys,
        "extra_keys": extra_keys,
        "selected_cited_keys": selected_cited,
        "extra_cited_keys": extra_cited,
        "other_cited_keys": other_cited,
        "bib_keys_not_cited": bib_not_cited,
        "counts": {
            "cited_total": len(cited_keys),
            "selected_total": len(selected_keys),
            "selected_cited_total": len(selected_cited),
            "extra_total": len(extra_keys),
            "extra_cited_total": len(extra_cited),
            "bib_not_cited_total": len(bib_not_cited),
        },
    }

def render_reference_usage_markdown(usage: dict[str, Any]) -> str:
    lines = ["# Mapa de uso das referências", ""]
    counts = usage.get("counts", {})
    lines += [
        f"- Citações únicas no paper: {counts.get('cited_total', 0)}",
        f"- Chaves de artigos selecionados: {counts.get('selected_total', 0)}",
        f"- Chaves de selecionados efetivamente citadas: {counts.get('selected_cited_total', 0)}",
        f"- Chaves de artigos extras: {counts.get('extra_total', 0)}",
        f"- Chaves de extras efetivamente citadas: {counts.get('extra_cited_total', 0)}",
        f"- Entradas do .bib não citadas: {counts.get('bib_not_cited_total', 0)}",
        "",
        "## Selecionados citados",
    ]
    sel = usage.get("selected_cited_keys", [])
    lines += [f"- `{k}`" for k in sel] if sel else ["- Nenhum"]
    lines += ["", "## Extras citados"]
    ex = usage.get("extra_cited_keys", [])
    lines += [f"- `{k}`" for k in ex] if ex else ["- Nenhum"]
    lines += ["", "## Outras chaves citadas"]
    other = usage.get("other_cited_keys", [])
    lines += [f"- `{k}`" for k in other] if other else ["- Nenhuma"]
    lines += ["", "## Entradas do .bib não citadas"]
    notc = usage.get("bib_keys_not_cited", [])
    lines += [f"- `{k}`" for k in notc] if notc else ["- Nenhuma"]
    lines += [""]
    return "\n".join(lines)

def make_bib_key(authors: list[str], year: str | None, title: str) -> str:
    surname = slugify((authors[0].split()[-1] if authors else "anon"))
    year_part = re.search(r"(19|20)\d{2}", year or "")
    year_str = year_part.group(0) if year_part else "sd"
    title_words = [w for w in slugify(title).split("_") if w][:2]
    return f"{surname}{year_str}_{'_'.join(title_words) if title_words else 'trabalho'}"

def unique_key(candidate: str, used: set[str]) -> str:
    key = candidate
    idx = 2
    while key in used:
        key = f"{candidate}_{idx}"
        idx += 1
    used.add(key)
    return key

def render_biblatex_entry(key: str, meta: InferredBibMetadata) -> str:
    fields: list[tuple[str, str | None]] = []
    if meta.authors:
        fields.append(("author", " and ".join(meta.authors)))
    fields.append(("title", meta.title))
    fields.append(("year", meta.year))
    fields.append(("journaltitle", meta.journaltitle))
    fields.append(("doi", meta.doi))
    fields.append(("url", meta.url))
    fields.append(("note", meta.note))
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields if v)
    return f"@{meta.entry_type}{{{key},\n  {body}\n}}"

def collect_readable_files(raw_items: list[Any]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_items or []:
        if not str(raw).strip():
            continue
        p = Path(os.path.expanduser(str(raw))).resolve()
        candidates: list[Path] = []
        if p.is_dir():
            candidates = [c for c in sorted(p.rglob("*")) if c.is_file() and c.suffix.lower() in READABLE_SUFFIXES]
        elif p.is_file() and p.suffix.lower() in READABLE_SUFFIXES:
            candidates = [p]
        for c in candidates:
            key = str(c)
            if key not in seen:
                out.append(c)
                seen.add(key)
    return out

def infer_bib_metadata_for_doc(client: OpenAI, model: str, doc: SourceDoc) -> InferredBibMetadata:
    prompt = textwrap.dedent(
        f"""
        Extraia metadados bibliográficos de um artigo a partir do trecho abaixo.
        Retorne JSON estruturado com:
        - entry_type (article, misc)
        - title
        - authors
        - year
        - journaltitle
        - doi
        - url
        - note

        Se algum campo estiver ausente, deixe nulo.
        Documento: {doc.label}
        Caminho: {doc.path}

        Texto:
        {shorten_text(doc.extracted_text, 16000)}
        """
    ).strip()
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=InferredBibMetadata,
    )
    if resp.output_parsed is None:
        raise RuntimeError("IA não retornou metadados bibliográficos estruturados.")
    return resp.output_parsed

def build_bib_entries_for_extra_docs(client: OpenAI, model: str, docs: list[SourceDoc], existing_keys: list[str]) -> tuple[list[SourceDoc], list[str], list[str]]:
    used = set(existing_keys)
    added_entries: list[str] = []
    added_keys: list[str] = []

    for doc in docs:
        try:
            meta = infer_bib_metadata_for_doc(client, model, doc)
        except Exception:
            meta = InferredBibMetadata(
                entry_type="misc",
                title=Path(doc.path).stem.replace("_", " ").replace("-", " "),
                url=doc.metadata.get("url") if isinstance(doc.metadata, dict) else None,
                note="Artigo extra adicionado fora da triagem da pesquisa; revisar metadados manualmente.",
            )

        key = unique_key(make_bib_key(meta.authors, meta.year, meta.title), used)
        entry = render_biblatex_entry(key, meta)
        doc.bib_key = key
        doc.metadata = {**doc.metadata, **meta.model_dump()}
        added_entries.append(entry)
        added_keys.append(key)

    return docs, added_entries, added_keys

def find_fallback_template() -> Path | None:
    candidates = [Path.cwd() / DEFAULT_PAPER_TEMPLATE, Path.cwd() / DEFAULT_FALLBACK_TEMPLATE]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

def build_default_paper_template() -> str:
    return textwrap.dedent(r"""
        #+TITLE: TÍTULO A SER GERADO PELA IA
        #+AUTHOR: Gustavo M. Mendes de Tarso
        #+LANGUAGE: pt_BR
        #+OPTIONS: toc:nil num:t ^:nil
        #+LATEX_CLASS: fgv-paper
        #+LATEX_HEADER: \institution{Faculdade Getúlio Vargas}
        #+LATEX_HEADER: \coursename{}
        #+LATEX_HEADER: \disciplinename{}
        #+LATEX_HEADER: \professorname{}
        #+LATEX_HEADER: \cityname{Brasília}
        #+LATEX_HEADER: \papertype{Texto gerado automaticamente pela IA após a conclusão do paper}
        #+LATEX_HEADER: \covernote{Nota filosófica a ser gerada pela IA após a conclusão do paper}

        #+begin_abstract
        Resumo a ser gerado pela IA.
        #+end_abstract

        * Introdução
        * Desenvolvimento
        * Considerações finais
        * Referências
        #+PRINT_BIBLIOGRAPHY:
        """).strip() + "\n"

def detect_research_output_dir(cfg: dict[str, Any]) -> Path:
    pipeline = cfg.get("pipeline", {})
    pesquisa_dir_existente = str(pipeline.get("pesquisa_dir_existente") or "").strip()
    if pesquisa_dir_existente:
        return Path(pesquisa_dir_existente).expanduser().resolve()

    saida = cfg.get("saida", {})
    base_dir = Path(saida.get("output_dir") or ".").expanduser().resolve()
    prefixo = (saida.get("prefixo") or "atividade").strip()
    create_subdir = bool(saida.get("criar_subdiretorio", True))
    return (base_dir / prefixo) if create_subdir else base_dir

def detect_research_paths(cfg: dict[str, Any]) -> ResearchPaths:
    root_dir = detect_research_output_dir(cfg)
    prefixo = (cfg.get("saida", {}).get("prefixo") or "atividade").strip()
    rp = ResearchPaths(root_dir=root_dir)
    candidates = {
        "org": root_dir / f"{prefixo}.org",
        "bib": root_dir / f"{prefixo}.bib",
        "debug": root_dir / f"{prefixo}_debug.json",
        "pdf": root_dir / f"{prefixo}.pdf",
        "prisma_svg": root_dir / f"{prefixo}_prisma.svg",
        "prisma_pdf": root_dir / f"{prefixo}_prisma.pdf",
    }
    rp.org_path = candidates["org"] if candidates["org"].exists() else None
    rp.bib_path = candidates["bib"] if candidates["bib"].exists() else None
    rp.debug_path = candidates["debug"] if candidates["debug"].exists() else None
    rp.pdf_path = candidates["pdf"] if candidates["pdf"].exists() else None
    rp.prisma_svg_path = candidates["prisma_svg"] if candidates["prisma_svg"].exists() else None
    rp.prisma_pdf_path = candidates["prisma_pdf"] if candidates["prisma_pdf"].exists() else None

    config_output = cfg.get("controle", {}).get("config_output")
    if config_output:
        cp = Path(config_output).expanduser().resolve()
        rp.config_path = cp if cp.exists() else None
    else:
        for pattern in ("*config*.toml",):
            matches = sorted(root_dir.glob(pattern))
            if matches:
                rp.config_path = matches[0]
                break

    cache_dir = root_dir / f"{prefixo}_fulltext_cache"
    rp.fulltext_cache_dir = cache_dir if cache_dir.exists() and cache_dir.is_dir() else None

    seen_selected: set[str] = set()
    if rp.debug_path and rp.debug_path.exists():
        try:
            debug_json = json.loads(rp.debug_path.read_text(encoding="utf-8"))
            for entry in (debug_json.get("selected_all") or []):
                if not isinstance(entry, dict):
                    continue
                rp.selected_entries.append(entry)
                pdf_raw = entry.get("downloaded_pdf_path")
                if pdf_raw:
                    pdf_path = Path(str(pdf_raw)).expanduser()
                    if not pdf_path.is_absolute():
                        pdf_path = (root_dir / pdf_path).resolve()
                    else:
                        pdf_path = pdf_path.resolve()
                    if pdf_path.exists() and pdf_path.is_file():
                        key = str(pdf_path)
                        if key not in seen_selected:
                            rp.selected_fulltext_paths.append(pdf_path)
                            seen_selected.add(key)
        except Exception as exc:
            debug_print(f"Falha ao ler debug JSON da pesquisa: {exc}")

    # fallback: se houver cache local, inclui somente o que casar com os selecionados
    if rp.fulltext_cache_dir and rp.selected_entries:
        by_name = {p.name: p for p in rp.fulltext_cache_dir.glob("*.pdf")}
        for entry in rp.selected_entries:
            pdf_raw = entry.get("downloaded_pdf_path")
            if not pdf_raw:
                continue
            name = Path(str(pdf_raw)).name
            cand = by_name.get(name)
            if cand and str(cand) not in seen_selected:
                rp.selected_fulltext_paths.append(cand)
                seen_selected.add(str(cand))

    # fallback residual: qualquer PDF do cache, se nada explícito foi encontrado
    if rp.fulltext_cache_dir and not rp.selected_fulltext_paths:
        for pdf in sorted(rp.fulltext_cache_dir.glob("*.pdf")):
            rp.selected_fulltext_paths.append(pdf)

    # backward compatibility: mantém também lista ampla de PDFs adicionais úteis
    for pdf in sorted(root_dir.glob("*.pdf")):
        if pdf in {rp.pdf_path, rp.prisma_pdf_path}:
            continue
        rp.fulltext_paths.append(pdf)
    if rp.fulltext_cache_dir:
        for pdf in sorted(rp.fulltext_cache_dir.glob("*.pdf")):
            if pdf not in rp.fulltext_paths:
                rp.fulltext_paths.append(pdf)
    return rp

def filter_research_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k in RESEARCH_SECTIONS}

def dumps_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(dumps_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        raise TypeError("Dict não deve ser serializado diretamente aqui")
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'

def dict_to_toml(cfg: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {dumps_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def run_research_stage(config_path: Path, script_path: Path, cwd: Path | None = None) -> None:
    ensure_command(sys.executable)
    if not script_path.exists():
        raise RuntimeError(f"Script de pesquisa não encontrado: {script_path}")
    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    debug_print("Executando etapa de pesquisa:", cmd)
    proc = subprocess.run(cmd, cwd=str(cwd or script_path.parent), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Etapa de pesquisa falhou com código {proc.returncode}.")

def extract_selected_titles_from_org(org_text: str) -> list[str]:
    titles: list[str] = []
    for m in re.finditer(r"\*\s+Texto selecionado\s+\d+.*?\n.*?\n([^\n]+)", org_text, flags=re.IGNORECASE | re.DOTALL):
        cand = m.group(1).strip()
        if cand and cand not in titles:
            titles.append(cand)
    return titles

def build_paper_context(cfg: dict[str, Any], research_paths: ResearchPaths) -> tuple[PaperContext, dict[str, Any]]:
    pesquisa = cfg.get("pesquisa", {})
    atividade = cfg.get("atividade", {})
    debug_json: dict[str, Any] = {}
    if research_paths.debug_path and research_paths.debug_path.exists():
        try:
            debug_json = json.loads(research_paths.debug_path.read_text(encoding="utf-8"))
        except Exception:
            debug_json = {}

    pergunta = None
    hipotese = None
    titulo_sugerido = pesquisa.get("trabalho") or None
    if isinstance(debug_json.get("proposal"), dict):
        proposal = debug_json["proposal"]
        pergunta = proposal.get("pergunta_pesquisa") or proposal.get("pergunta") or pergunta
        hipotese = proposal.get("hipotese") or hipotese
    if research_paths.org_path and research_paths.org_path.exists() and not titulo_sugerido:
        org_text = research_paths.org_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(?im)^#\+TITLE:\s*(.+)$", org_text)
        if m:
            titulo_sugerido = m.group(1).strip()

    context = PaperContext(
        tema=pesquisa.get("tema", ""),
        recorte=pesquisa.get("recorte", ""),
        objetivo=pesquisa.get("objetivo", ""),
        pergunta_pesquisa=pergunta,
        hipotese=hipotese,
        palavras_chave=list(pesquisa.get("palavras_chave") or []),
        titulo_sugerido=titulo_sugerido,
        tipo_estudo=pesquisa.get("tipo_estudo") or None,
        idiomas=list(pesquisa.get("idiomas") or ["português", "inglês"]),
        modo_origem=atividade.get("modo", "revisao_sistematica"),
        titulo_trabalho_base=pesquisa.get("trabalho") or None,
    )
    return context, debug_json

def maybe_rewrite_paper_context(
    client: OpenAI,
    model: str,
    cfg: dict[str, Any],
    context: PaperContext,
    guidance_docs: list[SourceDoc],
) -> tuple[PaperContext, dict[str, Any]]:
    paper = cfg.get("paper", {})
    usar_contexto_consolidado = bool(paper.get("usar_contexto_consolidado_da_pesquisa", True))
    reformular = bool(paper.get("reformular_tema_recorte_objetivo", False))
    modo_escrita = str(paper.get("modo_escrita") or "novo").strip().lower()

    if usar_contexto_consolidado and not reformular:
        return context, {"used": False, "source": "pesquisa_consolidada"}

    override_tema = str(paper.get("tema") or "").strip()
    override_recorte = str(paper.get("recorte") or "").strip()
    override_objetivo = str(paper.get("objetivo") or "").strip()
    if not reformular and (override_tema or override_recorte or override_objetivo):
        new_context = PaperContext(
            tema=override_tema or context.tema,
            recorte=override_recorte or context.recorte,
            objetivo=override_objetivo or context.objetivo,
            pergunta_pesquisa=context.pergunta_pesquisa,
            hipotese=context.hipotese,
            palavras_chave=context.palavras_chave,
            titulo_sugerido=context.titulo_sugerido,
            tipo_estudo=context.tipo_estudo,
            idiomas=context.idiomas,
            modo_origem=context.modo_origem,
            titulo_trabalho_base=context.titulo_trabalho_base,
        )
        return new_context, {"used": True, "source": "paper_override_manual"}

    prompt = textwrap.dedent(
        f"""
        Reformule, para a etapa de redação do paper, o tema, o recorte e o objetivo abaixo.

        Regras:
        - preserve o núcleo analítico da pesquisa já consolidada;
        - NÃO mude o assunto central;
        - apenas refine a formulação para a escrita do paper;
        - leve em conta o modo de escrita: {modo_escrita};
        - se o modo for "reescrever", priorize coesão e clareza do argumento;
        - se o modo for "expandir", mantenha o núcleo e permita maior abrangência analítica moderada;
        - retorne JSON estruturado.

        Contexto consolidado da pesquisa:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Orientações e artefatos disponíveis:
        {json.dumps(summarize_docs(guidance_docs), ensure_ascii=False, indent=2)}
        """
    ).strip()
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=RewrittenPaperContextOutput,
    )
    parsed = resp.output_parsed
    if parsed is None:
        return context, {"used": False, "source": "fallback_contexto_consolidado"}

    new_context = PaperContext(
        tema=parsed.tema or context.tema,
        recorte=parsed.recorte or context.recorte,
        objetivo=parsed.objetivo or context.objetivo,
        pergunta_pesquisa=parsed.pergunta_pesquisa or context.pergunta_pesquisa,
        hipotese=parsed.hipotese or context.hipotese,
        palavras_chave=context.palavras_chave,
        titulo_sugerido=parsed.titulo_sugerido or context.titulo_sugerido,
        tipo_estudo=context.tipo_estudo,
        idiomas=context.idiomas,
        modo_origem=context.modo_origem,
        titulo_trabalho_base=context.titulo_trabalho_base,
    )
    return new_context, {
        "used": True,
        "source": "ia_rewrite",
        "rationale": parsed.rationale,
        "modo_escrita": modo_escrita,
    }

def collect_guidance_docs(cfg: dict[str, Any], research_paths: ResearchPaths) -> list[SourceDoc]:
    docs: list[SourceDoc] = []

    def add(path: Path, kind: str, label: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            return
        try:
            if path.suffix.lower() in READABLE_SUFFIXES:
                text = read_text_file(path, 20000)
            else:
                text = shorten_text(path.read_text(encoding="utf-8", errors="ignore"), 20000)
            docs.append(SourceDoc(
                path=str(path),
                kind=kind,
                label=label or path.name,
                extracted_text=text,
            ))
        except Exception as exc:
            debug_print(f"Falha ao ler orientação {path}: {exc}")

    def add_from_value(raw_value: Any, kind: str, label: str) -> None:
        if raw_value is None:
            return
        raw = str(raw_value).strip()
        if not raw:
            return
        maybe_path = Path(os.path.expanduser(raw)).resolve()
        if maybe_path.exists() and maybe_path.is_file():
            add(maybe_path, kind, label)
            return
        docs.append(SourceDoc(
            path=f"inline:{kind}",
            kind=kind,
            label=label,
            extracted_text=shorten_text(raw, 20000),
        ))

    arquivo_orientacao = cfg.get("saida", {}).get("arquivo_orientacao")
    if arquivo_orientacao:
        add_from_value(arquivo_orientacao, "orientacao_externa", "orientacao_externa")

    if research_paths.org_path:
        add(research_paths.org_path, "pesquisa_org", "pesquisa_org")
    if research_paths.debug_path:
        add(research_paths.debug_path, "pesquisa_debug", "pesquisa_debug")

    paper = cfg.get("paper", {})
    orientacao = paper.get("orientacao")
    if orientacao:
        add_from_value(orientacao, "orientacao_paper", "orientacao_paper")

    for raw in paper.get("guidance_paths", []) or []:
        p = Path(os.path.expanduser(str(raw))).resolve()
        if p.exists():
            add(p, "guia_adicional")

    # reescrita/expansão do paper: usa o .org anterior como orientação
    if bool(paper.get("reescrever_a_partir_do_org_atual", False)) or bool(paper.get("preservar_estrutura_do_org_anterior", False)):
        org_existente_raw = str(paper.get("paper_org_existente") or "").strip()
        org_existente: Path | None = None
        if org_existente_raw:
            p = Path(os.path.expanduser(org_existente_raw)).resolve()
            if p.exists() and p.is_file():
                org_existente = p
        else:
            prefix = (paper.get("prefixo") or ((cfg.get("saida", {}).get("prefixo") or "atividade") + "_paper")).strip()
            output_dir_raw = paper.get("output_dir")
            if output_dir_raw:
                base = Path(output_dir_raw).expanduser().resolve()
                create_subdir = bool(paper.get("criar_subdiretorio", True))
                paper_dir = base / prefix if create_subdir else base
            else:
                paper_dir = build_paper_output_dir(cfg, research_paths.root_dir)
            candidate = paper_dir / f"{prefix}.org"
            if candidate.exists():
                org_existente = candidate

        if org_existente is not None:
            add(org_existente, "paper_org_anterior", "paper_org_anterior")

    return docs

def collect_base_docs(cfg: dict[str, Any], research_paths: ResearchPaths, max_files: int = 12) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    paper = cfg.get("paper", {})

    def add(path: Path, kind: str, label: str | None = None) -> None:
        if not path.exists() or path.suffix.lower() not in READABLE_SUFFIXES:
            return
        try:
            docs.append(SourceDoc(
                path=str(path),
                kind=kind,
                label=label or path.name,
                extracted_text=read_text_file(path, 25000),
            ))
        except Exception as exc:
            debug_print(f"Falha ao ler texto-base {path}: {exc}")

    usar_selecionados = bool(paper.get("usar_artigos_selecionados_pesquisa", True))
    permitir_correlata_extra = bool(paper.get("permitir_busca_correlata_extra", False))

    if usar_selecionados:
        for path in research_paths.selected_fulltext_paths[:max_files]:
            add(path, "texto_selecionado_formal")

        if not docs and research_paths.selected_entries:
            for entry in research_paths.selected_entries[:max_files]:
                title = str(entry.get("title") or "Sem título")
                abstract = str(entry.get("abstract") or entry.get("tldr") or "").strip()
                if not abstract:
                    continue
                docs.append(SourceDoc(
                    path=f"selected_all:{entry.get('paper_id') or slugify(title)}",
                    kind="texto_selecionado_resumo",
                    label=title,
                    extracted_text=shorten_text(abstract, 25000),
                    metadata={
                        "paper_id": entry.get("paper_id"),
                        "doi": entry.get("doi"),
                        "url": entry.get("url"),
                        "pdf_url": entry.get("pdf_url"),
                        "downloaded_pdf_path": entry.get("downloaded_pdf_path"),
                    },
                ))

    if permitir_correlata_extra:
        selected_set = {str(p.resolve()) for p in research_paths.selected_fulltext_paths}
        extra_count = 0
        for path in research_paths.fulltext_paths:
            if str(path.resolve()) in selected_set:
                continue
            add(path, "texto_correlato_extra_cache")
            extra_count += 1
            if len(docs) >= max_files + 8:
                break
        if extra_count:
            debug_print(f"Textos correlatos extras de cache adicionados: {extra_count}")

    if not docs and research_paths.org_path:
        add(research_paths.org_path, "texto_base_revisao", "pesquisa_org")
    if research_paths.pdf_path:
        add(research_paths.pdf_path, "relatorio_pesquisa_pdf", "pesquisa_pdf")
    return docs

def summarize_docs(docs: list[SourceDoc]) -> list[dict[str, Any]]:
    out = []
    for d in docs:
        out.append({
            "kind": d.kind,
            "label": d.label,
            "path": d.path,
            "excerpt": shorten_text(d.extracted_text, 2500),
            "bib_key": d.bib_key,
        })
    return out

def collect_extra_article_docs(cfg: dict[str, Any], max_files: int = 20) -> list[SourceDoc]:
    paper = cfg.get("paper", {})
    raw_items = paper.get("artigos_extras_paths", []) or []
    files = collect_readable_files(raw_items)[:max_files]
    docs: list[SourceDoc] = []
    for path in files:
        try:
            docs.append(SourceDoc(
                path=str(path),
                kind="artigo_extra",
                label=path.name,
                extracted_text=read_text_file(path, 25000),
                metadata={"source_path": str(path)},
            ))
        except Exception as exc:
            debug_print(f"Falha ao ler artigo extra {path}: {exc}")
    return docs

def infer_final_front_matter(client: OpenAI, model: str, context: PaperContext, org_text: str) -> FinalFrontMatterOutput:
    prompt = textwrap.dedent(
        f"""
        Gere os elementos finais de capa para um paper acadêmico em português.

        Contexto:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Conteúdo preliminar do paper:
        {shorten_text(org_text, 15000)}

        Regras:
        - title: título acadêmico final, claro e elegante.
        - paper_type: descrição curta do tipo do paper para a capa.
        - cover_note: nota curta, formal e compatível com a capa.
        """
    ).strip()
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=FinalFrontMatterOutput,
    )
    if resp.output_parsed is None:
        raise RuntimeError("A IA não retornou os elementos finais de capa.")
    return resp.output_parsed

def apply_final_front_matter(org_text: str, *, title: str, author: str, paper_type: str, cover_note: str, institution_name: str, course_name: str = "", discipline_name: str = "", professor_name: str = "", city_name: str = "Brasília") -> str:
    org_text = replace_org_header_line(org_text, "#+TITLE:", title)
    org_text = replace_org_header_line(org_text, "#+AUTHOR:", author)
    org_text = replace_latex_header_macro(org_text, "institution", institution_name)
    org_text = replace_latex_header_macro(org_text, "coursename", course_name)
    org_text = replace_latex_header_macro(org_text, "disciplinename", discipline_name)
    org_text = replace_latex_header_macro(org_text, "professorname", professor_name)
    org_text = replace_latex_header_macro(org_text, "cityname", city_name)
    org_text = replace_latex_header_macro(org_text, "papertype", paper_type)
    org_text = replace_latex_header_macro(org_text, "covernote", cover_note)
    return org_text

def build_paper_prompt(
    cfg: dict[str, Any],
    context: PaperContext,
    template_text: str,
    base_docs: list[SourceDoc],
    guidance_docs: list[SourceDoc],
    bib_keys: list[str],
    bib_entries: list[str],
    style: str,
) -> str:
    paper = cfg.get("paper", {})
    modo_escrita = str(paper.get("modo_escrita") or "novo").strip().lower()
    perfil_redacao = str(paper.get("perfil_redacao") or "academico_equilibrado").strip().lower()
    priorizar_citacoes = bool(paper.get("priorizar_citacoes_dos_selecionados", True))
    usar_contexto_consolidado = bool(paper.get("usar_contexto_consolidado_da_pesquisa", True))
    extras_so_complementam = bool(paper.get("extras_so_complementam", True))
    minimo_citacoes = int(paper.get("minimo_citacoes_dos_selecionados", 0) or 0)
    preservar_estrutura = bool(paper.get("preservar_estrutura_do_org_anterior", False))

    selected_keys = sorted({d.bib_key for d in base_docs if d.kind.startswith("texto_selecionado") and d.bib_key})
    extra_keys = sorted({d.bib_key for d in base_docs if d.kind == "artigo_extra" and d.bib_key})

    instrucoes_modo = {
        "novo": "Escreva um paper novo, mas usando a pesquisa consolidada como base metodológica e analítica.",
        "reescrever": "Reescreva o paper com maior coesão, clareza e densidade argumentativa, aproveitando o paper anterior se ele estiver entre as orientações.",
        "expandir": "Mantenha a linha argumentativa principal e expanda o paper com mais desenvolvimento analítico, usando os textos extras quando úteis.",
    }.get(modo_escrita, "Escreva um paper acadêmico coeso a partir da pesquisa consolidada.")

    instrucoes_perfil = {
        "academico_equilibrado": "Adote redação acadêmica equilibrada, com boa relação entre síntese, discussão teórica e fluidez.",
        "mais_teorico": "Privilegie densidade conceitual, diálogo teórico e elaboração interpretativa mais forte.",
        "mais_discursivo": "Privilegie fluidez discursiva, encadeamento argumentativo e texto mais ensaístico, sem perder rigor acadêmico.",
        "mais_sintetico": "Privilegie concisão, objetividade e alta compressão informacional, evitando expansões desnecessárias.",
    }.get(perfil_redacao, "Adote redação acadêmica equilibrada e coesa.")

    regra_citacao = "Priorize as citações dos artigos formalmente selecionados na pesquisa." if priorizar_citacoes else "Use as referências de forma equilibrada, sem obrigação de priorizar os selecionados."
    regra_extras = "Os artigos extras só podem complementar a argumentação; não substituem o papel central dos artigos formalmente selecionados." if extras_so_complementam else "Os artigos extras podem ter papel relevante, desde que não descaracterizem o núcleo da pesquisa."
    regra_estrutura = "Se houver um org anterior do paper entre as orientações, preserve sua estrutura principal e melhore o texto dentro dessa arquitetura." if preservar_estrutura else "Se houver um org anterior do paper entre as orientações, use-o apenas como referência, sem obrigação de preservar a mesma estrutura."
    contexto_regra = (
        "Tema, recorte e objetivo já consolidados pela pesquisa devem ser tratados como referência principal."
        if usar_contexto_consolidado else
        "É permitido trabalhar com formulação refinada do tema, recorte e objetivo para a escrita do paper, sem mudar o núcleo temático."
    )

    limites = {
        "limite_palavras_total": paper.get("limite_palavras_total"),
        "limite_palavras_introducao": paper.get("limite_palavras_introducao"),
        "limite_palavras_revisao": paper.get("limite_palavras_revisao"),
        "limite_palavras_conclusao": paper.get("limite_palavras_conclusao"),
    }
    limites_dict = {k: v for k, v in limites.items() if v not in (None, "", 0)}
    limites_txt = json.dumps(limites_dict, ensure_ascii=False, indent=2) if limites_dict else "Nenhum limite explícito informado."

    minimo_txt = ""
    if minimo_citacoes > 0 and selected_keys:
        minimo_txt = f"13. Cite pelo menos {minimo_citacoes} chaves dentre os artigos formalmente selecionados, preferindo estas quando pertinentes: {json.dumps(selected_keys, ensure_ascii=False)}."
    elif minimo_citacoes > 0:
        minimo_txt = f"13. Tente preservar no mínimo {minimo_citacoes} citações dos artigos formalmente selecionados, se houver chaves disponíveis no material de base."

    return textwrap.dedent(
        f"""
        Gere um paper acadêmico completo em Org-mode.

        Regras obrigatórias:
        1. Preserve o cabeçalho técnico do template (linhas #+...).
        2. O paper deve ser escrito em português, em tom acadêmico, argumentativo e coeso.
        3. Use apenas citações com as chaves bibliográficas fornecidas.
        4. Não invente chaves bibliográficas.
        5. Use citações nativas do Org Cite, como [cite:@chave] ou [cite/t:@chave].
        6. Não escreva numeração manual nos títulos.
        7. Não crie manualmente uma seção de referências fora do mecanismo de bibliografia do template.
        8. Use o estilo bibliográfico final {style.upper()}.
        9. {instrucoes_modo}
        10. {instrucoes_perfil}
        11. {regra_citacao}
        12. {regra_extras}
        {minimo_txt}
        14. {contexto_regra}
        15. {regra_estrutura}
        16. Se os textos-base forem insuficientes em algum ponto, trate essa limitação de forma honesta.
        17. Respeite, quando possível, os limites de extensão abaixo:
        {limites_txt}

        Contexto do paper:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Chaves bibliográficas de artigos formalmente selecionados:
        {json.dumps(selected_keys, ensure_ascii=False, indent=2)}

        Chaves bibliográficas de artigos extras:
        {json.dumps(extra_keys, ensure_ascii=False, indent=2)}

        Template Org:
        {shorten_text(template_text, 30000)}

        Chaves bibliográficas disponíveis:
        {json.dumps(bib_keys, ensure_ascii=False, indent=2)}

        Entradas bib disponíveis:
        {shorten_text(json.dumps(bib_entries, ensure_ascii=False, indent=2), 25000)}

        Textos-base:
        {json.dumps(summarize_docs(base_docs), ensure_ascii=False, indent=2)}

        Orientações e artefatos da pesquisa:
        {json.dumps(summarize_docs(guidance_docs), ensure_ascii=False, indent=2)}

        Retorne apenas o conteúdo completo do arquivo .org final.
        """
    ).strip()

def generate_paper_org(client: OpenAI, model: str, cfg: dict[str, Any], context: PaperContext, template_text: str, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], bib_filename: str, bib_entries: list[str], bib_keys: list[str], style: str) -> tuple[str, str]:
    prompt = build_paper_prompt(cfg, context, template_text, base_docs, guidance_docs, bib_keys, bib_entries, style)
    resp = client.responses.create(model=model, input=prompt)
    org_text = resp.output_text.strip()
    org_text = apply_citation_style(org_text, bib_filename, style)
    org_text = ensure_cover_command(org_text)
    org_text = cleanup_generated_org(org_text)
    return org_text, prompt

def build_paper_output_dir(cfg: dict[str, Any], research_root: Path) -> Path:
    paper = cfg.get("paper", {})
    prefix = (paper.get("prefixo") or ((cfg.get("saida", {}).get("prefixo") or "atividade") + "_paper")).strip()
    output_dir_raw = paper.get("output_dir")
    if output_dir_raw:
        base = Path(output_dir_raw).expanduser().resolve()
    else:
        base = research_root
    create_subdir = bool(paper.get("criar_subdiretorio", True))
    return base / prefix if create_subdir else base

def build_bundle_dir(cfg: dict[str, Any], research_root: Path) -> Path:
    pipeline = cfg.get("pipeline", {})
    bundle_dir_raw = pipeline.get("bundle_dir")
    if bundle_dir_raw:
        return Path(bundle_dir_raw).expanduser().resolve()
    return research_root / "paper_bundle"

def copy_if_exists(src: Path | None, dest: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)

def build_bundle(cfg: dict[str, Any], research_paths: ResearchPaths, paper_context: PaperContext, debug_json: dict[str, Any], bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = bundle_dir / "referencias"
    base_dir = bundle_dir / "textos_base"
    orient_dir = bundle_dir / "orientacoes"

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "mode_origem": cfg.get("atividade", {}).get("modo"),
        "research_root": str(research_paths.root_dir),
        "paper_context": asdict(paper_context),
        "artifacts": {},
    }

    artifacts = manifest["artifacts"]
    artifacts["research_org"] = copy_if_exists(research_paths.org_path, orient_dir / (research_paths.org_path.name if research_paths.org_path else "pesquisa.org"))
    artifacts["research_debug_json"] = copy_if_exists(research_paths.debug_path, orient_dir / (research_paths.debug_path.name if research_paths.debug_path else "pesquisa_debug.json"))
    artifacts["research_pdf"] = copy_if_exists(research_paths.pdf_path, orient_dir / (research_paths.pdf_path.name if research_paths.pdf_path else "pesquisa.pdf"))
    artifacts["research_prisma_svg"] = copy_if_exists(research_paths.prisma_svg_path, orient_dir / (research_paths.prisma_svg_path.name if research_paths.prisma_svg_path else "prisma.svg"))
    artifacts["research_prisma_pdf"] = copy_if_exists(research_paths.prisma_pdf_path, orient_dir / (research_paths.prisma_pdf_path.name if research_paths.prisma_pdf_path else "prisma.pdf"))
    artifacts["research_bib"] = copy_if_exists(research_paths.bib_path, refs_dir / (research_paths.bib_path.name if research_paths.bib_path else "references.bib"))
    artifacts["research_config"] = copy_if_exists(research_paths.config_path, orient_dir / (research_paths.config_path.name if research_paths.config_path else "config.toml"))

    copied_fulltexts: list[str] = []
    for pdf in (research_paths.selected_fulltext_paths or research_paths.fulltext_paths):
        target = base_dir / pdf.name
        copy_if_exists(pdf, target)
        copied_fulltexts.append(str(target))
    artifacts["fulltexts"] = copied_fulltexts
    artifacts["selected_entries_json"] = None
    selected_entries_path = bundle_dir / "selected_entries.json"
    if research_paths.selected_entries:
        write_text(selected_entries_path, json.dumps(research_paths.selected_entries, ensure_ascii=False, indent=2))
        artifacts["selected_entries_json"] = str(selected_entries_path)

    context_path = bundle_dir / "contexto_paper.json"
    write_text(context_path, json.dumps({
        "paper_context": asdict(paper_context),
        "debug_json_excerpt": debug_json,
        "config_excerpt": {
            "atividade": cfg.get("atividade", {}),
            "pesquisa": cfg.get("pesquisa", {}),
            "bibliografia": cfg.get("bibliografia", {}),
        },
    }, ensure_ascii=False, indent=2))
    manifest["context_json"] = str(context_path)

    manifest_path = bundle_dir / "manifest.json"
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest_path

def update_bundle_with_paper(
    manifest_path: Path | None,
    *,
    org_path: Path,
    bib_path: Path,
    context_json_path: Path,
    prompt_audit_path: Path,
    provenance_path: Path | None = None,
    extra_docs: list[SourceDoc] | None = None,
    pdf_path: Path | None = None,
) -> None:
    if manifest_path is None or not manifest_path.exists():
        return

    bundle_dir = manifest_path.parent
    paper_dir = bundle_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})

    def copy_local(src: Path, key: str) -> str:
        dest = paper_dir / src.name
        shutil.copy2(src, dest)
        artifacts[key] = str(dest)
        return str(dest)

    copy_local(org_path, "paper_org")
    copy_local(bib_path, "paper_bib")
    copy_local(context_json_path, "paper_context_json")
    copy_local(prompt_audit_path, "paper_prompt_audit")
    if provenance_path and provenance_path.exists():
        copy_local(provenance_path, "paper_provenance_json")
    if pdf_path and pdf_path.exists():
        copy_local(pdf_path, "paper_pdf")

    copied_extras: list[str] = []
    for doc in extra_docs or []:
        p = Path(doc.path)
        if p.exists() and p.is_file():
            dest = paper_dir / "extras" / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            copied_extras.append(str(dest))
    artifacts["paper_extra_articles"] = copied_extras

    manifest["updated_at"] = datetime.now().isoformat()
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))


def assemble_delivery_package(
    cfg: dict[str, Any],
    *,
    research_paths: ResearchPaths,
    paper_output_dir: Path,
    paper_prefix: str,
    manifest_path: Path | None,
    org_path: Path,
    bib_path: Path,
    context_json_path: Path,
    prompt_audit_path: Path,
    provenance_path: Path,
    reference_usage_json_path: Path,
    reference_usage_md_path: Path,
    section_limits_json_path: Path,
    paper_pdf_path: Path | None = None,
) -> Path | None:
    entrega = cfg.get("entrega", {})
    if not bool(entrega.get("gerar_pacote_final", True)):
        return None

    package_dir = paper_output_dir / "entrega_final"
    package_dir.mkdir(parents=True, exist_ok=True)

    def copy_optional(src: Path | None, subname: str | None = None) -> str | None:
        if src is None or not src.exists():
            return None
        dest = package_dir / (subname or src.name)
        shutil.copy2(src, dest)
        return str(dest)

    copied = {
        "pesquisa_org": copy_optional(research_paths.org_path),
        "pesquisa_bib": copy_optional(research_paths.bib_path),
        "pesquisa_debug": copy_optional(research_paths.debug_path),
        "pesquisa_pdf": copy_optional(research_paths.pdf_path),
        "paper_org": copy_optional(org_path),
        "paper_bib": copy_optional(bib_path),
        "paper_contexto": copy_optional(context_json_path),
        "paper_prompts": copy_optional(prompt_audit_path),
        "paper_proveniencia": copy_optional(provenance_path),
        "reference_usage_json": copy_optional(reference_usage_json_path),
        "reference_usage_md": copy_optional(reference_usage_md_path),
        "section_limits_json": copy_optional(section_limits_json_path),
        "bundle_manifest": copy_optional(manifest_path),
    }
    if paper_pdf_path and bool(entrega.get("incluir_paper_pdf", True)):
        copied["paper_pdf"] = copy_optional(paper_pdf_path)
    if research_paths.prisma_svg_path and bool(entrega.get("incluir_prisma_svg", True)):
        copied["pesquisa_prisma_svg"] = copy_optional(research_paths.prisma_svg_path)
    if research_paths.prisma_pdf_path and bool(entrega.get("incluir_prisma_pdf", True)):
        copied["pesquisa_prisma_pdf"] = copy_optional(research_paths.prisma_pdf_path)

    readme = package_dir / "README_entrega.md"
    lines = [
        f"# Pacote final de entrega — {paper_prefix}",
        "",
        "Este diretório reúne os principais artefatos finais do pipeline integrado.",
        "",
        "## Itens copiados",
    ]
    lines.extend(f"- **{k}**: `{v}`" for k, v in copied.items() if v)
    lines.append("")
    write_text(readme, "\n".join(lines))
    return package_dir


def build_latex_env(latex_extra_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if latex_extra_path:
        resolved = latex_extra_path.expanduser().resolve()
        latex_dir = resolved.parent if resolved.is_file() else resolved
        texinputs_prefix = f"{latex_dir.as_posix()}//:"
        env["TEXINPUTS"] = texinputs_prefix + env.get("TEXINPUTS", "")
        env["BIBINPUTS"] = texinputs_prefix + env.get("BIBINPUTS", "")
        env["BSTINPUTS"] = texinputs_prefix + env.get("BSTINPUTS", "")
    return env

def run_compile_sequence(org_path: Path, *, emacs_init: Path | None = None, academic_writing: Path | None = None, latex_extra_path: Path | None = None) -> Path:
    emacs = ensure_command("emacs")
    ensure_command("lualatex")
    ensure_command("biber")
    export_el = org_path.parent / f"{org_path.stem}_export_pdf.el"
    export_code = textwrap.dedent(
        f"""
        (require 'package)
        (package-initialize)
        (require 'org)
        (require 'ox)
        (require 'ox-latex)
        (require 'oc)
        (require 'oc-biblatex)
        (setq org-export-use-babel nil)
        (setq org-confirm-babel-evaluate nil)
        (setq org-latex-pdf-process
              '("lualatex -interaction=nonstopmode -file-line-error %f"
                "biber %b"
                "lualatex -interaction=nonstopmode -file-line-error %f"
                "lualatex -interaction=nonstopmode -file-line-error %f"))
        (find-file "{org_path.as_posix()}")
        (org-latex-export-to-pdf)
        """
    ).strip() + "\n"
    write_text(export_el, export_code)
    cmd = [emacs, "--batch", "-Q"]
    if emacs_init is not None:
        cmd.extend(["-l", str(emacs_init)])
    if academic_writing is not None:
        cmd.extend(["-l", str(academic_writing)])
    cmd.extend(["-l", str(export_el)])
    proc = subprocess.run(cmd, cwd=str(org_path.parent), capture_output=True, text=True, env=build_latex_env(latex_extra_path))
    debug_print("Compilando PDF:", cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Falha ao exportar PDF via Emacs batch.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    pdf_path = org_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"Exportação concluída sem erro, mas o PDF não foi encontrado: {pdf_path}")
    return pdf_path

def build_provenance_payload(
    cfg: dict[str, Any],
    research_paths: ResearchPaths,
    paper_context: PaperContext,
    base_docs: list[SourceDoc],
    guidance_docs: list[SourceDoc],
    extra_docs: list[SourceDoc],
    bib_keys: list[str],
    bib_entries: list[str],
    context_origin_info: dict[str, Any],
    template_path: Path | None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    paper = cfg.get("paper", {})
    return {
        "generated_at": datetime.now().isoformat(),
        "pipeline": {
            "executar_pesquisa": bool(cfg.get("pipeline", {}).get("executar_pesquisa", True)),
            "executar_paper": bool(cfg.get("pipeline", {}).get("executar_paper", True)),
            "pesquisa_dir_existente": str(cfg.get("pipeline", {}).get("pesquisa_dir_existente") or ""),
        },
        "paper_controls": {
            "modo_escrita": str(paper.get("modo_escrita") or "novo"),
            "perfil_redacao": str(paper.get("perfil_redacao") or "academico_equilibrado"),
            "usar_bib_da_pesquisa": bool(paper.get("usar_bib_da_pesquisa", True)),
            "incluir_artigos_extras_no_bib": bool(paper.get("incluir_artigos_extras_no_bib", True)),
            "permitir_busca_correlata_extra": bool(paper.get("permitir_busca_correlata_extra", False)),
            "priorizar_citacoes_dos_selecionados": bool(paper.get("priorizar_citacoes_dos_selecionados", True)),
            "extras_so_complementam": bool(paper.get("extras_so_complementam", True)),
            "minimo_citacoes_dos_selecionados": int(paper.get("minimo_citacoes_dos_selecionados", 0) or 0),
            "preservar_estrutura_do_org_anterior": bool(paper.get("preservar_estrutura_do_org_anterior", False)),
            "usar_contexto_consolidado_da_pesquisa": bool(paper.get("usar_contexto_consolidado_da_pesquisa", True)),
            "reformular_tema_recorte_objetivo": bool(paper.get("reformular_tema_recorte_objetivo", False)),
            "limites_palavras": {
                "total": paper.get("limite_palavras_total"),
                "introducao": paper.get("limite_palavras_introducao"),
                "revisao": paper.get("limite_palavras_revisao"),
                "conclusao": paper.get("limite_palavras_conclusao"),
            },
        },
        "context_origin": context_origin_info,
        "paper_context_final": asdict(paper_context),
        "template_path": str(template_path) if template_path else None,
        "bundle_manifest": str(manifest_path) if manifest_path else None,
        "research_artifacts": {
            "root_dir": str(research_paths.root_dir),
            "org_path": str(research_paths.org_path) if research_paths.org_path else None,
            "bib_path": str(research_paths.bib_path) if research_paths.bib_path else None,
            "debug_path": str(research_paths.debug_path) if research_paths.debug_path else None,
            "pdf_path": str(research_paths.pdf_path) if research_paths.pdf_path else None,
            "fulltext_cache_dir": str(research_paths.fulltext_cache_dir) if research_paths.fulltext_cache_dir else None,
        },
        "selected_entries_count": len(research_paths.selected_entries),
        "selected_fulltext_paths_used": [d.path for d in base_docs if d.kind.startswith("texto_selecionado")],
        "extra_article_paths_used": [d.path for d in extra_docs],
        "guidance_sources_used": [d.path for d in guidance_docs],
        "bib_keys_count": len(bib_keys),
        "bib_entries_count": len(bib_entries),
    }

def build_dry_run_report(cfg: dict[str, Any], research_cfg: dict[str, Any], research_script: Path, research_paths: ResearchPaths) -> dict[str, Any]:
    paper = cfg.get("paper", {})
    return {
        "generated_at": datetime.now().isoformat(),
        "dry_run": True,
        "pipeline": {**cfg.get("pipeline", {}), "executar_bundle": bool(cfg.get("pipeline", {}).get("executar_bundle", cfg.get("pipeline", {}).get("criar_bundle", True)))},
        "research_script_exists": research_script.exists(),
        "research_output_dir": str(research_paths.root_dir),
        "research_artifacts_exist": {
            "org": bool(research_paths.org_path and research_paths.org_path.exists()),
            "bib": bool(research_paths.bib_path and research_paths.bib_path.exists()),
            "debug": bool(research_paths.debug_path and research_paths.debug_path.exists()),
            "pdf": bool(research_paths.pdf_path and research_paths.pdf_path.exists()),
            "fulltext_cache_dir": bool(research_paths.fulltext_cache_dir and research_paths.fulltext_cache_dir.exists()),
        },
        "counts": {
            "selected_entries": len(research_paths.selected_entries),
            "selected_fulltexts": len(research_paths.selected_fulltext_paths),
            "fulltexts_total": len(research_paths.fulltext_paths),
        },
        "paper_plan": {
            "modo_escrita": str(paper.get("modo_escrita") or "novo"),
            "perfil_redacao": str(paper.get("perfil_redacao") or "academico_equilibrado"),
            "usar_artigos_selecionados_pesquisa": bool(paper.get("usar_artigos_selecionados_pesquisa", True)),
            "usar_bib_da_pesquisa": bool(paper.get("usar_bib_da_pesquisa", True)),
            "incluir_artigos_extras_no_bib": bool(paper.get("incluir_artigos_extras_no_bib", True)),
            "permitir_busca_correlata_extra": bool(paper.get("permitir_busca_correlata_extra", False)),
            "priorizar_citacoes_dos_selecionados": bool(paper.get("priorizar_citacoes_dos_selecionados", True)),
            "extras_so_complementam": bool(paper.get("extras_so_complementam", True)),
            "minimo_citacoes_dos_selecionados": int(paper.get("minimo_citacoes_dos_selecionados", 0) or 0),
            "preservar_estrutura_do_org_anterior": bool(paper.get("preservar_estrutura_do_org_anterior", False)),
            "usar_contexto_consolidado_da_pesquisa": bool(paper.get("usar_contexto_consolidado_da_pesquisa", True)),
            "reformular_tema_recorte_objetivo": bool(paper.get("reformular_tema_recorte_objetivo", False)),
            "limites_palavras": {
                "total": paper.get("limite_palavras_total"),
                "introducao": paper.get("limite_palavras_introducao"),
                "revisao": paper.get("limite_palavras_revisao"),
                "conclusao": paper.get("limite_palavras_conclusao"),
            },
            "artigos_extras_paths": cfg.get("paper", {}).get("artigos_extras_paths", []),
        },
        "research_config_excerpt": {
            "atividade": research_cfg.get("atividade", {}),
            "pesquisa": research_cfg.get("pesquisa", {}),
            "saida": research_cfg.get("saida", {}),
        },
    }

# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline integrado de pesquisa + paper usando TOML unificado.")
    p.add_argument("--config", required=True, help="Arquivo TOML unificado do pipeline.")
    p.add_argument("--model", default=None, help="Override do modelo OpenAI para a etapa do paper.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global DEBUG
    load_env()
    args = parse_args()
    DEBUG = bool(args.debug)

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    pipeline = cfg.get("pipeline", {})

    executar_pesquisa = bool(pipeline.get("executar_pesquisa", True))
    executar_paper = bool(pipeline.get("executar_paper", True))
    executar_bundle = bool(pipeline.get("executar_bundle", pipeline.get("criar_bundle", True)))
    dry_run = bool(cfg.get("controle", {}).get("dry_run", False))

    research_script = Path(pipeline.get("script_pesquisa") or DEFAULT_RESEARCH_SCRIPT).expanduser().resolve()
    research_cfg = filter_research_config(cfg)

    # prepara TOML temporário só com as seções aceitas pelo gerador de pesquisa
    research_output_dir = detect_research_output_dir(cfg)
    temp_cfg_path = research_output_dir / "pipeline_research_config.toml"
    research_output_dir.mkdir(parents=True, exist_ok=True)
    write_text(temp_cfg_path, dict_to_toml(research_cfg))

    research_paths = detect_research_paths(cfg)

    if dry_run:
        report = build_dry_run_report(cfg, research_cfg, research_script, research_paths)
        dry_run_path = research_output_dir / "pipeline_dry_run_report.json"
        write_text(dry_run_path, json.dumps(report, ensure_ascii=False, indent=2))
        print("Dry run concluído. Nenhuma chamada à OpenAI ou execução pesada foi realizada.")
        print(f"- Relatório: {dry_run_path}")
        return 0

    if executar_pesquisa:
        run_research_stage(temp_cfg_path, research_script)
        research_paths = detect_research_paths(cfg)
    if research_paths.org_path is None:
        raise RuntimeError(f"Não foi possível localizar o .org da pesquisa em {research_paths.root_dir}.")

    paper_context, debug_json = build_paper_context(cfg, research_paths)

    manifest_path: Path | None = None
    if executar_bundle:
        manifest_path = build_bundle(cfg, research_paths, paper_context, debug_json, build_bundle_dir(cfg, research_paths.root_dir))

    if not executar_paper:
        print("Pesquisa concluída. Etapa do paper desativada no TOML. O .org da pesquisa foi preservado.")
        if manifest_path:
            print(f"Bundle: {manifest_path}")
        return 0

    client, model = make_client(args.model or cfg.get("openai", {}).get("model"))

    # paper output
    paper_output_dir = build_paper_output_dir(cfg, research_paths.root_dir)
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    paper = cfg.get("paper", {})
    paper_prefix = (paper.get("prefixo") or ((cfg.get("saida", {}).get("prefixo") or "atividade") + "_paper")).strip()
    paper_bib_name = f"{paper_prefix}.bib"

    # template do paper
    # prioridade: template_paper.org; fallback organizacional: template_research.org
    template_path = None
    if paper.get("template_org"):
        template_path = Path(paper["template_org"]).expanduser().resolve()
    elif find_fallback_template():
        template_path = find_fallback_template()

    template_text = read_template_raw(template_path) if template_path and template_path.exists() else build_default_paper_template()

    # docs e bibliografia
    base_docs = collect_base_docs(cfg, research_paths)
    guidance_docs = collect_guidance_docs(cfg, research_paths)
    extra_docs = collect_extra_article_docs(cfg)
    if extra_docs:
        debug_print(f"Artigos extras carregados: {len(extra_docs)}")
    base_docs.extend(extra_docs)

    paper = cfg.get("paper", {})
    usar_bib_da_pesquisa = bool(paper.get("usar_bib_da_pesquisa", True))
    incluir_artigos_extras_no_bib = bool(paper.get("incluir_artigos_extras_no_bib", True))
    style = (paper.get("estilo_citacao") or cfg.get("bibliografia", {}).get("estilo_citacao") or DEFAULT_STYLE)

    bib_entries, bib_keys = (parse_bib_entries(research_paths.bib_path) if usar_bib_da_pesquisa else ([], []))
    if extra_docs and incluir_artigos_extras_no_bib:
        extra_docs, extra_bib_entries, extra_bib_keys = build_bib_entries_for_extra_docs(client, model, extra_docs, bib_keys)
        bib_entries.extend(extra_bib_entries)
        bib_keys.extend(extra_bib_keys)
    if not bib_entries:
        debug_print("Nenhum .bib disponível para o paper; bibliografia ficará vazia até revisão posterior.")

    assign_bib_keys_to_selected_docs(base_docs, research_paths, bib_entries)

    paper_context, context_origin_info = maybe_rewrite_paper_context(client, model, cfg, paper_context, guidance_docs)

    org_text, prompt_text = generate_paper_org(
        client=client,
        model=model,
        cfg=cfg,
        context=paper_context,
        template_text=template_text,
        base_docs=base_docs,
        guidance_docs=guidance_docs,
        bib_filename=paper_bib_name,
        bib_entries=bib_entries,
        bib_keys=bib_keys,
        style=style,
    )

    front = infer_final_front_matter(client, model, paper_context, org_text)
    atividade = cfg.get("atividade", {})
    org_text = apply_final_front_matter(
        org_text,
        title=front.title.strip(),
        author=str(atividade.get("aluno") or DEFAULT_AUTHOR),
        paper_type=front.paper_type.strip(),
        cover_note=front.cover_note.strip(),
        institution_name=str(paper.get("institution_name") or DEFAULT_INSTITUTION),
        course_name=str(atividade.get("curso") or ""),
        discipline_name=str(atividade.get("disciplina") or ""),
        professor_name=str(atividade.get("professor") or ""),
        city_name=str(atividade.get("polo") or "Brasília"),
    )

    # grava artefatos do paper
    org_path = paper_output_dir / f"{paper_prefix}.org"
    bib_path = paper_output_dir / paper_bib_name
    context_json_path = paper_output_dir / f"{paper_prefix}_contexto.json"
    prompt_audit_path = paper_output_dir / f"{paper_prefix}_prompts_auditoria.txt"
    provenance_path = paper_output_dir / f"{paper_prefix}_proveniencia.json"

    write_text(org_path, org_text)
    write_text(bib_path, "\n\n".join(bib_entries).strip() + ("\n" if bib_entries else ""))
    write_text(context_json_path, json.dumps({
        "generated_at": datetime.now().isoformat(),
        "paper_context": asdict(paper_context),
        "research_paths": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(research_paths).items()},
        "config_excerpt": {
            "atividade": cfg.get("atividade", {}),
            "pesquisa": cfg.get("pesquisa", {}),
            "paper": cfg.get("paper", {}),
        },
        "bundle_manifest": str(manifest_path) if manifest_path else None,
        "context_origin_info": context_origin_info,
        "bib_keys": bib_keys,
        "base_docs": [asdict(d) for d in base_docs],
        "guidance_docs": [asdict(d) for d in guidance_docs],
        "selected_entries": research_paths.selected_entries,
        "selected_fulltext_paths": [str(p) for p in research_paths.selected_fulltext_paths],
        "extra_article_paths": [d.path for d in extra_docs],
    }, ensure_ascii=False, indent=2))
    write_text(prompt_audit_path, "===== generate_paper_org =====\n" + prompt_text + "\n")
    write_text(provenance_path, json.dumps(
        build_provenance_payload(
            cfg=cfg,
            research_paths=research_paths,
            paper_context=paper_context,
            base_docs=base_docs,
            guidance_docs=guidance_docs,
            extra_docs=extra_docs,
            bib_keys=bib_keys,
            bib_entries=bib_entries,
            context_origin_info=context_origin_info,
            template_path=template_path,
            manifest_path=manifest_path,
        ),
        ensure_ascii=False,
        indent=2,
    ))

    reference_usage = build_reference_usage_map(org_text, base_docs, extra_docs, bib_keys)
    reference_usage_json_path = paper_output_dir / f"{paper_prefix}_uso_referencias.json"
    reference_usage_md_path = paper_output_dir / f"{paper_prefix}_uso_referencias.md"
    write_text(reference_usage_json_path, json.dumps(reference_usage, ensure_ascii=False, indent=2))
    write_text(reference_usage_md_path, render_reference_usage_markdown(reference_usage))

    section_limits_json_path = paper_output_dir / f"{paper_prefix}_limites_secoes.json"
    write_text(section_limits_json_path, json.dumps({
        "word_counts": count_org_words_per_top_section(org_text),
        "configured_limits": {
            "total": paper.get("limite_palavras_total"),
            "introducao": paper.get("limite_palavras_introducao"),
            "revisao": paper.get("limite_palavras_revisao"),
            "conclusao": paper.get("limite_palavras_conclusao"),
        }
    }, ensure_ascii=False, indent=2))

    # compila PDF do paper se pedido
    paper_pdf_path = None
    if bool(paper.get("exportar_pdf", cfg.get("saida", {}).get("exportar_pdf", False))):
        latex = cfg.get("latex", {})
        emacs_init = Path(latex["org_latex_class_init"]).expanduser().resolve() if latex.get("org_latex_class_init") else None
        academic_writing = emacs_init
        latex_extra_path = Path(latex["latex_extra_path"]).expanduser().resolve() if latex.get("latex_extra_path") else None
        paper_pdf_path = run_compile_sequence(org_path, emacs_init=None, academic_writing=academic_writing, latex_extra_path=latex_extra_path)

    print("\nArquivos gerados:")
    print(f"- Pesquisa ORG: {research_paths.org_path}")
    if research_paths.bib_path:
        print(f"- Pesquisa BIB: {research_paths.bib_path}")
    if research_paths.debug_path:
        print(f"- Pesquisa DEBUG: {research_paths.debug_path}")
    if research_paths.pdf_path:
        print(f"- Pesquisa PDF: {research_paths.pdf_path}")
    if research_paths.fulltext_cache_dir:
        print(f"- Pesquisa FULLTEXT CACHE: {research_paths.fulltext_cache_dir}")
    if research_paths.selected_fulltext_paths:
        print(f"- PDFs selecionados usados no paper: {len(research_paths.selected_fulltext_paths)}")
    if extra_docs:
        print(f"- Artigos extras usados no paper: {len(extra_docs)}")
    print(f"- Paper ORG: {org_path}")
    print(f"- Paper BIB: {bib_path}")
    print(f"- Paper CONTEXTO: {context_json_path}")
    print(f"- Paper PROVENIÊNCIA: {provenance_path}")
    print(f"- Paper PROMPTS: {prompt_audit_path}")
    print(f"- Uso de referências (JSON): {reference_usage_json_path}")
    print(f"- Uso de referências (MD): {reference_usage_md_path}")
    print(f"- Contagem/limites por seção: {section_limits_json_path}")
    if manifest_path:
        update_bundle_with_paper(
            manifest_path,
            org_path=org_path,
            bib_path=bib_path,
            context_json_path=context_json_path,
            prompt_audit_path=prompt_audit_path,
            provenance_path=provenance_path,
            extra_docs=extra_docs,
            pdf_path=paper_pdf_path,
        )

    package_dir = assemble_delivery_package(
        cfg,
        research_paths=research_paths,
        paper_output_dir=paper_output_dir,
        paper_prefix=paper_prefix,
        manifest_path=manifest_path,
        org_path=org_path,
        bib_path=bib_path,
        context_json_path=context_json_path,
        prompt_audit_path=prompt_audit_path,
        provenance_path=provenance_path,
        reference_usage_json_path=reference_usage_json_path,
        reference_usage_md_path=reference_usage_md_path,
        section_limits_json_path=section_limits_json_path,
        paper_pdf_path=paper_pdf_path,
    )

    minimo_sel = int(paper.get("minimo_citacoes_dos_selecionados", 0) or 0)
    if minimo_sel > 0:
        cited_sel = int(reference_usage.get("counts", {}).get("selected_cited_total", 0))
        if cited_sel < minimo_sel:
            print(f"Aviso: o paper citou {cited_sel} chave(s) de artigos selecionados, abaixo do mínimo configurado ({minimo_sel}).")

    if paper_pdf_path:
        print(f"- Paper PDF: {paper_pdf_path}")
    if manifest_path:
        print(f"- Bundle MANIFEST: {manifest_path}")
    if package_dir:
        print(f"- Pacote final de entrega: {package_dir}")

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        if DEBUG:
            traceback.print_exc()
        raise SystemExit(1)

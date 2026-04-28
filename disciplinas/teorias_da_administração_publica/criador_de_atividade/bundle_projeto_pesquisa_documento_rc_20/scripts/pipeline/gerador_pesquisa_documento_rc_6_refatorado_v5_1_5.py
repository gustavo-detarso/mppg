#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline integrado de pesquisa + documento acadêmico para o projeto MPPG, com suporte a mock_run.

Objetivo:
1. Ler um TOML unificado.
2. Extrair dele um TOML compatível com o gerador_pesquisa_rc_2.py.
3. Executar opcionalmente o gerador_pesquisa_rc_2.py (PRISMA ou empírico).
4. Montar um bundle de handoff com os artefatos gerados.
5. Gerar opcionalmente um documento acadêmico em Org-mode a partir do contexto e dos artefatos da pesquisa.
6. Preservar o .org da pesquisa e o .org do documento acadêmico como artefatos distintos.

Este script não tenta reimplementar o motor metodológico do gerador_pesquisa_rc_2.py.
Ele o reutiliza como etapa de pesquisa e assume a etapa redacional do documento acadêmico, consumindo explicitamente os PDFs selecionados na pesquisa quando disponíveis.
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
DEFAULT_RESEARCH_SCRIPT = "./gerador_pesquisa_rc_2.py"
DEFAULT_AUTHOR = "Gustavo M. Mendes de Tarso"
DEFAULT_INSTITUTION = "Fundação Getúlio Vargas"
DEFAULT_SCHOOL = ""
DEFAULT_PAPER_TEMPLATE = "template_paper.org"
DEFAULT_DISSERTATION_TEMPLATE = "template_dissertacao.org"
DEFAULT_FALLBACK_TEMPLATE = "template_research.org"
RESEARCH_BRIDGE_OUT_ORIENTACAO = "_".join(["arquivo", "orientacao"])
RESEARCH_BRIDGE_TRIAGEM_PROMPT = "_".join(["triagem", "prompt", "path"])
RESEARCH_BRIDGE_TRIAGEM_EXTRAS = "_".join(["diretivas", "extras"])
DEFAULT_STYLE = "apa"
DEBUG = False

DEFAULT_DOC_TYPE = "paper"
DEFAULT_DOC_PREFIX_BY_TYPE = {
    "paper": "paper",
    "dissertacao": "dissertacao",
}

def normalize_document_type(raw: str | None) -> str:
    value = (raw or DEFAULT_DOC_TYPE).strip().lower()
    aliases = {
        "paper": "paper",
        "artigo": "paper",
        "papel": "paper",
        "documento": "paper",
        "dissertacao": "dissertacao",
        "dissertação": "dissertacao",
        "thesis": "dissertacao",
    }
    return aliases.get(value, value)


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
class DocumentContext:
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
    documento_type: str
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

class RewrittenDocumentContextOutput(BaseModel):
    tema: str
    recorte: str
    objetivo: str
    pergunta_pesquisa: str | None = None
    hipotese: str | None = None
    titulo_sugerido: str | None = None
    rationale: str | None = None

class DissertationFrontSectionsOutput(BaseModel):
    dedicatoria: str = ""
    agradecimentos: str = ""
    epigrafe_texto: str = ""
    epigrafe_autor: str = ""
    resumo: str
    palavras_chave: list[str] = Field(default_factory=list)
    abstract: str
    keywords: list[str] = Field(default_factory=list)
    lista_ilustracoes: list[str] = Field(default_factory=list)
    lista_tabelas: list[str] = Field(default_factory=list)
    siglas: list[str] = Field(default_factory=list)
    simbolos: list[str] = Field(default_factory=list)
    glossario: list[str] = Field(default_factory=list)

class DissertationStageBodyOutput(BaseModel):
    org_body: str

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def debug_print(*parts: object) -> None:
    if DEBUG:
        print("[DEBUG]", *parts, file=sys.stderr)

def safe_resolve_user_path(raw: Any, base_dir: Path | None = None) -> Path | None:
    text = str(raw).strip()
    if not text:
        return None
    looks_like_path = (
        text.startswith(("./", "../", "~/", "/"))
        or os.sep in text
        or (os.altsep is not None and os.altsep in text)
        or bool(Path(text).suffix)
    )
    if ("\n" in text) or (len(text) > 240 and not looks_like_path):
        return None
    try:
        path = Path(os.path.expanduser(text))
        if not path.is_absolute():
            path = ((base_dir or Path.cwd()) / path).resolve()
        else:
            path = path.resolve()
        return path
    except OSError:
        return None
    except Exception:
        return None


def get_config_base_dir(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("__config_dir__")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return Path.cwd().resolve()


def resolve_configured_path(raw: Any, cfg: dict[str, Any], *, base_dir: Path | None = None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    bases: list[Path] = []
    if base_dir is not None:
        bases.append(base_dir.resolve())
    cfg_base = get_config_base_dir(cfg)
    if cfg_base not in bases:
        bases.append(cfg_base)
    cwd = Path.cwd().resolve()
    if cwd not in bases:
        bases.append(cwd)
    cfg_parent = cfg_base.parent.resolve()
    if cfg_parent not in bases:
        bases.append(cfg_parent)
    for candidate_base in bases:
        resolved = safe_resolve_user_path(text, base_dir=candidate_base)
        if resolved is not None and resolved.exists():
            return resolved
    return safe_resolve_user_path(text, base_dir=bases[0] if bases else None)

def safe_path_exists_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False
    except Exception:
        return False

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


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value

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

    maybe_path = safe_resolve_user_path(raw)
    if safe_path_exists_file(maybe_path):
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

def remove_latex_header_macro(org_text: str, macro: str) -> str:
    """Remove uma macro #+LATEX_HEADER de linha única, se existir.

    Usada para campos opcionais: se o TOML deixar o campo vazio, a macro
    não deve sequer aparecer no .org gerado.
    """
    pattern = re.compile(rf'^\s*#\+LATEX_HEADER:\s*\\{re.escape(macro)}\{{.*\}}\s*$', re.MULTILINE)
    return pattern.sub('', org_text)

def replace_or_insert_latex_header_macro(org_text: str, macro: str, new_value: str) -> str:
    updated = replace_latex_header_macro(org_text, macro, new_value)
    if updated != org_text:
        return updated
    lines = org_text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#+LATEX_HEADER:"):
            insert_at = i + 1
    lines.insert(insert_at, f"#+LATEX_HEADER: \\{macro}{{{new_value}}}")
    return "\n".join(lines)

def set_required_latex_header_macro(org_text: str, macro: str, value: str) -> str:
    return replace_or_insert_latex_header_macro(org_text, macro, str(value or '').strip())

def set_optional_latex_header_macro(org_text: str, macro: str, value: str | None) -> str:
    clean = re.sub(r"\s+", " ", str(value or "").strip())
    if not clean or is_placeholder_value(clean):
        return remove_latex_header_macro(org_text, macro)
    return replace_or_insert_latex_header_macro(org_text, macro, clean)

def set_optional_date_latex_header_macro(org_text: str, macro: str, value: str | None) -> str:
    clean = re.sub(r"\s+", " ", str(value or "").strip())
    # Data vazia ou placeholder de formulário deve desaparecer do .org.
    # Se o usuário escrever explicitamente "A definir", o valor é mantido.
    if not clean or re.search(r"[_]{2,}|/_[^ ]|^__/.+", clean):
        return remove_latex_header_macro(org_text, macro)
    return replace_or_insert_latex_header_macro(org_text, macro, clean)

def insert_latex_header_line(org_text: str, line: str) -> str:
    lines = org_text.splitlines()
    insert_at = 0
    for i, current in enumerate(lines):
        if current.startswith("#+LATEX_HEADER:"):
            insert_at = i + 1
    lines.insert(insert_at, line)
    return "\n".join(lines)

def split_title_and_subtitle(title: str) -> tuple[str, str]:
    raw = re.sub(r"\s+", " ", str(title or "").strip())
    if not raw:
        return "TÍTULO DA DISSERTAÇÃO", ""
    for sep in (" — ", " – ", ": "):
        if sep in raw:
            a, b = raw.split(sep, 1)
            return a.strip(), b.strip()
    return raw, ""


def derive_area_from_course(course_name: str) -> str:
    text = re.sub(r"\s+", " ", str(course_name or "").strip())
    if not text:
        return "Políticas Públicas e Governo"
    for prefix in ("Mestrado Profissional em ", "Mestrado em ", "Master in "):
        if text.lower().startswith(prefix.lower()):
            remainder = text[len(prefix):].strip()
            if remainder:
                return remainder
    return text



def is_placeholder_value(value: str | None) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text:
        return True
    placeholders = {
        "seu nome",
        "nome completo do aluno",
        "nome do aluno",
        "autor da epígrafe",
        "orientador(a) a definir",
        "a definir",
        "[inserir]",
        "título do apêndice",
        "titulo do apendice",
        "título do anexo",
        "titulo do anexo",
    }
    if text in placeholders:
        return True
    return text.startswith("[inserir")

def normalize_banca_member(raw: Any) -> dict[str, str] | None:
    """Normaliza um item da banca vindo do TOML.

    Formatos aceitos:
      [[documento.banca]]
      nome = "Prof. Dr. Nome"
      funcao = "Orientador"
      instituicao = "Fundação Getúlio Vargas"

    Também aceita string no formato:
      "Orientador | Prof. Dr. Nome | Fundação Getúlio Vargas"
    """
    if isinstance(raw, dict):
        nome = str(raw.get("nome") or raw.get("name") or "").strip()
        funcao = str(raw.get("funcao") or raw.get("função") or raw.get("papel") or raw.get("tipo") or raw.get("role") or "").strip()
        instituicao = str(raw.get("instituicao") or raw.get("instituição") or raw.get("vinculo") or raw.get("vínculo") or raw.get("affiliation") or "").strip()
    else:
        parts = [part.strip() for part in str(raw or "").split("|")]
        if len(parts) >= 3:
            funcao, nome, instituicao = parts[0], parts[1], " | ".join(parts[2:])
        elif len(parts) == 2:
            funcao, nome, instituicao = parts[0], parts[1], ""
        else:
            funcao, nome, instituicao = "", str(raw or "").strip(), ""

    if is_placeholder_value(nome):
        return None
    if is_placeholder_value(funcao):
        funcao = ""
    if is_placeholder_value(instituicao):
        instituicao = ""
    return {"funcao": funcao, "nome": nome, "instituicao": instituicao}


def collect_banca_members(documento: dict[str, Any]) -> list[dict[str, str]]:
    """Coleta a banca examinadora dinâmica definida no TOML.

    A fonte preferencial é [[documento.banca]] ou [[documento.banca_examinadora]].
    Mantém compatibilidade com membro_banca_1, membro_banca_2 etc., mas sem
    preencher placeholders automáticos.
    """
    raw_members = documento.get("banca") or documento.get("banca_examinadora") or []
    members: list[dict[str, str]] = []
    if isinstance(raw_members, list):
        for raw in raw_members:
            parsed = normalize_banca_member(raw)
            if parsed:
                members.append(parsed)
    elif raw_members:
        parsed = normalize_banca_member(raw_members)
        if parsed:
            members.append(parsed)

    if members:
        return members

    # Compatibilidade com campos antigos. A função/cargo pode ser informada em
    # funcao_banca_1, papel_banca_1, tipo_banca_1 etc.
    for idx in range(1, 11):
        raw_name = documento.get(f"membro_banca_{idx}")
        if raw_name is None:
            continue
        nome = str(raw_name or "").strip()
        if is_placeholder_value(nome):
            continue
        funcao = str(
            documento.get(f"funcao_banca_{idx}")
            or documento.get(f"função_banca_{idx}")
            or documento.get(f"papel_banca_{idx}")
            or documento.get(f"tipo_banca_{idx}")
            or ""
        ).strip()
        instituicao = str(
            documento.get(f"instituicao_banca_{idx}")
            or documento.get(f"instituição_banca_{idx}")
            or documento.get(f"vinculo_banca_{idx}")
            or documento.get(f"vínculo_banca_{idx}")
            or ""
        ).strip()
        parsed = normalize_banca_member({"nome": nome, "funcao": funcao, "instituicao": instituicao})
        if parsed:
            members.append(parsed)
    return members


def clear_banca_headers(org_text: str) -> str:
    for macro in ("membrobanca", "membrobancaum", "membrobancadois", "membrobancatres"):
        # remove também a macro dinâmica com três argumentos.
        if macro == "membrobanca":
            org_text = re.sub(r'^\s*#\+LATEX_HEADER:\s*\\membrobanca\{.*\}\{.*\}\{.*\}\s*$', '', org_text, flags=re.MULTILINE)
        else:
            org_text = remove_latex_header_macro(org_text, macro)
    return org_text


def apply_banca_headers(org_text: str, documento: dict[str, Any]) -> str:
    org_text = clear_banca_headers(org_text)
    members = collect_banca_members(documento)
    for member in members:
        funcao = member.get("funcao", "")
        nome = member.get("nome", "")
        instituicao = member.get("instituicao", "")
        org_text = insert_latex_header_line(
            org_text,
            f"#+LATEX_HEADER: \\membrobanca{{{funcao}}}{{{nome}}}{{{instituicao}}}",
        )
    return org_text


def apply_dissertation_template_metadata(org_text: str, cfg: dict[str, Any], final_title: str) -> str:
    atividade = cfg.get("atividade", {})
    documento = cfg.get("documento", {})
    institution = str(documento.get("institution_name") or DEFAULT_INSTITUTION or "Fundação Getúlio Vargas").strip()
    institution_upper = institution.upper()
    raw_author = str(atividade.get("aluno") or DEFAULT_AUTHOR).strip()
    author = DEFAULT_AUTHOR if is_placeholder_value(raw_author) else raw_author
    raw_course_name = str(atividade.get("curso") or "Mestrado Profissional em Políticas Públicas e Governo").strip()
    course_name = "Mestrado Profissional em Políticas Públicas e Governo" if is_placeholder_value(raw_course_name) else raw_course_name
    raw_professor_name = str(atividade.get("professor") or "").strip()
    professor_name = "" if is_placeholder_value(raw_professor_name) else raw_professor_name
    raw_city_name = str(atividade.get("polo") or "Brasília").strip()
    city_name = "Brasília" if is_placeholder_value(raw_city_name) else raw_city_name
    raw_school_name = str(documento.get("school_name") or DEFAULT_SCHOOL).strip()
    school_name = "" if is_placeholder_value(raw_school_name) else raw_school_name
    raw_program_name = str(documento.get("program_name") or course_name).strip()
    program_name = course_name if is_placeholder_value(raw_program_name) else raw_program_name
    raw_area_name = str(documento.get("area_de_concentracao") or derive_area_from_course(course_name)).strip()
    area_name = derive_area_from_course(course_name) if is_placeholder_value(raw_area_name) else raw_area_name
    linha_pesquisa = str(documento.get("linha_pesquisa") or documento.get("linha_de_pesquisa") or "").strip()
    coorientador = str(documento.get("coorientador") or documento.get("co_orientador") or documento.get("coorientadora") or "").strip()
    dissertation_year = str(documento.get("ano") or datetime.now().year)
    title_main, title_sub = split_title_and_subtitle(final_title)
    default_nature_target = school_name if school_name else institution
    connector = " da Fundação Getúlio Vargas" if school_name and school_name.strip().lower() != institution.strip().lower() else ""
    nature_text = str(documento.get("natureza_trabalho") or f"Dissertação apresentada à {default_nature_target}{connector}, como requisito para obtenção do título de Mestre em {area_name}.").strip()

    required_replacements = {
        "autor": author,
        "titulo": title_main,
        "cidade": city_name,
        "ano": dissertation_year,
        "instituicao": institution_upper,
        "programa": program_name,
        "curso": course_name,
        "areadeconcentracao": area_name,
        "naturezatrabalho": nature_text,
    }
    optional_replacements = {
        "subtitulo": title_sub,
        "escola": school_name,
        "linhapesquisa": linha_pesquisa,
        "orientador": professor_name,
        "coorientador": coorientador,
    }

    for macro, value in required_replacements.items():
        org_text = set_required_latex_header_macro(org_text, macro, value)
    for macro, value in optional_replacements.items():
        org_text = set_optional_latex_header_macro(org_text, macro, value)

    org_text = set_optional_date_latex_header_macro(org_text, "dataaprovacao", documento.get("data_aprovacao"))
    org_text = apply_banca_headers(org_text, documento)

    # Remove linhas em branco excessivas criadas pela retirada de macros opcionais.
    org_text = re.sub(r"\n{3,}", "\n\n", org_text)
    return org_text

def apply_citation_style(org_text: str, bib_filename: str, style: str) -> str:
    style = (style or DEFAULT_STYLE).strip().lower() or DEFAULT_STYLE
    bib_name = Path(str(bib_filename)).name
    cite_line = f"#+CITE_EXPORT: biblatex {style}"
    org_text = re.sub(r"(?im)^\s*#\+CITE_EXPORT:.*\n?", "", org_text)
    org_text = re.sub(r"(?im)^\s*#\+BIBLIOGRAPHY:\s+.*\n?", "", org_text)
    org_text = re.sub(r"(?im)^\s*#\+PRINT_BIBLIOGRAPHY:.*\n?", "", org_text)
    org_text = re.sub(r"(?im)^\s*#\+LATEX_HEADER:\s*\\ExecuteBibliographyOptions\{.*\}\s*\n?", "", org_text)
    org_text = re.sub(r"(?im)^\s*#\+LATEX_HEADER:\s*\\PassOptionsToPackage\{.*\}\{biblatex\}\s*\n?", "", org_text)
    lines = [ln for ln in org_text.splitlines() if ln.strip()]
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#+LATEX_CLASS") or line.startswith("#+LATEX_CLASS_OPTIONS"):
            insert_at = i + 1
    lines.insert(insert_at, cite_line)
    lines.append(f"#+BIBLIOGRAPHY: {bib_name}")
    return "\n".join(lines).strip() + "\n"

def ensure_document_class(org_text: str, doc_type: str) -> str:
    class_name = "fgv-dissertacao" if normalize_document_type(doc_type) == "dissertacao" else "fgv-paper"
    org_text = re.sub(r"(?im)^\s*#\+LATEX_CLASS:.*\n?", "", org_text)
    return f"#+LATEX_CLASS: {class_name}\n" + org_text.lstrip("\n")

def normalize_bibliography_block(org_text: str) -> str:
    org_text = re.sub(r"(?im)^\s*#\+PRINT_BIBLIOGRAPHY:.*\n?", "", org_text)
    org_text = re.sub(r"(?im)^\*+\s+(Refer[êe]ncias|Bibliography)\s*\n?", "", org_text)
    org_text = org_text.rstrip() + "\n\n#+PRINT_BIBLIOGRAPHY:\n"
    return org_text

def ensure_cover_command(org_text: str) -> str:
    if "\\usepapercover" not in org_text or "#+LATEX: \\makemytitle" in org_text:
        return org_text
    marker = "#+begin_abstract"
    if marker in org_text:
        return org_text.replace(marker, "#+LATEX: \\makemytitle\n\n" + marker, 1)
    return org_text + "\n#+LATEX: \\makemytitle\n"

def cleanup_generated_org(org_text: str) -> str:
    org_text = re.sub(r"(?im)^\s*[,;:]+\s*$\n?", "", org_text)
    org_text = re.sub(r"(?im)^\s*P\d+\\?\}\s*$\n?", "", org_text)
    org_text = re.sub(r"(?im)^\s*<empty citation>\s*$\n?", "", org_text)
    org_text = re.sub(r"\n{3,}", "\n\n", org_text)
    org_text = re.sub(r"[ \t]+\n", "\n", org_text)
    org_text = re.sub(r"(?ms)^\*\s+EP[ÍI]GRAFE\n(?:(?!^\* ).)*?(?:Autor da ep[ií]grafe|\[\.\.\.\]|“\[\.\.\.\]”).*?(?=^\* |\Z)", "", org_text)
    org_text = re.sub(r"(?ms)^\*\s+AP[ÊE]NDICE\s+A\s+[—-]\s+T[ÍI]TULO DO AP[ÊE]NDICE.*?(?=^\* |\Z)", "", org_text)
    org_text = re.sub(r"(?ms)^\*\s+ANEXO\s+A\s+[—-]\s+T[ÍI]TULO DO ANEXO.*?(?=^\* |\Z)", "", org_text)
    return org_text.strip() + "\n"

def strip_org_cite_directives(doc: str) -> str:
    banned = ("#+cite_export:", "#+bibliography:", "#+print_bibliography:")
    kept = []
    for line in doc.splitlines():
        if line.strip().lower().startswith(banned):
            continue
        kept.append(line)
    return "\n".join(kept).strip() + "\n"


def org_uses_citation_pipeline(doc: str) -> bool:
    if not doc or not doc.strip():
        return False
    patterns = (
        r"(?im)^\s*#\+cite_export:",
        r"(?im)^\s*#\+bibliography:",
        r"(?im)^\s*#\+print_bibliography:",
        r"\[cite:[^\]]+\]",
    )
    return any(re.search(pattern, doc) for pattern in patterns)

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
            key = _bib_key_from_filename(p, bib_entries)
        if key is None:
            title_norm = normalize_title_loose(doc.label)
            key = by_title.get(title_norm)
        if key is None:
            label_norm = normalize_title_loose(doc.label)
            for bib in bib_entries:
                meta = parse_bib_entry_meta(bib)
                bib_key = meta.get("key")
                bib_title_norm = normalize_title_loose(str(meta.get("title") or ""))
                if bib_key and label_norm and bib_title_norm and (label_norm in bib_title_norm or bib_title_norm in label_norm):
                    key = str(bib_key)
                    break
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


def normalize_heading_loose(text: str) -> str:
    return normalize_title_loose(text or "")


def classify_top_section_bucket(title: str) -> str:
    t = normalize_heading_loose(title)
    if any(x in t for x in ("introducao",)):
        return "introducao"
    if any(x in t for x in ("referencial", "revisao", "fundamentacao", "marco teorico")):
        return "referencial_teorico"
    if any(x in t for x in ("metodologia", "metodo", "procedimentos", "estrategia de pesquisa")):
        return "metodologia"
    if any(x in t for x in ("resultados", "discussao", "analise", "achados", "implicacoes")):
        return "resultados_discussao"
    if any(x in t for x in ("consideracoes finais", "conclusao", "conclusoes", "sintese final", "agenda futura")):
        return "conclusao"
    return "outros"


def aggregate_section_buckets(word_counts: dict[str, int]) -> dict[str, int]:
    buckets = {
        "introducao": 0,
        "referencial_teorico": 0,
        "metodologia": 0,
        "resultados_discussao": 0,
        "conclusao": 0,
        "outros": 0,
        "documento_total": int(word_counts.get("documento_total", 0) or 0),
    }
    for title, words in word_counts.items():
        if title == "documento_total":
            continue
        bucket = classify_top_section_bucket(title)
        buckets[bucket] += int(words or 0)
    return buckets


def dissertation_generation_targets(cfg: dict[str, Any]) -> dict[str, int]:
    documento = cfg.get("documento", {})
    doc_type_local = normalize_document_type(documento.get("tipo_documento"))
    if doc_type_local != "dissertacao":
        return {
            "documento_total_min": int(documento.get("limite_palavras_total") or 6000),
            "introducao_min": int(documento.get("limite_palavras_introducao") or 800),
            "referencial_teorico_min": 1800,
            "metodologia_min": 1000,
            "resultados_discussao_min": 1800,
            "conclusao_min": int(documento.get("limite_palavras_conclusao") or 800),
        }
    return {
        "documento_total_min": int(documento.get("min_palavras_total") or documento.get("limite_palavras_total") or 14000),
        "documento_total_alvo": int(documento.get("alvo_palavras_total") or 18000),
        "introducao_min": int(documento.get("min_palavras_introducao") or documento.get("limite_palavras_introducao") or 1400),
        "referencial_teorico_min": int(documento.get("min_palavras_referencial") or documento.get("limite_palavras_revisao") or 4200),
        "metodologia_min": int(documento.get("min_palavras_metodologia") or 1800),
        "resultados_discussao_min": int(documento.get("min_palavras_resultados") or 5200),
        "conclusao_min": int(documento.get("min_palavras_conclusao") or documento.get("limite_palavras_conclusao") or 1400),
    }


def should_expand_dissertation(cfg: dict[str, Any], org_text: str, selected_keys: list[str]) -> tuple[bool, dict[str, Any]]:
    documento = cfg.get("documento", {})
    doc_type_local = normalize_document_type(documento.get("tipo_documento"))
    if doc_type_local != "dissertacao":
        return False, {}
    counts = count_org_words_per_top_section(org_text)
    buckets = aggregate_section_buckets(counts)
    targets = dissertation_generation_targets(cfg)
    cited = extract_cited_keys_from_org(org_text)
    cited_selected = [k for k in cited if k in set(selected_keys)]
    reasons: list[str] = []
    if buckets.get("documento_total", 0) < targets.get("documento_total_min", 0):
        reasons.append(f"documento_total<{targets.get('documento_total_min')}")
    for bucket_key, target_key in (
        ("introducao", "introducao_min"),
        ("referencial_teorico", "referencial_teorico_min"),
        ("metodologia", "metodologia_min"),
        ("resultados_discussao", "resultados_discussao_min"),
        ("conclusao", "conclusao_min"),
    ):
        if buckets.get(bucket_key, 0) < targets.get(target_key, 0):
            reasons.append(f"{bucket_key}<{targets.get(target_key)}")
    return bool(reasons), {
        "word_counts": counts,
        "bucket_counts": buckets,
        "targets": targets,
        "selected_cited_total": len(cited_selected),
        "selected_keys_total": len(selected_keys),
        "reasons": reasons,
    }


def build_fulltext_cache_pdf_paths(research_paths: ResearchPaths) -> list[Path]:
    """Return every PDF actually present in the research fulltext cache.

    The dissertation citation-coverage target is now the download/cache folder
    (prefixo_fulltext_cache/), not an arbitrary minimum number of selected items.
    """
    seen: set[str] = set()
    paths: list[Path] = []

    def add(path: Path | None) -> None:
        if not path:
            return
        try:
            p = Path(path).expanduser()
        except Exception:
            return
        if not p.exists() or p.suffix.lower() != ".pdf":
            return
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            paths.append(p)

    if research_paths.fulltext_cache_dir and research_paths.fulltext_cache_dir.exists():
        for pdf in sorted(research_paths.fulltext_cache_dir.glob("*.pdf")):
            add(pdf)
    for pdf in research_paths.fulltext_paths:
        add(pdf)
    for pdf in research_paths.selected_fulltext_paths:
        add(pdf)
    return paths


def _bib_key_from_filename(path: Path, bib_entries: list[str]) -> str | None:
    stem_norm = normalize_title_loose(path.stem)
    if not stem_norm:
        return None
    for entry in bib_entries:
        key = bib_entry_key(entry)
        if key and normalize_title_loose(key) == stem_norm:
            return key
    for entry in bib_entries:
        key = bib_entry_key(entry)
        if not key:
            continue
        key_norm = normalize_title_loose(key)
        if key_norm and (key_norm in stem_norm or stem_norm in key_norm):
            return key
    return None


def match_bib_key_for_fulltext_pdf(path: Path, research_paths: ResearchPaths, bib_entries: list[str]) -> str | None:
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    name = path.name
    stem_norm = normalize_title_loose(path.stem)

    for entry in research_paths.selected_entries:
        raw = entry.get("downloaded_pdf_path") or entry.get("pdf_path") or entry.get("local_pdf_path")
        if raw:
            try:
                p = Path(str(raw)).expanduser()
                if str(p.resolve()) == resolved or p.name == name:
                    key = match_bib_key_for_selected(entry, bib_entries)
                    if key:
                        return key
            except Exception:
                if Path(str(raw)).name == name:
                    key = match_bib_key_for_selected(entry, bib_entries)
                    if key:
                        return key

    key = _bib_key_from_filename(path, bib_entries)
    if key:
        return key

    for entry in research_paths.selected_entries:
        title = str(entry.get("title") or "")
        title_norm = normalize_title_loose(title)
        if title_norm and stem_norm and (title_norm in stem_norm or stem_norm in title_norm):
            key = match_bib_key_for_selected(entry, bib_entries)
            if key:
                return key

    for bib in bib_entries:
        meta = parse_bib_entry_meta(bib)
        key = meta.get("key")
        title_norm = normalize_title_loose(str(meta.get("title") or ""))
        if key and title_norm and stem_norm and (title_norm in stem_norm or stem_norm in title_norm):
            return str(key)
    return None


def build_fulltext_cache_citation_keys(research_paths: ResearchPaths, bib_entries: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    keys: list[str] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pdf in build_fulltext_cache_pdf_paths(research_paths):
        key = match_bib_key_for_fulltext_pdf(pdf, research_paths, bib_entries)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
        elif not key:
            unresolved.append({"path": str(pdf), "filename": pdf.name})
    return keys, unresolved


def build_selected_corpus_catalog(research_paths: ResearchPaths, bib_entries: list[str], limit: int = 40) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for pdf in build_fulltext_cache_pdf_paths(research_paths):
        key = match_bib_key_for_fulltext_pdf(pdf, research_paths, bib_entries)
        if key:
            seen_keys.add(key)
        meta = None
        if key:
            for bib in bib_entries:
                if bib_entry_key(bib) == key:
                    meta = parse_bib_entry_meta(bib)
                    break
        catalog.append({
            "title": (meta or {}).get("title") or pdf.stem,
            "year": (meta or {}).get("year"),
            "venue": None,
            "doi": (meta or {}).get("doi"),
            "bib_key": key,
            "keywords": [],
            "paper_id": None,
            "has_pdf": True,
            "pdf_path": str(pdf),
            "source": "fulltext_cache",
            "abstract_excerpt": "",
        })
    for entry in research_paths.selected_entries:
        key = match_bib_key_for_selected(entry, bib_entries)
        if key and key in seen_keys:
            continue
        catalog.append({
            "title": entry.get("title"),
            "year": entry.get("year"),
            "venue": entry.get("venue"),
            "doi": entry.get("doi"),
            "bib_key": key,
            "keywords": entry.get("keywords") or [],
            "paper_id": entry.get("paper_id"),
            "has_pdf": bool(entry.get("downloaded_pdf_path") or entry.get("pdf_url")),
            "source": "selected_all",
            "abstract_excerpt": shorten_text(str(entry.get("abstract") or entry.get("tldr") or ""), 900),
        })
    return catalog[:limit]

def build_reference_usage_map(org_text: str, base_docs: list[SourceDoc], extra_docs: list[SourceDoc], bib_keys: list[str], required_keys: list[str] | None = None) -> dict[str, Any]:
    cited_keys = extract_cited_keys_from_org(org_text)
    selected_keys = sorted({d.bib_key for d in base_docs if d.kind.startswith("texto_selecionado") and d.bib_key})
    extra_keys = sorted({d.bib_key for d in extra_docs if d.bib_key})
    required_keys = sorted({k for k in (required_keys or selected_keys) if k})
    selected_cited = [k for k in cited_keys if k in selected_keys]
    extra_cited = [k for k in cited_keys if k in extra_keys]
    required_cited = [k for k in cited_keys if k in required_keys]
    required_not_cited = [k for k in required_keys if k not in cited_keys]
    cited_known = set(selected_keys) | set(extra_keys) | set(required_keys)
    other_cited = [k for k in cited_keys if k not in cited_known]
    bib_not_cited = [k for k in bib_keys if k not in cited_keys]

    return {
        "cited_keys": cited_keys,
        "selected_keys": selected_keys,
        "extra_keys": extra_keys,
        "required_fulltext_cache_keys": required_keys,
        "selected_cited_keys": selected_cited,
        "extra_cited_keys": extra_cited,
        "required_fulltext_cache_cited_keys": required_cited,
        "required_fulltext_cache_not_cited_keys": required_not_cited,
        "other_cited_keys": other_cited,
        "bib_keys_not_cited": bib_not_cited,
        "counts": {
            "cited_total": len(cited_keys),
            "selected_total": len(selected_keys),
            "selected_cited_total": len(selected_cited),
            "extra_total": len(extra_keys),
            "extra_cited_total": len(extra_cited),
            "required_fulltext_cache_total": len(required_keys),
            "required_fulltext_cache_cited_total": len(required_cited),
            "required_fulltext_cache_not_cited_total": len(required_not_cited),
            "bib_not_cited_total": len(bib_not_cited),
        },
    }

def render_reference_usage_markdown(usage: dict[str, Any]) -> str:
    lines = ["# Mapa de uso das referências", ""]
    counts = usage.get("counts", {})
    lines += [
        f"- Citações únicas no documento: {counts.get('cited_total', 0)}",
        f"- Chaves de artigos selecionados: {counts.get('selected_total', 0)}",
        f"- Chaves de selecionados efetivamente citadas: {counts.get('selected_cited_total', 0)}",
        f"- Chaves obrigatórias do fulltext_cache: {counts.get('required_fulltext_cache_total', 0)}",
        f"- Chaves do fulltext_cache efetivamente citadas: {counts.get('required_fulltext_cache_cited_total', 0)}",
        f"- Chaves do fulltext_cache não citadas: {counts.get('required_fulltext_cache_not_cited_total', 0)}",
        f"- Chaves de artigos extras: {counts.get('extra_total', 0)}",
        f"- Chaves de extras efetivamente citadas: {counts.get('extra_cited_total', 0)}",
        f"- Entradas do .bib não citadas: {counts.get('bib_not_cited_total', 0)}",
        "",
        "## Selecionados citados",
    ]
    sel = usage.get("selected_cited_keys", [])
    lines += [f"- `{k}`" for k in sel] if sel else ["- Nenhum"]
    req = usage.get("required_fulltext_cache_not_cited_keys", [])
    lines += ["", "## Fulltext cache obrigatório não citado"]
    lines += [f"- `{k}`" for k in req] if req else ["- Nenhum"]
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

def collect_readable_files(raw_items: list[Any], base_dir: Path | None = None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_items or []:
        if not str(raw).strip():
            continue
        p = safe_resolve_user_path(raw, base_dir=base_dir)
        if p is None:
            continue
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

def find_fallback_template(doc_type: str = DEFAULT_DOC_TYPE) -> Path | None:
    doc_type = normalize_document_type(doc_type)
    preferred = DEFAULT_DISSERTATION_TEMPLATE if doc_type == "dissertacao" else DEFAULT_PAPER_TEMPLATE
    candidates = [
        Path.cwd() / "templates" / preferred,
        Path.cwd() / preferred,
        Path.cwd() / "templates" / DEFAULT_FALLBACK_TEMPLATE,
        Path.cwd() / DEFAULT_FALLBACK_TEMPLATE,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

def build_default_document_template(doc_type) -> str:
    class_name = "fgv-dissertacao" if normalize_document_type(doc_type) == "dissertacao" else "fgv-paper"
    return textwrap.dedent(f"""
        #+TITLE: TÍTULO A SER GERADO PELA IA
        #+AUTHOR: Gustavo M. Mendes de Tarso
        #+LANGUAGE: pt_BR
        #+OPTIONS: toc:nil num:t ^:nil
        #+LATEX_CLASS: {class_name}
        #+LATEX_HEADER: \institution{{Faculdade Getúlio Vargas}}
        #+LATEX_HEADER: \coursename{{}}
        #+LATEX_HEADER: \disciplinename{{}}
        #+LATEX_HEADER: \professorname{{}}
        #+LATEX_HEADER: \cityname{{Brasília}}
        #+LATEX_HEADER: \papertype{{Texto gerado automaticamente pela IA após a conclusão do documento}}
        #+LATEX_HEADER: \covernote{{Nota filosófica a ser gerada pela IA após a conclusão do documento}}

        #+begin_abstract
        Resumo a ser gerado pela IA.
        #+end_abstract

        * Introdução
        * Desenvolvimento
        * Considerações finais
        #+PRINT_BIBLIOGRAPHY:
        """).strip() + "\n"

def detect_research_output_dir(cfg: dict[str, Any]) -> Path:
    pipeline = cfg.get("pipeline", {})
    pesquisa_dir_existente = resolve_configured_path(pipeline.get("pesquisa_dir_existente"), cfg)
    if pesquisa_dir_existente is not None:
        return pesquisa_dir_existente

    saida = cfg.get("saida", {})
    base_dir = resolve_configured_path(saida.get("output_dir") or ".", cfg)
    if base_dir is None:
        base_dir = get_config_base_dir(cfg)
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
        cp = resolve_configured_path(config_output, cfg)
        rp.config_path = cp if cp is not None and cp.exists() else None
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

def _ensure_list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []

def collect_orientation_values(section: dict[str, Any] | None) -> list[str]:
    values: list[str] = []
    if not isinstance(section, dict):
        return values
    seen: set[str] = set()

    def add(raw: Any) -> None:
        for item in _ensure_list_of_strings(raw):
            if item not in seen:
                values.append(item)
                seen.add(item)

    add(section.get("orientacoes_paths"))
    add(section.get("orientacao_inline"))
    return values

def resolve_orientation_contents(values: list[str], *, max_chars: int = 40000, base_dir: Path | None = None) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for idx, raw in enumerate(values, start=1):
        label = f"orientacao_{idx}"
        candidate = safe_resolve_user_path(raw, base_dir=base_dir)
        if safe_path_exists_file(candidate):
            try:
                if candidate.suffix.lower() in READABLE_SUFFIXES:
                    text = read_text_file(candidate, max_chars=max_chars)
                else:
                    text = shorten_text(candidate.read_text(encoding="utf-8", errors="ignore"), max_chars)
                chunks.append((str(candidate), text))
                continue
            except Exception as exc:
                debug_print(f"Falha ao ler orientação {candidate}: {exc}")
        chunks.append((f"inline:{label}", shorten_text(str(raw), max_chars)))
    return chunks

def write_combined_orientation_file(values: list[str], output_path: Path, *, title: str, base_dir: Path | None = None) -> Path | None:
    chunks = resolve_orientation_contents(values, max_chars=50000, base_dir=base_dir)
    if not chunks:
        return None
    parts: list[str] = [f"# {title}"]
    for idx, (source, text) in enumerate(chunks, start=1):
        parts.append(f"\n## Bloco {idx} — {source}\n")
        parts.append(text.strip())
    write_text(output_path, "\n\n".join(parts).strip() + "\n")
    return output_path

def filter_research_config(cfg: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    research_cfg = {k: json.loads(json.dumps(v)) for k, v in cfg.items() if k in RESEARCH_SECTIONS}

    saida = research_cfg.setdefault("saida", {})
    triagem = research_cfg.setdefault("triagem", {})
    config_base_dir = get_config_base_dir(cfg)

    saida_values = collect_orientation_values(cfg.get("saida", {}))
    triagem_values = collect_orientation_values(cfg.get("triagem", {}))

    saida.pop("orientacoes_paths", None)
    saida.pop("orientacao_inline", None)
    triagem.pop("orientacoes_paths", None)
    triagem.pop("orientacao_inline", None)

    if saida_values:
        combined = write_combined_orientation_file(
            saida_values,
            work_dir / "_orientacoes_saida_combinadas.txt",
            title="Orientações gerais da pesquisa",
            base_dir=config_base_dir,
        )
        if combined is not None:
            saida[RESEARCH_BRIDGE_OUT_ORIENTACAO] = str(combined)

    if triagem_values:
        combined = write_combined_orientation_file(
            triagem_values,
            work_dir / "_orientacoes_triagem_combinadas.txt",
            title="Orientações específicas de triagem",
            base_dir=config_base_dir,
        )
        if combined is not None:
            triagem[RESEARCH_BRIDGE_TRIAGEM_PROMPT] = str(combined)
            triagem[RESEARCH_BRIDGE_TRIAGEM_EXTRAS] = ""

    return research_cfg

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
        raise TypeError("Dict deve ser serializado como subtabela TOML, não como valor escalar")
    text = str(value).replace("\\", "\\\\").replace('"', '\"')
    return f'"{text}"'


def _emit_toml_section(lines: list[str], section_name: str, values: dict[str, Any]) -> None:
    """Serializa uma seção TOML preservando subtabelas, como [atividade.metadados]."""
    scalar_items: list[tuple[str, Any]] = []
    nested_items: list[tuple[str, dict[str, Any]]] = []

    for key, value in values.items():
        if isinstance(value, dict):
            nested_items.append((str(key), value))
        else:
            scalar_items.append((str(key), value))

    if scalar_items:
        lines.append(f"[{section_name}]")
        for key, value in scalar_items:
            lines.append(f"{key} = {dumps_toml_value(value)}")
        lines.append("")

    for key, value in nested_items:
        _emit_toml_section(lines, f"{section_name}.{key}", value)


def dict_to_toml(cfg: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        _emit_toml_section(lines, str(section), values)
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

def build_document_context(cfg: dict[str, Any], research_paths: ResearchPaths) -> tuple[DocumentContext, dict[str, Any]]:
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

    context = DocumentContext(
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

def maybe_rewrite_document_context(
    client: OpenAI,
    model: str,
    cfg: dict[str, Any],
    context: DocumentContext,
    guidance_docs: list[SourceDoc],
) -> tuple[DocumentContext, dict[str, Any]]:
    documento = cfg.get("documento", {})
    usar_contexto_consolidado = bool(documento.get("usar_contexto_consolidado_da_pesquisa", True))
    reformular = bool(documento.get("reformular_tema_recorte_objetivo", False))
    modo_escrita = str(documento.get("modo_escrita") or "novo").strip().lower()

    if usar_contexto_consolidado and not reformular:
        return context, {"used": False, "source": "pesquisa_consolidada"}

    override_tema = str(documento.get("tema") or "").strip()
    override_recorte = str(documento.get("recorte") or "").strip()
    override_objetivo = str(documento.get("objetivo") or "").strip()
    if not reformular and (override_tema or override_recorte or override_objetivo):
        new_context = DocumentContext(
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
        Reformule, para a etapa de redação do documento, o tema, o recorte e o objetivo abaixo.

        Regras:
        - preserve o núcleo analítico da pesquisa já consolidada;
        - NÃO mude o assunto central;
        - apenas refine a formulação para a escrita do documento;
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
        text_format=RewrittenDocumentContextOutput,
    )
    parsed = resp.output_parsed
    if parsed is None:
        return context, {"used": False, "source": "fallback_contexto_consolidado"}

    new_context = DocumentContext(
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
    config_base_dir = get_config_base_dir(cfg)

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

    def add_values(values: list[str], kind_prefix: str) -> None:
        for idx, raw in enumerate(values, start=1):
            raw = str(raw).strip()
            if not raw:
                continue
            maybe_path = safe_resolve_user_path(raw, base_dir=config_base_dir)
            if safe_path_exists_file(maybe_path):
                add(maybe_path, kind_prefix, f"{kind_prefix}_{idx}")
            else:
                docs.append(SourceDoc(
                    path=f"inline:{kind_prefix}_{idx}",
                    kind=kind_prefix,
                    label=f"{kind_prefix}_{idx}",
                    extracted_text=shorten_text(raw, 20000),
                ))

    # orientações gerais da pesquisa
    saida_values = collect_orientation_values(cfg.get("saida", {}))
    add_values(saida_values, "orientacao_pesquisa")

    # artefatos da pesquisa como orientação
    if research_paths.org_path:
        add(research_paths.org_path, "pesquisa_org", "pesquisa_org")
    if research_paths.debug_path:
        add(research_paths.debug_path, "pesquisa_debug", "pesquisa_debug")

    # orientações específicas do documento acadêmico
    documento_values = collect_orientation_values(cfg.get("documento", {}))
    add_values(documento_values, "orientacao_documento")

    # reescrita/expansão do documento acadêmico: usa o .org anterior como orientação
    documento = cfg.get("documento", {})
    if bool(documento.get("reescrever_a_partir_do_org_atual", False)):
        org_existente_raw = str(documento.get("documento_org_existente") or "").strip()
        org_existente: Path | None = None
        if org_existente_raw:
            p = safe_resolve_user_path(org_existente_raw, base_dir=config_base_dir)
            if p is None:
                p = Path(os.path.expanduser(org_existente_raw)).resolve()
            if p.exists() and p.is_file():
                org_existente = p
        else:
            doc_type_local = normalize_document_type(documento.get("tipo_documento"))
            prefix = (documento.get("prefixo") or ((cfg.get("saida", {}).get("prefixo") or "atividade") + f"_{DEFAULT_DOC_PREFIX_BY_TYPE.get(doc_type_local, 'documento')}" )).strip()
            output_dir_raw = documento.get("output_dir")
            if output_dir_raw:
                base = Path(output_dir_raw).expanduser().resolve()
                create_subdir = bool(documento.get("criar_subdiretorio", True))
                documento_dir = base / prefix if create_subdir else base
            else:
                documento_dir = build_document_output_dir(cfg, research_paths.root_dir)
            candidate = documento_dir / f"{prefix}.org"
            if candidate.exists():
                org_existente = candidate

        if org_existente is not None:
            add(org_existente, "paper_org_anterior", "paper_org_anterior")

    return docs

def collect_base_docs(cfg: dict[str, Any], research_paths: ResearchPaths, max_files: int | None = None) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    documento = cfg.get("documento", {})

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

    usar_selecionados = bool(documento.get("usar_artigos_selecionados_pesquisa", True))
    permitir_correlata_extra = bool(documento.get("permitir_busca_correlata_extra", False))
    citar_todos_fulltext_cache = True if normalize_document_type(documento.get("tipo_documento")) == "dissertacao" else bool(documento.get("citar_todos_fulltext_cache", True))
    if max_files is None:
        doc_type_local = normalize_document_type(documento.get("tipo_documento"))
        if citar_todos_fulltext_cache and doc_type_local == "dissertacao":
            default_max = max(len(build_fulltext_cache_pdf_paths(research_paths)), len(research_paths.selected_fulltext_paths), 9999)
        else:
            default_max = 18 if doc_type_local == "dissertacao" else 12
        max_files = int(documento.get("max_selected_base_docs", default_max) or default_max)

    if usar_selecionados:
        source_paths = build_fulltext_cache_pdf_paths(research_paths) if citar_todos_fulltext_cache else research_paths.selected_fulltext_paths
        for path in source_paths[:max_files]:
            add(path, "texto_selecionado_fulltext_cache" if citar_todos_fulltext_cache else "texto_selecionado_formal")

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

def summarize_docs(docs: list[SourceDoc], excerpt_chars: int = 2500) -> list[dict[str, Any]]:
    out = []
    for d in docs:
        excerpt = shorten_text(d.extracted_text, excerpt_chars)
        out.append({
            "kind": d.kind,
            "label": d.label,
            "path": d.path,
            "excerpt": excerpt,
            "excerpt_word_count": count_words(excerpt),
            "fulltext_word_count": count_words(d.extracted_text),
            "bib_key": d.bib_key,
        })
    return out

def collect_extra_article_docs(cfg: dict[str, Any], max_files: int = 20) -> list[SourceDoc]:
    documento = cfg.get("documento", {})
    raw_items = documento.get("artigos_extras_paths", []) or []
    files = collect_readable_files(raw_items, get_config_base_dir(cfg))[:max_files]
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

def infer_final_front_matter(client: OpenAI, model: str, context: DocumentContext, org_text: str) -> FinalFrontMatterOutput:
    prompt = textwrap.dedent(
        f"""
        Gere os elementos finais de capa para um documento acadêmico em português.

        Contexto:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Conteúdo preliminar do documento:
        {shorten_text(org_text, 15000)}

        Regras:
        - title: título acadêmico final, claro e elegante.
        - documento_type: descrição curta do tipo do documento para a capa.
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

def apply_final_front_matter(org_text: str, *, title: str, author: str, documento_type: str, cover_note: str, institution_name: str, course_name: str = "", discipline_name: str = "", professor_name: str = "", city_name: str = "Brasília") -> str:
    org_text = replace_org_header_line(org_text, "#+TITLE:", title)
    org_text = replace_org_header_line(org_text, "#+AUTHOR:", author)
    org_text = replace_latex_header_macro(org_text, "institution", institution_name)
    org_text = replace_latex_header_macro(org_text, "coursename", course_name)
    org_text = replace_latex_header_macro(org_text, "disciplinename", discipline_name)
    org_text = replace_latex_header_macro(org_text, "professorname", professor_name)
    org_text = replace_latex_header_macro(org_text, "cityname", city_name)
    org_text = replace_latex_header_macro(org_text, "papertype", documento_type)
    org_text = replace_latex_header_macro(org_text, "covernote", cover_note)
    return org_text

def build_paper_prompt(
    cfg: dict[str, Any],
    context: DocumentContext,
    template_text: str,
    base_docs: list[SourceDoc],
    guidance_docs: list[SourceDoc],
    bib_keys: list[str],
    bib_entries: list[str],
    style: str,
    selected_corpus_catalog: list[dict[str, Any]] | None = None,
) -> str:
    documento = cfg.get("documento", {})
    doc_type_local = normalize_document_type(documento.get("tipo_documento"))
    modo_escrita = str(documento.get("modo_escrita") or "novo").strip().lower()
    perfil_redacao = str(documento.get("perfil_redacao") or "academico_equilibrado").strip().lower()
    priorizar_citacoes = bool(documento.get("priorizar_citacoes_dos_selecionados", True))
    usar_contexto_consolidado = bool(documento.get("usar_contexto_consolidado_da_pesquisa", True))
    extras_so_complementam = bool(documento.get("extras_so_complementam", True))
    preservar_estrutura = bool(documento.get("preservar_estrutura_do_org_anterior", False))

    selected_keys = sorted({d.bib_key for d in base_docs if d.kind.startswith("texto_selecionado") and d.bib_key})
    extra_keys = sorted({d.bib_key for d in base_docs if d.kind == "artigo_extra" and d.bib_key})
    excerpt_chars = 4200 if doc_type_local == "dissertacao" else 2500
    targets = dissertation_generation_targets(cfg)

    instrucoes_modo = {
        "novo": "Escreva um documento novo, mas usando a pesquisa consolidada como base metodológica e analítica.",
        "reescrever": "Reescreva o documento com maior coesão, clareza e densidade argumentativa, aproveitando o documento anterior se ele estiver entre as orientações.",
        "expandir": "Mantenha a linha argumentativa principal e expanda o documento com mais desenvolvimento analítico, usando os textos extras quando úteis.",
    }.get(modo_escrita, "Escreva um documento acadêmico coeso a partir da pesquisa consolidada.")

    instrucoes_perfil = {
        "academico_equilibrado": "Adote redação acadêmica equilibrada, com boa relação entre síntese, discussão teórica e fluidez.",
        "mais_teorico": "Privilegie densidade conceitual, diálogo teórico e elaboração interpretativa mais forte.",
        "mais_discursivo": "Privilegie fluidez discursiva, encadeamento argumentativo e texto mais ensaístico, sem perder rigor acadêmico.",
        "mais_sintetico": "Privilegie concisão, objetividade e alta compressão informacional, evitando expansões desnecessárias.",
    }.get(perfil_redacao, "Adote redação acadêmica equilibrada e coesa.")

    regra_citacao = "Priorize as citações dos artigos formalmente selecionados na pesquisa." if priorizar_citacoes else "Use as referências de forma equilibrada, sem obrigação de priorizar os selecionados."
    regra_extras = "Os artigos extras só podem complementar a argumentação; não substituem o papel central dos artigos formalmente selecionados." if extras_so_complementam else "Os artigos extras podem ter papel relevante, desde que não descaracterizem o núcleo da pesquisa."
    regra_estrutura = "Se houver um org anterior do documento entre as orientações, preserve sua estrutura principal e melhore o texto dentro dessa arquitetura." if preservar_estrutura else "Se houver um org anterior do documento entre as orientações, use-o apenas como referência, sem obrigação de preservar a mesma estrutura."
    contexto_regra = (
        "Tema, recorte e objetivo já consolidados pela pesquisa devem ser tratados como referência principal."
        if usar_contexto_consolidado else
        "É permitido trabalhar com formulação refinada do tema, recorte e objetivo para a escrita do documento, sem mudar o núcleo temático."
    )

    limites = {
        "limite_palavras_total": documento.get("limite_palavras_total"),
        "limite_palavras_introducao": documento.get("limite_palavras_introducao"),
        "limite_palavras_revisao": documento.get("limite_palavras_revisao"),
        "limite_palavras_conclusao": documento.get("limite_palavras_conclusao"),
    }
    limites_dict = {k: v for k, v in limites.items() if v not in (None, "", 0)}
    limites_txt = json.dumps(limites_dict, ensure_ascii=False, indent=2) if limites_dict else "Nenhum limite explícito informado."

    cobertura_fulltext_txt = (
        "13. A cobertura bibliográfica obrigatória da dissertação é definida pelos PDFs/textos efetivamente baixados no diretório *_fulltext_cache/: todas as chaves mapeadas a esses arquivos devem ser interpretadas e citadas ao longo do texto, de forma substantiva e não meramente ornamental."
    )

    dissertation_rules = ""
    dissertation_targets = ""
    if doc_type_local == "dissertacao":
        dissertation_rules = textwrap.dedent(f"""
        18. Este trabalho deve ter fôlego de dissertação de mestrado baseada em revisão de literatura robusta, e não de paper curto ou resumo expandido.
        19. Desenvolva em profundidade os conceitos, os debates teóricos, os contrastes entre autores, as lacunas da literatura e as implicações para o governo público federal brasileiro.
        20. Absorva de modo explícito o material já consolidado na pesquisa, aproveitando a massa crítica de artigos selecionados e seus resumos/trechos.
        21. Amplie especialmente as seções de referencial teórico, metodologia e resultados/discussão; evite compressão excessiva.
        22. Em vez de apenas resumir autores, compare abordagens, explicite convergências e divergências e construa sínteses analíticas próprias.
        23. Sempre que houver literatura suficiente, subdivida a discussão em eixos temáticos densos e bem desenvolvidos, com transições argumentativas claras.
        24. Não reduza a metodologia a um parágrafo curto: explique estratégia de busca, critérios, corpus, limites e racionalidade analítica em densidade compatível com dissertação.
        25. Em resultados e discussão, explore de forma aprofundada aplicações da IA, modelos de governança, ética, accountability, compliance, riscos, capacidades estatais, assimetrias institucionais e implicações para a APF.
        26. Não encurte o documento para caber em poucas páginas. O alvo de desenvolvimento textual é compatível com um corpo principal de dissertação.
        27. Metas mínimas desejáveis de extensão textual por bloco analítico:
        {json.dumps(targets, ensure_ascii=False, indent=2)}
        """).strip()
        dissertation_targets = textwrap.dedent("""
        Estrutura desejável para dissertação:
        - Introdução substantiva, com contextualização, delimitação, problema, objetivos e justificativa bem desenvolvidos.
        - Referencial teórico aprofundado, com múltiplos subtópicos e comparação entre autores e correntes.
        - Metodologia detalhada, absorvendo os elementos relevantes da pesquisa PRISMA já realizada.
        - Resultados e discussão como núcleo do trabalho, com desenvolvimento significativamente mais longo do que introdução e conclusão.
        - Considerações finais com síntese, limitações e agenda futura bem explicitadas.
        """).strip()

    return textwrap.dedent(
        f"""
        Gere um documento acadêmico completo em Org-mode.

        Regras obrigatórias:
        1. Preserve o cabeçalho técnico do template (linhas #+...).
        2. O documento deve ser escrito em português, em tom acadêmico, argumentativo e coeso.
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
        {cobertura_fulltext_txt}
        14. {contexto_regra}
        15. {regra_estrutura}
        16. Se os textos-base forem insuficientes em algum ponto, trate essa limitação de forma honesta.
        17. Respeite, quando possível, os limites de extensão abaixo:
        {limites_txt}
        {dissertation_rules}

        Contexto do documento:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Estrutura e profundidade esperadas:
        {dissertation_targets or 'Use a estrutura do template e desenvolva cada seção com densidade acadêmica apropriada.'}

        Chaves bibliográficas de artigos formalmente selecionados:
        {json.dumps(selected_keys, ensure_ascii=False, indent=2)}

        Chaves bibliográficas de artigos extras:
        {json.dumps(extra_keys, ensure_ascii=False, indent=2)}

        Catálogo do corpus selecionado da pesquisa:
        {shorten_text(json.dumps(selected_corpus_catalog or [], ensure_ascii=False, indent=2), 40000)}

        Template Org:
        {shorten_text(template_text, 30000)}

        Chaves bibliográficas disponíveis:
        {json.dumps(bib_keys, ensure_ascii=False, indent=2)}

        Entradas bib disponíveis:
        {shorten_text(json.dumps(bib_entries, ensure_ascii=False, indent=2), 25000)}

        Textos-base:
        {json.dumps(summarize_docs(base_docs, excerpt_chars=excerpt_chars), ensure_ascii=False, indent=2)}

        Orientações e artefatos da pesquisa:
        {json.dumps(summarize_docs(guidance_docs, excerpt_chars=3200), ensure_ascii=False, indent=2)}

        Retorne apenas o conteúdo completo do arquivo .org final.
        """
    ).strip()

def cleanup_placeholder_sections_in_tex_ready_text(text: str) -> str:
    text = re.sub(r"(?im)^\s*[,;:]+\s*$\n?", "", text)
    text = re.sub(r"(?im)^\s*P\d+\\?\}\s*$\n?", "", text)
    text = re.sub(r"(?im)^\s*<empty citation>\s*$\n?", "", text)
    return text


def normalize_org_citations(text: str) -> str:
    text = text.replace("<empty citation>", "")
    text = text.replace("[][]", "")
    return text

def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:org|markdown|md|text)?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def sanitize_generated_org_fragment(text: str) -> str:
    text = strip_code_fences(text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?im)^\s*#\+.*$\n?", "", text)
    text = re.sub(r"(?im)^\s*<empty citation>\s*$\n?", "", text)
    text = text.replace("[][]", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def strip_redundant_leading_heading(text: str, heading_level: int, heading_title: str) -> str:
    body = sanitize_generated_org_fragment(text)
    if not body:
        return body
    stars = "*" * max(int(heading_level or 1), 1)
    wanted = normalize_heading_loose(heading_title)
    pattern = re.compile(rf"(?ms)^\s*{re.escape(stars)}\s+(?P<title>.+?)\s*\n")
    while True:
        m = pattern.match(body)
        if not m:
            break
        found = normalize_heading_loose(m.group("title"))
        if found != wanted:
            break
        body = body[m.end():].lstrip("\n")
    return body.strip()


def _find_section_body_span(org_text: str, heading_level: int, heading_title: str) -> tuple[int, int]:
    stars = "*" * heading_level
    pattern = re.compile(rf"(?m)^{re.escape(stars)}\s+{re.escape(heading_title)}\s*$")
    m = pattern.search(org_text)
    if not m:
        raise ValueError(f"Seção não encontrada: nivel={heading_level} titulo={heading_title!r}")
    pos = m.end()
    if pos < len(org_text) and org_text[pos] == "\n":
        pos += 1
    drawer = re.compile(r"(?ms)^:PROPERTIES:\n.*?\n:END:\n")
    dm = drawer.match(org_text[pos:])
    if dm:
        pos += dm.end()
    while pos < len(org_text) and org_text[pos] == "\n":
        pos += 1
    next_heading = re.compile(rf"(?m)^(?:\*{{1,{heading_level}}})\s+")
    nm = next_heading.search(org_text, pos)
    end = nm.start() if nm else len(org_text)
    return pos, end


def replace_section_body(org_text: str, heading_level: int, heading_title: str, new_body: str) -> str:
    start, end = _find_section_body_span(org_text, heading_level, heading_title)
    body = strip_redundant_leading_heading(new_body, heading_level, heading_title)
    prefix = org_text[:start].rstrip() + "\n\n"
    suffix = org_text[end:].lstrip("\n")
    if body:
        return prefix + body + "\n\n" + suffix
    return prefix + suffix


def extract_section_body(org_text: str, heading_level: int, heading_title: str) -> str:
    start, end = _find_section_body_span(org_text, heading_level, heading_title)
    return org_text[start:end].strip()


def extract_subheading_titles(section_body: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"(?m)^\*\*\s+(.+)$", section_body)]


def build_bib_meta_lookup(bib_entries: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in bib_entries:
        meta = parse_bib_entry_meta(entry)
        key = str(meta.get("key") or "").strip()
        if key:
            out[key] = meta
    return out


def chunk_keys_weighted(keys: list[str], weights: list[float]) -> list[list[str]]:
    if not weights:
        return []
    if not keys:
        return [[] for _ in weights]
    total = float(sum(weights)) or 1.0
    raw_sizes = [len(keys) * (w / total) for w in weights]
    sizes = [int(x) for x in raw_sizes]
    remainder = len(keys) - sum(sizes)
    order = sorted(range(len(weights)), key=lambda i: (raw_sizes[i] - sizes[i]), reverse=True)
    for i in range(remainder):
        sizes[order[i % len(order)]] += 1
    chunks: list[list[str]] = []
    cursor = 0
    for size in sizes:
        chunks.append(keys[cursor:cursor + size])
        cursor += size
    if cursor < len(keys):
        chunks[-1].extend(keys[cursor:])
    return chunks


def build_corpus_overview(selected_corpus_catalog: list[dict[str, Any]] | None, limit: int = 24) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in (selected_corpus_catalog or [])[:limit]:
        out.append({
            "bib_key": item.get("bib_key"),
            "title": item.get("title"),
            "year": item.get("year"),
            "venue": item.get("venue"),
            "keywords": item.get("keywords") or [],
        })
    return out


def summarize_docs_by_keys(base_docs: list[SourceDoc], keys: list[str], excerpt_chars: int = 1200, max_docs: int = 8) -> list[dict[str, Any]]:
    chosen: list[SourceDoc] = []
    key_set = set(keys)
    for doc in base_docs:
        if doc.bib_key and doc.bib_key in key_set:
            chosen.append(doc)
    if not chosen:
        chosen = [d for d in base_docs if d.kind.startswith("texto_selecionado")][:max_docs]
    return summarize_docs(chosen[:max_docs], excerpt_chars=excerpt_chars)


def build_dissertation_front_sections_prompt(
    cfg: dict[str, Any],
    context: DocumentContext,
    template_text: str,
    guidance_docs: list[SourceDoc],
    selected_corpus_catalog: list[dict[str, Any]] | None = None,
) -> str:
    atividade = cfg.get("atividade", {})
    documento = cfg.get("documento", {})
    return textwrap.dedent(
        f"""
        Gere os elementos pré-textuais de uma dissertação acadêmica em português, em formato estruturado.

        Regras:
        1. Não use placeholders genéricos como "[...]", "Autor da epígrafe" ou "A definir", salvo quando não houver dado objetivo para nomes de banca/orientação, os quais não fazem parte desta etapa.
        2. DEDICATÓRIA e AGRADECIMENTOS devem ser breves, elegantes e plausíveis.
        3. A EPÍGRAFE é opcional; se não houver boa opção, devolva epigrafe_texto e epigrafe_autor vazios.
        4. O RESUMO deve ter densidade acadêmica e refletir a dissertação de revisão de literatura.
        5. O ABSTRACT deve ser tradução acadêmica consistente do resumo.
        6. As listas devem ser enxutas e compatíveis com o documento.
        7. O glossário deve conter termos realmente úteis ao tema.

        Contexto consolidado:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Metadados do trabalho:
        {json.dumps({
            "aluno": atividade.get("aluno"),
            "curso": atividade.get("curso"),
            "disciplina": atividade.get("disciplina"),
            "professor": atividade.get("professor"),
            "institution_name": documento.get("institution_name"),
            "school_name": documento.get("school_name"),
        }, ensure_ascii=False, indent=2)}

        Estrutura do template:
        {shorten_text(template_text, 12000)}

        Catálogo do corpus selecionado:
        {json.dumps(build_corpus_overview(selected_corpus_catalog), ensure_ascii=False, indent=2)}

        Orientações e artefatos relevantes:
        {json.dumps(summarize_docs(guidance_docs, excerpt_chars=1800), ensure_ascii=False, indent=2)}
        """
    ).strip()


def build_dissertation_stage_plan(template_text: str, cfg: dict[str, Any], bib_keys: list[str]) -> list[dict[str, Any]]:
    targets = dissertation_generation_targets(cfg)
    citation_target_keys = list(cfg.get("__citation_target_keys__") or bib_keys)
    top_sections = [
        ("INTRODUÇÃO", "introducao", targets.get("introducao_min", 1200)),
        ("REFERENCIAL TEÓRICO", "referencial_teorico", targets.get("referencial_teorico_min", 4000)),
        ("METODOLOGIA", "metodologia", targets.get("metodologia_min", 1800)),
        ("RESULTADOS E DISCUSSÃO", "resultados_discussao", targets.get("resultados_discussao_min", 5000)),
        ("CONSIDERAÇÕES FINAIS", "conclusao", targets.get("conclusao_min", 1200)),
    ]
    plan: list[dict[str, Any]] = []
    subsection_counts: list[int] = []
    by_top: list[list[str]] = []
    for top_title, _, _ in top_sections:
        body = extract_section_body(template_text, 1, top_title)
        subs = extract_subheading_titles(body)
        if not subs:
            subs = [top_title]
        by_top.append(subs)
        subsection_counts.append(len(subs))
    key_chunks = chunk_keys_weighted(citation_target_keys, [max(c, 1) for c in subsection_counts])
    chunk_idx = 0
    for (top_title, bucket, bucket_min), subs in zip(top_sections, by_top):
        per_sub_min = max(220, int(bucket_min / max(len(subs), 1)))
        template_body = extract_section_body(template_text, 1, top_title)
        siblings = subs[:]
        for sub_title in subs:
            mandatory_keys = key_chunks[chunk_idx] if chunk_idx < len(key_chunks) else []
            chunk_idx += 1
            plan.append({
                "top_title": top_title,
                "sub_title": sub_title,
                "bucket": bucket,
                "min_words": per_sub_min,
                "mandatory_keys": mandatory_keys,
                "all_keys": bib_keys,
                "citation_target_keys": citation_target_keys,
                "template_body": template_body,
                "sibling_subtitles": siblings,
            })
    return plan


def build_dissertation_subsection_prompt(
    cfg: dict[str, Any],
    context: DocumentContext,
    stage: dict[str, Any],
    bib_meta_lookup: dict[str, dict[str, Any]],
    base_docs: list[SourceDoc],
    guidance_docs: list[SourceDoc],
    selected_corpus_catalog: list[dict[str, Any]] | None = None,
    previous_excerpt: str = "",
) -> str:
    top_title = stage["top_title"]
    sub_title = stage["sub_title"]
    mandatory_keys = stage.get("mandatory_keys", [])
    all_keys = stage.get("all_keys", [])
    min_words = int(stage.get("min_words", 320))
    sibling_subtitles = stage.get("sibling_subtitles", [])
    focus_map = {
        "INTRODUÇÃO": "Contextualize o problema, delimite o objeto, explicite problema, objetivos e justificativa com densidade de dissertação.",
        "REFERENCIAL TEÓRICO": "Aprofunde conceitos, compare autores, explicite convergências, divergências e lacunas, evitando resumo superficial.",
        "METODOLOGIA": "Explique em detalhe o desenho da revisão, o corpus, critérios, bases, limites e racionalidade analítica.",
        "RESULTADOS E DISCUSSÃO": "Faça análise densa do corpus, compare abordagens, desenvolva implicações para a APF e articule achados com a literatura.",
        "CONSIDERAÇÕES FINAIS": "Sintetize criticamente, explicite limites e desdobre agenda futura sem mera repetição mecânica.",
    }
    bib_subset = [bib_meta_lookup[k] for k in mandatory_keys if k in bib_meta_lookup]
    docs_subset = summarize_docs_by_keys(base_docs, mandatory_keys, excerpt_chars=1000, max_docs=8)
    return textwrap.dedent(
        f"""
        Escreva APENAS o corpo em Org-mode da subseção ** {sub_title} que pertence à seção * {top_title}.

        Regras obrigatórias:
        1. NÃO repita o título de nível 1.
        2. Retorne o texto iniciando com a heading de nível 2 correspondente, isto é: ** {sub_title}
        3. Use exclusivamente citações nativas do Org Cite, como [cite:@chave].
        4. Não invente chaves bibliográficas.
        5. Desenvolva a subseção com densidade compatível com dissertação, evitando síntese curta.
        6. Meta mínima de extensão desta subseção: cerca de {min_words} palavras.
        7. Nesta subseção, use de forma orgânica e explícita estas chaves obrigatórias, cada uma ao menos uma vez, salvo incompatibilidade temática absoluta: {json.dumps(mandatory_keys, ensure_ascii=False)}.
        8. O projeto global exige que TODOS os PDFs/textos do diretório *_fulltext_cache/ sejam interpretados pela IA e distribuídos ao longo da dissertação. Aqui, sua responsabilidade principal é usar os documentos resumidos e metadados desta etapa para construir argumento substantivo, comparativo e citado, não apenas mencionar autores.
        9. Subtítulos irmãos desta seção para manter coerência global: {json.dumps(sibling_subtitles, ensure_ascii=False)}.
        10. {focus_map.get(top_title, '')}
        11. Evite listas excessivas; priorize parágrafos analíticos e comparativos.

        Contexto consolidado:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Catálogo do corpus selecionado:
        {json.dumps(build_corpus_overview(selected_corpus_catalog), ensure_ascii=False, indent=2)}

        Metadados bibliográficos prioritários para esta subseção:
        {json.dumps(bib_subset, ensure_ascii=False, indent=2)}

        Documentos resumidos mais relevantes para esta subseção:
        {json.dumps(docs_subset, ensure_ascii=False, indent=2)}

        Chaves globais disponíveis em toda a dissertação:
        {json.dumps(all_keys, ensure_ascii=False, indent=2)}

        Estrutura da seção no template:
        {shorten_text(stage.get('template_body', ''), 6000)}

        Último contexto já redigido para manter continuidade:
        {shorten_text(previous_excerpt, 5000)}

        Orientações e artefatos relevantes:
        {json.dumps(summarize_docs(guidance_docs[:6], excerpt_chars=1400), ensure_ascii=False, indent=2)}
        """
    ).strip()


def build_dissertation_subsection_repair_prompt(
    cfg: dict[str, Any],
    context: DocumentContext,
    stage: dict[str, Any],
    current_body: str,
    missing_keys: list[str],
    bib_meta_lookup: dict[str, dict[str, Any]],
    base_docs: list[SourceDoc],
) -> str:
    docs_subset = summarize_docs_by_keys(base_docs, missing_keys or stage.get("mandatory_keys", []), excerpt_chars=1000, max_docs=8)
    metas = [bib_meta_lookup[k] for k in missing_keys if k in bib_meta_lookup]
    return textwrap.dedent(
        f"""
        Reescreva a subseção abaixo em Org-mode, preservando sua linha argumentativa, mas corrigindo duas insuficiências: falta de cobertura bibliográfica obrigatória e/ou extensão abaixo do desejado.

        Subseção: ** {stage['sub_title']}
        Seção: * {stage['top_title']}
        Meta mínima aproximada de palavras: {stage.get('min_words', 320)}
        Chaves obrigatórias que ainda precisam aparecer organicamente: {json.dumps(missing_keys, ensure_ascii=False)}
        Use apenas Org Cite ([cite:@chave]).
        Retorne o texto completo da subseção, começando com ** {stage['sub_title']}.

        Contexto:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Metadados das chaves faltantes:
        {json.dumps(metas, ensure_ascii=False, indent=2)}

        Documentos resumidos para apoiar a revisão:
        {json.dumps(docs_subset, ensure_ascii=False, indent=2)}

        Subseção atual:
        {shorten_text(current_body, 12000)}
        """
    ).strip()


def maybe_repair_dissertation_subsection(
    client: OpenAI,
    model: str,
    cfg: dict[str, Any],
    context: DocumentContext,
    stage: dict[str, Any],
    body: str,
    bib_meta_lookup: dict[str, dict[str, Any]],
    base_docs: list[SourceDoc],
) -> tuple[str, str | None, dict[str, Any]]:
    cited = set(extract_cited_keys_from_org(body))
    mandatory = [k for k in stage.get("mandatory_keys", []) if k]
    missing = [k for k in mandatory if k not in cited]
    words = count_words(body)
    min_words = int(stage.get("min_words", 320))
    needs_repair = bool(missing or words < int(min_words * 0.75))
    diagnostics = {
        "top_title": stage.get("top_title"),
        "sub_title": stage.get("sub_title"),
        "word_count": words,
        "min_words": min_words,
        "mandatory_keys": mandatory,
        "missing_keys": missing,
        "needs_repair": needs_repair,
    }
    if not needs_repair:
        return body, None, diagnostics
    prompt = build_dissertation_subsection_repair_prompt(cfg, context, stage, body, missing, bib_meta_lookup, base_docs)
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=DissertationStageBodyOutput,
    )
    if resp.output_parsed is None:
        raise RuntimeError(f"A IA não retornou a subseção reparada para {stage.get('sub_title')}.")
    repaired = sanitize_generated_org_fragment(resp.output_parsed.org_body)
    return repaired, prompt, diagnostics


def build_missing_bibliography_patch_prompt(
    cfg: dict[str, Any],
    context: DocumentContext,
    missing_keys: list[str],
    bib_meta_lookup: dict[str, dict[str, Any]],
    base_docs: list[SourceDoc],
    current_org: str,
) -> str:
    docs_subset = summarize_docs_by_keys(base_docs, missing_keys, excerpt_chars=1000, max_docs=12)
    metas = [bib_meta_lookup[k] for k in missing_keys if k in bib_meta_lookup]
    return textwrap.dedent(
        f"""
        O documento acadêmico abaixo ainda não utilizou todas as chaves bibliográficas exigidas dos artigos com PDF no diretório *_fulltext_cache/. Gere APENAS uma nova subseção de Org-mode para ser inserida em * RESULTADOS E DISCUSSÃO, com heading de nível 2, intitulada ** Síntese integradora ampliada do corpus.

        Regras:
        1. Use organicamente TODAS estas chaves faltantes ao menos uma vez: {json.dumps(missing_keys, ensure_ascii=False)}.
        2. Use apenas citações Org Cite ([cite:@chave]).
        3. A subseção deve ser analítica, comparativa e aprofundada, não uma lista mecânica de referências.
        4. Interprete os documentos resumidos correspondentes e conecte as referências faltantes aos eixos de governança, ética, compliance, risco, capacidades estatais e implicações para a APF, conforme pertinente.
        5. Não trate a citação como adereço: para cada chave faltante, incorpore pelo menos uma contribuição substantiva do texto ao argumento.

        Contexto:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Metadados bibliográficos das chaves faltantes:
        {json.dumps(metas, ensure_ascii=False, indent=2)}

        Documentos resumidos correspondentes:
        {json.dumps(docs_subset, ensure_ascii=False, indent=2)}

        Documento atual:
        {shorten_text(current_org, 20000)}
        """
    ).strip()


def build_dissertation_org_from_stages(
    client: OpenAI,
    model: str,
    cfg: dict[str, Any],
    context: DocumentContext,
    template_text: str,
    base_docs: list[SourceDoc],
    guidance_docs: list[SourceDoc],
    bib_filename: str,
    bib_entries: list[str],
    bib_keys: list[str],
    style: str,
    selected_corpus_catalog: list[dict[str, Any]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    prompt_parts: list[str] = []
    stage_diagnostics: list[dict[str, Any]] = []
    working_org = template_text

    front_prompt = build_dissertation_front_sections_prompt(cfg, context, template_text, guidance_docs, selected_corpus_catalog=selected_corpus_catalog)
    prompt_parts.append("===== front_sections =====\n" + front_prompt + "\n")
    front_resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": front_prompt}],
        text_format=DissertationFrontSectionsOutput,
    )
    if front_resp.output_parsed is None:
        raise RuntimeError("A IA não retornou os elementos pré-textuais da dissertação.")
    front = front_resp.output_parsed

    if front.dedicatoria.strip():
        working_org = replace_section_body(working_org, 1, "DEDICATÓRIA", front.dedicatoria.strip())
    if front.agradecimentos.strip():
        working_org = replace_section_body(working_org, 1, "AGRADECIMENTOS", front.agradecimentos.strip())
    if front.epigrafe_texto.strip():
        ep = front.epigrafe_texto.strip()
        if front.epigrafe_autor.strip():
            ep += "\n" + front.epigrafe_autor.strip()
        working_org = replace_section_body(working_org, 1, "EPÍGRAFE", ep)
    else:
        working_org = replace_section_body(working_org, 1, "EPÍGRAFE", "")
    resumo_body = front.resumo.strip()
    if front.palavras_chave:
        resumo_body += "\n\nPalavras-chave: " + "; ".join([p.strip() for p in front.palavras_chave if str(p).strip()]) + "."
    working_org = replace_section_body(working_org, 1, "RESUMO", resumo_body)
    abstract_body = front.abstract.strip()
    if front.keywords:
        abstract_body += "\n\nKeywords: " + "; ".join([p.strip() for p in front.keywords if str(p).strip()]) + "."
    working_org = replace_section_body(working_org, 1, "ABSTRACT", abstract_body)
    working_org = replace_section_body(working_org, 1, "LISTA DE ILUSTRAÇÕES", "\n".join(f"- {x}" for x in front.lista_ilustracoes if str(x).strip()))
    working_org = replace_section_body(working_org, 1, "LISTA DE TABELAS", "\n".join(f"- {x}" for x in front.lista_tabelas if str(x).strip()))
    working_org = replace_section_body(working_org, 1, "LISTA DE ABREVIATURAS E SIGLAS", "\n".join(f"- {x}" for x in front.siglas if str(x).strip()))
    working_org = replace_section_body(working_org, 1, "LISTA DE SÍMBOLOS", "\n".join(f"- {x}" for x in front.simbolos if str(x).strip()))
    if front.glossario:
        working_org = replace_section_body(working_org, 1, "GLOSSÁRIO", "\n".join(f"- {x}" for x in front.glossario if str(x).strip()))

    bib_meta_lookup = build_bib_meta_lookup(bib_entries)
    plan = build_dissertation_stage_plan(template_text, cfg, bib_keys)
    previous_excerpt = ""
    for idx, stage in enumerate(plan, start=1):
        print(f"[5/6] Gerando subseção {idx}/{len(plan)}: {stage['top_title']} :: {stage['sub_title']}...", flush=True)
        stage_prompt = build_dissertation_subsection_prompt(
            cfg=cfg,
            context=context,
            stage=stage,
            bib_meta_lookup=bib_meta_lookup,
            base_docs=base_docs,
            guidance_docs=guidance_docs,
            selected_corpus_catalog=selected_corpus_catalog,
            previous_excerpt=previous_excerpt,
        )
        prompt_parts.append(f"===== stage {idx} {stage['top_title']} / {stage['sub_title']} =====\n" + stage_prompt + "\n")
        stage_resp = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": stage_prompt}],
            text_format=DissertationStageBodyOutput,
        )
        if stage_resp.output_parsed is None:
            raise RuntimeError(f"A IA não retornou a subseção {stage['sub_title']!r}.")
        stage_body = sanitize_generated_org_fragment(stage_resp.output_parsed.org_body)
        stage_body, repair_prompt, diagnostics = maybe_repair_dissertation_subsection(
            client=client,
            model=model,
            cfg=cfg,
            context=context,
            stage=stage,
            body=stage_body,
            bib_meta_lookup=bib_meta_lookup,
            base_docs=base_docs,
        )
        stage_diagnostics.append(diagnostics)
        if repair_prompt:
            prompt_parts.append(f"===== stage {idx} repair {stage['top_title']} / {stage['sub_title']} =====\n" + repair_prompt + "\n")
        working_org = replace_section_body(working_org, 2, stage["sub_title"], stage_body)
        previous_excerpt = stage_body

    cited_after = set(extract_cited_keys_from_org(working_org))
    citation_target_keys = list(cfg.get("__citation_target_keys__") or bib_keys)
    missing_global = [k for k in citation_target_keys if k and k not in cited_after]
    coverage_info = {
        "total_bib_keys": len(bib_keys),
        "citation_target_source": cfg.get("__citation_target_source__", "bib_keys"),
        "citation_target_total": len(citation_target_keys),
        "citation_target_keys": citation_target_keys,
        "cited_after_stages": len(cited_after),
        "missing_after_stages": missing_global,
    }
    if missing_global:
        patch_prompt = build_missing_bibliography_patch_prompt(cfg, context, missing_global, bib_meta_lookup, base_docs, working_org)
        prompt_parts.append("===== missing_bibliography_patch =====\n" + patch_prompt + "\n")
        patch_resp = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": patch_prompt}],
            text_format=DissertationStageBodyOutput,
        )
        if patch_resp.output_parsed is None:
            raise RuntimeError("A IA não retornou a subseção integradora do corpus faltante.")
        patch_body = sanitize_generated_org_fragment(patch_resp.output_parsed.org_body)
        current_results = extract_section_body(working_org, 1, "RESULTADOS E DISCUSSÃO")
        current_results = (current_results.rstrip() + "\n\n" + patch_body.strip()).strip()
        working_org = replace_section_body(working_org, 1, "RESULTADOS E DISCUSSÃO", current_results)
        coverage_info["applied_patch_for_missing_keys"] = True
        coverage_info["missing_before_patch"] = missing_global

    working_org = ensure_document_class(working_org, "dissertacao")
    working_org = apply_citation_style(working_org, bib_filename, style)
    working_org = normalize_bibliography_block(working_org)
    working_org = ensure_cover_command(working_org)
    working_org = cleanup_generated_org(working_org)
    return working_org, "\n".join(prompt_parts), {"staged_generation": True, "section_diagnostics": stage_diagnostics, "coverage": coverage_info}


def generate_paper_org(client: OpenAI, model: str, cfg: dict[str, Any], context: DocumentContext, template_text: str, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], bib_filename: str, bib_entries: list[str], bib_keys: list[str], style: str, selected_corpus_catalog: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    doc_type = normalize_document_type(cfg.get("documento", {}).get("tipo_documento"))
    staged_generation = bool(cfg.get("documento", {}).get("geracao_em_etapas", doc_type == "dissertacao"))
    if doc_type == "dissertacao" and staged_generation:
        org_text, prompt_audit, staged_diagnostics = build_dissertation_org_from_stages(
            client=client,
            model=model,
            cfg=cfg,
            context=context,
            template_text=template_text,
            base_docs=base_docs,
            guidance_docs=guidance_docs,
            bib_filename=bib_filename,
            bib_entries=bib_entries,
            bib_keys=bib_keys,
            style=style,
            selected_corpus_catalog=selected_corpus_catalog,
        )
        cfg["__staged_generation_diagnostics__"] = staged_diagnostics
        return org_text, prompt_audit
    prompt = build_paper_prompt(cfg, context, template_text, base_docs, guidance_docs, bib_keys, bib_entries, style, selected_corpus_catalog=selected_corpus_catalog)
    resp = client.responses.create(model=model, input=prompt)
    org_text = resp.output_text.strip()
    org_text = ensure_document_class(org_text, doc_type)
    org_text = apply_citation_style(org_text, bib_filename, style)
    org_text = normalize_bibliography_block(org_text)
    org_text = ensure_cover_command(org_text)
    org_text = cleanup_generated_org(org_text)
    return org_text, prompt

def build_expansion_prompt(cfg: dict[str, Any], context: DocumentContext, current_org: str, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], diagnostics: dict[str, Any], bib_keys: list[str], selected_corpus_catalog: list[dict[str, Any]] | None = None) -> str:
    targets = diagnostics.get("targets", {})
    reasons = diagnostics.get("reasons", [])
    return textwrap.dedent(
        f"""
        Reescreva e expanda o documento acadêmico em Org-mode abaixo, preservando o cabeçalho técnico e a estrutura geral, mas aumentando substancialmente a densidade analítica e a extensão do texto.

        Contexto consolidado:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Diagnóstico do rascunho atual:
        {json.dumps(diagnostics, ensure_ascii=False, indent=2)}

        Regras obrigatórias:
        1. Preserve o formato Org-mode, a estrutura do template e o uso de Org Cite.
        2. Não invente chaves; use apenas estas chaves bibliográficas disponíveis: {json.dumps(bib_keys, ensure_ascii=False)}.
        3. Corrija compressões excessivas e desenvolva melhor as seções sinalizadas no diagnóstico.
        4. Para dissertação, o texto precisa ter fôlego analítico, comparando autores, desenvolvendo conceitos, explicitando lacunas e aprofundando implicações para a APF.
        5. Aprofunde especialmente as seções insuficientes segundo as metas mínimas abaixo:
        {json.dumps(targets, ensure_ascii=False, indent=2)}
        6. Razões da expansão necessária: {json.dumps(reasons, ensure_ascii=False)}.
        7. Use de modo mais denso o corpus formalmente selecionado e sua bibliografia.
        8. Não reduza o texto já existente; preserve o que estiver bom e amplie o que estiver curto.

        Catálogo do corpus selecionado:
        {shorten_text(json.dumps(selected_corpus_catalog or [], ensure_ascii=False, indent=2), 30000)}

        Textos-base resumidos:
        {json.dumps(summarize_docs(base_docs, excerpt_chars=3000), ensure_ascii=False, indent=2)}

        Orientações e artefatos da pesquisa:
        {json.dumps(summarize_docs(guidance_docs, excerpt_chars=2600), ensure_ascii=False, indent=2)}

        Documento atual a ser expandido:
        {shorten_text(current_org, 50000)}

        Retorne apenas o .org final completo e expandido.
        """
    ).strip()


def maybe_expand_dissertation_org(client: OpenAI, model: str, cfg: dict[str, Any], context: DocumentContext, org_text: str, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], bib_filename: str, bib_entries: list[str], bib_keys: list[str], style: str, selected_corpus_catalog: list[dict[str, Any]] | None = None) -> tuple[str, str | None, dict[str, Any] | None]:
    documento = cfg.get("documento", {})
    doc_type_local = normalize_document_type(documento.get("tipo_documento"))
    if doc_type_local != "dissertacao":
        return org_text, None, {}
    if bool(documento.get("geracao_em_etapas", True)):
        diagnostics = cfg.get("__staged_generation_diagnostics__") or {"staged_generation": True, "skipped_whole_document_expansion": True}
        return org_text, None, diagnostics
    selected_keys = sorted({d.bib_key for d in base_docs if d.kind.startswith("texto_selecionado") and d.bib_key})
    should_expand, diagnostics = should_expand_dissertation(cfg, org_text, selected_keys)
    if not should_expand:
        return org_text, None, diagnostics
    prompt = build_expansion_prompt(cfg, context, org_text, base_docs, guidance_docs, diagnostics, bib_keys, selected_corpus_catalog=selected_corpus_catalog)
    resp = client.responses.create(model=model, input=prompt)
    expanded = resp.output_text.strip()
    expanded = ensure_document_class(expanded, doc_type_local)
    expanded = apply_citation_style(expanded, bib_filename, style)
    expanded = normalize_bibliography_block(expanded)
    expanded = ensure_cover_command(expanded)
    expanded = cleanup_generated_org(expanded)
    return expanded, prompt, diagnostics

def build_document_output_dir(cfg: dict[str, Any], research_root: Path) -> Path:
    documento = cfg.get("documento", {})
    doc_type = normalize_document_type(documento.get("tipo_documento"))
    default_suffix = DEFAULT_DOC_PREFIX_BY_TYPE.get(doc_type, "documento")
    prefix = (documento.get("prefixo") or ((cfg.get("saida", {}).get("prefixo") or "atividade") + f"_{default_suffix}")).strip()
    output_dir_raw = documento.get("output_dir")
    if output_dir_raw:
        base = resolve_configured_path(output_dir_raw, cfg)
        if base is None:
            base = research_root
    else:
        base = research_root
    create_subdir = bool(documento.get("criar_subdiretorio", True))
    if not create_subdir:
        return base
    if base.name == prefix:
        return base
    return base / prefix

def build_bundle_dir(cfg: dict[str, Any], research_root: Path) -> Path:
    pipeline = cfg.get("pipeline", {})
    bundle_dir_raw = pipeline.get("bundle_dir")
    if bundle_dir_raw:
        resolved = resolve_configured_path(bundle_dir_raw, cfg)
        if resolved is not None:
            return resolved
    return research_root / "documento_bundle"

def copy_if_exists(src: Path | None, dest: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)

def build_bundle(cfg: dict[str, Any], research_paths: ResearchPaths, documento_context: DocumentContext, debug_json: dict[str, Any], bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = bundle_dir / "referencias"
    base_dir = bundle_dir / "textos_base"
    orient_dir = bundle_dir / "orientacoes"

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "mode_origem": cfg.get("atividade", {}).get("modo"),
        "research_root": str(research_paths.root_dir),
        "documento_context": asdict(documento_context),
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
        "documento_context": asdict(documento_context),
        "debug_json_excerpt": debug_json,
        "config_excerpt": {
            "atividade": cfg.get("atividade", {}),
            "pesquisa": cfg.get("pesquisa", {}),
            "bibliografia": cfg.get("bibliografia", {}),
        },
    }, ensure_ascii=False, indent=2, default=str))
    manifest["context_json"] = str(context_path)

    manifest_path = bundle_dir / "manifest.json"
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest_path

def update_bundle_with_documento(
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
    documento_dir = bundle_dir / "paper"
    documento_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})

    def copy_local(src: Path, key: str) -> str:
        dest = documento_dir / src.name
        shutil.copy2(src, dest)
        artifacts[key] = str(dest)
        return str(dest)

    copy_local(org_path, "documento_org")
    copy_local(bib_path, "documento_bib")
    copy_local(context_json_path, "documento_context_json")
    copy_local(prompt_audit_path, "documento_prompt_audit")
    if provenance_path and provenance_path.exists():
        copy_local(provenance_path, "documento_provenance_json")
    if pdf_path and pdf_path.exists():
        copy_local(pdf_path, "documento_pdf")

    copied_extras: list[str] = []
    for doc in extra_docs or []:
        p = Path(doc.path)
        if p.exists() and p.is_file():
            dest = documento_dir / "extras" / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            copied_extras.append(str(dest))
    artifacts["documento_extra_articles"] = copied_extras

    manifest["updated_at"] = datetime.now().isoformat()
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))


def assemble_delivery_package(
    cfg: dict[str, Any],
    *,
    research_paths: ResearchPaths,
    documento_output_dir: Path,
    documento_prefix: str,
    manifest_path: Path | None,
    org_path: Path,
    bib_path: Path,
    context_json_path: Path,
    prompt_audit_path: Path,
    provenance_path: Path,
    reference_usage_json_path: Path,
    reference_usage_md_path: Path,
    section_limits_json_path: Path,
    documento_pdf_path: Path | None = None,
) -> Path | None:
    entrega = cfg.get("entrega", {})
    if not bool(entrega.get("gerar_pacote_final", True)):
        return None

    package_dir = documento_output_dir / "entrega_final"
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
        "documento_org": copy_optional(org_path),
        "documento_bib": copy_optional(bib_path),
        "documento_contexto": copy_optional(context_json_path),
        "documento_prompts": copy_optional(prompt_audit_path),
        "documento_proveniencia": copy_optional(provenance_path),
        "reference_usage_json": copy_optional(reference_usage_json_path),
        "reference_usage_md": copy_optional(reference_usage_md_path),
        "section_limits_json": copy_optional(section_limits_json_path),
        "bundle_manifest": copy_optional(manifest_path),
    }
    if documento_pdf_path and bool(entrega.get("incluir_documento_pdf", True)):
        copied["documento_pdf"] = copy_optional(documento_pdf_path)
    if research_paths.prisma_svg_path and bool(entrega.get("incluir_prisma_svg", True)):
        copied["pesquisa_prisma_svg"] = copy_optional(research_paths.prisma_svg_path)
    if research_paths.prisma_pdf_path and bool(entrega.get("incluir_prisma_pdf", True)):
        copied["pesquisa_prisma_pdf"] = copy_optional(research_paths.prisma_pdf_path)

    readme = package_dir / "README_entrega.md"
    lines = [
        f"# Pacote final de entrega — {documento_prefix}",
        "",
        "Este diretório reúne os principais artefatos finais do pipeline integrado.",
        "",
        "## Itens copiados",
    ]
    lines.extend(f"- **{k}**: `{v}`" for k, v in copied.items() if v)
    lines.append("")
    write_text(readme, "\n".join(lines))
    return package_dir



def extract_citation_style_from_org(org_text: str) -> str:
    m = re.search(r"(?im)^\s*#\+CITE_EXPORT:\s*biblatex(?:\s+([A-Za-z0-9_\-]+))?", org_text)
    if m and m.group(1):
        return m.group(1).strip().lower()
    return DEFAULT_STYLE


def postprocess_generated_tex_for_bibliography(tex_path: Path, bib_filename: str, style: str) -> None:
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    bib_name = Path(str(bib_filename)).name
    style = (style or DEFAULT_STYLE).strip().lower() or DEFAULT_STYLE

    tex = re.sub(r"(?im)^\\PassOptionsToPackage\{.*\}\{biblatex\}\s*$\n?", "", tex)
    tex = re.sub(r"(?im)^\\ExecuteBibliographyOptions\{.*\}\s*$\n?", "", tex)
    tex = re.sub(r"(?ms)^sortcites=true,.*?(?=\\begin\{document\})", "", tex)
    tex = re.sub(r"(?ms)^pdfauthor=.*?(?=\\usepackage(?:\[[^\]]*\])?\{biblatex\}|\\begin\{document\})", "", tex)
    tex = re.sub(r"(?ms)^\\hypersetup\{.*?\}\s*", "", tex)
    tex = tex.replace("[][]", "")
    tex = tex.replace("<empty citation>", "")
    tex = re.sub(r"(?m)^\s*[,;:]+\s*$\n?", "", tex)
    tex = re.sub(r"(?m)^\s*P\d+\\?\}\s*$\n?", "", tex)
    tex = re.sub(r"\\dataaprovacao\{[^}]*__[^}]*\}", r"\\dataaprovacao{A definir}", tex)
    tex = re.sub(r"\\dataaprovacao\{\s*\}", r"\\dataaprovacao{A definir}", tex)
    tex = re.sub(r"\\autor\{\s*Seu nome\s*\}", rf"\\autor{{{DEFAULT_AUTHOR}}}", tex)
    tex = re.sub(r"\\author\{\s*Seu nome\s*\}", rf"\\author{{{DEFAULT_AUTHOR}}}", tex)

    if re.search(r"\\usepackage(?:\[[^\]]*\])?\{biblatex\}", tex):
        tex = re.sub(
            r"\\usepackage(?:\[[^\]]*\])?\{biblatex\}",
            rf"\\usepackage[backend=biber,style={style}]{{biblatex}}",
            tex,
            count=1,
        )
    else:
        tex = tex.replace(r"\begin{document}", f"\\usepackage[backend=biber,style={style}]{{biblatex}}\n\\addbibresource{{{bib_name}}}\n\\begin{{document}}", 1)

    tex = re.sub(r"\\addbibresource\{[^}]+\}", rf"\\addbibresource{{{bib_name}}}", tex)
    if not re.search(r"\\addbibresource\{[^}]+\}", tex):
        tex = tex.replace(
            rf"\usepackage[backend=biber,style={style}]{{biblatex}}",
            f"\\usepackage[backend=biber,style={style}]{{biblatex}}\n\\addbibresource{{{bib_name}}}",
            1,
        )

    tex = tex.replace(r"\maketitle", "")

    cleaned_lines = []
    seen_addbib = False
    for line in tex.splitlines():
        stripped = line.strip()
        if stripped.startswith("sortcites=true") or stripped.startswith("pdfauthor=") or stripped.startswith("pdftitle=") or stripped.startswith("pdfkeywords=") or stripped.startswith("pdfsubject=") or stripped.startswith("pdfcreator=") or stripped.startswith("pdflang="):
            continue
        if stripped in {",", ";", ":", "P26}", "P26\\}"}:
            continue
        if re.fullmatch(r"P\d+\\?\}", stripped):
            continue
        if stripped.startswith(r"\addbibresource{"):
            if seen_addbib:
                continue
            line = rf"\addbibresource{{{bib_name}}}"
            seen_addbib = True
        cleaned_lines.append(line)

    tex = "\n".join(cleaned_lines).strip() + "\n"
    tex_path.write_text(tex, encoding="utf-8")

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
    lualatex = ensure_command("lualatex")
    org_text = org_path.read_text(encoding="utf-8", errors="ignore")
    uses_citations = org_uses_citation_pipeline(org_text)
    style = extract_citation_style_from_org(org_text)
    export_el = org_path.parent / f"{org_path.stem}_export_pdf.el"
    tex_path = org_path.with_suffix('.tex')
    bib_path = org_path.with_suffix('.bib')

    init_loader = ""
    if academic_writing is not None:
        init_loader = f'(load-file "{academic_writing.as_posix()}")\n'

    cite_requirements = ""
    cite_guard = ""
    export_call = "(org-latex-export-to-latex)"

    if uses_citations:
        ensure_command("biber")
        cite_requirements = "(ignore-errors (require 'oc))\n(ignore-errors (require 'oc-biblatex))\n"
        cite_guard = textwrap.dedent("""
        (setq org-cite-insert-processor 'basic
              org-cite-follow-processor 'basic
              org-cite-activate-processor 'basic
              org-cite-export-processors '((latex biblatex) (t basic)))
        (setq-local org-cite-export-processors '((latex biblatex) (t basic)))
        """).strip() + "\n"
    else:
        cite_guard = textwrap.dedent("""
        (setq org-cite-insert-processor 'basic
              org-cite-follow-processor 'basic
              org-cite-activate-processor 'basic
              org-cite-export-processors '((latex basic) (t basic)))
        (setq-local org-cite-export-processors '((latex basic) (t basic)))
        (setq-local org-cite-global-bibliography nil)
        (when (boundp 'org-export-with-cite-processors)
          (setq org-export-with-cite-processors nil)
          (setq-local org-export-with-cite-processors nil))
        (when (boundp 'org-export-process-citations)
          (setq org-export-process-citations nil)
          (setq-local org-export-process-citations nil))
        """).strip() + "\n"
        export_call = "(let ((org-cite-export-processors '((latex basic) (t basic))) (org-export-with-cite-processors nil) (org-export-process-citations nil)) (org-latex-export-to-latex nil nil nil nil '(:with-cite-processors nil)))"

    export_code = textwrap.dedent(f"""
    {init_loader}(require 'org)
    (require 'ox)
    (require 'ox-latex)
    {cite_requirements}(find-file "{org_path.as_posix()}")
    (setq-local org-export-use-babel nil)
    (setq-local org-confirm-babel-evaluate nil)
    {cite_guard}{export_call}
    """).strip() + "\n"
    write_text(export_el, export_code)

    cmd = [emacs, "--batch", "-Q"]
    if emacs_init is not None:
        cmd.extend(["-l", str(emacs_init)])
    cmd.extend(["-l", str(export_el)])
    env = build_latex_env(latex_extra_path)
    proc = subprocess.run(cmd, cwd=str(org_path.parent), capture_output=True, text=True, env=env)
    debug_print("Exportando TEX:", cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Falha ao exportar TEX via Emacs batch.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    if not tex_path.exists():
        raise RuntimeError(f"Exportação concluída sem erro, mas o TEX não foi encontrado: {tex_path}")

    postprocess_generated_tex_for_bibliography(tex_path, bib_path.name, style)

    commands = [[lualatex, "-interaction=nonstopmode", "-file-line-error", tex_path.name]]
    if uses_citations:
        biber = ensure_command("biber")
        commands.append([biber, tex_path.stem])
        commands.append([lualatex, "-interaction=nonstopmode", "-file-line-error", tex_path.name])
        commands.append([lualatex, "-interaction=nonstopmode", "-file-line-error", tex_path.name])
    else:
        commands.append([lualatex, "-interaction=nonstopmode", "-file-line-error", tex_path.name])

    combined_stdout = []
    combined_stderr = []
    for one in commands:
        proc = subprocess.run(one, cwd=str(org_path.parent), capture_output=True, text=True, env=env)
        debug_print("Compilando PDF:", one)
        combined_stdout.append(proc.stdout)
        combined_stderr.append(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                "Falha ao compilar PDF do TEX pós-processado.\n"
                f"Comando: {' '.join(one)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

    pdf_path = org_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(
            "Compilação concluída sem erro, mas o PDF não foi encontrado.\n"
            f"STDOUT final:\n{''.join(combined_stdout)[-4000:]}\nSTDERR final:\n{''.join(combined_stderr)[-4000:]}"
        )
    return pdf_path

def build_provenance_payload(
    cfg: dict[str, Any],
    research_paths: ResearchPaths,
    documento_context: DocumentContext,
    base_docs: list[SourceDoc],
    guidance_docs: list[SourceDoc],
    extra_docs: list[SourceDoc],
    bib_keys: list[str],
    bib_entries: list[str],
    context_origin_info: dict[str, Any],
    template_path: Path | None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    documento = cfg.get("documento", {})
    return {
        "generated_at": datetime.now().isoformat(),
        "pipeline": {
            "executar_pesquisa": bool(cfg.get("pipeline", {}).get("executar_pesquisa", True)),
            "executar_documento": bool(cfg.get("pipeline", {}).get("executar_documento", True)),
            "pesquisa_dir_existente": str(cfg.get("pipeline", {}).get("pesquisa_dir_existente") or ""),
        },
        "documento_controls": {
            "modo_escrita": str(documento.get("modo_escrita") or "novo"),
            "perfil_redacao": str(documento.get("perfil_redacao") or "academico_equilibrado"),
            "usar_bib_da_pesquisa": bool(documento.get("usar_bib_da_pesquisa", True)),
            "incluir_artigos_extras_no_bib": bool(documento.get("incluir_artigos_extras_no_bib", True)),
            "permitir_busca_correlata_extra": bool(documento.get("permitir_busca_correlata_extra", False)),
            "priorizar_citacoes_dos_selecionados": bool(documento.get("priorizar_citacoes_dos_selecionados", True)),
            "extras_so_complementam": bool(documento.get("extras_so_complementam", True)),
            "preservar_estrutura_do_org_anterior": bool(documento.get("preservar_estrutura_do_org_anterior", False)),
            "usar_contexto_consolidado_da_pesquisa": bool(documento.get("usar_contexto_consolidado_da_pesquisa", True)),
            "reformular_tema_recorte_objetivo": bool(documento.get("reformular_tema_recorte_objetivo", False)),
            "limites_palavras": {
                "total": documento.get("limite_palavras_total"),
                "introducao": documento.get("limite_palavras_introducao"),
                "revisao": documento.get("limite_palavras_revisao"),
                "conclusao": documento.get("limite_palavras_conclusao"),
            },
        },
        "context_origin": context_origin_info,
        "documento_context_final": asdict(documento_context),
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
    documento = cfg.get("documento", {})
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
            "modo_escrita": str(documento.get("modo_escrita") or "novo"),
            "perfil_redacao": str(documento.get("perfil_redacao") or "academico_equilibrado"),
            "usar_artigos_selecionados_pesquisa": bool(documento.get("usar_artigos_selecionados_pesquisa", True)),
            "usar_bib_da_pesquisa": bool(documento.get("usar_bib_da_pesquisa", True)),
            "incluir_artigos_extras_no_bib": bool(documento.get("incluir_artigos_extras_no_bib", True)),
            "permitir_busca_correlata_extra": bool(documento.get("permitir_busca_correlata_extra", False)),
            "priorizar_citacoes_dos_selecionados": bool(documento.get("priorizar_citacoes_dos_selecionados", True)),
            "extras_so_complementam": bool(documento.get("extras_so_complementam", True)),
            "preservar_estrutura_do_org_anterior": bool(documento.get("preservar_estrutura_do_org_anterior", False)),
            "usar_contexto_consolidado_da_pesquisa": bool(documento.get("usar_contexto_consolidado_da_pesquisa", True)),
            "reformular_tema_recorte_objetivo": bool(documento.get("reformular_tema_recorte_objetivo", False)),
            "limites_palavras": {
                "total": documento.get("limite_palavras_total"),
                "introducao": documento.get("limite_palavras_introducao"),
                "revisao": documento.get("limite_palavras_revisao"),
                "conclusao": documento.get("limite_palavras_conclusao"),
            },
            "artigos_extras_paths": cfg.get("documento", {}).get("artigos_extras_paths", []),
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
    p = argparse.ArgumentParser(description="Pipeline integrado de pesquisa + documento usando TOML unificado.")
    p.add_argument("--config", required=True, help="Arquivo TOML unificado do pipeline.")
    p.add_argument("--model", default=None, help="Override do modelo OpenAI para a etapa do documento.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mock run: gera toda a cadeia de artefatos com conteúdo sintético
# ---------------------------------------------------------------------------
def _mock_lipsum(topic: str, idx: int = 1) -> str:
    base = [
        f"Texto simulado para validação estrutural do pipeline sobre {topic}.",
        "Este conteúdo foi gerado em mock_run para testar a escrita de arquivos, a montagem do pacote e a compilação opcional do PDF.",
        "O objetivo é preservar uma semântica acadêmica mínima, com encadeamento coerente entre problema, método, resultados e implicações.",
        "As passagens incluem citações fictícias, tabelas de exemplo e referências simuladas para testar bibliografia, sumário e renderização."
    ]
    return " ".join(base + [f"Parágrafo sintético {idx}."])

def _mock_write_pdf(path: Path, title: str) -> None:
    # PDF mínimo e válido para testes de existência/empacotamento.
    payload = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 95 >>
stream
BT
/F1 18 Tf
72 780 Td
({title[:60].replace('(', '[').replace(')', ']')}) Tj
0 -28 Td
/F1 11 Tf
(Mock PDF gerado em mock_run para validar a cadeia de artefatos.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000063 00000 n 
0000000122 00000 n 
0000000251 00000 n 
0000000397 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
467
%%EOF
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("latin-1", errors="ignore"))

def _mock_make_bib_entries(n: int, topic: str) -> tuple[list[str], list[str], list[str]]:
    keys: list[str] = []
    entries: list[str] = []
    titles: list[str] = []
    for i in range(1, n + 1):
        key = f"mock{i:02d}"
        title = f"Mock study {i:02d} on {topic}"
        year = 2018 + (i % 8)
        keys.append(key)
        titles.append(title)
        entries.append(textwrap.dedent(f"""
        @article{{{key},
          author = {{Autor Mock {i} and Coautor Simulado {i}}},
          title = {{{title}}},
          year = {{{year}}},
          journaltitle = {{Journal of Synthetic Public Sector AI Studies}},
          volume = {{{1 + (i % 9)}}},
          number = {{{1 + (i % 4)}}},
          pages = {{{10+i}-{18+i}}},
          doi = {{10.9999/mock.{i:02d}}},
          url = {{https://example.org/mock/{i:02d}}}
        }}
        """).strip())
    return entries, keys, titles

def _mock_make_prisma_svg(path: Path, total: int, selected: int) -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">
  <style>
    .b {{ fill:#f8fafc; stroke:#334155; stroke-width:2; rx:16; ry:16; }}
    .t {{ font: 18px Arial, sans-serif; fill:#0f172a; }}
    .s {{ font: 15px Arial, sans-serif; fill:#334155; }}
    .a {{ stroke:#475569; stroke-width:3; marker-end:url(#m); fill:none; }}
  </style>
  <defs><marker id="m" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#475569"/></marker></defs>
  <rect x="320" y="35" width="260" height="80" class="b"/><text x="350" y="70" class="t">Registros identificados</text><text x="350" y="98" class="s">n = {total}</text>
  <rect x="320" y="165" width="260" height="80" class="b"/><text x="350" y="200" class="t">Após deduplicação</text><text x="350" y="228" class="s">n = {max(total-8, selected)}</text>
  <rect x="320" y="295" width="260" height="80" class="b"/><text x="350" y="330" class="t">Triagem de títulos/resumos</text><text x="350" y="358" class="s">n = {max(total-15, selected)}</text>
  <rect x="320" y="425" width="260" height="80" class="b"/><text x="350" y="460" class="t">Estudos selecionados</text><text x="350" y="488" class="s">n = {selected}</text>
  <line x1="450" y1="115" x2="450" y2="165" class="a"/><line x1="450" y1="245" x2="450" y2="295" class="a"/><line x1="450" y1="375" x2="450" y2="425" class="a"/>
</svg>'''
    write_text(path, svg)

def _mock_make_selected_entries(root_dir: Path, prefixo: str, titles: list[str], keys: list[str]) -> tuple[list[dict[str, Any]], list[Path]]:
    cache_dir = root_dir / f"{prefixo}_fulltext_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    pdfs: list[Path] = []
    for i, (title, key) in enumerate(zip(titles, keys), start=1):
        pdf_path = cache_dir / f"{key}.pdf"
        _mock_write_pdf(pdf_path, title)
        pdfs.append(pdf_path)
        entries.append({
            "title": title,
            "abstract": _mock_lipsum(title, i),
            "year": str(2018 + (i % 8)),
            "authors": [f"Autor Mock {i}", f"Coautor Simulado {i}"],
            "source": "mock_run",
            "doi": f"10.9999/mock.{i:02d}",
            "url": f"https://example.org/mock/{i:02d}",
            "downloaded_pdf_path": str(pdf_path),
            "bib_key": key,
            "reason_selected": "Seleção sintética para validação completa da cadeia."
        })
    return entries, pdfs

def _mock_build_research_org(cfg: dict[str, Any], prefixo: str, titles: list[str], keys: list[str]) -> str:
    pesquisa = cfg.get("pesquisa", {})
    topic = pesquisa.get("tema") or "tema de pesquisa"
    rows = ["| ID | Título | Ano | Eixo | Chave |", "|-"]
    for i, (title, key) in enumerate(zip(titles, keys), start=1):
        eixo = ["governança", "ética", "compliance", "riscos"][i % 4]
        rows.append(f"| {i} | {title} | {2018 + (i % 8)} | {eixo} | {key} |")
    cites = " ".join(f"[cite:@{k}]" for k in keys[: min(6, len(keys))])
    rows_text = "\n".join(rows)
    selected_blocks = []
    for i, title in enumerate(titles[: min(10, len(titles))], start=1):
        selected_blocks.append(
            f"** Texto selecionado {i}\n{title}\n{_mock_lipsum(title, i+2)}\n"
        )
    selected_text = "\n".join(selected_blocks)
    return textwrap.dedent(f"""
    #+TITLE: {prefixo}
    #+AUTHOR: {cfg.get("atividade", {}).get("aluno") or DEFAULT_AUTHOR}
    #+LANGUAGE: pt_BR
    #+OPTIONS: toc:t num:t
    #+CITE_EXPORT: biblatex apa
    #+BIBLIOGRAPHY: {prefixo}.bib

    * Introdução
    {_mock_lipsum(topic, 1)}

    * Estratégia de busca
    {_mock_lipsum(topic, 2)}

    * Panorama preliminar
    {cites}

    * Tabela sintética de estudos selecionados
    {rows_text}

    * Estudos selecionados
    {selected_text}

    * Referências
    #+PRINT_BIBLIOGRAPHY:
    """).strip() + "\n"

def _mock_build_document_org(cfg: dict[str, Any], documento_prefix: str, doc_type: str, titles: list[str], keys: list[str]) -> str:
    pesquisa = cfg.get("pesquisa", {})
    tema = pesquisa.get("tema") or "tema"
    atividade = cfg.get("atividade", {})
    class_name = "fgv-dissertacao" if doc_type == "dissertacao" else "fgv-paper"
    cite_block = " ".join(f"[cite:@{k}]" for k in keys[: min(10, len(keys))])
    table = "\n".join([
        "| Eixo | Descrição | Intensidade | Referência |",
        "|-",
        f"| Governança | Estruturas decisórias e responsabilização | Alta | {keys[0] if keys else 'mock01'} |",
        f"| Ética | Transparência, vieses e supervisão humana | Alta | {keys[1] if len(keys)>1 else 'mock02'} |",
        f"| Compliance | Conformidade regulatória e controles | Média | {keys[2] if len(keys)>2 else 'mock03'} |",
        f"| Gestão de riscos | Monitoramento e mitigação | Alta | {keys[3] if len(keys)>3 else 'mock04'} |",
    ])
    corpo = textwrap.dedent(f"""
    #+TITLE: {documento_prefix}
    #+AUTHOR: {atividade.get("aluno") or DEFAULT_AUTHOR}
    #+LANGUAGE: pt_BR
    #+OPTIONS: toc:t num:t
    #+LATEX_COMPILER: lualatex
    #+LATEX_CLASS: {class_name}
    #+CITE_EXPORT: biblatex apa
    #+BIBLIOGRAPHY: {documento_prefix}.bib

    * RESUMO
    {_mock_lipsum(tema, 1)}

    * ABSTRACT
    {_mock_lipsum(tema, 2)}

    * INTRODUÇÃO
    {_mock_lipsum(tema, 3)}

    * REFERENCIAL TEÓRICO
    {cite_block}

    * METODOLOGIA
    {_mock_lipsum(tema, 4)}

    * RESULTADOS E DISCUSSÃO
    {_mock_lipsum(tema, 5)}

    ** Tabela sintética de eixos analíticos
    {table}

    ** Discussão integrada
    {' '.join(f"[cite:@{k}]" for k in keys[: min(8, len(keys))])}

    * CONSIDERAÇÕES FINAIS
    {_mock_lipsum(tema, 6)}

    * REFERÊNCIAS
    #+PRINT_BIBLIOGRAPHY:
    """).strip() + "\n"
    return corpo

def run_mock_pipeline(
    cfg: dict[str, Any],
    *,
    research_script: Path,
    research_output_dir: Path,
    temp_cfg_path: Path,
    executar_pesquisa: bool,
    executar_documento: bool,
    executar_bundle: bool,
) -> int:
    controle = cfg.get("controle", {})
    documento = cfg.get("documento", {})
    mock_seed = int(controle.get("mock_seed", 42) or 42)
    mock_registros = int(controle.get("mock_quantidade_registros", cfg.get("busca", {}).get("quantidade_selecionados", 12)) or 12)
    mock_registros = max(4, mock_registros)
    mock_gerar_pdf = bool(controle.get("mock_gerar_pdf", True))
    random.seed(mock_seed)

    research_output_dir.mkdir(parents=True, exist_ok=True)
    prefixo = (cfg.get("saida", {}).get("prefixo") or "atividade").strip()
    topic = str(cfg.get("pesquisa", {}).get("tema") or "tema de pesquisa").strip() or "tema de pesquisa"

    bib_entries, bib_keys, titles = _mock_make_bib_entries(mock_registros, shorten_text(topic, 60))
    selected_entries, selected_pdfs = _mock_make_selected_entries(research_output_dir, prefixo, titles, bib_keys)

    if executar_pesquisa or not detect_research_paths(cfg).org_path:
        org_path = research_output_dir / f"{prefixo}.org"
        bib_path = research_output_dir / f"{prefixo}.bib"
        debug_path = research_output_dir / f"{prefixo}_debug.json"
        pdf_path = research_output_dir / f"{prefixo}.pdf"
        prisma_svg_path = research_output_dir / f"{prefixo}_prisma.svg"
        prisma_pdf_path = research_output_dir / f"{prefixo}_prisma.pdf"

        write_text(org_path, _mock_build_research_org(cfg, prefixo, titles, bib_keys))
        write_text(bib_path, "\n\n".join(bib_entries).strip() + "\n")
        write_text(debug_path, json.dumps({
            "mock_run": True,
            "generated_at": datetime.now().isoformat(),
            "proposal": {
                "tema": cfg.get("pesquisa", {}).get("tema"),
                "recorte": cfg.get("pesquisa", {}).get("recorte"),
                "objetivo": cfg.get("pesquisa", {}).get("objetivo"),
                "pergunta_pesquisa": f"Como {topic.lower()} é tratado na literatura do setor público?",
                "hipotese": "A adoção responsável depende de governança, salvaguardas éticas e gestão de riscos."
            },
            "selected_all": selected_entries,
            "selected_count": len(selected_entries),
            "mock_seed": mock_seed,
        }, ensure_ascii=False, indent=2))
        _mock_make_prisma_svg(prisma_svg_path, total=max(mock_registros + 18, mock_registros), selected=mock_registros)
        if mock_gerar_pdf:
            _mock_write_pdf(pdf_path, prefixo)
            _mock_write_pdf(prisma_pdf_path, f"{prefixo} prisma")

    research_paths = detect_research_paths(cfg)
    print("[2/6] Carregando contexto e artefatos da pesquisa...", flush=True)
    documento_context, debug_json = build_document_context(cfg, research_paths)

    manifest_path: Path | None = None
    if executar_bundle:
        print("[3/6] Montando bundle intermediário...", flush=True)
        manifest_path = build_bundle(cfg, research_paths, documento_context, debug_json, build_bundle_dir(cfg, research_paths.root_dir))

    if not executar_documento:
        report_path = research_output_dir / "pipeline_mock_run_report.json"
        write_text(report_path, json.dumps({
            "generated_at": datetime.now().isoformat(),
            "mock_run": True,
            "document_stage": False,
            "research_root": str(research_output_dir),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "artifacts": {
                "research_org": str(research_paths.org_path) if research_paths.org_path else None,
                "research_bib": str(research_paths.bib_path) if research_paths.bib_path else None,
                "research_debug": str(research_paths.debug_path) if research_paths.debug_path else None,
            }
        }, ensure_ascii=False, indent=2))
        print("Mock run concluído. Artefatos sintéticos de pesquisa foram gerados.")
        print(f"- Relatório: {report_path}")
        return 0

    documento_output_dir = build_document_output_dir(cfg, research_paths.root_dir)
    documento_output_dir.mkdir(parents=True, exist_ok=True)
    doc_type = normalize_document_type(documento.get("tipo_documento"))
    documento_prefix = (documento.get("prefixo") or ((cfg.get("saida", {}).get("prefixo") or "atividade") + f"_{DEFAULT_DOC_PREFIX_BY_TYPE.get(doc_type, 'documento')}")).strip()
    documento_bib_name = f"{documento_prefix}.bib"

    base_docs: list[SourceDoc] = []
    for entry, pdf, key in zip(selected_entries, selected_pdfs, bib_keys):
        base_docs.append(SourceDoc(
            path=str(pdf),
            kind="texto_selecionado_mock",
            label=str(entry.get("title")),
            extracted_text=_mock_lipsum(str(entry.get("title")), 7),
            summary="Documento sintético para validação estrutural.",
            bib_key=key,
            metadata={"mock_run": True, "entry": entry},
        ))

    guidance_docs = collect_guidance_docs(cfg, research_paths)
    extra_docs: list[SourceDoc] = []

    org_text = _mock_build_document_org(cfg, documento_prefix, doc_type, titles, bib_keys)
    prompt_text = textwrap.dedent(f"""    MOCK RUN — nenhum prompt real enviado à OpenAI.
    Tema: {cfg.get("pesquisa", {}).get("tema")}
    Tipo de documento: {doc_type}
    Quantidade de referências sintéticas: {len(bib_keys)}
    """)

    org_path = documento_output_dir / f"{documento_prefix}.org"
    bib_path = documento_output_dir / documento_bib_name
    context_json_path = documento_output_dir / f"{documento_prefix}_contexto.json"
    prompt_audit_path = documento_output_dir / f"{documento_prefix}_prompts_auditoria.txt"
    provenance_path = documento_output_dir / f"{documento_prefix}_proveniencia.json"
    reference_usage_json_path = documento_output_dir / f"{documento_prefix}_uso_referencias.json"
    reference_usage_md_path = documento_output_dir / f"{documento_prefix}_uso_referencias.md"
    section_limits_json_path = documento_output_dir / f"{documento_prefix}_limites_secoes.json"
    documento_pdf_path = documento_output_dir / f"{documento_prefix}.pdf" if mock_gerar_pdf else None

    write_text(org_path, org_text)
    write_text(bib_path, "\n\n".join(bib_entries).strip() + "\n")
    write_text(context_json_path, json.dumps({
        "generated_at": datetime.now().isoformat(),
        "mock_run": True,
        "documento_context": asdict(documento_context),
        "research_paths": json_safe(asdict(research_paths)),
        "config_excerpt": {
            "atividade": cfg.get("atividade", {}),
            "pesquisa": cfg.get("pesquisa", {}),
            "documento": cfg.get("documento", {}),
        },
        "bundle_manifest": str(manifest_path) if manifest_path else None,
        "context_origin_info": {"used": False, "source": "mock_run"},
        "bib_keys": bib_keys,
        "base_docs": [asdict(d) for d in base_docs],
        "guidance_docs": [asdict(d) for d in guidance_docs],
        "selected_entries": selected_entries,
        "selected_fulltext_paths": [str(p) for p in selected_pdfs],
        "extra_article_paths": [],
    }, ensure_ascii=False, indent=2, default=str))
    write_text(prompt_audit_path, prompt_text)
    write_text(provenance_path, json.dumps(
        build_provenance_payload(
            cfg=cfg,
            research_paths=research_paths,
            documento_context=documento_context,
            base_docs=base_docs,
            guidance_docs=guidance_docs,
            extra_docs=extra_docs,
            bib_keys=bib_keys,
            bib_entries=bib_entries,
            context_origin_info={"used": False, "source": "mock_run"},
            template_path=Path(str(documento.get("template_org") or "")) if documento.get("template_org") else None,
            manifest_path=manifest_path,
        ),
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
    reference_usage = build_reference_usage_map(org_text, base_docs, extra_docs, bib_keys, required_keys=cfg.get("__citation_target_keys__"))
    write_text(reference_usage_json_path, json.dumps(reference_usage, ensure_ascii=False, indent=2))
    write_text(reference_usage_md_path, render_reference_usage_markdown(reference_usage))
    write_text(section_limits_json_path, json.dumps({
        "mock_run": True,
        "word_counts": count_org_words_per_top_section(org_text),
        "configured_limits": {
            "total": documento.get("limite_palavras_total"),
            "introducao": documento.get("limite_palavras_introducao"),
            "revisao": documento.get("limite_palavras_revisao"),
            "conclusao": documento.get("limite_palavras_conclusao"),
        }
    }, ensure_ascii=False, indent=2))

    if documento_pdf_path:
        _mock_write_pdf(documento_pdf_path, documento_prefix)

    if manifest_path:
        update_bundle_with_documento(
            manifest_path,
            org_path=org_path,
            bib_path=bib_path,
            context_json_path=context_json_path,
            prompt_audit_path=prompt_audit_path,
            provenance_path=provenance_path,
            extra_docs=extra_docs,
            pdf_path=documento_pdf_path,
        )

    package_dir = assemble_delivery_package(
        cfg,
        research_paths=research_paths,
        documento_output_dir=documento_output_dir,
        documento_prefix=documento_prefix,
        manifest_path=manifest_path,
        org_path=org_path,
        bib_path=bib_path,
        context_json_path=context_json_path,
        prompt_audit_path=prompt_audit_path,
        provenance_path=provenance_path,
        reference_usage_json_path=reference_usage_json_path,
        reference_usage_md_path=reference_usage_md_path,
        section_limits_json_path=section_limits_json_path,
        documento_pdf_path=documento_pdf_path,
    )

    report_path = research_output_dir / "pipeline_mock_run_report.json"
    write_text(report_path, json.dumps({
        "generated_at": datetime.now().isoformat(),
        "mock_run": True,
        "mock_seed": mock_seed,
        "mock_quantidade_registros": mock_registros,
        "mock_gerar_pdf": mock_gerar_pdf,
        "research_root": str(research_output_dir),
        "document_root": str(documento_output_dir),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "delivery_package": str(package_dir) if package_dir else None,
        "artifacts": {
            "research_org": str(research_paths.org_path) if research_paths.org_path else None,
            "research_bib": str(research_paths.bib_path) if research_paths.bib_path else None,
            "research_debug": str(research_paths.debug_path) if research_paths.debug_path else None,
            "research_pdf": str(research_paths.pdf_path) if research_paths.pdf_path else None,
            "research_prisma_svg": str(research_paths.prisma_svg_path) if research_paths.prisma_svg_path else None,
            "document_org": str(org_path),
            "document_bib": str(bib_path),
            "document_context": str(context_json_path),
            "document_prompt_audit": str(prompt_audit_path),
            "document_provenance": str(provenance_path),
            "reference_usage_json": str(reference_usage_json_path),
            "reference_usage_md": str(reference_usage_md_path),
            "section_limits_json": str(section_limits_json_path),
            "document_pdf": str(documento_pdf_path) if documento_pdf_path else None,
        }
    }, ensure_ascii=False, indent=2))

    print("Mock run concluído. Toda a cadeia de artefatos sintéticos foi gerada.")
    print(f"- Relatório: {report_path}")
    print(f"- Pesquisa ORG: {research_paths.org_path}")
    print(f"- Documento ORG: {org_path}")
    print(f"- Documento BIB: {bib_path}")
    if documento_pdf_path:
        print(f"- Documento PDF sintético: {documento_pdf_path}")
    if package_dir:
        print(f"- Pacote final de entrega: {package_dir}")
    return 0


def main() -> int:
    global DEBUG
    load_env()
    args = parse_args()
    DEBUG = bool(args.debug)

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    cfg["__config_dir__"] = str(cfg_path.parent)
    pipeline = cfg.get("pipeline", {})

    executar_pesquisa = bool(pipeline.get("executar_pesquisa", True))
    executar_documento = bool(pipeline.get("executar_documento", True))
    executar_bundle = bool(pipeline.get("executar_bundle", pipeline.get("criar_bundle", True)))
    dry_run = bool(cfg.get("controle", {}).get("dry_run", False))
    mock_run = bool(cfg.get("controle", {}).get("mock_run", False))

    research_script = resolve_configured_path(pipeline.get("script_pesquisa") or DEFAULT_RESEARCH_SCRIPT, cfg)
    if research_script is None:
        research_script = Path(DEFAULT_RESEARCH_SCRIPT).expanduser().resolve()

    # prepara TOML temporário só com as seções aceitas pelo gerador de pesquisa
    research_output_dir = detect_research_output_dir(cfg)
    temp_cfg_path = research_output_dir / "pipeline_research_config.toml"
    research_output_dir.mkdir(parents=True, exist_ok=True)
    research_cfg = filter_research_config(cfg, research_output_dir)
    write_text(temp_cfg_path, dict_to_toml(research_cfg))

    research_paths = detect_research_paths(cfg)

    if dry_run:
        report = build_dry_run_report(cfg, research_cfg, research_script, research_paths)
        dry_run_path = research_output_dir / "pipeline_dry_run_report.json"
        write_text(dry_run_path, json.dumps(report, ensure_ascii=False, indent=2))
        print("Dry run concluído. Nenhuma chamada à OpenAI ou execução pesada foi realizada.")
        print(f"- Relatório: {dry_run_path}")
        return 0

    if mock_run:
        return run_mock_pipeline(
            cfg,
            research_script=research_script,
            research_output_dir=research_output_dir,
            temp_cfg_path=temp_cfg_path,
            executar_pesquisa=executar_pesquisa,
            executar_documento=executar_documento,
            executar_bundle=executar_bundle,
        )

    if executar_pesquisa:
        print("[1/6] Executando etapa de pesquisa...", flush=True)
        run_research_stage(temp_cfg_path, research_script)
        research_paths = detect_research_paths(cfg)
    if research_paths.org_path is None:
        raise RuntimeError(f"Não foi possível localizar o .org da pesquisa em {research_paths.root_dir}.")

    print("[2/6] Carregando contexto e artefatos da pesquisa...", flush=True)
    documento_context, debug_json = build_document_context(cfg, research_paths)

    manifest_path: Path | None = None
    if executar_bundle:
        print("[3/6] Montando bundle intermediário...", flush=True)
        manifest_path = build_bundle(cfg, research_paths, documento_context, debug_json, build_bundle_dir(cfg, research_paths.root_dir))

    if not executar_documento:
        print("Pesquisa concluída. Etapa do documento desativada no TOML. O .org da pesquisa foi preservado.")
        if manifest_path:
            print(f"Bundle: {manifest_path}")
        return 0

    print("[4/6] Inicializando cliente OpenAI para geração da dissertação...", flush=True)
    client, model = make_client(args.model or cfg.get("openai", {}).get("model"))

    # documento output
    documento_output_dir = build_document_output_dir(cfg, research_paths.root_dir)
    documento_output_dir.mkdir(parents=True, exist_ok=True)
    documento = cfg.get("documento", {})
    doc_type = normalize_document_type(documento.get("tipo_documento"))
    documento_prefix = (documento.get("prefixo") or ((cfg.get("saida", {}).get("prefixo") or "atividade") + f"_{DEFAULT_DOC_PREFIX_BY_TYPE.get(doc_type, 'documento')}" )).strip()
    documento_bib_name = f"{documento_prefix}.bib"

    # template do documento
    # prioridade: template_paper.org; fallback organizacional: template_research.org
    template_path = None
    doc_type = normalize_document_type(documento.get("tipo_documento"))
    if documento.get("template_org"):
        template_path = resolve_configured_path(documento["template_org"], cfg)
    else:
        template_path = find_fallback_template(doc_type)

    template_text = read_template_raw(template_path) if template_path and template_path.exists() else build_default_document_template(doc_type)

    # docs e bibliografia
    base_docs = collect_base_docs(cfg, research_paths)
    guidance_docs = collect_guidance_docs(cfg, research_paths)
    extra_docs = collect_extra_article_docs(cfg)
    if extra_docs:
        debug_print(f"Artigos extras carregados: {len(extra_docs)}")
    base_docs.extend(extra_docs)

    documento = cfg.get("documento", {})
    usar_bib_da_pesquisa = bool(documento.get("usar_bib_da_pesquisa", True))
    incluir_artigos_extras_no_bib = bool(documento.get("incluir_artigos_extras_no_bib", True))
    style = (documento.get("estilo_citacao") or cfg.get("bibliografia", {}).get("estilo_citacao") or DEFAULT_STYLE)

    bib_entries, bib_keys = (parse_bib_entries(research_paths.bib_path) if usar_bib_da_pesquisa else ([], []))
    if extra_docs and incluir_artigos_extras_no_bib:
        extra_docs, extra_bib_entries, extra_bib_keys = build_bib_entries_for_extra_docs(client, model, extra_docs, bib_keys)
        bib_entries.extend(extra_bib_entries)
        bib_keys.extend(extra_bib_keys)
    if not bib_entries:
        debug_print("Nenhum .bib disponível para o documento; bibliografia ficará vazia até revisão posterior.")

    assign_bib_keys_to_selected_docs(base_docs, research_paths, bib_entries)
    fulltext_citation_keys, unresolved_fulltext_pdfs = build_fulltext_cache_citation_keys(research_paths, bib_entries)
    if fulltext_citation_keys:
        cfg["__citation_target_keys__"] = fulltext_citation_keys
        cfg["__citation_target_source__"] = "fulltext_cache"
    else:
        cfg["__citation_target_keys__"] = sorted({d.bib_key for d in base_docs if d.kind.startswith("texto_selecionado") and d.bib_key})
        cfg["__citation_target_source__"] = "selected_base_docs"
    if unresolved_fulltext_pdfs:
        debug_print(f"PDFs do fulltext_cache sem chave BibTeX mapeada: {len(unresolved_fulltext_pdfs)}")
    selected_corpus_catalog = build_selected_corpus_catalog(research_paths, bib_entries, limit=max(40, len(fulltext_citation_keys) + 10))

    documento_context, context_origin_info = maybe_rewrite_document_context(client, model, cfg, documento_context, guidance_docs)

    print("[5/6] Gerando ORG da dissertação...", flush=True)
    org_text, prompt_text = generate_paper_org(
        client=client,
        model=model,
        cfg=cfg,
        context=documento_context,
        template_text=template_text,
        base_docs=base_docs,
        guidance_docs=guidance_docs,
        bib_filename=documento_bib_name,
        bib_entries=bib_entries,
        bib_keys=bib_keys,
        style=style,
        selected_corpus_catalog=selected_corpus_catalog,
    )
    expansion_prompt_text = None
    expansion_diagnostics = None
    org_text, expansion_prompt_text, expansion_diagnostics = maybe_expand_dissertation_org(
        client=client,
        model=model,
        cfg=cfg,
        context=documento_context,
        org_text=org_text,
        base_docs=base_docs,
        guidance_docs=guidance_docs,
        bib_filename=documento_bib_name,
        bib_entries=bib_entries,
        bib_keys=bib_keys,
        style=style,
        selected_corpus_catalog=selected_corpus_catalog,
    )

    front = infer_final_front_matter(client, model, documento_context, org_text)
    atividade = cfg.get("atividade", {})
    final_title = front.title.strip()
    org_text = apply_final_front_matter(
        org_text,
        title=final_title,
        author=str(atividade.get("aluno") or DEFAULT_AUTHOR),
        documento_type=front.documento_type.strip(),
        cover_note=front.cover_note.strip(),
        institution_name=str(documento.get("institution_name") or DEFAULT_INSTITUTION),
        course_name=str(atividade.get("curso") or ""),
        discipline_name=str(atividade.get("disciplina") or ""),
        professor_name=str(atividade.get("professor") or ""),
        city_name=str(atividade.get("polo") or "Brasília"),
    )
    if normalize_document_type(documento.get("tipo_documento")) == "dissertacao":
        org_text = apply_dissertation_template_metadata(org_text, cfg, final_title)

    # grava artefatos do documento
    org_path = documento_output_dir / f"{documento_prefix}.org"
    bib_path = documento_output_dir / documento_bib_name
    context_json_path = documento_output_dir / f"{documento_prefix}_contexto.json"
    prompt_audit_path = documento_output_dir / f"{documento_prefix}_prompts_auditoria.txt"
    provenance_path = documento_output_dir / f"{documento_prefix}_proveniencia.json"

    write_text(org_path, org_text)
    write_text(bib_path, "\n\n".join(bib_entries).strip() + ("\n" if bib_entries else ""))
    write_text(context_json_path, json.dumps({
        "generated_at": datetime.now().isoformat(),
        "documento_context": asdict(documento_context),
        "research_paths": json_safe(asdict(research_paths)),
        "config_excerpt": {
            "atividade": cfg.get("atividade", {}),
            "pesquisa": cfg.get("pesquisa", {}),
            "documento": cfg.get("documento", {}),
        },
        "bundle_manifest": str(manifest_path) if manifest_path else None,
        "context_origin_info": context_origin_info,
        "bib_keys": bib_keys,
        "base_docs": [asdict(d) for d in base_docs],
        "guidance_docs": [asdict(d) for d in guidance_docs],
        "selected_entries": research_paths.selected_entries,
        "selected_fulltext_paths": [str(p) for p in research_paths.selected_fulltext_paths],
        "selected_corpus_catalog": selected_corpus_catalog,
        "citation_target_source": cfg.get("__citation_target_source__"),
        "citation_target_keys": cfg.get("__citation_target_keys__", []),
        "unresolved_fulltext_pdfs": unresolved_fulltext_pdfs,
        "expansion_diagnostics": expansion_diagnostics,
        "extra_article_paths": [d.path for d in extra_docs],
    }, ensure_ascii=False, indent=2))
    prompt_audit_content = "===== generate_paper_org =====\n" + prompt_text + "\n"
    if expansion_prompt_text:
        prompt_audit_content += "\n===== maybe_expand_dissertation_org =====\n" + expansion_prompt_text + "\n"
        prompt_audit_content += "\n===== diagnostics =====\n" + json.dumps(expansion_diagnostics or {}, ensure_ascii=False, indent=2) + "\n"
    write_text(prompt_audit_path, prompt_audit_content)
    write_text(provenance_path, json.dumps(
        build_provenance_payload(
            cfg=cfg,
            research_paths=research_paths,
            documento_context=documento_context,
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
        default=str,
    ))

    reference_usage = build_reference_usage_map(org_text, base_docs, extra_docs, bib_keys, required_keys=cfg.get("__citation_target_keys__"))
    reference_usage_json_path = documento_output_dir / f"{documento_prefix}_uso_referencias.json"
    reference_usage_md_path = documento_output_dir / f"{documento_prefix}_uso_referencias.md"
    write_text(reference_usage_json_path, json.dumps(reference_usage, ensure_ascii=False, indent=2))
    write_text(reference_usage_md_path, render_reference_usage_markdown(reference_usage))

    section_limits_json_path = documento_output_dir / f"{documento_prefix}_limites_secoes.json"
    write_text(section_limits_json_path, json.dumps({
        "word_counts": count_org_words_per_top_section(org_text),
        "configured_limits": {
            "total": documento.get("limite_palavras_total"),
            "introducao": documento.get("limite_palavras_introducao"),
            "revisao": documento.get("limite_palavras_revisao"),
            "conclusao": documento.get("limite_palavras_conclusao"),
        }
    }, ensure_ascii=False, indent=2))

    # compila PDF do documento se pedido
    documento_pdf_path = None
    if bool(documento.get("exportar_pdf", cfg.get("saida", {}).get("exportar_pdf", False))):
        print("[6/6] Compilando PDF da dissertação...", flush=True)
        latex = cfg.get("latex", {})
        academic_writing = resolve_configured_path(latex.get("org_latex_class_init"), cfg) if latex.get("org_latex_class_init") else None
        emacs_init = resolve_configured_path(latex.get("emacs_init"), cfg) if latex.get("emacs_init") else None
        latex_extra_path = resolve_configured_path(latex.get("latex_extra_path"), cfg) if latex.get("latex_extra_path") else None
        try:
            documento_pdf_path = run_compile_sequence(org_path, emacs_init=emacs_init, academic_writing=academic_writing, latex_extra_path=latex_extra_path)
        except Exception as exc:
            pdf_error_path = documento_output_dir / f"{documento_prefix}_pdf_erro.txt"
            write_text(
                pdf_error_path,
                "A compilação do PDF do documento falhou, mas o .org da dissertação foi preservado.\n\n"
                f"Documento ORG: {org_path}\n"
                f"Documento BIB: {bib_path}\n\n"
                f"Erro:\n{exc}\n",
            )
            print("Aviso: a compilação do PDF do documento falhou, mas os artefatos textuais foram preservados.")
            print(f"- Documento ORG preservado em: {org_path}")
            print(f"- Log simplificado da falha do PDF: {pdf_error_path}")
            documento_pdf_path = None

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
        print(f"- PDFs selecionados usados no documento: {len(research_paths.selected_fulltext_paths)}")
    if extra_docs:
        print(f"- Artigos extras usados no documento: {len(extra_docs)}")
    print(f"- Documento ORG: {org_path}")
    print(f"- Documento BIB: {bib_path}")
    print(f"- Documento CONTEXTO: {context_json_path}")
    print(f"- Documento PROVENIÊNCIA: {provenance_path}")
    print(f"- Documento PROMPTS: {prompt_audit_path}")
    print(f"- Diretório do documento: {documento_output_dir}")
    print(f"- Uso de referências (JSON): {reference_usage_json_path}")
    print(f"- Uso de referências (MD): {reference_usage_md_path}")
    print(f"- Contagem/limites por seção: {section_limits_json_path}")
    if manifest_path:
        update_bundle_with_documento(
            manifest_path,
            org_path=org_path,
            bib_path=bib_path,
            context_json_path=context_json_path,
            prompt_audit_path=prompt_audit_path,
            provenance_path=provenance_path,
            extra_docs=extra_docs,
            pdf_path=documento_pdf_path,
        )

    package_dir = assemble_delivery_package(
        cfg,
        research_paths=research_paths,
        documento_output_dir=documento_output_dir,
        documento_prefix=documento_prefix,
        manifest_path=manifest_path,
        org_path=org_path,
        bib_path=bib_path,
        context_json_path=context_json_path,
        prompt_audit_path=prompt_audit_path,
        provenance_path=provenance_path,
        reference_usage_json_path=reference_usage_json_path,
        reference_usage_md_path=reference_usage_md_path,
        section_limits_json_path=section_limits_json_path,
        documento_pdf_path=documento_pdf_path,
    )

    req_total = int(reference_usage.get("counts", {}).get("required_fulltext_cache_total", 0))
    req_cited = int(reference_usage.get("counts", {}).get("required_fulltext_cache_cited_total", 0))
    if req_total and req_cited < req_total:
        print(f"Aviso: o documento citou {req_cited}/{req_total} chave(s) dos PDFs presentes no fulltext_cache. Verifique o mapa de uso de referências para as chaves faltantes.")

    if documento_pdf_path:
        print(f"- Documento PDF: {documento_pdf_path}")
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

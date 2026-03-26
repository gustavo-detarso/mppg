#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador autocontido de paper em Org-mode a partir de template .org,
com suporte a textos-base, orientações do professor, .zip de artigos, inferência via OpenAI,
referências correlatas opcionais em múltiplas bases e compilação opcional.

Fluxo:
1. Lê .env (OPENAI_API_KEY e, opcionalmente, chaves de Semantic Scholar / Scopus / Web of Science).
2. Pergunta, em ordem lógica, os campos do template e os metadados estratégicos.
3. Infere primeiro um contexto-base amplo e, em seguida, afunila o paper em segunda etapa.
4. Opcionalmente busca artigos correlatos com texto completo verificável.
5. Gera .org, .bib, .json de contexto e .txt de auditoria de prompts.
6. Opcionalmente compila com: lualatex -> biber -> lualatex -> lualatex.
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
import traceback
import zipfile
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

try:
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None

try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.completion import PathCompleter, WordCompleter
    PROMPT_TOOLKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    pt_prompt = None
    PathCompleter = None
    WordCompleter = None
    PROMPT_TOOLKIT_AVAILABLE = False


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
DEFAULT_TEMPLATE_ORG = "template_paper.org"
FALLBACK_TEMPLATE_ORG = "template.org"
DEFAULT_BASENAME = "paper"
DEFAULT_STYLE = "apa"
DEFAULT_ACADEMIC_WRITING = os.getenv("ORG_LATEX_CLASS_INIT", "/home/gustavodetarso/.emacs.d/lisp/academic-writing.el")
DEFAULT_LATEX_EXTRA_PATH = os.getenv("LATEX_EXTRA_PATH", "/home/gustavodetarso/texmf/tex/latex/fgv/fgv-paper.sty")
STATE_FILE = ".gerar_paper_org_ai_state.json"
PROMPTS_DIR_NAME = "prompts"
PROMPT_INFERIR_CONTEXTO = "inferir_contexto_base.txt"
PROMPT_AFUNILAR_PAPER = "afunilar_paper.txt"
PROMPT_GERAR_TITULO_CAPA = "gerar_titulo_e_capa.txt"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"
WOS_URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"
DEBUG = False
DEFAULT_AUTHOR = "Gustavo M. Mendes de Tarso"
DEFAULT_INSTITUTION = "Faculdade Getúlio Vargas"
AUTO_HEADER_TITLE = "TÍTULO A SER GERADO PELA IA"
AUTO_HEADER_PAPER_TYPE = "Texto gerado automaticamente pela IA após a conclusão do paper"
AUTO_HEADER_COVER_NOTE = "Nota filosófica a ser gerada pela IA após a conclusão do paper"



@dataclass
class SourceDoc:
    path: str
    kind: str
    label: str
    extracted_text: str
    summary: str | None = None
    bib_key: str | None = None
    bib_entry: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputItem:
    path: Path
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".org", ".rst", ".tex", ".json", ".csv", ".yaml", ".yml", ".xml"}
SUPPORTED_BINARY_SUFFIXES = {".pdf", ".docx"}
SUPPORTED_DOC_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_BINARY_SUFFIXES
SUPPORTED_CONTAINER_SUFFIXES = {".zip"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_DOC_SUFFIXES | SUPPORTED_CONTAINER_SUFFIXES


@dataclass
class PaperContext:
    tema: str
    recorte: str
    objetivo: str
    pergunta_pesquisa: str | None = None
    hipotese: str | None = None
    palavras_chave: list[str] = field(default_factory=list)
    titulo_sugerido: str | None = None
    tipo_estudo_correlato: str = "artigo acadêmico"
    idiomas: list[str] = field(default_factory=lambda: ["português", "inglês"])


@dataclass
class CandidatePaper:
    paper_id: str
    title: str
    abstract: str
    year: int | None
    venue: str | None
    authors: list[str]
    url: str | None
    pdf_url: str | None
    doi: str | None
    source: str
    full_text_verified: bool = False
    downloaded_pdf_path: str | None = None
    tldr: str | None = None


@dataclass
class SourceFetchResult:
    source: str
    query: str
    retrieved: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class SearchAudit:
    per_source: list[SourceFetchResult]
    identified_total: int
    duplicates_removed: int
    other_removed: int
    screened_total: int
    warnings: list[str] = field(default_factory=list)


class ThemeCombinationOutput(BaseModel):
    tema_adicional_1: str
    tema_adicional_2: str
    justificativa: str | None = None


class BaseContextInferenceOutput(BaseModel):
    tema: str
    recorte_amplo: str
    objetivo_amplo: str
    pergunta_ampla: str | None = None
    hipotese_ampla: str | None = None
    palavras_chave: list[str] = Field(default_factory=list)
    temas_adicionais_candidatos: list[str] = Field(default_factory=list)
    combinacoes_sugeridas: list[ThemeCombinationOutput] = Field(default_factory=list)
    observacoes_sobre_limites_do_recorte: str | None = None


class FinalContextInferenceOutput(BaseModel):
    tema_final: str
    recorte_final: str
    objetivo_final: str
    pergunta_pesquisa: str
    hipotese: str | None = None
    estrutura_sugerida: list[str] = Field(default_factory=list)
    autores_centrais: list[str] = Field(default_factory=list)
    palavras_chave: list[str] = Field(default_factory=list)
    titulo_provisorio: str | None = None


class BibMetadataOutput(BaseModel):
    entry_type: str = "article"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    journaltitle: str | None = None
    booktitle: str | None = None
    publisher: str | None = None
    school: str | None = None
    doi: str | None = None
    url: str | None = None
    note: str | None = None


class QuerySuggestionOutput(BaseModel):
    palavras_chave: list[str]
    query_geral: str
    query_semantic: str | None = None
    query_scopus: str | None = None
    query_wos: str | None = None


class RankedRefItem(BaseModel):
    paper_id: str
    justificativa: str


class RankedReferenceOutput(BaseModel):
    selecionados: list[RankedRefItem]
    observacao: str | None = None


class FinalFrontMatterOutput(BaseModel):
    title: str
    paper_type: str
    cover_note: str


@dataclass
class TemplateField:
    key: str
    default: str
    line: str
    category: str
    prompt: str


def load_env() -> None:
    load_dotenv(override=False)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def load_state() -> dict[str, Any]:
    path = script_dir() / STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(data: dict[str, Any]) -> None:
    path = script_dir() / STATE_FILE
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY não encontrado no .env ou no ambiente.")
    return OpenAI(api_key=key)

def debug_print(*parts: object) -> None:
    if DEBUG:
        print("[DEBUG]", *parts, file=sys.stderr)


def default_emacs_init() -> Path | None:
    candidates = [Path.home() / ".emacs.d" / "init.el", Path.home() / ".config" / "emacs" / "init.el"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_template_path() -> Path:
    candidates = [script_dir() / DEFAULT_TEMPLATE_ORG, script_dir() / FALLBACK_TEMPLATE_ORG]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return script_dir() / DEFAULT_TEMPLATE_ORG


def default_academic_writing() -> Path | None:
    candidates = [
        Path(DEFAULT_ACADEMIC_WRITING).expanduser(),
        script_dir() / 'academic-writing.el',
        Path.home() / '.emacs.d' / 'lisp' / 'academic-writing.el',
        Path.home() / '.config' / 'emacs' / 'lisp' / 'academic-writing.el',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def ensure_command_available(name: str) -> str:
    cmd = shutil.which(name)
    if not cmd:
        raise RuntimeError(f"Comando obrigatório não encontrado no PATH: {name}")
    return cmd


def run_checked(cmd: list[str], *, cwd: str | None = None, label: str | None = None) -> subprocess.CompletedProcess[str]:
    debug_print("Executando", label or "comando", cmd, f"cwd={cwd or os.getcwd()}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    debug_print("rc=", proc.returncode)
    if proc.stdout.strip():
        debug_print("stdout:\n" + proc.stdout[-4000:])
    if proc.stderr.strip():
        debug_print("stderr:\n" + proc.stderr[-4000:])
    return proc


def preflight_checks(*, exportar_pdf: bool, emacs_init: Path | None, academic_writing: Path | None = None, latex_extra_path: Path | None = None) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY não encontrado no .env ou no ambiente.")
    if exportar_pdf:
        ensure_command_available("emacs")
        ensure_command_available("lualatex")
        ensure_command_available("biber")
        if emacs_init is not None and not emacs_init.exists():
            raise RuntimeError(f"Arquivo init do Emacs não encontrado: {emacs_init}")
        if academic_writing is not None and not academic_writing.exists():
            raise RuntimeError(f"Arquivo academic-writing.el não encontrado: {academic_writing}")
        if latex_extra_path is not None and not latex_extra_path.exists():
            raise RuntimeError(f"Caminho LaTeX extra não encontrado: {latex_extra_path}")


def _path_completer(only_directories: bool = False):
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None
    return PathCompleter(expanduser=True, only_directories=only_directories)


def _word_completer(words: Iterable[str]):
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None
    return WordCompleter(list(words), ignore_case=True, sentence=True)


def _prompt_raw(label: str, default: str | None = None, completer=None) -> str:
    shown = f" [{default}]" if default not in (None, "") else ""
    if PROMPT_TOOLKIT_AVAILABLE:
        value = pt_prompt(f"{label}{shown}: ", completer=completer)
    else:
        value = input(f"{label}{shown}: ")
    value = value.strip()
    return value if value else (default or "")


def prompt_text(label: str, default: str | None = None, *, required: bool = True, completer=None) -> str:
    while True:
        value = _prompt_raw(label, default, completer=completer)
        if value or not required:
            return value
        print("Valor obrigatório.")


def prompt_yes_no(label: str, default: bool = True) -> bool:
    default_txt = "s" if default else "n"
    while True:
        raw = _prompt_raw(f"{label} (s/n)", default_txt, completer=_word_completer(["s", "n", "sim", "nao", "não"]))
        raw = raw.lower()
        if raw in {"s", "sim", "y", "yes"}:
            return True
        if raw in {"n", "nao", "não", "no"}:
            return False
        print("Responda s ou n.")


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = _prompt_raw(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Digite um inteiro válido.")


def prompt_path(label: str, default: str | None = None, *, must_exist: bool = False, only_directories: bool = False) -> Path:
    while True:
        raw = _prompt_raw(label, default, completer=_path_completer(only_directories=only_directories))
        p = Path(os.path.expanduser(raw)).resolve()
        if must_exist and not p.exists():
            print(f"Caminho inexistente: {p}")
            continue
        if only_directories and p.exists() and not p.is_dir():
            print("Informe um diretório.")
            continue
        return p


def prompt_multi_paths(label: str, *, required: bool = True) -> list[Path]:
    items: list[Path] = []
    print(f"{label}. Informe um caminho por vez. Aceita arquivo, diretório ou .zip. Enter vazio encerra.")
    while True:
        p = prompt_path("Caminho", None, must_exist=False)
        raw = str(p)
        if raw == str(Path(".").resolve()):
            raw = ""
        if not raw:
            if items or not required:
                return items
            print("Pelo menos um caminho é obrigatório.")
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            print("Caminho inválido.")
            continue
        items.append(path.resolve())


def normalize_style(style: str) -> str:
    style = (style or DEFAULT_STYLE).strip().lower()
    if style not in {"apa", "abnt"}:
        raise ValueError("Estilo deve ser 'apa' ou 'abnt'.")
    return style


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


def shorten_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit].strip() + ("…" if len(text) > limit else "")


def read_template_raw(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if raw.lstrip().startswith("# -*- mode: snippet -*-"):
        marker = "\n# --\n"
        pos = raw.find(marker)
        if pos != -1:
            return raw[pos + len(marker):]
    return raw


def normalize_key(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_template_fields(template_raw: str) -> list[TemplateField]:
    fields: list[TemplateField] = []
    seen: set[str] = set()
    for line in template_raw.splitlines():
        m = re.search(r"\$\{\d+:([^}]+)\}", line)
        if not m:
            continue
        default = m.group(1)
        key = ""
        category = "estrategico"
        prompt = default
        stripped = line.strip()
        if stripped.startswith("#+TITLE:"):
            key, category, prompt = "title", "academico", "Título do paper"
        elif stripped.startswith("#+AUTHOR:"):
            key, category, prompt = "author", "academico", "Autor"
        elif "\\programname{" in line:
            key, category, prompt = "program_name", "academico", "Programa"
        elif "\\coursename{" in line:
            key, category, prompt = "course_name", "academico", "Curso"
        elif "\\disciplinename{" in line:
            key, category, prompt = "discipline_name", "academico", "Disciplina"
        elif "\\professorname{" in line:
            key, category, prompt = "professor_name", "academico", "Professor"
        elif "\\cityname{" in line:
            key, category, prompt = "city_name", "academico", "Cidade"
        elif "\\papertype{" in line:
            key, category, prompt = "paper_type", "academico", "Tipo do paper"
        elif "\\covernote{" in line:
            key, category, prompt = "cover_note", "academico", "Nota da capa"
        elif stripped.startswith("#+BIBLIOGRAPHY:"):
            key, category, prompt = "bibliography_file", "academico", "Arquivo .bib"
        elif re.match(r"^[A-Z0-9_]+:\s*\$\{\d+:", stripped):
            left = stripped.split(":", 1)[0]
            key = normalize_key(left)
            category = "estrategico"
            prompt = left.replace("_", " ").title()
        if not key:
            continue
        if key in seen:
            continue
        fields.append(TemplateField(key=key, default=default, line=line, category=category, prompt=prompt))
        seen.add(key)
    return fields


def prompt_template_fields(fields: list[TemplateField], state: dict[str, Any], bib_default: str) -> tuple[dict[str, str], dict[str, str]]:
    academic: dict[str, str] = {}
    strategic: dict[str, str] = {}
    ordered = [f for f in fields if f.category == "academico"] + [f for f in fields if f.category == "estrategico"]
    print("\n=== Metadados acadêmicos do template ===")
    skipped_notice_printed = False
    for field in ordered:
        default = state.get(field.key, field.default)
        if field.key == "bibliography_file":
            default = bib_default
        if field.category == "academico":
            if field.key == "title":
                academic[field.key] = AUTO_HEADER_TITLE
                if not skipped_notice_printed:
                    print("Título, tipo do paper e nota da capa serão gerados pela IA ao final. Autor virá com padrão editável.")
                    skipped_notice_printed = True
                continue
            if field.key == "author":
                academic[field.key] = prompt_text(field.prompt, DEFAULT_AUTHOR, required=False)
                if not academic[field.key].strip():
                    academic[field.key] = DEFAULT_AUTHOR
                continue
            if field.key == "institution_name":
                academic[field.key] = prompt_text(field.prompt, DEFAULT_INSTITUTION, required=False)
                if not academic[field.key].strip():
                    academic[field.key] = DEFAULT_INSTITUTION
                continue
            if field.key == "course_name":
                academic[field.key] = ""
                continue
            if field.key == "paper_type":
                academic[field.key] = AUTO_HEADER_PAPER_TYPE
                continue
            if field.key == "cover_note":
                academic[field.key] = AUTO_HEADER_COVER_NOTE
                continue
            if field.key == "bibliography_file":
                academic[field.key] = bib_default
                continue
            academic[field.key] = prompt_text(field.prompt, str(default) if default is not None else "", required=False)
        else:
            strategic[field.key] = str(default) if default is not None else ""
    return academic, strategic


def prompt_strategy_fields(fields: list[TemplateField], defaults: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    strategic_fields = [f for f in fields if f.category == "estrategico"]
    if strategic_fields:
        print("\n=== Metadados estratégicos do paper ===")
    for field in strategic_fields:
        out[field.key] = prompt_text(field.prompt, defaults.get(field.key, field.default), required=False)
    return out


def materialize_template(template_raw: str, fields: list[TemplateField], answers: dict[str, str]) -> str:
    out = template_raw
    for field in fields:
        pattern = re.escape(field.line)
        # substitui todos os placeholders com o mesmo valor padrão naquela linha
        val = answers.get(field.key, field.default)
        out = out.replace(field.line, re.sub(r"\$\{\d+:([^}]+)\}", val.replace("\\", "\\\\"), field.line))
    return out


def _read_pdf_text(path: Path, max_chars: int = 30000) -> str:
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


def _read_docx_text(path: Path, max_chars: int = 30000) -> str:
    if docx is not None:
        d = docx.Document(str(path))
        return shorten_text("\n".join(p.text for p in d.paragraphs if p.text), max_chars)
    with zipfile.ZipFile(path, "r") as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = re.sub(r"<[^>]+>", " ", xml)
    return shorten_text(re.sub(r"\s+", " ", xml), max_chars)


def read_text_file(path: Path, max_chars: int = 30000) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return shorten_text(path.read_text(encoding="utf-8", errors="ignore"), max_chars)
    if suffix == ".pdf":
        return _read_pdf_text(path, max_chars)
    if suffix == ".docx":
        return _read_docx_text(path, max_chars)
    return shorten_text(path.read_text(encoding="utf-8", errors="ignore"), max_chars)


def sanitize_member_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._/-]+", "_", name).strip("/\\")
    clean = clean.replace("..", "_")
    return clean or "arquivo"


def extract_supported_members_from_zip(zip_path: Path, workspace_dir: Path) -> list[InputItem]:
    items: list[InputItem] = []
    target_root = workspace_dir / f"zip_{slugify(zip_path.stem)}_{zip_path.stat().st_mtime_ns}"
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_name = sanitize_member_name(member.filename)
            suffix = Path(member_name).suffix.lower()
            if suffix not in SUPPORTED_DOC_SUFFIXES:
                continue
            dest = target_root / member_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src_fh, open(dest, "wb") as dst_fh:
                shutil.copyfileobj(src_fh, dst_fh)
            items.append(InputItem(path=dest, label=f"{zip_path.name}:{member.filename}", metadata={"container": str(zip_path), "member": member.filename}))
    return items


def collect_input_items(paths: list[Path], workspace_dir: Path) -> list[InputItem]:
    items: list[InputItem] = []
    seen_labels: set[str] = set()

    def add_item(item: InputItem) -> None:
        label = item.label
        if label in seen_labels:
            idx = 2
            base = label
            while f"{base} [{idx}]" in seen_labels:
                idx += 1
            item.label = f"{base} [{idx}]"
        seen_labels.add(item.label)
        items.append(item)

    def walk(p: Path) -> None:
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    walk(child)
            return
        suffix = p.suffix.lower()
        if suffix in SUPPORTED_CONTAINER_SUFFIXES:
            for item in extract_supported_members_from_zip(p, workspace_dir):
                add_item(item)
            return
        if suffix in SUPPORTED_DOC_SUFFIXES:
            add_item(InputItem(path=p, label=p.name, metadata={"source": str(p)}))

    for path in paths:
        walk(path)
    return items


def build_source_docs(items: list[InputItem], kind: str, max_chars: int = 30000) -> list[SourceDoc]:
    docs = []
    for item in items:
        docs.append(SourceDoc(path=str(item.path), kind=kind, label=item.label, extracted_text=read_text_file(item.path, max_chars=max_chars), metadata=dict(item.metadata)))
    return docs


def compact_doc_payload(docs: list[SourceDoc], limit: int = 6000) -> list[dict[str, Any]]:
    return [
        {
            "label": d.label,
            "kind": d.kind,
            "path": d.path,
            "summary": d.summary,
            "text_excerpt": shorten_text(d.extracted_text, limit),
            "bib_key": d.bib_key,
        }
        for d in docs
    ]


def append_prompt_log(store: list[tuple[str, str]], section: str, content: str) -> None:
    store.append((section, content.strip()))


def prompts_dir() -> Path:
    cwd_candidate = Path.cwd() / PROMPTS_DIR_NAME
    if cwd_candidate.exists():
        return cwd_candidate
    return script_dir() / PROMPTS_DIR_NAME


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def load_prompt_template(filename: str) -> str:
    base = prompts_dir()
    path = base / filename
    if not path.exists():
        raise RuntimeError(f"Prompt não encontrado em {path}. Crie o arquivo em ./{PROMPTS_DIR_NAME}/{filename}.")
    return path.read_text(encoding="utf-8")


def render_prompt_file(filename: str, **kwargs: Any) -> str:
    template = load_prompt_template(filename)
    data = {k: (json.dumps(v, ensure_ascii=False, indent=2) if isinstance(v, (dict, list)) else str(v)) for k, v in kwargs.items()}
    return template.format_map(SafeDict(data)).strip()


def infer_base_context_with_ai(client: OpenAI, model: str, template_text: str, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], prompt_log: list[tuple[str, str]]) -> BaseContextInferenceOutput:
    prompt = render_prompt_file(
        PROMPT_INFERIR_CONTEXTO,
        template_text=shorten_text(template_text, 18000),
        base_docs=compact_doc_payload(base_docs, 5000),
        guidance_docs=compact_doc_payload(guidance_docs, 5000),
    )
    append_prompt_log(prompt_log, "infer_base_context_with_ai", prompt)
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=BaseContextInferenceOutput,
    )
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError("A IA não retornou contexto-base estruturado.")
    return parsed


def choose_theme_combination(base_context: BaseContextInferenceOutput) -> tuple[str, str, str | None]:
    combos = base_context.combinacoes_sugeridas or []
    if combos:
        print("\n=== Combinações sugeridas de temas adicionais ===")
        for idx, combo in enumerate(combos, start=1):
            just = f" — {combo.justificativa}" if combo.justificativa else ""
            print(f"{idx}) {combo.tema_adicional_1} + {combo.tema_adicional_2}{just}")
    else:
        print("\nA IA não sugeriu combinações fechadas de temas adicionais.")
        if base_context.temas_adicionais_candidatos:
            print("Temas adicionais candidatos:", ", ".join(base_context.temas_adicionais_candidatos))

    if combos and not prompt_yes_no("Deseja escolher manualmente a combinação de temas adicionais?", default=False):
        chosen = combos[0]
        print(f"Usando combinação sugerida pela IA: {chosen.tema_adicional_1} + {chosen.tema_adicional_2}")
        return chosen.tema_adicional_1, chosen.tema_adicional_2, chosen.justificativa

    if combos:
        idx = prompt_int("Número da combinação desejada (0 para informar manualmente)", 1)
        if 1 <= idx <= len(combos):
            chosen = combos[idx - 1]
            return chosen.tema_adicional_1, chosen.tema_adicional_2, chosen.justificativa

    t1 = prompt_text("Tema adicional 1", (base_context.temas_adicionais_candidatos[0] if base_context.temas_adicionais_candidatos else ""), required=False)
    t2_default = base_context.temas_adicionais_candidatos[1] if len(base_context.temas_adicionais_candidatos) > 1 else ""
    t2 = prompt_text("Tema adicional 2", t2_default, required=False)
    return t1, t2, None


def narrow_context_with_ai(client: OpenAI, model: str, template_text: str, base_context: BaseContextInferenceOutput, tema_adicional_1: str, tema_adicional_2: str, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], prompt_log: list[tuple[str, str]]) -> FinalContextInferenceOutput:
    prompt = render_prompt_file(
        PROMPT_AFUNILAR_PAPER,
        template_text=shorten_text(template_text, 18000),
        contexto_base=base_context.model_dump(),
        tema_adicional_1=tema_adicional_1,
        tema_adicional_2=tema_adicional_2,
        base_docs=compact_doc_payload(base_docs, 5000),
        guidance_docs=compact_doc_payload(guidance_docs, 5000),
    )
    append_prompt_log(prompt_log, "narrow_context_with_ai", prompt)
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=FinalContextInferenceOutput,
    )
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError("A IA não retornou contexto final estruturado.")
    return parsed


def build_strategy_answers_from_inference(template_fields: list[TemplateField], defaults: dict[str, str], base_context: BaseContextInferenceOutput, final_context: FinalContextInferenceOutput, tema_adicional_1: str, tema_adicional_2: str) -> dict[str, str]:
    out = dict(defaults)
    mapping = {
        "tema_principal": final_context.tema_final,
        "tema_adicional_1": tema_adicional_1,
        "tema_adicional_2": tema_adicional_2,
        "autor_obrigatorio_1": final_context.autores_centrais[0] if len(final_context.autores_centrais) > 0 else "",
        "autor_obrigatorio_2": final_context.autores_centrais[1] if len(final_context.autores_centrais) > 1 else "",
        "autor_obrigatorio_3": final_context.autores_centrais[2] if len(final_context.autores_centrais) > 2 else "",
        "autor_complementar_1": final_context.autores_centrais[3] if len(final_context.autores_centrais) > 3 else "",
        "autor_complementar_2": final_context.autores_centrais[4] if len(final_context.autores_centrais) > 4 else "",
        "recorte_empirico": final_context.recorte_final,
        "pergunta_de_pesquisa": final_context.pergunta_pesquisa,
        "pergunta_pesquisa": final_context.pergunta_pesquisa,
        "hipotese": final_context.hipotese or "",
        "objetivo_geral": final_context.objetivo_final,
        "metodologia": "Análise bibliográfica e argumentativa orientada pelo roteiro da disciplina e pelos textos-base.",
        "conclusao_esperada": base_context.observacoes_sobre_limites_do_recorte or "",
    }
    valid_keys = {f.key for f in template_fields}
    for k, v in mapping.items():
        if k in valid_keys:
            out[k] = v
    return out


def infer_context_with_ai(client: OpenAI, model: str, template_text: str, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], prompt_log: list[tuple[str, str]]) -> tuple[BaseContextInferenceOutput, FinalContextInferenceOutput, str, str, PaperContext, dict[str, str]]:
    base_context = infer_base_context_with_ai(client, model, template_text, base_docs, guidance_docs, prompt_log)
    print("\n=== Contexto-base inferido pela IA ===")
    print(json.dumps(base_context.model_dump(), ensure_ascii=False, indent=2))
    tema_adicional_1, tema_adicional_2, _ = choose_theme_combination(base_context)
    final_context = narrow_context_with_ai(
        client, model, template_text, base_context, tema_adicional_1, tema_adicional_2, base_docs, guidance_docs, prompt_log
    )
    print("\n=== Contexto final afunilado pela IA ===")
    print(json.dumps(final_context.model_dump(), ensure_ascii=False, indent=2))
    context = PaperContext(
        tema=final_context.tema_final,
        recorte=final_context.recorte_final,
        objetivo=final_context.objetivo_final,
        pergunta_pesquisa=final_context.pergunta_pesquisa,
        hipotese=final_context.hipotese,
        palavras_chave=final_context.palavras_chave,
        titulo_sugerido=final_context.titulo_provisorio,
    )
    return base_context, final_context, tema_adicional_1, tema_adicional_2, context, {
        "tema_principal": final_context.tema_final,
        "tema_adicional_1": tema_adicional_1,
        "tema_adicional_2": tema_adicional_2,
        "objetivo_geral": final_context.objetivo_final,
        "recorte_empirico": final_context.recorte_final,
        "pergunta_de_pesquisa": final_context.pergunta_pesquisa,
        "hipotese": final_context.hipotese or "",
    }


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


def render_biblatex_entry(key: str, meta: BibMetadataOutput) -> str:
    entry_type = meta.entry_type or "article"
    fields: list[tuple[str, str | None]] = []
    if meta.authors:
        fields.append(("author", " and ".join(meta.authors)))
    fields.append(("title", meta.title))
    fields.append(("year", meta.year))
    fields.append(("journaltitle", meta.journaltitle))
    fields.append(("booktitle", meta.booktitle))
    fields.append(("publisher", meta.publisher))
    fields.append(("school", meta.school))
    fields.append(("doi", meta.doi))
    fields.append(("url", meta.url))
    fields.append(("note", meta.note))
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields if v)
    return f"@{entry_type}{{{key},\n  {body}\n}}"


def ai_extract_bib_metadata(client: OpenAI, model: str, doc: SourceDoc, prompt_log: list[tuple[str, str]]) -> BibMetadataOutput:
    prompt = textwrap.dedent(
        f"""
        Extraia metadados bibliográficos em formato estruturado a partir do documento abaixo.
        Se algum campo estiver ausente, deixe-o nulo.
        Defina entry_type entre article, book, incollection, thesis, report, misc.

        Arquivo: {doc.label}
        Texto extraído:
        {shorten_text(doc.extracted_text, 16000)}
        """
    ).strip()
    append_prompt_log(prompt_log, f"ai_extract_bib_metadata::{doc.label}", prompt)
    resp = client.responses.parse(model=model, input=[{"role": "user", "content": prompt}], text_format=BibMetadataOutput)
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError("A IA não retornou metadados estruturados.")
    return parsed


def build_base_doc_bibliography(client: OpenAI, model: str, docs: list[SourceDoc], prompt_log: list[tuple[str, str]]) -> list[SourceDoc]:
    used: set[str] = set()
    out = []
    for doc in docs:
        try:
            meta = ai_extract_bib_metadata(client, model, doc, prompt_log)
        except Exception:
            meta = BibMetadataOutput(entry_type="misc", title=Path(doc.path).stem.replace("_", " "), note="Metadados incompletos; revisar manualmente.")
        key = unique_key(make_bib_key(meta.authors, meta.year, meta.title), used)
        doc.bib_key = key
        doc.bib_entry = render_biblatex_entry(key, meta)
        doc.metadata = meta.model_dump()
        out.append(doc)
    return out


def summarize_document(client: OpenAI, model: str, doc: SourceDoc, prompt_log: list[tuple[str, str]], limit: int = 1000) -> str:
    prompt = textwrap.dedent(
        f"""
        Resuma em português, em um único parágrafo denso, o documento abaixo para subsidiar a redação de um paper.
        Priorize argumento central, objeto, método, achados e utilidade analítica.

        Documento: {doc.label}
        Texto extraído:
        {shorten_text(doc.extracted_text, 15000)}
        """
    ).strip()
    append_prompt_log(prompt_log, f"summarize_document::{doc.label}", prompt)
    resp = client.responses.create(model=model, input=prompt)
    return shorten_text(resp.output_text.strip(), limit)


def prompt_sources(default: list[str] | None = None) -> list[str]:
    base_default = ",".join(default or ["semantic_scholar"])
    print("Bases disponíveis: 1) semantic_scholar  2) scopus  3) web_of_science")
    raw = prompt_text("Bases separadas por vírgula (nomes ou números)", base_default, completer=_word_completer(["1", "2", "3", "semantic_scholar", "scopus", "web_of_science", "wos"]))
    alias = {
        "1": "semantic_scholar", "2": "scopus", "3": "web_of_science",
        "semantic": "semantic_scholar", "semantic_scholar": "semantic_scholar", "semantic scholar": "semantic_scholar",
        "scopus": "scopus", "wos": "web_of_science", "web_of_science": "web_of_science", "web of science": "web_of_science",
    }
    out = []
    for tok in raw.split(","):
        t = alias.get(tok.strip().lower(), tok.strip().lower())
        if t in {"semantic_scholar", "scopus", "web_of_science"} and t not in out:
            out.append(t)
    out = out or ["semantic_scholar"]
    print("Bases selecionadas:", ", ".join(out))
    return out


def prompt_list(label: str, default: str = "") -> list[str]:
    raw = prompt_text(label, default, required=False)
    return [x.strip() for x in raw.split(",") if x.strip()]


def languages_include_both_pt_en(idiomas: Iterable[str]) -> bool:
    toks = {i.strip().lower() for i in idiomas}
    return any("port" in t for t in toks) and any("ing" in t or t == "en" for t in toks)


def build_query_for_source(source: str, tema: str, palavras: list[str], tipo_estudo: str, *, recorte: str = "") -> str:
    terms = [f'"{x}"' if " " in x else x for x in (palavras or [tema])]
    if recorte:
        terms.append(f'"{recorte}"' if " " in recorte else recorte)
    if tipo_estudo:
        terms.append(f'"{tipo_estudo}"' if " " in tipo_estudo else tipo_estudo)
    nucleus = " OR ".join(dict.fromkeys(terms))
    if source == "scopus":
        return f"TITLE-ABS-KEY({nucleus})"
    if source == "web_of_science":
        return f"TS=({nucleus})"
    return nucleus


def suggest_queries_with_ai(client: OpenAI, model: str, tema: str, recorte: str, objetivo: str, bases: list[str], tipo_estudo: str, idiomas: list[str], prompt_log: list[tuple[str, str]]) -> QuerySuggestionOutput:
    prompt = textwrap.dedent(
        f"""
        Sugira palavras-chave e strings de busca acadêmicas para localizar artigos correlatos.
        Tema: {tema}
        Recorte: {recorte}
        Objetivo: {objetivo}
        Tipo de estudo: {tipo_estudo}
        Bases: {', '.join(bases)}
        Idiomas: {', '.join(idiomas)}

        Retorne JSON com: palavras_chave, query_geral, query_semantic, query_scopus, query_wos.
        """
    ).strip()
    append_prompt_log(prompt_log, "suggest_queries_with_ai", prompt)
    resp = client.responses.parse(model=model, input=[{"role": "user", "content": prompt}], text_format=QuerySuggestionOutput)
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError("A IA não retornou queries estruturadas.")
    return parsed


def configure_keywords_and_queries(client: OpenAI, model: str, context: PaperContext, bases: list[str], tipo_estudo: str, idiomas: list[str], prompt_log: list[tuple[str, str]]) -> tuple[list[str], dict[str, str | None]]:
    suggestion: QuerySuggestionOutput | None = None
    if prompt_yes_no("Deseja que a IA sugira palavras-chave e queries?", default=True):
        try:
            suggestion = suggest_queries_with_ai(client, model, context.tema, context.recorte, context.objetivo, bases, tipo_estudo, idiomas, prompt_log)
            print("\nSugestão de palavras-chave:", ", ".join(suggestion.palavras_chave))
            if "semantic_scholar" in bases and suggestion.query_semantic:
                print("Semantic Scholar:", suggestion.query_semantic)
            if "scopus" in bases and suggestion.query_scopus:
                print("Scopus:", suggestion.query_scopus)
            if "web_of_science" in bases and suggestion.query_wos:
                print("Web of Science:", suggestion.query_wos)
        except Exception as exc:
            print(f"Aviso: não foi possível sugerir queries com IA: {exc}")
    kw_default = ",".join(suggestion.palavras_chave if suggestion else (context.palavras_chave or [context.tema]))
    palavras = prompt_list("Palavras-chave (separadas por vírgula)", kw_default)
    default_queries: dict[str, str | None] = {
        "geral": suggestion.query_geral if suggestion else None,
        "semantic_scholar": suggestion.query_semantic if suggestion and suggestion.query_semantic else build_query_for_source("semantic_scholar", context.tema, palavras, tipo_estudo, recorte=context.recorte),
        "scopus": suggestion.query_scopus if suggestion and suggestion.query_scopus else build_query_for_source("scopus", context.tema, palavras, tipo_estudo, recorte=context.recorte),
        "web_of_science": suggestion.query_wos if suggestion and suggestion.query_wos else build_query_for_source("web_of_science", context.tema, palavras, tipo_estudo, recorte=context.recorte),
    }
    queries: dict[str, str | None] = {"geral": None, "semantic_scholar": None, "scopus": None, "web_of_science": None}
    editar = prompt_yes_no("Deseja editar manualmente as queries sugeridas/geradas?", default=False if suggestion else True)
    if editar:
        default_general = default_queries["geral"] or default_queries.get(bases[0]) or ""
        queries["geral"] = prompt_text("Query geral opcional", default_general, required=False) or None
        if "semantic_scholar" in bases:
            queries["semantic_scholar"] = prompt_text("Query Semantic Scholar", default_queries["semantic_scholar"] or "", required=False) or None
        if "scopus" in bases:
            queries["scopus"] = prompt_text("Query Scopus", default_queries["scopus"] or "", required=False) or None
        if "web_of_science" in bases:
            queries["web_of_science"] = prompt_text("Query Web of Science", default_queries["web_of_science"] or "", required=False) or None
    else:
        queries["geral"] = default_queries["geral"] or default_queries.get(bases[0])
        for base in bases:
            queries[base] = default_queries[base]
    return palavras, queries


def semantic_headers() -> dict[str, str]:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": key} if key else {}


def scopus_headers() -> dict[str, str] | None:
    key = os.getenv("SCOPUS_API_KEY")
    if not key:
        return None
    headers = {"X-ELS-APIKey": key, "Accept": "application/json"}
    inst = os.getenv("SCOPUS_INSTTOKEN")
    if inst:
        headers["X-ELS-Insttoken"] = inst
    return headers


def wos_headers() -> dict[str, str] | None:
    key = os.getenv("WOS_API_KEY")
    return {"X-ApiKey": key} if key else None


def fetch_json(url: str, *, headers=None, params=None, timeout=30) -> dict[str, Any]:
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_semantic_candidates(query: str, limit: int) -> tuple[list[CandidatePaper], SourceFetchResult]:
    data = fetch_json(S2_URL, headers=semantic_headers(), params={
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,venue,authors,tldr,url,openAccessPdf,externalIds",
    })
    candidates = []
    for item in data.get("data", []):
        external = item.get("externalIds") or {}
        pdf_url = ((item.get("openAccessPdf") or {}).get("url") if isinstance(item.get("openAccessPdf"), dict) else None)
        candidates.append(CandidatePaper(
            paper_id=item.get("paperId") or slugify(item.get("title", "")),
            title=item.get("title") or "Sem título",
            abstract=item.get("abstract") or "",
            year=item.get("year"),
            venue=item.get("venue"),
            authors=[a.get("name") for a in item.get("authors", []) if a.get("name")],
            url=item.get("url"),
            pdf_url=pdf_url,
            doi=external.get("DOI"),
            source="semantic_scholar",
            full_text_verified=bool(pdf_url),
            tldr=((item.get("tldr") or {}).get("text") if isinstance(item.get("tldr"), dict) else None),
        ))
    return candidates, SourceFetchResult(source="semantic_scholar", query=query, retrieved=len(candidates))


def fetch_scopus_candidates(query: str, limit: int) -> tuple[list[CandidatePaper], SourceFetchResult]:
    headers = scopus_headers()
    if not headers:
        return [], SourceFetchResult(source="scopus", query=query, retrieved=0, warnings=["SCOPUS_API_KEY ausente; Scopus ignorado."])
    data = fetch_json(SCOPUS_URL, headers=headers, params={"query": query, "count": limit})
    entries = (((data or {}).get("search-results") or {}).get("entry") or [])
    candidates = []
    for item in entries:
        links = item.get("link") or []
        url = None
        pdf_url = None
        for lk in links:
            href = lk.get("@href")
            if href and not url:
                url = href
            if href and href.lower().endswith(".pdf"):
                pdf_url = href
        creators = str(item.get("dc:creator") or "")
        authors = [x.strip() for x in creators.split(";") if x.strip()] if creators else []
        year = str(item.get("prism:coverDate") or "")[:4]
        candidates.append(CandidatePaper(
            paper_id=item.get("dc:identifier") or slugify(item.get("dc:title", "")),
            title=item.get("dc:title") or "Sem título",
            abstract=item.get("dc:description") or "",
            year=int(year) if year.isdigit() else None,
            venue=item.get("prism:publicationName"),
            authors=authors,
            url=url,
            pdf_url=pdf_url,
            doi=item.get("prism:doi"),
            source="scopus",
            full_text_verified=bool(pdf_url),
        ))
    return candidates, SourceFetchResult(source="scopus", query=query, retrieved=len(candidates))


def fetch_wos_candidates(query: str, limit: int) -> tuple[list[CandidatePaper], SourceFetchResult]:
    headers = wos_headers()
    if not headers:
        return [], SourceFetchResult(source="web_of_science", query=query, retrieved=0, warnings=["WOS_API_KEY ausente; Web of Science ignorado."])
    data = fetch_json(WOS_URL, headers=headers, params={"q": query, "limit": limit})
    hits = (data or {}).get("hits") or []
    candidates = []
    for item in hits:
        links = item.get("links") or []
        url = None
        pdf_url = None
        for lk in links:
            href = lk.get("url") or lk.get("href")
            if href and not url:
                url = href
            if href and href.lower().endswith(".pdf"):
                pdf_url = href
        names = item.get("names") or {}
        authors = []
        if isinstance(names, dict):
            for a in names.get("authors", []):
                nm = a.get("displayName") or a.get("fullName")
                if nm:
                    authors.append(nm)
        source_meta = item.get("source") or {}
        year = source_meta.get("publishYear") if isinstance(source_meta, dict) else None
        identifiers = item.get("identifiers") or {}
        candidates.append(CandidatePaper(
            paper_id=item.get("uid") or slugify(item.get("title", "")),
            title=item.get("title") or "Sem título",
            abstract=item.get("abstract") or "",
            year=int(str(year)) if str(year).isdigit() else None,
            venue=source_meta.get("sourceTitle") if isinstance(source_meta, dict) else None,
            authors=authors,
            url=url,
            pdf_url=pdf_url,
            doi=identifiers.get("doi") if isinstance(identifiers, dict) else None,
            source="web_of_science",
            full_text_verified=bool(pdf_url),
        ))
    return candidates, SourceFetchResult(source="web_of_science", query=query, retrieved=len(candidates))


def dedupe_candidates(candidates: list[CandidatePaper]) -> tuple[list[CandidatePaper], int]:
    seen: dict[str, CandidatePaper] = {}
    out: list[CandidatePaper] = []
    duplicates = 0
    for cand in candidates:
        key = (cand.doi or "").lower().strip() or slugify(cand.title)
        if key in seen:
            duplicates += 1
            prev = seen[key]
            if not prev.pdf_url and cand.pdf_url:
                prev.pdf_url = cand.pdf_url
                prev.full_text_verified = cand.full_text_verified
            continue
        seen[key] = cand
        out.append(cand)
    return out, duplicates


def download_pdf(url: str, dest: Path) -> Path:
    resp = requests.get(url, stream=True, timeout=45, allow_redirects=True)
    resp.raise_for_status()
    ctype = (resp.headers.get("content-type") or "").lower()
    if "pdf" not in ctype and not str(resp.url).lower().endswith(".pdf"):
        raise RuntimeError("URL não retornou PDF.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                fh.write(chunk)
    return dest


def ensure_candidate_readable(cand: CandidatePaper, cache_dir: Path) -> CandidatePaper:
    if cand.downloaded_pdf_path and Path(cand.downloaded_pdf_path).exists():
        cand.full_text_verified = True
        return cand
    for url in [cand.pdf_url, cand.url]:
        if not url:
            continue
        try:
            fname = slugify(cand.paper_id or cand.title)[:80] + ".pdf"
            dest = download_pdf(url, cache_dir / fname)
            cand.downloaded_pdf_path = str(dest)
            cand.full_text_verified = True
            return cand
        except Exception:
            continue
    cand.full_text_verified = False
    return cand


def collect_candidates(bases: list[str], queries: dict[str, str | None], limit: int, output_dir: Path, basename: str) -> tuple[list[CandidatePaper], SearchAudit]:
    all_candidates: list[CandidatePaper] = []
    per_source: list[SourceFetchResult] = []
    warnings: list[str] = []
    for source in bases:
        query = queries.get(source) or queries.get("geral") or ""
        try:
            if source == "semantic_scholar":
                cands, audit = fetch_semantic_candidates(query, limit)
            elif source == "scopus":
                cands, audit = fetch_scopus_candidates(query, limit)
            elif source == "web_of_science":
                cands, audit = fetch_wos_candidates(query, limit)
            else:
                continue
        except Exception as exc:
            warnings.append(f"Falha ao consultar {source}: {exc}")
            continue
        per_source.append(audit)
        warnings.extend(audit.warnings)
        all_candidates.extend(cands)
    identified_total = len(all_candidates)
    deduped, duplicates_removed = dedupe_candidates(all_candidates)
    verified: list[CandidatePaper] = []
    removed_other = 0
    cache_dir = output_dir / f"{basename}_fulltext_cache"
    for cand in deduped:
        cand = ensure_candidate_readable(cand, cache_dir)
        if cand.full_text_verified:
            verified.append(cand)
        else:
            removed_other += 1
    return verified, SearchAudit(per_source=per_source, identified_total=identified_total, duplicates_removed=duplicates_removed, other_removed=removed_other, screened_total=len(verified), warnings=warnings)


def rank_correlated_candidates(client: OpenAI, model: str, context: PaperContext, base_docs: list[SourceDoc], candidates: list[CandidatePaper], max_items: int, prompt_log: list[tuple[str, str]]) -> RankedReferenceOutput:
    payload = [
        {
            "paper_id": c.paper_id,
            "title": c.title,
            "year": c.year,
            "authors": c.authors,
            "venue": c.venue,
            "doi": c.doi,
            "url": c.url,
            "pdf_url": c.pdf_url,
            "summary": shorten_text(c.abstract or c.tldr or c.title, 1200),
        }
        for c in candidates
    ]
    prompt = textwrap.dedent(
        f"""
        Selecione até {max_items} referências correlatas realmente úteis para apoiar um paper acadêmico.

        Contexto do paper:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Textos-base já fornecidos:
        {json.dumps(compact_doc_payload(base_docs, 2500), ensure_ascii=False, indent=2)}

        Candidatos recuperados:
        {json.dumps(payload, ensure_ascii=False, indent=2)}

        Regras:
        - selecione até {max_items};
        - priorize aderência temática e complementaridade em relação aos textos-base;
        - não selecione estudos redundantes se houver alternativas melhores;
        - devolva somente paper_id e justificativa.
        """
    ).strip()
    append_prompt_log(prompt_log, "rank_correlated_candidates", prompt)
    resp = client.responses.parse(model=model, input=[{"role": "user", "content": prompt}], text_format=RankedReferenceOutput)
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError("A IA não retornou ranking estruturado.")
    return parsed


def build_candidate_bib_entry(candidate: CandidatePaper, used_keys: set[str]) -> tuple[str, str]:
    key = unique_key(make_bib_key(candidate.authors, str(candidate.year or ""), candidate.title), used_keys)
    meta = BibMetadataOutput(
        entry_type="article",
        title=candidate.title,
        authors=candidate.authors,
        year=str(candidate.year) if candidate.year else None,
        journaltitle=candidate.venue,
        doi=candidate.doi,
        url=candidate.pdf_url or candidate.url,
    )
    return key, render_biblatex_entry(key, meta)


def run_related_search_flow(client: OpenAI, model: str, context: PaperContext, output_dir: Path, basename: str, prompt_log: list[tuple[str, str]]) -> tuple[list[SourceDoc], dict[str, Any]]:
    if not prompt_yes_no("Deseja buscar artigos correlatos adicionais para ampliar as referências?", default=True):
        return [], {"used": False}
    max_refs = prompt_int("Quantas referências correlatas extras, no máximo, podem ser incluídas?", 5)
    quantity = max(prompt_int("Quantos candidatos correlatos recuperar para ranqueamento?", 15), max_refs)
    usar_fluxo = prompt_yes_no("Deseja seguir o mesmo fluxo do gerar_atividade para bases, palavras-chave e queries?", default=True)
    if not usar_fluxo:
        return [], {"used": False, "note": "fluxo de busca desativado"}

    bases = prompt_sources(["semantic_scholar"])
    tipo_estudo = prompt_text("Tipo de estudo correlato a priorizar", context.tipo_estudo_correlato, required=False)
    idiomas = prompt_list("Idiomas aceitáveis (separados por vírgula)", ",".join(context.idiomas)) or context.idiomas
    query_bilingue = prompt_yes_no("Deseja montar/sugerir queries bilíngues (português + inglês)?", default=languages_include_both_pt_en(idiomas))
    palavras, queries = configure_keywords_and_queries(client, model, context, bases, tipo_estudo, idiomas, prompt_log)
    base_fallbacks = {
        "semantic_scholar": build_query_for_source("semantic_scholar", context.tema, palavras, tipo_estudo, recorte=context.recorte),
        "scopus": build_query_for_source("scopus", context.tema, palavras, tipo_estudo, recorte=context.recorte),
        "web_of_science": build_query_for_source("web_of_science", context.tema, palavras, tipo_estudo, recorte=context.recorte),
    }
    queries["geral"] = queries.get("geral") or queries.get(bases[0]) or base_fallbacks[bases[0]]
    for base in bases:
        queries[base] = queries.get(base) or base_fallbacks[base]
    candidates, audit = collect_candidates(bases, queries, quantity, output_dir, basename)
    if not candidates:
        return [], {"used": True, "note": "nenhum candidato correlato recuperado", "audit": asdict(audit)}
    ranked = rank_correlated_candidates(client, model, context, [], candidates, max_refs, prompt_log)
    chosen_ids = {x.paper_id for x in ranked.selecionados[:max_refs]}
    chosen = [c for c in candidates if c.paper_id in chosen_ids]
    docs: list[SourceDoc] = []
    used_keys: set[str] = set()
    for cand in chosen:
        extracted = shorten_text(cand.abstract or cand.tldr or cand.title, 3000)
        if cand.downloaded_pdf_path and Path(cand.downloaded_pdf_path).exists():
            try:
                extracted = read_text_file(Path(cand.downloaded_pdf_path), 20000)
            except Exception:
                pass
        key, bib_entry = build_candidate_bib_entry(cand, used_keys)
        docs.append(SourceDoc(
            path=str(cand.downloaded_pdf_path or cand.url or cand.pdf_url or ""),
            kind="correlata",
            label=cand.title,
            extracted_text=extracted,
            summary=shorten_text(extracted, 1200),
            bib_key=key,
            bib_entry=bib_entry,
            metadata={"paper_id": cand.paper_id, "title": cand.title, "year": cand.year, "authors": cand.authors, "venue": cand.venue, "doi": cand.doi, "url": cand.pdf_url or cand.url},
        ))
    return docs, {"used": True, "observacao": ranked.observacao, "selecionados": [x.model_dump() for x in ranked.selecionados], "audit": asdict(audit), "queries": queries, "bases": bases, "palavras_chave": palavras, "query_bilingue": query_bilingue}


def apply_citation_style(org_text: str, bib_filename: str, style: str) -> str:
    style = normalize_style(style)
    if re.search(r"(?im)^#\+CITE_EXPORT:", org_text):
        org_text = re.sub(r"(?im)^#\+CITE_EXPORT:.*$", f"#+CITE_EXPORT: biblatex backend=biber,style={style},sortcites=true,sorting=nyt,giveninits=true,maxcitenames=2,maxbibnames=20,uniquelist=minyear", org_text)
    else:
        lines = org_text.splitlines()
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("#+LATEX_CLASS_OPTIONS"):
                insert_at = i + 1
                break
        lines.insert(insert_at, f"#+CITE_EXPORT: biblatex backend=biber,style={style},sortcites=true,sorting=nyt,giveninits=true,maxcitenames=2,maxbibnames=20,uniquelist=minyear")
        org_text = "\n".join(lines)
    if re.search(r"(?im)^#\+BIBLIOGRAPHY:\s+.*$", org_text):
        org_text = re.sub(r"(?im)^#\+BIBLIOGRAPHY:\s+.*$", f"#+BIBLIOGRAPHY: {bib_filename}", org_text)
    else:
        org_text += f"\n#+BIBLIOGRAPHY: {bib_filename}\n"
    return org_text


def cleanup_generated_org(org_text: str) -> str:
    org_text = org_text.replace("* Referências\n#+PRINT_BIBLIOGRAPHY:", "#+PRINT_BIBLIOGRAPHY:")
    org_text = re.sub(r"\n{3,}", "\n\n", org_text)
    return org_text.strip() + "\n"



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


def infer_final_front_matter(client: OpenAI, model: str, context: PaperContext, org_text: str, prompt_log: list[tuple[str, str]]) -> FinalFrontMatterOutput:
    prompt = render_prompt_file(
        PROMPT_GERAR_TITULO_CAPA,
        contexto=json.dumps(asdict(context), ensure_ascii=False, indent=2),
        org_text=shorten_text(org_text, 25000),
    )
    append_prompt_log(prompt_log, "infer_final_front_matter", prompt)
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=FinalFrontMatterOutput,
    )
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError("A IA não retornou os elementos finais de capa.")
    return parsed




def normalize_cover_parts(paper_type: str, cover_note: str) -> tuple[str, str]:
    pt = (paper_type or "").strip()
    cn = (cover_note or "").strip()
    pt = re.sub(r"[\s\.;:!?]+$", "", pt)
    cn = re.sub(r"^[\s\.;:!?]+", "", cn)
    return pt, cn

def apply_final_front_matter(org_text: str, *, title: str, author: str, paper_type: str, cover_note: str, course_name: str = "", institution_name: str = DEFAULT_INSTITUTION) -> str:
    paper_type, cover_note = normalize_cover_parts(paper_type, cover_note)
    org_text = replace_org_header_line(org_text, "#+TITLE:", title)
    org_text = replace_org_header_line(org_text, "#+AUTHOR:", author)
    org_text = replace_latex_header_macro(org_text, "institution", institution_name)
    org_text = replace_latex_header_macro(org_text, "coursename", course_name)
    org_text = replace_latex_header_macro(org_text, "papertype", paper_type)
    org_text = replace_latex_header_macro(org_text, "covernote", cover_note)
    return org_text


def ensure_cover_command(org_text: str) -> str:
    if "\\usepapercover" not in org_text or "#+LATEX: \\makemytitle" in org_text:
        return org_text
    marker = "#+begin_abstract"
    if marker in org_text:
        return org_text.replace(marker, "#+LATEX: \\makemytitle\n\n" + marker, 1)
    return org_text + "\n#+LATEX: \\makemytitle\n"


def generate_paper_org(client: OpenAI, model: str, template_text: str, context: PaperContext, base_docs: list[SourceDoc], guidance_docs: list[SourceDoc], correlated_docs: list[SourceDoc], bib_filename: str, style: str, prompt_log: list[tuple[str, str]]) -> str:
    sources_payload = [{"kind": d.kind, "label": d.label, "bib_key": d.bib_key, "summary": d.summary, "path": d.path} for d in [*base_docs, *correlated_docs]]
    prompt = textwrap.dedent(
        f"""
        Gere um paper acadêmico completo em Org-mode a partir do template fornecido.

        Regras obrigatórias:
        1. Preserve o cabeçalho técnico do template (#+...).
        2. O arquivo final deve apontar para o arquivo bibliográfico: {bib_filename}.
        3. O estilo bibliográfico final deve ser {style.upper()} via BibLaTeX/Org Cite.
        4. Use citações nativas do Org com chaves EXATAS no formato [cite/t:@chave] ou [cite:@chave].
        5. Cite somente as chaves fornecidas. Não invente chaves.
        6. Respeite os comentários-guia invisíveis do template.
        7. Produza texto corrido, acadêmico e argumentativo.
        8. Não escreva numeração manual nos títulos.
        9. Não crie seção manual de referências se houver #+PRINT_BIBLIOGRAPHY:.
        10. As referências correlatas são complementares e não substituem os textos-base.

        Contexto do paper:
        {json.dumps(asdict(context), ensure_ascii=False, indent=2)}

        Template Org:
        {shorten_text(template_text, 30000)}

        Fontes citáveis disponíveis:
        {json.dumps(sources_payload, ensure_ascii=False, indent=2)}

        Orientações do professor:
        {json.dumps(compact_doc_payload(guidance_docs, 4000), ensure_ascii=False, indent=2)}

        Retorne apenas o conteúdo completo do arquivo .org final.
        """
    ).strip()
    append_prompt_log(prompt_log, "generate_paper_org", prompt)
    resp = client.responses.create(model=model, input=prompt)
    org_text = resp.output_text.strip()
    org_text = apply_citation_style(org_text, bib_filename, style)
    return cleanup_generated_org(org_text)


def split_bib_entries(text: str) -> list[str]:
    """Divide um .bib em entradas individuais de maneira simples e robusta."""
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
                    entry = text[at:j+1].strip()
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
    m = re.match(r"\s*@[^\{]+\{\s*([^,]+)\s*,", entry, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def load_existing_bib_entries(paths: list[Path]) -> tuple[list[str], set[str], list[str]]:
    merged: list[str] = []
    seen_keys: set[str] = set()
    loaded_files: list[str] = []
    for path in paths:
        if not path.exists() or path.suffix.lower() != '.bib':
            continue
        loaded_files.append(str(path))
        text = path.read_text(encoding='utf-8', errors='ignore')
        for entry in split_bib_entries(text):
            key = bib_entry_key(entry)
            if not key:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(entry.strip())
    return merged, seen_keys, loaded_files


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_prompt_audit(path: Path, items: list[tuple[str, str]]) -> None:
    chunks = [f"===== PROMPT {i:02d} | {section} =====\n{content}\n" for i, (section, content) in enumerate(items, start=1)]
    write_text(path, "\n\n".join(chunks).strip() + "\n")


def _build_latex_env(latex_extra_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if latex_extra_path:
        resolved = latex_extra_path.expanduser().resolve()
        latex_dir = resolved.parent if resolved.is_file() else resolved
        texinputs_prefix = f"{latex_dir.as_posix()}//:"
        env["TEXINPUTS"] = texinputs_prefix + env.get("TEXINPUTS", "")
        env["BIBINPUTS"] = texinputs_prefix + env.get("BIBINPUTS", "")
        env["BSTINPUTS"] = texinputs_prefix + env.get("BSTINPUTS", "")
    return env


def run_compile_sequence(org_path: Path, *, emacs_init: Path | None = None, academic_writing: Path | None = None, latex_extra_path: Path | None = None) -> Path | None:
    emacs = ensure_command_available("emacs")
    ensure_command_available("lualatex")
    ensure_command_available("biber")
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
    emacs_cmd = [emacs, "--batch", "-Q"]
    if emacs_init is not None:
        emacs_cmd.extend(["-l", str(emacs_init)])
    if academic_writing is not None:
        emacs_cmd.extend(["-l", str(academic_writing)])
    emacs_cmd.extend(["-l", str(export_el)])
    proc = subprocess.run(
        emacs_cmd,
        cwd=str(org_path.parent),
        capture_output=True,
        text=True,
        env=_build_latex_env(latex_extra_path),
    )
    debug_print("Executando exportação Org->PDF", emacs_cmd, f"cwd={org_path.parent}")
    if proc.stdout.strip():
        debug_print("stdout:\n" + proc.stdout[-4000:])
    if proc.stderr.strip():
        debug_print("stderr:\n" + proc.stderr[-4000:])
    if proc.returncode != 0:
        raise RuntimeError(
            f"Falha ao exportar PDF via Emacs batch.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    pdf_path = org_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"A exportação terminou sem erro, mas o PDF não foi encontrado: {pdf_path}")
    return pdf_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gerador autocontido de paper em Org-mode com IA e busca correlata opcional.")
    p.add_argument("--template")
    p.add_argument("--output-dir")
    p.add_argument("--basename")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--citation-style", choices=["apa", "abnt"], default=None)
    p.add_argument("--exportar-pdf", action="store_true")
    p.add_argument("--emacs-init", help="Caminho para um init.el a ser carregado antes da exportação Org->PDF")
    p.add_argument("--academic-writing", help="Caminho para o academic-writing.el que registra a classe LaTeX fgv-paper")
    p.add_argument("--latex-extra-path", help="Diretório ou arquivo LaTeX extra (ex.: fgv-paper.sty) para TEXINPUTS")
    p.add_argument("--debug", action="store_true", help="Mostra logs detalhados e traceback completo em caso de erro")
    p.add_argument("--preflight-only", action="store_true", help="Somente valida ambiente e encerra")
    p.add_argument("--nao-salvar-prompts", dest="salvar_prompts", action="store_false")
    p.set_defaults(salvar_prompts=True)
    return p.parse_args()


def main() -> int:
    global DEBUG
    load_env()
    args = parse_args()
    DEBUG = bool(args.debug)
    state = load_state()
    prompt_log: list[tuple[str, str]] = []
    emacs_init_default = args.emacs_init or state.get("last_emacs_init")
    academic_writing_default = args.academic_writing or state.get("last_academic_writing")
    latex_extra_default = args.latex_extra_path or state.get("last_latex_extra_path")
    emacs_init = Path(os.path.expanduser(emacs_init_default)).resolve() if emacs_init_default else default_emacs_init()
    academic_writing = Path(os.path.expanduser(academic_writing_default)).resolve() if academic_writing_default else default_academic_writing()
    latex_extra_path = Path(os.path.expanduser(latex_extra_default)).resolve() if latex_extra_default else (Path(DEFAULT_LATEX_EXTRA_PATH).expanduser().resolve() if Path(DEFAULT_LATEX_EXTRA_PATH).expanduser().exists() else None)

    client = make_client()
    model = args.model or DEFAULT_MODEL

    template_default = args.template or state.get("last_template") or str(default_template_path())
    output_default = args.output_dir or state.get("last_output_dir") or "."
    basename_default = args.basename or state.get("last_basename") or DEFAULT_BASENAME
    style_default = normalize_style(args.citation_style or state.get("citation_style") or DEFAULT_STYLE)

    template_path = prompt_path("Template .org do paper (indique apenas um arquivo .org)", template_default, must_exist=True)
    if template_path.suffix.lower() != ".org":
        raise RuntimeError(f"O template do paper deve ser um arquivo .org. Recebido: {template_path}")
    output_root_dir = prompt_path("Diretório de saída", output_default, only_directories=True)
    basename = prompt_text("Nome-base dos arquivos de saída", basename_default)
    output_dir = output_root_dir / basename
    output_dir.mkdir(parents=True, exist_ok=True)
    input_workspace_dir = output_dir / ".paper_inputs_tmp"
    input_workspace_dir.mkdir(parents=True, exist_ok=True)
    style = normalize_style(prompt_text("Estilo bibliográfico/citacional (apa ou abnt)", style_default, completer=_word_completer(["apa", "abnt"])))
    exportar_pdf = True if args.exportar_pdf else prompt_yes_no("Deseja também exportar automaticamente para PDF ao final?", default=bool(state.get("last_export_pdf", False)))
    if exportar_pdf:
        aw_default = str(academic_writing) if academic_writing else (academic_writing_default or "")
        use_aw = prompt_yes_no("Usar o arquivo academic-writing.el para registrar a classe LaTeX?", default=bool(aw_default))
        if use_aw:
            aw_value = prompt_path("Arquivo academic-writing.el", aw_default, must_exist=True)
            academic_writing = Path(os.path.expanduser(aw_value)).resolve()
        else:
            academic_writing = None
        if args.emacs_init or state.get("last_emacs_init"):
            if prompt_yes_no("Deseja carregar também um init.el do Emacs antes da exportação?", default=False):
                ei_default = str(emacs_init) if emacs_init else (emacs_init_default or "")
                ei_value = prompt_path("Arquivo init.el do Emacs", ei_default, must_exist=True)
                emacs_init = Path(os.path.expanduser(ei_value)).resolve()
            else:
                emacs_init = None
        else:
            emacs_init = None
        le_default = str(latex_extra_path) if latex_extra_path else (latex_extra_default or "")
        if prompt_yes_no("Deseja informar um caminho LaTeX extra (TEXINPUTS), como o fgv-paper.sty?", default=bool(le_default)):
            le_value = prompt_path("Arquivo ou diretório LaTeX extra", le_default, must_exist=True)
            latex_extra_path = Path(os.path.expanduser(le_value)).resolve()
        else:
            latex_extra_path = None
    else:
        emacs_init = None
        academic_writing = None
        latex_extra_path = None

    preflight_checks(exportar_pdf=bool(exportar_pdf), emacs_init=emacs_init, academic_writing=academic_writing, latex_extra_path=latex_extra_path)
    if args.preflight_only:
        print("Pré-check concluído com sucesso.")
        if academic_writing is not None:
            print(f"academic-writing.el: {academic_writing}")
        if emacs_init is not None:
            print(f"Init do Emacs: {emacs_init}")
        if latex_extra_path is not None:
            print(f"Caminho LaTeX extra: {latex_extra_path}")
        return 0

    base_paths = prompt_multi_paths("Selecione os documentos-base do paper", required=True)
    guidance_paths = prompt_multi_paths("Selecione os documentos de orientação específica do professor", required=False)

    bib_filename = f"{basename}.bib"
    imported_bib_paths: list[Path] = []
    if prompt_yes_no("Deseja importar um .bib existente para complementar/mesclar as referências?", default=False):
        imported_bib_paths = [p for p in prompt_multi_paths("Selecione um ou mais arquivos .bib existentes", required=True) if p.suffix.lower() == ".bib"]
        if not imported_bib_paths:
            print("Nenhum arquivo .bib válido foi selecionado; seguirei sem importação de .bib existente.")
    template_raw = read_template_raw(template_path)
    template_fields = parse_template_fields(template_raw)
    academic_answers, strategy_defaults = prompt_template_fields(template_fields, state, bib_filename)
    academic_answers["bibliography_file"] = bib_filename
    template_text = materialize_template(template_raw, template_fields, {**academic_answers, **strategy_defaults})
    template_text = apply_citation_style(template_text, bib_filename, style)

    base_items = collect_input_items(base_paths, input_workspace_dir)
    guidance_items = collect_input_items(guidance_paths, input_workspace_dir) if guidance_paths else []
    if not base_items:
        raise RuntimeError("Nenhum documento-base suportado foi encontrado nos caminhos informados.")
    base_docs = build_source_docs(base_items, "base", max_chars=40000)
    guidance_docs = build_source_docs(guidance_items, "orientacao", max_chars=25000) if guidance_items else []
    debug_print(f"Documentos-base coletados: {len(base_docs)}")
    debug_print(f"Documentos de orientação coletados: {len(guidance_docs)}")

    base_context, narrowed_context, tema_adicional_1, tema_adicional_2, context, strategy_answers = infer_context_with_ai(
        client, model, template_text, base_docs, guidance_docs, prompt_log
    )
    strategy_answers = build_strategy_answers_from_inference(
        template_fields, strategy_defaults, base_context, narrowed_context, tema_adicional_1, tema_adicional_2
    )

    base_docs = build_base_doc_bibliography(client, model, base_docs, prompt_log)
    for doc in base_docs:
        try:
            doc.summary = summarize_document(client, model, doc, prompt_log)
        except Exception:
            doc.summary = shorten_text(doc.extracted_text, 1200)

    correlated_docs, related_info = run_related_search_flow(client, model, context, output_dir, basename, prompt_log)

    imported_bib_entries, used_keys, imported_bib_files = load_existing_bib_entries(imported_bib_paths)
    bib_entries: list[str] = list(imported_bib_entries)
    for doc in [*base_docs, *correlated_docs]:
        if doc.bib_key is None:
            doc.bib_key = unique_key(slugify(Path(doc.path).stem), used_keys)
        else:
            doc.bib_key = unique_key(doc.bib_key, used_keys)
        if doc.bib_entry is None:
            meta = BibMetadataOutput(entry_type="misc", title=Path(doc.path).stem.replace("_", " "), note="Metadados incompletos; revisar manualmente.")
            doc.bib_entry = render_biblatex_entry(doc.bib_key, meta)
        else:
            doc.bib_entry = re.sub(r"^@([^{]+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{doc.bib_key},", doc.bib_entry, count=1)
        bib_entries.append(doc.bib_entry.strip())

    final_answers = dict(academic_answers)
    final_answers.update(strategy_defaults)
    final_answers.update({k: v for k, v in strategy_answers.items() if v not in (None, "")})
    final_answers["title"] = AUTO_HEADER_TITLE
    final_answers["author"] = academic_answers.get("author", DEFAULT_AUTHOR) or DEFAULT_AUTHOR
    final_answers["institution_name"] = academic_answers.get("institution_name", DEFAULT_INSTITUTION) or DEFAULT_INSTITUTION
    final_answers["course_name"] = ""
    final_answers["paper_type"] = AUTO_HEADER_PAPER_TYPE
    final_answers["cover_note"] = AUTO_HEADER_COVER_NOTE
    final_answers.setdefault("tema_principal", context.tema)
    final_answers.setdefault("recorte_empirico", context.recorte)
    final_answers.setdefault("objetivo_geral", context.objetivo)
    if context.pergunta_pesquisa:
        final_answers.setdefault("pergunta_de_pesquisa", context.pergunta_pesquisa)
    if context.hipotese:
        final_answers.setdefault("hipotese", context.hipotese)
    template_text_final = materialize_template(template_raw, template_fields, final_answers)
    template_text_final = apply_citation_style(template_text_final, bib_filename, style)

    resumo_campos = {
        "template": str(template_path),
        "saida": str(output_dir),
        "saida_pai": str(output_root_dir),
        "basename": basename,
        "estilo": style,
        "titulo": final_answers.get("title", ""),
        "autor": final_answers.get("author", DEFAULT_AUTHOR),
        "disciplina": final_answers.get("discipline_name", ""),
        "professor": final_answers.get("professor_name", ""),
        "tema": context.tema,
        "recorte": context.recorte,
        "objetivo": context.objetivo,
        "refs_base": len(base_docs),
        "refs_correlatas": len(correlated_docs),
    }
    print("\n=== Resumo consolidado ===")
    print(json.dumps(resumo_campos, ensure_ascii=False, indent=2))
    if not prompt_yes_no("Deseja seguir com a geração do paper com esses dados?", default=True):
        raise RuntimeError("Geração cancelada pelo usuário antes da chamada à OpenAI.")

    org_text = generate_paper_org(client, model, template_text_final, context, base_docs, guidance_docs, correlated_docs, bib_filename, style, prompt_log)
    front_matter = infer_final_front_matter(client, model, context, org_text, prompt_log)
    org_text = apply_final_front_matter(
        org_text,
        title=front_matter.title.strip(),
        author=final_answers.get("author", DEFAULT_AUTHOR) or DEFAULT_AUTHOR,
        paper_type=front_matter.paper_type.strip(),
        cover_note=front_matter.cover_note.strip(),
        course_name="",
    )

    bib_path = output_dir / bib_filename
    org_path = output_dir / f"{basename}.org"
    json_path = output_dir / f"{basename}_contexto.json"
    prompt_audit_path = output_dir / f"{basename}_prompts_auditoria.txt"

    write_text(bib_path, "\n\n".join(bib_entries).strip() + "\n")
    write_text(org_path, org_text)
    write_text(json_path, json.dumps({
        "generated_at": datetime.now().isoformat(),
        "template": str(template_path),
        "citation_style": style,
        "last_emacs_init": str(emacs_init) if emacs_init else "",
        "template_field_answers": final_answers,
        "context": asdict(context),
        "base_context": base_context.model_dump(),
        "narrowed_context": narrowed_context.model_dump(),
        "base_docs": [asdict(d) for d in base_docs],
        "guidance_docs": [asdict(d) for d in guidance_docs],
        "correlated_docs": [asdict(d) for d in correlated_docs],
        "related_info": related_info,
        "imported_bib_files": imported_bib_files,
    }, ensure_ascii=False, indent=2))
    if args.salvar_prompts:
        write_prompt_audit(prompt_audit_path, prompt_log)

    pdf_path = None
    if exportar_pdf:
        pdf_path = run_compile_sequence(org_path, emacs_init=emacs_init, academic_writing=academic_writing, latex_extra_path=latex_extra_path)

    print("\nArquivos gerados:")
    print(f"- Org: {org_path}")
    print(f"- Bib: {bib_path}")
    if imported_bib_paths:
        print(f"- .bib(s) importado(s): {', '.join(str(p) for p in imported_bib_paths)}")
    print(f"- Contexto JSON: {json_path}")
    if args.salvar_prompts:
        print(f"- Auditoria de prompts: {prompt_audit_path}")
    if pdf_path:
        print(f"- PDF: {pdf_path}")

    save_state({
        "last_template": str(template_path),
        "last_output_dir": str(output_root_dir),
        "last_project_dir": str(output_dir),
        "last_basename": basename,
        "citation_style": style,
        "last_export_pdf": bool(exportar_pdf),
        "last_emacs_init": str(emacs_init) if emacs_init else "",
        "last_academic_writing": str(academic_writing) if academic_writing else "",
        "last_latex_extra_path": str(latex_extra_path) if latex_extra_path else "",
        "last_imported_bib_files": [str(p) for p in imported_bib_paths],
        **{k: v for k, v in final_answers.items() if isinstance(v, str)},
    })
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

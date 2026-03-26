#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera uma atividade acadêmica em Org-mode com fluxograma PRISMA em SVG,
a partir de busca em múltiplas bases (Semantic Scholar, Scopus e Web of Science)
e análise via OpenAI API.

A v3.5 mantém o autocomplete interativo de caminhos no terminal via prompt_toolkit
(quando instalado), com fallback automático para input() simples, preserva o
comportamento de padrões apenas entre colchetes, mantém a opção
interativa/por flag para limpar arquivos auxiliares ao final e passa a sugerir
palavras-chave e strings de busca automaticamente via OpenAI, com revisão do
usuário antes da execução.

Fluxo recomendado:
1. O script coleta os parâmetros por flags ou por perguntas interativas no terminal.
2. Busca candidatos nas bases selecionadas.
3. Deduplica, aplica filtro temporal e envia os candidatos para a OpenAI.
4. A OpenAI devolve JSON estruturado para a triagem e a análise final.
5. O Python monta o .org e o SVG localmente, de forma determinística.

Dependências sugeridas:
    pip install openai requests python-dotenv pydantic prompt_toolkit

Variáveis de ambiente aceitas:
    OPENAI_API_KEY
    SEMANTIC_SCHOLAR_API_KEY   (opcional)
    SCOPUS_API_KEY             (necessária para Scopus)
    SCOPUS_INSTTOKEN           (opcional, conforme entitlement Elsevier)
    WOS_API_KEY                (necessária para Web of Science Starter API)

Exemplo:
    python gerar_atividade_prisma_api_multibase_interativo.py \
      --disciplina "Teorias de Administração Pública" \
      --professor "Bernardo Buta" \
      --curso "Mestrado de Políticas Públicas e Governo" \
      --tema "stakeholders" \
      --recorte "artigo de revisão recente" \
      --objetivo "localizar um artigo de revisão recente e analisá-lo" \
      --palavras-chave "stakeholder,stakeholders,stakeholder engagement" \
      --bases "semantic_scholar,scopus,web_of_science" \
      --aluno "Gustavo M. Mendes de Tarso"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from datetime import datetime
from urllib.parse import urljoin
from dataclasses import dataclass, field
from html import escape, unescape
from pathlib import Path
from typing import Iterable, Literal

import requests
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.completion import PathCompleter, WordCompleter
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - dependência opcional em runtime
    pt_prompt = None
    PathCompleter = None
    WordCompleter = None
    PROMPT_TOOLKIT_AVAILABLE = False


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_BASES = ["semantic_scholar"]
DEFAULT_TIPO_ESTUDO = "artigo de revisão"
DEFAULT_PERIODO = "2022-2026"
DEFAULT_IDIOMAS = "inglês,português"
DEFAULT_TRIAGEM = 12
DEFAULT_POLO = "Brasília"
DEFAULT_TURMA = "T-01"
DEFAULT_CITATION_STYLE = "ABNT"
DEFAULT_ORG_MODEL_FILENAME = "template.org"
DEFAULT_ORG_LATEX_CLASS_INIT = "/home/gustavodetarso/.emacs.d/lisp/academic-writing.el"
DEFAULT_LATEX_EXTRA_PATH = "/home/gustavodetarso/texmf/tex/latex/fgv/fgv-paper.sty"
DEFAULT_PDF_EXPORT_COMMAND = ""
DEFAULT_FGV_LOGO_PATH = "/home/gustavodetarso/Documentos/.share/mgs_org/fgv.png"

PT_MONTHS = {1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio", 6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro"}

def today_pt_br() -> str:
    now = datetime.now()
    return f"{now.day} de {PT_MONTHS[now.month]} de {now.year}"

CITATION_STYLE_CHOICES = {
    "abnt": "ABNT",
    "apa": "APA",
    "chicago": "Chicago",
    "mla": "MLA",
    "vancouver": "Vancouver",
}

REMOVABLE_OUTPUT_SUFFIXES = {".aux", ".log", ".out", ".tex", ".toc", ".fls", ".fdb_latexmk", ".bbl", ".blg", ".synctex.gz", ".pdf_tex", ".xdv"}
REMOVABLE_OUTPUT_EXACT_NAMES = {"svg-inkscape"}

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
WOS_STARTER_URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"

SOURCE_LABELS = {
    "semantic_scholar": "Semantic Scholar",
    "scopus": "Scopus",
    "web_of_science": "Web of Science",
}

PRISMA_STATEMENT_REF = (
    "PAGE, Matthew J. et al. *The PRISMA 2020 statement: an updated guideline "
    "for reporting systematic reviews*. BMJ, 2021."
)


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR / ".env"
CONTROLLED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "SCOPUS_API_KEY",
    "SCOPUS_INSTTOKEN",
    "WOS_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "OPENAI_MODEL",
)


def load_local_env_file() -> Path | None:
    """Carrega sempre o .env do diretório do script, sobrescrevendo o ambiente."""
    if not ENV_FILE.exists():
        return None
    values = dotenv_values(ENV_FILE)
    for key in CONTROLLED_ENV_KEYS:
        value = values.get(key)
        if value not in (None, ""):
            os.environ[key] = str(value).strip()
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return ENV_FILE


def fallback_work_title(config: "Config") -> str:
    tema = config.tema.strip()
    recorte = config.recorte.strip()
    tipo = config.tipo_estudo.strip()
    base = f"{tema}: análise de {tipo}"
    if recorte:
        return textwrap.shorten(f"{base} sobre {recorte}", width=140, placeholder="…")
    return textwrap.shorten(base, width=140, placeholder="…")


def normalize_citation_style(value: str | None) -> str:
    if not value:
        return DEFAULT_CITATION_STYLE
    raw = value.strip()
    lower = raw.lower()
    aliases = {
        "abnt": "ABNT",
        "apa": "APA",
        "apa 7": "APA",
        "apa 7th": "APA",
        "apa7": "APA",
        "chicago": "Chicago",
        "chicago author-date": "Chicago",
        "mla": "MLA",
        "vancouver": "Vancouver",
    }
    return aliases.get(lower, raw)


def format_prisma_statement_reference(style: str) -> str:
    style = normalize_citation_style(style)
    refs = {
        "ABNT": "PAGE, Matthew J. et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ, v. 372, p. n71, 2021.",
        "APA": "Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. BMJ, 372, n71.",
        "Chicago": "Page, Matthew J., Joanne E. McKenzie, Patrick M. Bossuyt, Isabelle Boutron, Tammy C. Hoffmann, Cynthia D. Mulrow, et al. 2021. 'The PRISMA 2020 Statement: An Updated Guideline for Reporting Systematic Reviews.' BMJ 372: n71.",
        "MLA": "Page, Matthew J., et al. 'The PRISMA 2020 Statement: An Updated Guideline for Reporting Systematic Reviews.' BMJ, vol. 372, 2021, p. n71.",
        "Vancouver": "Page MJ, McKenzie JE, Bossuyt PM, Boutron I, Hoffmann TC, Mulrow CD, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ. 2021;372:n71.",
    }
    return refs.get(style, refs["ABNT"])



def study_type_terms(tipo_estudo: str) -> list[str]:
    """Extrai termos úteis do tipo de estudo para palavras-chave e queries."""
    raw = (tipo_estudo or "").strip()
    if not raw:
        return []
    lower = raw.lower()
    buckets: list[str] = []

    if "systematic" in lower or "sistem" in lower:
        buckets.extend(["systematic review", "revisão sistemática"])
    if "literature review" in lower or "revisão de literatura" in lower:
        buckets.extend(["literature review", "revisão de literatura"])
    if "scoping" in lower:
        buckets.extend(["scoping review", "revisão de escopo"])
    if "integrative" in lower or "integrativa" in lower:
        buckets.extend(["integrative review", "revisão integrativa"])
    if "narrative" in lower or "narrativa" in lower:
        buckets.extend(["narrative review", "revisão narrativa"])
    if "meta-analysis" in lower or "meta analysis" in lower or "meta-análise" in lower or "metanálise" in lower:
        buckets.extend(["meta-analysis", "meta-análise"])
    if "review" in lower or "revis" in lower:
        buckets.extend(["review", "revisão"])

    buckets.append(raw)

    seen: set[str] = set()
    out: list[str] = []
    for item in buckets:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def ensure_study_type_in_keywords(palavras: list[str], tipo_estudo: str) -> list[str]:
    base = [str(p).strip() for p in palavras if str(p).strip()]
    extras = study_type_terms(tipo_estudo)
    seen: set[str] = set()
    out: list[str] = []
    for item in base + extras:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def build_study_type_clause(tipo_estudo: str) -> str:
    terms = study_type_terms(tipo_estudo)
    if not terms:
        terms = ["systematic review", "literature review", "review"]
    return " OR ".join(f'"{t}"' if " " in t else t for t in terms)


BILINGUAL_TERM_MAP: dict[str, list[str]] = {
    "governanca": ["governance", "public governance", "governança"],
    "governança": ["governance", "public governance", "governanca"],
    "governance": ["governança", "governanca", "public governance"],
    "governanca publica": ["public governance", "governança pública"],
    "governança pública": ["public governance", "governanca publica"],
    "public governance": ["governança pública", "governanca publica"],
    "administracao publica": ["public administration", "administração pública"],
    "administração pública": ["public administration", "administracao publica"],
    "public administration": ["administração pública", "administracao publica"],
    "servico publico": ["public service", "setor público", "serviço público"],
    "serviço público": ["public service", "setor público", "servico publico"],
    "public service": ["serviço público", "servico publico", "setor público"],
    "setor publico": ["public sector", "setor público"],
    "setor público": ["public sector", "setor publico"],
    "public sector": ["setor público", "setor publico"],
    "integridade": ["integrity"],
    "integrity": ["integridade"],
    "controle interno": ["internal control", "controles internos"],
    "controles internos": ["internal controls", "controle interno"],
    "internal control": ["controle interno", "controles internos"],
    "internal controls": ["controle interno", "controles internos"],
    "accountability": ["accountability", "prestação de contas"],
    "prestacao de contas": ["accountability", "prestação de contas"],
    "prestação de contas": ["accountability", "prestacao de contas"],
    "gestao de riscos": ["risk management", "gestão de riscos"],
    "gestão de riscos": ["risk management", "gestao de riscos"],
    "risk management": ["gestão de riscos", "gestao de riscos"],
    "compliance": ["compliance"],
    "stakeholders": ["partes interessadas"],
    "stakeholder": ["parte interessada"],
    "partes interessadas": ["stakeholders"],
    "parte interessada": ["stakeholder"],
}


def _normalize_key(text: str) -> str:
    return slugify(text).replace('_', ' ').strip()


def languages_include_both_pt_en(idiomas: list[str] | None) -> bool:
    langs = ' '.join((idiomas or [])).lower()
    has_pt = any(tok in langs for tok in ['portugues', 'português', 'pt-br', 'pt_br', 'pt'])
    has_en = any(tok in langs for tok in ['ingles', 'inglês', 'english', 'en'])
    return has_pt and has_en


def expand_bilingual_terms(terms: list[str], tema: str = '', recorte: str = '', idiomas: list[str] | None = None, enabled: bool = False) -> list[str]:
    if not enabled:
        return [t for t in terms if t]

    pool = [t.strip() for t in terms if t and str(t).strip()]
    for extra in [tema, recorte]:
        if extra and extra.strip():
            pool.append(extra.strip())

    out: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip()
        if not term:
            return
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)

    normalized_pool = [(_normalize_key(item), item) for item in pool]
    for _, original in normalized_pool:
        add(original)

    for normalized, original in normalized_pool:
        for key, values in BILINGUAL_TERM_MAP.items():
            if key and key in normalized:
                for value in values:
                    add(value)

    if languages_include_both_pt_en(idiomas):
        for term in list(out):
            normalized = _normalize_key(term)
            for key, values in BILINGUAL_TERM_MAP.items():
                if key == normalized:
                    for value in values:
                        add(value)

    return out


class CandidateDecision(BaseModel):
    paper_id: str
    stage: Literal["incluido", "excluido_triagem", "excluido_elegibilidade"]
    tipo_estudo: str
    motivo: str


class TriageOutput(BaseModel):
    pergunta_orientadora: str
    introducao: str
    base_logica_busca: str
    criterios_inclusao: list[str]
    criterios_exclusao: list[str]
    observacao_metodologica: str
    selected_paper_id: str
    selected_paper_justification: str
    decisions: list[CandidateDecision]


class PaperAnalysisOutput(BaseModel):
    referencia_completa: str
    problema_objetivo: str
    argumento_central: str
    desenho_pesquisa: str
    principais_achados: str
    contribuicao_estudo: str
    justificativa_selecao_final: str
    texto_corrido_entrega: str


class SearchSuggestionOutput(BaseModel):
    palavras_chave: list[str]
    termos_relacionados: list[str]
    query_semantic: str
    query_scopus: str
    query_wos: str
    observacoes: str


class WorkTitleOutput(BaseModel):
    titulo_trabalho: str = Field(description="Título acadêmico em português para a atividade")


@dataclass
class Config:
    disciplina: str
    professor: str
    curso: str
    tema: str
    recorte: str
    objetivo: str
    bases: list[str]
    tipo_estudo: str
    estilo_citacao: str
    periodo: str
    idiomas: list[str]
    palavras_chave: list[str]
    query_bilingue: bool
    aluno: str
    turma: str
    polo: str
    trabalho: str
    titulo_trabalho: str | None
    quantidade_triagem: int
    model: str
    output_dir: Path
    org_modelo: Path
    arquivo_orientacao: Path | None
    texto_orientacao_extra: str | None
    prefixo: str
    query_geral: str | None
    query_semantic: str | None
    query_scopus: str | None
    query_wos: str | None
    nao_interativo: bool
    exportar_pdf: bool
    salvar_busca_bruta_json: bool
    gerar_env_example: bool
    remover_auxiliares: bool
    incluir_resumo_artigo_ia: bool
    org_latex_class_init: Path | None
    latex_extra_path: Path | None
    comando_exportacao_pdf: str | None
    fgv_logo_path: Path

    @property
    def bases_label(self) -> str:
        return ", ".join(SOURCE_LABELS.get(src, src) for src in self.bases)

    def source_query(self, source: str) -> str:
        if source == "semantic_scholar" and self.query_semantic:
            return self.query_semantic
        if source == "scopus" and self.query_scopus:
            return self.query_scopus
        if source == "web_of_science" and self.query_wos:
            return self.query_wos
        if self.query_geral:
            return self.query_geral
        return build_query_for_source(
            source,
            self.tema,
            self.palavras_chave,
            self.tipo_estudo,
            recorte=self.recorte,
            idiomas=self.idiomas,
            query_bilingue=self.query_bilingue,
        )


@dataclass
class CandidatePaper:
    paper_id: str
    title: str
    abstract: str
    year: int | None
    venue: str | None
    publication_date: str | None
    authors: list[str]
    tldr: str | None
    url: str | None
    pdf_url: str | None
    doi: str | None
    sources: list[str] = field(default_factory=list)
    source_ids: dict[str, str] = field(default_factory=dict)
    full_text_verified: bool = False
    full_text_source: str | None = None
    full_text_note: str | None = None
    downloaded_pdf_path: str | None = None
    downloaded_pdf_note: str | None = None

    def short_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "authors": self.authors,
            "doi": self.doi,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "tldr": self.tldr,
            "abstract": self.abstract,
            "full_text_verified": self.full_text_verified,
            "full_text_source": self.full_text_source,
            "full_text_note": self.full_text_note,
            "downloaded_pdf_path": self.downloaded_pdf_path,
            "downloaded_pdf_note": self.downloaded_pdf_note,
            "sources": [SOURCE_LABELS.get(s, s) for s in self.sources],
            "source_ids": self.source_ids,
        }


@dataclass
class SourceFetchResult:
    source: str
    query: str
    retrieved: int
    candidates: list[CandidatePaper]
    warnings: list[str] = field(default_factory=list)
    raw_payloads: list[dict] = field(default_factory=list)


@dataclass
class SearchAudit:
    per_source: list[SourceFetchResult]
    identified_total: int
    duplicates_removed: int
    other_removed: int
    screened_total: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class PrismaCounts:
    identified: int
    removed_pre: int
    duplicates_removed: int
    other_removed: int
    screened: int
    excluded_screening: int
    full_text_sought: int
    not_retrieved: int
    full_text_assessed: int
    excluded_full_text: int
    included_qualitative: int


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9áàâãéêíóôõúç\s_-]", "", text, flags=re.I)
    repl = {
        "á": "a", "à": "a", "â": "a", "ã": "a",
        "é": "e", "ê": "e", "í": "i", "ó": "o",
        "ô": "o", "õ": "o", "ú": "u", "ç": "c",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_") or "atividade_prisma"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


TEXT_GUIDANCE_SUFFIXES = {".txt", ".md", ".org", ".rst", ".tex", ".yaml", ".yml", ".json", ".csv", ".xml"}


def _strip_xml_like_tags(text: str) -> str:
    text = re.sub(r"<w:tab[^>]*/>", "\t", text)
    text = re.sub(r"<w:br[^>]*/>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_orientation_file(path: Path, max_chars: int = 20000) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_GUIDANCE_SUFFIXES:
        return read_text(path)[:max_chars]
    if suffix == ".docx":
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        return _strip_xml_like_tags(xml)[:max_chars]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            chunks = []
            for page in reader.pages[:20]:
                chunks.append(page.extract_text() or "")
            return re.sub(r"\s+", " ", "\n".join(chunks)).strip()[:max_chars]
        except Exception:
            pdftotext = shutil.which("pdftotext")
            if pdftotext:
                try:
                    result = subprocess.run([pdftotext, str(path), "-"], capture_output=True, text=True, check=True)
                    return re.sub(r"\s+", " ", result.stdout).strip()[:max_chars]
                except Exception as exc:
                    raise RuntimeError(f"Não foi possível extrair texto do PDF: {exc}") from exc
            raise RuntimeError("Não foi possível extrair texto do PDF. Instale pypdf ou pdftotext.")
    raise RuntimeError(f"Formato de arquivo de orientação não suportado: {suffix}")


def setup_logging(output_dir: Path, prefixo: str) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{prefixo}.log"
    logger = logging.getLogger(f"atividade_prisma.{prefixo}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False
    return logger, log_path


def write_env_example(path: Path) -> None:
    content = textwrap.dedent("""
    # OpenAI
    OPENAI_API_KEY=
    OPENAI_MODEL=gpt-5.4

    # Bases acadêmicas
    SEMANTIC_SCHOLAR_API_KEY=
    SCOPUS_API_KEY=
    SCOPUS_INSTTOKEN=
    WOS_API_KEY=

    # Opcional
    HTTP_PROXY=
    HTTPS_PROXY=
    """).strip() + "\n"
    write_text(path, content)


def save_raw_search_jsons(output_dir: Path, prefixo: str, audit: SearchAudit) -> list[Path]:
    paths: list[Path] = []
    for item in audit.per_source:
        path = output_dir / f"{prefixo}_{item.source}_raw.json"
        payload = {
            "source": item.source,
            "label": SOURCE_LABELS.get(item.source, item.source),
            "query": item.query,
            "retrieved": item.retrieved,
            "warnings": item.warnings,
            "saved_at": datetime.now().isoformat(),
            "pages": item.raw_payloads,
        }
        write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        paths.append(path)
    return paths


def save_source_logs(output_dir: Path, prefixo: str, audit: SearchAudit) -> list[Path]:
    paths: list[Path] = []
    for item in audit.per_source:
        path = output_dir / f"{prefixo}_{item.source}.log"
        lines = [
            f"Fonte: {SOURCE_LABELS.get(item.source, item.source)}",
            f"Query: {item.query}",
            f"Recuperados: {item.retrieved}",
            f"Warnings: {'; '.join(item.warnings) if item.warnings else 'nenhum'}",
            "Candidatos:",
        ]
        for idx, cand in enumerate(item.candidates, start=1):
            lines.extend([
                f"  {idx}. {cand.title}",
                f"     Ano: {cand.year or 's.d.'}",
                f"     DOI: {cand.doi or 'n/d'}",
                f"     URL: {cand.url or 'n/d'}",
                f"     ID fonte: {cand.source_ids.get(item.source, 'n/d')}",
            ])
        write_text(path, "\n".join(lines) + "\n")
        paths.append(path)
    return paths


def which_or_none(cmd: str) -> str | None:
    return shutil.which(cmd)


def convert_svg_to_pdf(svg_path: Path, logger: logging.Logger) -> Path:
    inkscape = which_or_none("inkscape")
    if not inkscape:
        raise RuntimeError("O comando 'inkscape' não foi encontrado no PATH. Ele é necessário para converter o fluxograma SVG em PDF.")
    pdf_path = svg_path.with_suffix('.pdf')
    cmd = [inkscape, str(svg_path), f"--export-filename={pdf_path}"]
    proc = subprocess.run(cmd, cwd=str(svg_path.parent), capture_output=True, text=True)
    if proc.stdout:
        logger.info("Saída da conversão SVG->PDF (stdout): %s", proc.stdout.strip()[:4000])
    if proc.stderr:
        logger.info("Saída da conversão SVG->PDF (stderr): %s", proc.stderr.strip()[:4000])
    if proc.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"Falha ao converter SVG em PDF com Inkscape (código {proc.returncode}).")
    logger.info("PDF do fluxograma gerado em %s", pdf_path)
    return pdf_path


def _build_latex_env(logger: logging.Logger, latex_extra_path: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if latex_extra_path:
        resolved_extra = latex_extra_path.resolve()
        latex_dir_path = resolved_extra.parent if resolved_extra.is_file() else resolved_extra
        latex_dir = latex_dir_path.as_posix()
        texinputs_prefix = f"{latex_dir}//:"
        env["TEXINPUTS"] = texinputs_prefix + env.get("TEXINPUTS", "")
        env["BIBINPUTS"] = texinputs_prefix + env.get("BIBINPUTS", "")
        env["BSTINPUTS"] = texinputs_prefix + env.get("BSTINPUTS", "")
        logger.info("Usando caminho extra de classes/pacotes LaTeX: %s (diretório efetivo: %s)", resolved_extra, latex_dir)
    return env


def export_org_to_pdf_internal(org_path: Path, logger: logging.Logger,
                               org_latex_class_init: Path | None = None,
                               latex_extra_path: Path | None = None) -> Path | None:
    emacs = which_or_none("emacs")
    lualatex = which_or_none("lualatex")
    if not emacs or not lualatex:
        logger.warning("Exportação automática para PDF ignorada: 'emacs' ou 'lualatex' não encontrados no PATH.")
        return None

    elisp_path = org_path.parent / f"{org_path.stem}_export_pdf.el"
    init_loader = ""
    if org_latex_class_init:
        init_loader = f'(load-file "{org_latex_class_init.as_posix()}")\n'

    elisp = textwrap.dedent(f"""
    (require 'package)
    (package-initialize)
    {init_loader}(require 'org)
    (require 'ox)
    (require 'ox-latex)
    (require 'oc)
    (require 'oc-biblatex)
    (setq org-export-use-babel nil)
    (setq org-confirm-babel-evaluate nil)
    (setq org-latex-pdf-process
          '("lualatex -shell-escape -interaction nonstopmode -output-directory %o %f"
            "biber %b"
            "lualatex -shell-escape -interaction nonstopmode -output-directory %o %f"
            "lualatex -shell-escape -interaction nonstopmode -output-directory %o %f"))
    (find-file "{org_path.as_posix()}")
    (org-latex-export-to-pdf)
    """).strip() + "\n"
    write_text(elisp_path, elisp)
    logger.info("Tentando exportar PDF automaticamente via Emacs batch + LuaLaTeX (fallback interno).")

    env = _build_latex_env(logger, latex_extra_path)
    if org_latex_class_init:
        logger.info("Carregando arquivo Emacs de registro da classe LaTeX: %s", org_latex_class_init)

    cmd = [emacs, "--batch", "-Q"]
    if org_latex_class_init:
        cmd.extend(["-l", str(org_latex_class_init)])
    cmd.extend(["-l", str(elisp_path)])

    proc = subprocess.run(cmd, cwd=str(org_path.parent), capture_output=True, text=True, env=env)
    logger.info("Saída do Emacs batch (stdout): %s", (proc.stdout or "").strip()[:4000])
    if proc.stderr:
        logger.info("Saída do Emacs batch (stderr): %s", proc.stderr.strip()[:4000])
    if proc.returncode != 0:
        logger.warning("Exportação automática para PDF falhou com código %s.", proc.returncode)
        return None
    pdf_path = org_path.with_suffix('.pdf')
    if pdf_path.exists():
        logger.info("PDF gerado com sucesso: %s", pdf_path)
        return pdf_path
    logger.warning("O processo terminou sem erro, mas o PDF final não foi localizado em %s.", pdf_path)
    return None


def export_org_to_pdf_external(org_path: Path, bib_path: Path | None, logger: logging.Logger,
                               command_template: str,
                               org_latex_class_init: Path | None = None,
                               latex_extra_path: Path | None = None) -> Path | None:
    pdf_path = org_path.with_suffix('.pdf')
    latex_dir_resolved = None
    class_init_resolved = None
    if latex_extra_path:
        latex_resolved = latex_extra_path.resolve()
        latex_dir_resolved = latex_resolved.parent if latex_resolved.is_file() else latex_resolved
    if org_latex_class_init:
        class_init_resolved = org_latex_class_init.resolve()

    placeholders = {
        'org': org_path.as_posix(),
        'org_dir': org_path.parent.as_posix(),
        'org_stem': org_path.stem,
        'pdf': pdf_path.as_posix(),
        'pdf_dir': pdf_path.parent.as_posix(),
        'class_init': class_init_resolved.as_posix() if class_init_resolved else '',
        'latex_path': latex_extra_path.resolve().as_posix() if latex_extra_path else '',
        'latex_dir': latex_dir_resolved.as_posix() if latex_dir_resolved else '',
        'bib': bib_path.as_posix() if bib_path else '',
    }
    try:
        command = command_template.format(**placeholders)
    except KeyError as exc:
        logger.warning("Comando de exportação PDF contém placeholder desconhecido: %s", exc)
        return None

    logger.info("Tentando exportar PDF via comando externo: %s", command)
    env = _build_latex_env(logger, latex_extra_path)
    proc = subprocess.run(command, cwd=str(org_path.parent), shell=True, capture_output=True, text=True, env=env)
    logger.info("Saída do comando externo (stdout): %s", (proc.stdout or '').strip()[:4000])
    if proc.stderr:
        logger.info("Saída do comando externo (stderr): %s", proc.stderr.strip()[:4000])
    if proc.returncode != 0:
        logger.warning("Exportação automática para PDF via comando externo falhou com código %s.", proc.returncode)
        return None
    if pdf_path.exists():
        logger.info("PDF gerado com sucesso: %s", pdf_path)
        return pdf_path
    logger.warning("O comando externo terminou sem erro, mas o PDF final não foi localizado em %s.", pdf_path)
    return None


def export_org_to_pdf(org_path: Path, bib_path: Path | None, logger: logging.Logger,
                      org_latex_class_init: Path | None = None,
                      latex_extra_path: Path | None = None,
                      command_template: str | None = None) -> Path | None:
    if command_template:
        return export_org_to_pdf_external(org_path, bib_path, logger, command_template, org_latex_class_init, latex_extra_path)
    return export_org_to_pdf_internal(org_path, logger, org_latex_class_init, latex_extra_path)


def cleanup_generated_files(output_dir: Path, prefixo: str, logger: logging.Logger) -> list[Path]:
    removed: list[Path] = []
    for path in output_dir.iterdir():
        if path.name == ".env.example":
            continue
        if path.is_dir():
            if path.name in REMOVABLE_OUTPUT_EXACT_NAMES:
                try:
                    shutil.rmtree(path)
                    removed.append(path)
                except Exception as exc:
                    logger.warning("Não foi possível remover diretório auxiliar %s: %s", path, exc)
            continue
        if not path.name.startswith(prefixo):
            continue
        if path.suffix.lower() in {".org", ".pdf", ".svg", ".json", ".bib"}:
            continue
        should_remove = False
        if path.name.endswith("_export_pdf.el"):
            should_remove = True
        elif path.suffix.lower() in REMOVABLE_OUTPUT_SUFFIXES:
            should_remove = True
        if should_remove:
            try:
                path.unlink()
                removed.append(path)
            except Exception as exc:
                logger.warning("Não foi possível remover arquivo auxiliar %s: %s", path, exc)
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera atividade acadêmica em Org-mode + SVG PRISMA usando OpenAI e múltiplas bases."
    )
    parser.add_argument("--disciplina")
    parser.add_argument("--professor")
    parser.add_argument("--curso")
    parser.add_argument("--tema")
    parser.add_argument("--recorte")
    parser.add_argument("--objetivo")
    parser.add_argument("--bases", help="Bases separadas por vírgula: semantic_scholar,scopus,web_of_science")
    parser.add_argument("--tipo-estudo")
    parser.add_argument("--estilo-citacao")
    parser.add_argument("--periodo", default=DEFAULT_PERIODO)
    parser.add_argument("--idiomas")
    parser.add_argument("--palavras-chave")
    parser.add_argument("--sugerir-palavras-chave-ia", action="store_true", help="Usa a OpenAI para sugerir palavras-chave e queries por base antes da busca.")
    parser.add_argument("--aluno")
    parser.add_argument("--turma", default=DEFAULT_TURMA)
    parser.add_argument("--polo", default=DEFAULT_POLO)
    parser.add_argument("--trabalho", help="Título do trabalho; se informado, substitui a sugestão automática da IA.")
    parser.add_argument("--quantidade-triagem", type=int, default=DEFAULT_TRIAGEM)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--query-geral")
    parser.add_argument("--query-semantic")
    parser.add_argument("--query-scopus")
    parser.add_argument("--query-wos")
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query-bilingue", dest="query_bilingue", action="store_true", help="Monta/sugere queries bilíngues em português e inglês quando possível.")
    query_group.add_argument("--sem-query-bilingue", dest="query_bilingue", action="store_false", help="Desativa a montagem bilíngue das queries.")
    parser.set_defaults(query_bilingue=None)
    parser.add_argument("--arquivo-orientacao", help="Arquivo opcional com parâmetros extras de orientação para o arquivo final.")
    parser.add_argument("--prefixo")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--org-modelo", default=DEFAULT_ORG_MODEL_FILENAME)
    parser.add_argument("--org-latex-class-init", default=DEFAULT_ORG_LATEX_CLASS_INIT, help="Arquivo .el que registra a classe LaTeX fgv-paper no Emacs batch (ex.: academic-writing.el).")
    parser.add_argument("--latex-extra-path", default=DEFAULT_LATEX_EXTRA_PATH, help="Diretório ou arquivo LaTeX da classe/pacotes (ex.: diretório com fgv-paper.sty/fgv-header.sty, ou o próprio arquivo fgv-paper.sty).")
    parser.add_argument("--comando-exportacao-pdf", default=DEFAULT_PDF_EXPORT_COMMAND, help="Comando externo para exportar o .org para PDF. Aceita placeholders como {org}, {org_dir}, {org_stem}, {pdf}, {pdf_dir}, {class_init}, {latex_path}, {latex_dir} e {bib}.")
    parser.add_argument("--fgv-logo-path", default=DEFAULT_FGV_LOGO_PATH, help="Caminho do logo da FGV a ser usado no cabeçalho do relatório.")
    parser.add_argument("--exportar-pdf", action="store_true", help="Tenta exportar o .org para PDF automaticamente.")
    parser.add_argument("--salvar-busca-bruta-json", action="store_true", help="Salva as respostas brutas das bases em JSON por fonte.")
    parser.add_argument("--gerar-env-example", action="store_true", help="Gera um arquivo .env.example no diretório de saída.")
    parser.add_argument("--remover-auxiliares", action="store_true", help="Remove arquivos auxiliares ao final, preservando apenas .org, .pdf, .svg e .json do trabalho atual.")
    resumo_group = parser.add_mutually_exclusive_group()
    resumo_group.add_argument("--incluir-resumo-artigo-ia", dest="incluir_resumo_artigo_ia", action="store_true", help="Inclui no .org a seção 'Resumo do artigo selecionado' gerada pela IA.")
    resumo_group.add_argument("--sem-resumo-artigo-ia", dest="incluir_resumo_artigo_ia", action="store_false", help="Não inclui no .org a seção 'Resumo do artigo selecionado' gerada pela IA.")
    parser.set_defaults(incluir_resumo_artigo_ia=None)
    parser.add_argument("--nao-interativo", action="store_true")
    return parser.parse_args()


def _supports_prompt_toolkit() -> bool:
    return PROMPT_TOOLKIT_AVAILABLE and bool(sys.stdin.isatty()) and bool(sys.stdout.isatty())


def _prompt_raw(label: str, default: str | None = None, completer=None, password: bool = False) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    message = f"{label}{suffix}: "
    if _supports_prompt_toolkit():
        kwargs = {"is_password": password}
        if completer is not None:
            kwargs["completer"] = completer
            kwargs["complete_while_typing"] = True
        try:
            return pt_prompt(message, **kwargs).strip()
        except TypeError:
            # Compatibilidade com variações de assinatura entre versões.
            kwargs.pop("is_password", None)
            if password:
                kwargs["password"] = True
            return pt_prompt(message, **kwargs).strip()
    return input(message).strip()


def _path_completer(only_directories: bool = False):
    if not _supports_prompt_toolkit():
        return None
    return PathCompleter(only_directories=only_directories, expanduser=True)


def _word_completer(words: list[str]):
    if not _supports_prompt_toolkit():
        return None
    return WordCompleter(words, ignore_case=True, sentence=True)


def prompt_text(label: str, current: str | None = None, default: str | None = None, force: bool = True) -> str:
    if not force and current is not None:
        return current
    base = current if current not in (None, "") else default
    while True:
        raw = _prompt_raw(label, str(base) if base is not None else None)
        if raw:
            return raw
        if base is not None:
            return str(base)


def prompt_path(
    label: str,
    current: str | None = None,
    default: str | None = None,
    *,
    force: bool = True,
    only_directories: bool = False,
    must_exist: bool = False,
) -> str:
    if not force and current is not None:
        return current
    base = current if current not in (None, "") else default
    while True:
        raw = _prompt_raw(label, str(base) if base is not None else None, completer=_path_completer(only_directories=only_directories))
        if not raw and base is not None:
            raw = str(base)
        expanded = os.path.expanduser(raw).strip()
        if not expanded:
            if must_exist:
                print("Informe um caminho válido.")
                continue
            return expanded
        path = Path(expanded)
        if must_exist and not path.exists():
            print(f"Caminho não encontrado: {path}")
            continue
        if only_directories and path.exists() and not path.is_dir():
            print(f"O caminho precisa ser um diretório: {path}")
            continue
        return expanded


def prompt_list(label: str, current_csv: str | None = None, default_csv: str | None = None, force: bool = True) -> list[str]:
    raw = prompt_text(label, current_csv, default_csv, force=force)
    return [x.strip() for x in raw.split(",") if x.strip()]


def prompt_int(label: str, current: int | None = None, default: int | None = None, force: bool = True) -> int:
    if not force and current is not None:
        return current
    base = current if current is not None else default
    while True:
        raw = _prompt_raw(label, str(base) if base is not None else None)
        if not raw and base is not None:
            return int(base)
        try:
            return int(raw)
        except ValueError:
            print("Digite um número inteiro válido.")


def prompt_yes_no(label: str, default: bool = True) -> bool:
    suffix = "S/n" if default else "s/N"
    raw = _prompt_raw(f"{label} [{suffix}]", None).lower()
    if not raw:
        return default
    return raw in {"s", "sim", "y", "yes"}


def normalize_sources(raw_sources: Iterable[str]) -> list[str]:
    norm = []
    seen = set()
    mapping = {
        "semantic": "semantic_scholar",
        "semantic_scholar": "semantic_scholar",
        "semantic-scholar": "semantic_scholar",
        "semanticscholar": "semantic_scholar",
        "scopus": "scopus",
        "wos": "web_of_science",
        "web_of_science": "web_of_science",
        "web-of-science": "web_of_science",
        "web of science": "web_of_science",
    }
    for item in raw_sources:
        key = mapping.get(item.strip().lower(), item.strip().lower())
        if key not in SOURCE_LABELS:
            raise ValueError(f"Base desconhecida: {item}")
        if key not in seen:
            seen.add(key)
            norm.append(key)
    return norm


def prompt_sources(current_csv: str | None, force: bool) -> list[str]:
    if not force and current_csv:
        return normalize_sources(current_csv.split(","))

    print("Bases disponíveis:")
    print("  1) Semantic Scholar")
    print("  2) Scopus")
    print("  3) Web of Science")
    default_txt = ",".join(DEFAULT_BASES)
    tokens_hint = [
        "1", "2", "3",
        "semantic_scholar", "semantic scholar", "semantic-scholar",
        "scopus",
        "web_of_science", "web of science", "web-of-science", "wos",
    ]
    raw = _prompt_raw(
        "Selecione as bases (números ou nomes, separados por vírgula)",
        default_txt,
        completer=_word_completer(tokens_hint),
    )
    if not raw:
        return list(DEFAULT_BASES)

    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    resolved = []
    num_map = {"1": "semantic_scholar", "2": "scopus", "3": "web_of_science"}
    for tok in tokens:
        resolved.append(num_map.get(tok, tok))
    return normalize_sources(resolved)


def _semantic_scholar_terms_for_query(terms: list[str]) -> list[str]:
    """Monta uma lista enxuta de termos para o Semantic Scholar.

    O endpoint /paper/search funciona melhor com texto curto e sem booleanos
    pesados (AND/OR/parênteses). Aqui priorizamos poucos termos únicos,
    preservando expressões compostas entre aspas na etapa final.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        term = re.sub(r'\s+', ' ', str(raw or '').strip())
        if not term:
            continue
        term = re.sub(r'^[()]+|[()]+$', '', term)
        term = re.sub(r'\b(?:AND|OR|NOT)\b', ' ', term, flags=re.IGNORECASE)
        term = re.sub(r'\s+', ' ', term).strip(' ,;')
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(term)

    phrases = [t for t in cleaned if ' ' in t]
    singles = [t for t in cleaned if ' ' not in t]

    ordered: list[str] = []
    for bucket in (phrases, singles):
        for item in bucket:
            if item not in ordered:
                ordered.append(item)

    return ordered[:8]


def simplify_semantic_scholar_query(query: str) -> str:
    """Reduz uma query arbitrária a um formato compatível com o Semantic Scholar."""
    raw_parts = re.split(r'\s+', query or '')
    parts: list[str] = []
    buffer: list[str] = []
    in_quotes = False
    for token in raw_parts:
        if not token:
            continue
        # preserva frases entre aspas quando possível
        if token.count('"') % 2 == 1:
            if not in_quotes:
                buffer = [token]
                in_quotes = True
            else:
                buffer.append(token)
                parts.append(' '.join(buffer).replace('"', '').strip())
                buffer = []
                in_quotes = False
            continue
        if in_quotes:
            buffer.append(token)
            continue
        parts.append(token)
    if buffer:
        parts.extend(buffer)

    simplified_terms = _semantic_scholar_terms_for_query(parts)
    return ' '.join(f'"{t}"' if ' ' in t else t for t in simplified_terms)


def build_query_for_source(
    source: str,
    tema: str,
    palavras: list[str],
    tipo_estudo: str,
    *,
    recorte: str = "",
    idiomas: list[str] | None = None,
    query_bilingue: bool = False,
) -> str:
    base_terms = palavras[:] if palavras else [tema]
    terms = ensure_study_type_in_keywords(base_terms, tipo_estudo)
    terms = expand_bilingual_terms(terms, tema=tema, recorte=recorte, idiomas=idiomas, enabled=query_bilingue)
    nucleus = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
    review = build_study_type_clause(tipo_estudo)
    if source == "semantic_scholar":
        semantic_terms = _semantic_scholar_terms_for_query(terms)
        if not semantic_terms:
            semantic_terms = _semantic_scholar_terms_for_query([tema, recorte, tipo_estudo])
        return ' '.join(f'"{t}"' if ' ' in t else t for t in semantic_terms)
    if source == "scopus":
        return f"TITLE-ABS-KEY(({nucleus}) AND ({review}))"
    if source == "web_of_science":
        return f"TS=(({nucleus}) AND ({review}))"
    raise ValueError(f"Base não suportada para query: {source}")


def prompt_choice(label: str, choices: dict[str, str], default: str) -> str:
    """Retorna a chave escolhida entre as opções fornecidas."""
    completer = _word_completer(list(choices.keys()) + list(choices.values()))
    while True:
        raw = _prompt_raw(label, default, completer=completer).strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        for key, alias in choices.items():
            if raw == alias.lower():
                return key
        print(f"Escolha inválida. Opções: {', '.join(f'{k}={v}' for k, v in choices.items())}")


def prompt_citation_style(current: str | None, force: bool) -> str:
    if not force and current:
        return normalize_citation_style(current)
    print("Estilos bibliográficos disponíveis:")
    print("  1) ABNT")
    print("  2) APA")
    print("  3) Chicago")
    print("  4) MLA")
    print("  5) Vancouver")
    choices = {
        "1": "ABNT",
        "2": "APA",
        "3": "Chicago",
        "4": "MLA",
        "5": "Vancouver",
        "abnt": "ABNT",
        "apa": "APA",
        "chicago": "Chicago",
        "mla": "MLA",
        "vancouver": "Vancouver",
    }
    selected = prompt_choice("Estilo de citações e referências bibliográficas", choices, "1")
    return normalize_citation_style(choices[selected])


def ensure_openai_api_key(interactive: bool) -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    if not interactive:
        return False
    raw = _prompt_raw("Chave da OpenAI (não será salva; Enter para cancelar)", None, password=True)
    if raw:
        os.environ["OPENAI_API_KEY"] = raw
        return True
    return False


def suggest_keywords_with_openai(
    tema: str,
    recorte: str,
    objetivo: str,
    tipo_estudo: str,
    bases: list[str],
    idiomas: list[str],
    model: str,
    query_bilingue: bool,
) -> SearchSuggestionOutput:
    client = make_openai_client()

    class _SearchSuggestionOutput(SearchSuggestionOutput):
        pass

    prompt = textwrap.dedent(
        f"""
        Você é um assistente de metodologia de busca bibliográfica.
        Gere sugestões iniciais de palavras-chave e strings de busca para bases acadêmicas.

        Tema central: {tema}
        Recorte específico: {recorte}
        Objetivo da atividade: {objetivo}
        Tipo de estudo priorizado: {tipo_estudo}
        Bases selecionadas: {', '.join(SOURCE_LABELS.get(b, b) for b in bases)}
        Idiomas aceitáveis: {', '.join(idiomas)}
        Query bilíngue (português + inglês): {'sim' if query_bilingue else 'não'}

        Regras gerais:
        - proponha palavras-chave diretamente relacionadas ao tema, ao recorte e ao objetivo;
        - inclua termos em português e inglês quando isso ampliar a busca;
        - monte queries bilíngues em português e inglês quando query_bilingue for verdadeiro;
        - mantenha foco em busca acadêmica, evitando termos vagos demais;
        - as queries devem priorizar {tipo_estudo};
        - inclua explicitamente o descritor do tipo de estudo também no campo palavras_chave;
        - não explique demais: devolva apenas os campos estruturados.

        Regras específicas por base:
        - query_semantic (Semantic Scholar):
          * NÃO use operadores booleanos como AND, OR ou NOT;
          * NÃO use parênteses;
          * use no máximo 8 termos ou expressões;
          * priorize inglês; inclua português apenas se for muito relevante;
          * use aspas apenas em expressões compostas;
          * devolva uma query curta, enxuta, sem sintaxe booleana, pronta para colar no Semantic Scholar.
        - query_scopus (Scopus):
          * use sintaxe booleana apropriada para Scopus;
          * pode usar TITLE-ABS-KEY, AND, OR e parênteses.
        - query_wos (Web of Science):
          * use sintaxe booleana apropriada para Web of Science;
          * pode usar TS=, AND, OR e parênteses.
        """
    ).strip()

    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=_SearchSuggestionOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("A OpenAI não retornou sugestões estruturadas de palavras-chave.")
    parsed.palavras_chave = ensure_study_type_in_keywords(parsed.palavras_chave, tipo_estudo)
    if query_bilingue:
        parsed.palavras_chave = expand_bilingual_terms(parsed.palavras_chave, tema=tema, recorte=recorte, idiomas=idiomas, enabled=True)
    # Sanitiza a query do Semantic Scholar para um formato realmente compatível.
    semantic_seed = parsed.query_semantic or ' '.join(parsed.palavras_chave[:6])
    parsed.query_semantic = simplify_semantic_scholar_query(semantic_seed)
    if not parsed.query_semantic.strip():
        parsed.query_semantic = build_query_for_source(
            'semantic_scholar', tema, parsed.palavras_chave, tipo_estudo, recorte=recorte, idiomas=idiomas, query_bilingue=query_bilingue
        )
    return parsed


def _print_keyword_suggestions(s: SearchSuggestionOutput, bases: list[str]) -> None:
    print("\nSugestões da IA")
    print("-" * 72)
    print("Palavras-chave sugeridas:")
    print("  " + ", ".join(s.palavras_chave))
    if s.termos_relacionados:
        print("Termos relacionados:")
        print("  " + ", ".join(s.termos_relacionados))
    if "semantic_scholar" in bases:
        print("\nQuery sugerida para Semantic Scholar:")
        print(s.query_semantic)
    if "scopus" in bases:
        print("\nQuery sugerida para Scopus:")
        print(s.query_scopus)
    if "web_of_science" in bases:
        print("\nQuery sugerida para Web of Science:")
        print(s.query_wos)
    if s.observacoes:
        print("\nObservações:")
        print(textwrap.fill(s.observacoes, width=88))
    print("-" * 72)


def configure_keywords_and_queries(
    args: argparse.Namespace,
    *,
    interactive: bool,
    tema: str,
    recorte: str,
    objetivo: str,
    bases: list[str],
    tipo_estudo: str,
    idiomas: list[str],
    model: str,
    query_bilingue: bool,
) -> tuple[list[str], str | None, str | None, str | None, str | None]:
    manual_csv = args.palavras_chave
    manual_words = [x.strip() for x in (manual_csv or "").split(",") if x.strip()]

    use_ai = bool(args.sugerir_palavras_chave_ia)
    if interactive:
        use_ai = prompt_yes_no(
            "Deseja que a IA sugira palavras-chave e queries por base a partir do tema, recorte e objetivo?",
            default=not bool(manual_words),
        )

    suggestion = None
    if use_ai:
        if not ensure_openai_api_key(interactive):
            if interactive:
                print("Não foi possível obter a chave da OpenAI. Seguindo para preenchimento manual.")
            use_ai = False
        else:
            while True:
                try:
                    suggestion = suggest_keywords_with_openai(
                        tema=tema,
                        recorte=recorte,
                        objetivo=objetivo,
                        tipo_estudo=tipo_estudo,
                        bases=bases,
                        idiomas=idiomas,
                        model=model,
                        query_bilingue=query_bilingue,
                    )
                except Exception as exc:
                    if interactive:
                        print(f"Falha ao gerar sugestões com a IA: {exc}")
                        print("Seguindo para preenchimento manual.")
                        use_ai = False
                        break
                    raise

                if not interactive:
                    palavras = ensure_study_type_in_keywords(suggestion.palavras_chave or manual_words or [tema], tipo_estudo)
                    query_geral = args.query_geral
                    query_semantic = args.query_semantic or (suggestion.query_semantic if "semantic_scholar" in bases else None)
                    query_scopus = args.query_scopus or (suggestion.query_scopus if "scopus" in bases else None)
                    query_wos = args.query_wos or (suggestion.query_wos if "web_of_science" in bases else None)
                    return palavras, query_geral, query_semantic, query_scopus, query_wos

                _print_keyword_suggestions(suggestion, bases)
                choice = prompt_choice(
                    "Ação sobre as sugestões [a=aceitar, e=editar, r=regenerar, m=manual]",
                    {"a": "aceitar", "e": "editar", "r": "regenerar", "m": "manual"},
                    "a",
                )
                if choice == "r":
                    continue
                if choice == "m":
                    use_ai = False
                    break
                if choice == "a":
                    palavras = ensure_study_type_in_keywords(suggestion.palavras_chave or manual_words or [tema], tipo_estudo)
                    return (
                        palavras,
                        None,
                        suggestion.query_semantic if "semantic_scholar" in bases else None,
                        suggestion.query_scopus if "scopus" in bases else None,
                        suggestion.query_wos if "web_of_science" in bases else None,
                    )
                if choice == "e":
                    palavras = ensure_study_type_in_keywords(prompt_list(
                        "Palavras-chave finais (separadas por vírgula)",
                        current_csv=",".join(suggestion.palavras_chave),
                        force=True,
                    ), tipo_estudo)
                    query_geral = None
                    query_semantic = args.query_semantic
                    query_scopus = args.query_scopus
                    query_wos = args.query_wos
                    if "semantic_scholar" in bases:
                        query_semantic = prompt_text("Query para Semantic Scholar", suggestion.query_semantic, force=True)
                    if "scopus" in bases:
                        query_scopus = prompt_text("Query para Scopus", suggestion.query_scopus, force=True)
                    if "web_of_science" in bases:
                        query_wos = prompt_text("Query para Web of Science", suggestion.query_wos, force=True)
                    return palavras, query_geral, query_semantic, query_scopus, query_wos

    palavras = ensure_study_type_in_keywords(prompt_list("Palavras-chave iniciais (separadas por vírgula)", args.palavras_chave, force=interactive) if interactive else (manual_words or [tema]), tipo_estudo)
    palavras = expand_bilingual_terms(palavras, tema=tema, recorte=recorte, idiomas=idiomas, enabled=query_bilingue)

    if interactive:
        auto_query = prompt_yes_no("Gerar automaticamente as queries específicas por base a partir do tema, do tipo de estudo e das palavras-chave?", default=True)
        if auto_query:
            return palavras, None, args.query_semantic, args.query_scopus, args.query_wos
        query_geral = prompt_text("Query geral única (deixe vazia se quiser informar por base)", args.query_geral, default="", force=True) or None
        query_semantic = args.query_semantic
        query_scopus = args.query_scopus
        query_wos = args.query_wos
        if not query_geral:
            if "semantic_scholar" in bases:
                query_semantic = prompt_text("Query para Semantic Scholar", args.query_semantic, force=True)
            if "scopus" in bases:
                query_scopus = prompt_text("Query para Scopus", args.query_scopus, force=True)
            if "web_of_science" in bases:
                query_wos = prompt_text("Query para Web of Science", args.query_wos, force=True)
        return palavras, query_geral, query_semantic, query_scopus, query_wos

    return palavras, args.query_geral, args.query_semantic, args.query_scopus, args.query_wos


def build_config(args: argparse.Namespace) -> Config:
    interactive = not args.nao_interativo

    curso = prompt_text("Curso/Programa", args.curso, force=interactive)
    if interactive:
        turma = prompt_text("Turma", args.turma, DEFAULT_TURMA, force=True)
        polo = prompt_text("Pólo", args.polo, DEFAULT_POLO, force=True)
        disciplina = prompt_text("Disciplina", args.disciplina, force=True)
        professor = prompt_text("Professor(a)", args.professor, force=True)
        aluno = prompt_text("Aluno(s)", args.aluno, force=True)
    else:
        turma = (args.turma or DEFAULT_TURMA).strip()
        polo = (args.polo or DEFAULT_POLO).strip()
        disciplina = (args.disciplina or "").strip()
        professor = (args.professor or "").strip()
        aluno = (args.aluno or "").strip()
    tema = prompt_text("Tema central", args.tema, force=interactive)
    recorte = prompt_text("Recorte específico", args.recorte, force=interactive)
    objetivo = prompt_text("Objetivo da atividade", args.objetivo, force=interactive)
    idiomas = prompt_list("Idiomas aceitáveis (separados por vírgula)", args.idiomas, DEFAULT_IDIOMAS, force=interactive)

    bases = prompt_sources(args.bases, force=interactive) if interactive else normalize_sources((args.bases or ",".join(DEFAULT_BASES)).split(","))

    tipo_estudo = prompt_text("Tipo de estudo a priorizar", args.tipo_estudo, DEFAULT_TIPO_ESTUDO, force=interactive)
    estilo_citacao = prompt_citation_style(args.estilo_citacao, force=interactive)
    incluir_resumo_artigo_ia = prompt_yes_no("Deseja incluir no .org a seção 'Resumo do artigo selecionado' gerada pela IA?", default=True) if interactive else (True if args.incluir_resumo_artigo_ia is None else bool(args.incluir_resumo_artigo_ia))
    query_bilingue_default = languages_include_both_pt_en(idiomas)
    query_bilingue = prompt_yes_no("Deseja montar/sugerir queries bilíngues (português + inglês)?", default=query_bilingue_default) if interactive else (query_bilingue_default if args.query_bilingue is None else bool(args.query_bilingue))
    periodo = prompt_text("Período desejado", args.periodo, DEFAULT_PERIODO, force=interactive)
    trabalho = (args.trabalho or "").strip()
    model = prompt_text("Modelo OpenAI", args.model, DEFAULT_MODEL, force=interactive)

    palavras, query_geral, query_semantic, query_scopus, query_wos = configure_keywords_and_queries(
        args,
        interactive=interactive,
        tema=tema,
        recorte=recorte,
        objetivo=objetivo,
        bases=bases,
        tipo_estudo=tipo_estudo,
        idiomas=idiomas,
        model=model,
        query_bilingue=query_bilingue,
    )

    prefixo_default = slugify(f"atividade_{tema}_prisma")
    prefixo = prompt_text("Prefixo de arquivos", args.prefixo, prefixo_default, force=interactive)
    output_dir = Path(prompt_path("Diretório de saída", args.output_dir, args.output_dir, force=interactive, only_directories=True)).expanduser().resolve()
    org_modelo = Path(prompt_path("Arquivo .org modelo", args.org_modelo, DEFAULT_ORG_MODEL_FILENAME, force=interactive, must_exist=False)).expanduser()
    if not org_modelo.is_absolute():
        org_modelo = (Path.cwd() / org_modelo).resolve()
    fgv_logo_path = Path(prompt_path("Caminho do logo da FGV para o cabeçalho", args.fgv_logo_path, DEFAULT_FGV_LOGO_PATH, force=interactive, must_exist=False, only_directories=False)).expanduser()
    if not fgv_logo_path.is_absolute():
        fgv_logo_path = (Path.cwd() / fgv_logo_path).resolve()

    arquivo_orientacao: Path | None = None
    texto_orientacao_extra: str | None = None
    usar_orientacao = bool(args.arquivo_orientacao)
    if interactive:
        usar_orientacao = prompt_yes_no("Deseja indicar um arquivo de orientação complementar com parâmetros extras para o arquivo final?", default=bool(args.arquivo_orientacao))
    if usar_orientacao:
        orient_path_txt = prompt_path("Arquivo de orientação complementar", args.arquivo_orientacao, args.arquivo_orientacao, force=interactive, must_exist=True) if interactive else (args.arquivo_orientacao or "")
        if orient_path_txt:
            arquivo_orientacao = Path(orient_path_txt).expanduser().resolve()
            try:
                texto_orientacao_extra = read_orientation_file(arquivo_orientacao)
            except Exception as exc:
                msg = f"Falha ao ler arquivo de orientação complementar ({arquivo_orientacao}): {exc}"
                if interactive:
                    print(msg)
                else:
                    raise RuntimeError(msg) from exc

    quantidade_triagem = prompt_int("Quantidade máxima de candidatos para triagem detalhada", args.quantidade_triagem, DEFAULT_TRIAGEM, force=interactive)
    exportar_pdf = prompt_yes_no("Tentar exportar automaticamente o .org para PDF ao final?", default=False) if interactive else bool(args.exportar_pdf)
    org_latex_class_init: Path | None = None
    latex_extra_path: Path | None = None
    comando_exportacao_pdf: str | None = None
    if exportar_pdf:
        class_init_default = args.org_latex_class_init or DEFAULT_ORG_LATEX_CLASS_INIT
        latex_path_default = args.latex_extra_path or DEFAULT_LATEX_EXTRA_PATH
        comando_default = (args.comando_exportacao_pdf or DEFAULT_PDF_EXPORT_COMMAND).strip()
        if interactive:
            usar_class_init = prompt_yes_no("Usar o arquivo .el padrão que registra a classe LaTeX fgv-paper?", default=bool(class_init_default))
            if usar_class_init and class_init_default:
                org_latex_class_init = Path(class_init_default).expanduser().resolve()
            else:
                tmp = prompt_path("Informe outro arquivo .el de registro da classe LaTeX [opcional]", None, None, force=True, must_exist=False)
                if tmp:
                    org_latex_class_init = Path(tmp).expanduser().resolve()
            usar_latex_extra = prompt_yes_no("Usar o caminho LaTeX padrão para classes/pacotes?", default=bool(latex_path_default))
            if usar_latex_extra and latex_path_default:
                latex_extra_path = Path(latex_path_default).expanduser().resolve()
            else:
                tmp = prompt_path("Informe outro caminho extra de classes/pacotes LaTeX [opcional]", None, None, force=True, must_exist=False, only_directories=False)
                if tmp:
                    latex_extra_path = Path(tmp).expanduser().resolve()
            comando_raw = _prompt_raw(
                "Comando externo de exportação PDF [opcional; placeholders: {org}, {org_dir}, {org_stem}, {pdf}, {pdf_dir}, {class_init}, {latex_path}, {latex_dir}, {bib}]",
                comando_default or None,
            )
            comando_exportacao_pdf = (comando_raw or comando_default).strip() or None
        else:
            if class_init_default:
                org_latex_class_init = Path(class_init_default).expanduser().resolve()
            if latex_path_default:
                latex_extra_path = Path(latex_path_default).expanduser().resolve()
            comando_exportacao_pdf = comando_default or None
    salvar_busca_bruta_json = prompt_yes_no("Salvar a busca bruta de cada base em JSON?", default=True) if interactive else bool(args.salvar_busca_bruta_json)
    gerar_env_example = prompt_yes_no("Gerar um arquivo .env.example no diretório de saída?", default=True) if interactive else bool(args.gerar_env_example)
    remover_auxiliares = prompt_yes_no(
        "Remover ao final os arquivos auxiliares (.aux, .log, .out, .tex, .toc, .fls, .fdb_latexmk, .bbl, .blg, .synctex.gz etc.), preservando apenas .org, .pdf, .svg e .json do trabalho atual?",
        default=False,
    ) if interactive else bool(args.remover_auxiliares)

    return Config(
        disciplina=disciplina,
        professor=professor,
        curso=curso,
        tema=tema,
        recorte=recorte,
        objetivo=objetivo,
        bases=bases,
        tipo_estudo=tipo_estudo,
        estilo_citacao=estilo_citacao,
        periodo=periodo,
        idiomas=idiomas,
        palavras_chave=palavras,
        query_bilingue=query_bilingue,
        aluno=aluno,
        turma=turma,
        polo=polo,
        trabalho=trabalho,
        titulo_trabalho=None,
        quantidade_triagem=quantidade_triagem,
        model=model,
        output_dir=output_dir,
        org_modelo=org_modelo,
        arquivo_orientacao=arquivo_orientacao,
        texto_orientacao_extra=texto_orientacao_extra,
        prefixo=prefixo,
        query_geral=query_geral,
        query_semantic=query_semantic,
        query_scopus=query_scopus,
        query_wos=query_wos,
        nao_interativo=args.nao_interativo,
        exportar_pdf=exportar_pdf,
        salvar_busca_bruta_json=salvar_busca_bruta_json,
        gerar_env_example=gerar_env_example,
        remover_auxiliares=remover_auxiliares,
        incluir_resumo_artigo_ia=incluir_resumo_artigo_ia,
        org_latex_class_init=org_latex_class_init,
        latex_extra_path=latex_extra_path,
        comando_exportacao_pdf=comando_exportacao_pdf,
        fgv_logo_path=fgv_logo_path,
    )



def generate_work_title_with_openai(client: OpenAI, config: Config, model_org_text: str) -> str:
    class _WorkTitleOutput(WorkTitleOutput):
        pass

    orientacao_extra = f"\n\nOrientações complementares para o arquivo final:\n{config.texto_orientacao_extra[:6000]}" if config.texto_orientacao_extra else ""

    prompt = textwrap.dedent(
        f"""
        Gere um título acadêmico em português para uma atividade de busca e análise bibliográfica.

        Dados da atividade:
        - Tema central: {config.tema}
        - Recorte específico: {config.recorte}
        - Objetivo da atividade: {config.objetivo}
        - Tipo de estudo priorizado: {config.tipo_estudo}
        - Bases consultadas: {config.bases_label}
        - Estilo bibliográfico escolhido: {config.estilo_citacao}

        Regras:
        - título claro, acadêmico e objetivo;
        - entre 8 e 20 palavras, preferencialmente;
        - evitar dois pontos desnecessários ou excesso de adjetivos;
        - refletir o tema, o recorte e o objetivo;
        - não mencionar PRISMA 2020 no título, a menos que seja realmente necessário;
        - produzir apenas um título final em português.
        {orientacao_extra}

        Modelo Org de referência (somente para inferir tom acadêmico):
        {model_org_text[:5000]}
        """
    ).strip()

    response = client.responses.parse(
        model=config.model,
        input=[{"role": "user", "content": prompt}],
        text_format=_WorkTitleOutput,
    )
    parsed = response.output_parsed
    if parsed is None or not parsed.titulo_trabalho.strip():
        return fallback_work_title(config)
    return textwrap.shorten(parsed.titulo_trabalho.strip(), width=160, placeholder="…")



def suggest_and_confirm_work_title(client: OpenAI | None, config: Config, model_org_text: str, interactive: bool) -> str:
    def _generate_once() -> str:
        if (config.trabalho or "").strip():
            return config.trabalho.strip()
        if client is None:
            return fallback_work_title(config)
        try:
            return generate_work_title_with_openai(client, config, model_org_text)
        except Exception:
            return fallback_work_title(config)

    title = _generate_once()
    if not interactive:
        return title

    while True:
        print("\nTítulo sugerido para o trabalho:")
        print(title)
        choice = prompt_choice(
            "Ação sobre o título [a=aceitar, e=editar, r=regenerar, m=manual]",
            {"a": "aceitar", "e": "editar", "r": "regenerar", "m": "manual"},
            "a",
        )
        if choice == "a":
            return title
        if choice == "e":
            return prompt_text("Título do trabalho", title, force=True)
        if choice == "m":
            return prompt_text("Título do trabalho", None, default=title, force=True)
        if choice == "r":
            # Se houve título passado por flag, regenerar passa a usar a IA.
            config.trabalho = ""
            title = _generate_once()


def load_model_org_text(path: Path) -> str:
    if path.exists():
        return read_text(path)
    return (
        "#+LANGUAGE: pt_BR\n"
        "#+OPTIONS: toc:nil num:t title:nil html-postamble:nil ^:{}\n"
        "#+STARTUP: indent\n"
        "#+LATEX_COMPILER: lualatex\n"
        "#+LATEX_CLASS: fgv-paper\n"
        "#+LATEX_CLASS_OPTIONS: [12pt,a4paper]\n"
        "#+LATEX_HEADER: \\usepackage{fgv-header}\n"
        "#+LATEX_HEADER: \\usepackage{float}\n"
        "#+LATEX_HEADER: \\usepackage{svg}\n"
    )


def make_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrado no ambiente.")
    return OpenAI(api_key=api_key)


def prompt_secret_if_missing(env_name: str, label: str, required: bool, interactive: bool) -> str | None:
    current = os.getenv(env_name)
    if current:
        return current
    if not interactive:
        return None
    if required:
        raw = _prompt_raw(f"{label} (não será salvo; deixe vazio para ignorar a base)", None, password=True)
        if raw:
            os.environ[env_name] = raw
            return raw
        return None
    raw = _prompt_raw(f"{label} (opcional; Enter para pular)", None, password=True)
    if raw:
        os.environ[env_name] = raw
        return raw
    return None


def semantic_scholar_headers(api_key: str | None) -> dict[str, str]:
    headers = {"User-Agent": "atividade-prisma-script/3.4"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def scopus_headers(api_key: str, insttoken: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key,
        "User-Agent": "atividade-prisma-script/3.4",
    }
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken
    return headers


def wos_headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "X-ApiKey": api_key,
        "User-Agent": "atividade-prisma-script/3.4",
    }


def first_nonempty(*values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value not in (None, "", [], {}):
            return value
    return None


def nested_get(data, *keys):
    cur = data
    for key in keys:
        if isinstance(cur, list):
            if not isinstance(key, int) or key >= len(cur):
                return None
            cur = cur[key]
        elif isinstance(cur, dict):
            if key not in cur:
                return None
            cur = cur[key]
        else:
            return None
    return cur


def year_from_text(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", str(text))
    return int(m.group(0)) if m else None


def author_list_from_semantic(item: dict) -> list[str]:
    return [a.get("name", "").strip() for a in item.get("authors", []) if a.get("name")]


def author_list_from_scopus(item: dict) -> list[str]:
    creators = []
    creator = item.get("dc:creator")
    if isinstance(creator, str) and creator.strip():
        creators.append(creator.strip())
    authors = item.get("author")
    if isinstance(authors, list):
        for a in authors:
            name = first_nonempty(a.get("authname"), a.get("ce:indexed-name"), a.get("ce:surname"))
            if isinstance(name, str) and name.strip() and name.strip() not in creators:
                creators.append(name.strip())
    return creators


def author_list_from_wos(item: dict) -> list[str]:
    names = []
    candidates = nested_get(item, "names", "authors")
    if isinstance(candidates, list):
        for a in candidates:
            name = first_nonempty(a.get("displayName"), a.get("fullName"), a.get("wosStandard"), a.get("name"))
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    elif isinstance(candidates, dict):
        for key in ("displayName", "fullName", "wosStandard", "name"):
            if candidates.get(key):
                names.append(str(candidates[key]).strip())
                break
    return names


def merge_candidate(into: CandidatePaper, other: CandidatePaper) -> CandidatePaper:
    if not into.abstract and other.abstract:
        into.abstract = other.abstract
    if not into.year and other.year:
        into.year = other.year
    if not into.venue and other.venue:
        into.venue = other.venue
    if not into.publication_date and other.publication_date:
        into.publication_date = other.publication_date
    if not into.authors and other.authors:
        into.authors = other.authors
    if not into.tldr and other.tldr:
        into.tldr = other.tldr
    if not into.url and other.url:
        into.url = other.url
    if not into.pdf_url and other.pdf_url:
        into.pdf_url = other.pdf_url
    if not into.doi and other.doi:
        into.doi = other.doi
    if not into.full_text_verified and other.full_text_verified:
        into.full_text_verified = other.full_text_verified
        into.full_text_source = other.full_text_source
        into.full_text_note = other.full_text_note
    elif not into.full_text_note and other.full_text_note:
        into.full_text_note = other.full_text_note
    for src in other.sources:
        if src not in into.sources:
            into.sources.append(src)
    into.source_ids.update(other.source_ids)
    return into


def dedupe_candidates(raw_items: Iterable[CandidatePaper]) -> tuple[list[CandidatePaper], int]:
    seen: dict[tuple[str, str], CandidatePaper] = {}
    duplicates_removed = 0
    for item in raw_items:
        title_key = re.sub(r"\s+", " ", item.title.lower()).strip()
        doi_key = (item.doi or "").lower().strip()
        key = (doi_key, title_key)
        if key in seen:
            duplicates_removed += 1
            merge_candidate(seen[key], item)
            continue
        seen[key] = item
    return list(seen.values()), duplicates_removed


def fetch_semantic_scholar_candidates(query: str, limit: int, api_key: str | None) -> SourceFetchResult:
    fields = ",".join([
        "paperId",
        "title",
        "abstract",
        "year",
        "venue",
        "publicationDate",
        "authors",
        "url",
        "externalIds",
        "tldr",
        "openAccessPdf",
    ])

    def _run(search_query: str) -> tuple[dict, list[dict]]:
        params = {"query": search_query, "limit": str(limit), "fields": fields}
        resp = requests.get(S2_SEARCH_URL, params=params, headers=semantic_scholar_headers(api_key), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        return payload, payload.get("data", []) or []

    payload, data = _run(query)
    effective_query = query
    raw_payloads: list[dict] = [payload]

    if not data:
        relaxed_query = simplify_semantic_scholar_query(query)
        if relaxed_query and relaxed_query != query:
            logging.info("Semantic Scholar retornou 0 resultados; tentando query simplificada: %s", relaxed_query)
            payload2, data2 = _run(relaxed_query)
            raw_payloads.append(payload2)
            if data2:
                payload, data = payload2, data2
                effective_query = relaxed_query

    papers: list[CandidatePaper] = []
    for item in data:
        tldr = item.get("tldr") or {}
        ext = item.get("externalIds") or {}
        open_pdf = item.get("openAccessPdf") or {}
        paper_id = item.get("paperId", "").strip()
        if not paper_id:
            continue
        papers.append(
            CandidatePaper(
                paper_id=f"semantic_scholar:{paper_id}",
                title=(item.get("title") or "Sem título").strip(),
                abstract=(item.get("abstract") or "").strip(),
                year=item.get("year"),
                venue=(item.get("venue") or "").strip() or None,
                publication_date=(item.get("publicationDate") or "").strip() or None,
                authors=author_list_from_semantic(item),
                tldr=(tldr.get("text") or "").strip() or None,
                url=(item.get("url") or "").strip() or None,
                pdf_url=(open_pdf.get("url") or "").strip() or None,
                doi=(ext.get("DOI") or "").strip() or None,
                sources=["semantic_scholar"],
                source_ids={"semantic_scholar": paper_id},
            )
        )
    return SourceFetchResult(source="semantic_scholar", query=effective_query, retrieved=len(papers), candidates=papers, raw_payloads=raw_payloads)


def fetch_scopus_candidates(query: str, limit: int, api_key: str, insttoken: str | None) -> SourceFetchResult:
    warnings: list[str] = []
    candidates: list[CandidatePaper] = []
    raw_payloads: list[dict] = []
    start = 0
    per_page = min(max(limit, 1), 25)
    while len(candidates) < limit:
        params = {
            "query": query,
            "count": str(min(per_page, limit - len(candidates))),
            "start": str(start),
            "view": "STANDARD",
        }
        resp = requests.get(SCOPUS_SEARCH_URL, params=params, headers=scopus_headers(api_key, insttoken), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        raw_payloads.append(payload)
        results = payload.get("search-results", {})
        entries = results.get("entry", []) or []
        if not entries:
            break
        for item in entries:
            title = (item.get("dc:title") or "Sem título").strip()
            doi = (item.get("prism:doi") or "").strip() or None
            cover = item.get("prism:coverDate")
            publication_name = (item.get("prism:publicationName") or "").strip() or None
            identifier = first_nonempty(item.get("eid"), item.get("dc:identifier"), doi, title)
            url = (item.get("prism:url") or "").strip() or None
            candidates.append(
                CandidatePaper(
                    paper_id=f"scopus:{identifier}",
                    title=title,
                    abstract=(item.get("dc:description") or "").strip(),
                    year=year_from_text(cover),
                    venue=publication_name,
                    publication_date=cover.strip() if isinstance(cover, str) and cover.strip() else None,
                    authors=author_list_from_scopus(item),
                    tldr=None,
                    url=url,
                    pdf_url=None,
                    doi=doi,
                    sources=["scopus"],
                    source_ids={"scopus": str(identifier)},
                )
            )
            if len(candidates) >= limit:
                break
        if len(entries) < per_page:
            break
        start += len(entries)
    if not candidates:
        warnings.append("Scopus não retornou resultados para a query informada.")
    return SourceFetchResult(source="scopus", query=query, retrieved=len(candidates), candidates=candidates, warnings=warnings, raw_payloads=raw_payloads)


def fetch_wos_candidates(query: str, limit: int, api_key: str) -> SourceFetchResult:
    warnings: list[str] = []
    candidates: list[CandidatePaper] = []
    raw_payloads: list[dict] = []
    page = 1
    per_page = min(max(limit, 1), 50)
    while len(candidates) < limit:
        params = {
            "q": query,
            "db": "WOS",
            "limit": str(min(per_page, limit - len(candidates))),
            "page": str(page),
            "detail": "short",
        }
        resp = requests.get(WOS_STARTER_URL, params=params, headers=wos_headers(api_key), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        raw_payloads.append(payload)
        hits = payload.get("hits") or payload.get("data") or payload.get("documents") or []
        if not hits:
            break
        for item in hits:
            title = first_nonempty(
                item.get("title"),
                nested_get(item, "titles", 0, "title"),
                nested_get(item, "names", "title"),
            )
            title = str(title).strip() if title else "Sem título"
            publication_date = first_nonempty(item.get("publishYear"), item.get("datePublished"), item.get("sourceDate"))
            venue = first_nonempty(
                nested_get(item, "source", "sourceTitle"),
                item.get("sourceTitle"),
                item.get("source"),
            )
            identifiers = item.get("identifiers") or {}
            doi = first_nonempty(identifiers.get("doi"), item.get("doi"))
            uid = first_nonempty(item.get("uid"), item.get("UID"), doi, title)
            links = item.get("links") or {}
            url = first_nonempty(links.get("record"), links.get("self"), item.get("url"))
            abstract = first_nonempty(item.get("abstract"), nested_get(item, "keywords", "authorKeywords"), "")
            candidates.append(
                CandidatePaper(
                    paper_id=f"web_of_science:{uid}",
                    title=title,
                    abstract=str(abstract).strip() if abstract else "",
                    year=year_from_text(str(publication_date) if publication_date is not None else None),
                    venue=str(venue).strip() if isinstance(venue, str) and venue.strip() else None,
                    publication_date=str(publication_date).strip() if publication_date is not None else None,
                    authors=author_list_from_wos(item),
                    tldr=None,
                    url=str(url).strip() if isinstance(url, str) and url.strip() else None,
                    pdf_url=None,
                    doi=str(doi).strip() if isinstance(doi, str) and doi.strip() else None,
                    sources=["web_of_science"],
                    source_ids={"web_of_science": str(uid)},
                )
            )
            if len(candidates) >= limit:
                break
        if len(hits) < per_page:
            break
        page += 1
    if not candidates:
        warnings.append("Web of Science não retornou resultados para a query informada.")
    return SourceFetchResult(source="web_of_science", query=query, retrieved=len(candidates), candidates=candidates, warnings=warnings, raw_payloads=raw_payloads)



PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
    "application/octet-stream",
}
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ActivityPrisma/3.7.9",
    "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
PDF_LINK_PATTERNS = [
    re.compile(r'<meta[^>]+name=["\\\']citation_pdf_url["\\\'][^>]+content=["\\\']([^"\\\']+)["\\\']', re.I),
    re.compile(r'href=["\\\']([^"\\\']+\.pdf(?:\?[^"\\\']*)?)["\\\']', re.I),
    re.compile(r'href=["\\\']([^"\\\']*download[^"\\\']*)["\\\']', re.I),
]

def is_probably_pdf_response(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    final_url = str(resp.url).lower()
    if ctype in PDF_CONTENT_TYPES:
        return True
    if final_url.endswith(".pdf"):
        return True
    disposition = (resp.headers.get("Content-Disposition") or "").lower()
    if ".pdf" in disposition:
        return True
    return False

def fetch_url_best_effort(url: str, timeout: int = 25) -> requests.Response | None:
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, allow_redirects=True, stream=True)
        return resp
    except Exception:
        return None

def verify_pdf_download_url(url: str) -> tuple[bool, str | None, str | None]:
    if not url:
        return False, None, "URL ausente"
    resp = fetch_url_best_effort(url)
    if resp is None:
        return False, None, "Falha ao acessar a URL"
    try:
        if resp.status_code >= 400:
            return False, None, f"HTTP {resp.status_code}"
        if is_probably_pdf_response(resp):
            return True, str(resp.url), None
        return False, None, "A URL não retornou um PDF verificável"
    finally:
        try:
            resp.close()
        except Exception:
            pass

def extract_pdf_links_from_html(html: str, base_url: str) -> list[str]:
    found: list[str] = []
    for pattern in PDF_LINK_PATTERNS:
        for match in pattern.findall(html):
            link = urljoin(base_url, match.strip())
            if link and link not in found:
                found.append(link)
    return found[:12]

def try_resolve_pdf_from_landing(url: str) -> tuple[str | None, str | None]:
    if not url:
        return None, "URL de landing ausente"
    resp = fetch_url_best_effort(url)
    if resp is None:
        return None, "Falha ao acessar a landing page"
    try:
        if resp.status_code >= 400:
            return None, f"Landing page retornou HTTP {resp.status_code}"
        if is_probably_pdf_response(resp):
            return str(resp.url), None
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
            return None, "Landing page não retornou HTML legível"
        try:
            html = resp.text[:400000]
        except Exception:
            return None, "Não foi possível ler o conteúdo HTML da landing page"
    finally:
        try:
            resp.close()
        except Exception:
            pass
    for link in extract_pdf_links_from_html(html, str(url)):
        ok, final_url, _note = verify_pdf_download_url(link)
        if ok and final_url:
            return final_url, None
    return None, "Nenhum link de PDF verificável foi encontrado na landing page"

def resolve_full_text_for_candidate(candidate: CandidatePaper) -> CandidatePaper:
    links_to_try: list[tuple[str, str]] = []
    if candidate.pdf_url:
        links_to_try.append((candidate.pdf_url, "pdf_url informado pela base"))
    if candidate.url:
        links_to_try.append((candidate.url, "URL principal do registro"))
    if candidate.doi:
        doi_url = candidate.doi if str(candidate.doi).lower().startswith("http") else f"https://doi.org/{candidate.doi}"
        links_to_try.append((doi_url, "landing page do DOI"))

    seen: set[str] = set()
    cleaned: list[tuple[str, str]] = []
    for link, source in links_to_try:
        link = (link or "").strip()
        if link and link not in seen:
            seen.add(link)
            cleaned.append((link, source))

    for link, source in cleaned:
        ok, final_url, note = verify_pdf_download_url(link)
        if ok and final_url:
            candidate.pdf_url = final_url
            candidate.full_text_verified = True
            candidate.full_text_source = source
            candidate.full_text_note = "PDF verificável acessível para leitura e download."
            return candidate
        resolved_url, landing_note = try_resolve_pdf_from_landing(link)
        if resolved_url:
            candidate.pdf_url = resolved_url
            candidate.full_text_verified = True
            candidate.full_text_source = f"{source} (resolução via landing page)"
            candidate.full_text_note = "PDF identificável e acessível a partir da landing page."
            return candidate
        candidate.full_text_note = note or landing_note or "Texto completo não verificável"

    candidate.full_text_verified = False
    if not candidate.full_text_note:
        candidate.full_text_note = "Não foi possível localizar um link verificável para download do texto completo."
    return candidate

def _pdf_cache_filename(candidate: CandidatePaper) -> str:
    seed = candidate.pdf_url or candidate.url or candidate.doi or candidate.paper_id
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:12]
    base = slugify(candidate.title)[:80] or slugify(candidate.paper_id)
    return f"{base}_{digest}.pdf"


def download_candidate_pdf(candidate: CandidatePaper, cache_dir: Path, timeout: int = 45, max_bytes: int = 30 * 1024 * 1024) -> CandidatePaper:
    if candidate.downloaded_pdf_path and Path(candidate.downloaded_pdf_path).exists():
        candidate.downloaded_pdf_note = candidate.downloaded_pdf_note or "PDF baixado localmente com sucesso."
        return candidate
    if not candidate.pdf_url:
        candidate.full_text_verified = False
        candidate.full_text_note = "Sem URL de PDF para download local."
        return candidate

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / _pdf_cache_filename(candidate)
    try:
        resp = requests.get(candidate.pdf_url, headers=REQUEST_HEADERS, timeout=timeout, allow_redirects=True, stream=True)
    except Exception as exc:
        candidate.full_text_verified = False
        candidate.full_text_note = f"Falha no download local do PDF: {exc}"
        return candidate

    try:
        if resp.status_code >= 400:
            candidate.full_text_verified = False
            candidate.full_text_note = f"Download local retornou HTTP {resp.status_code}"
            return candidate
        if not is_probably_pdf_response(resp):
            candidate.full_text_verified = False
            candidate.full_text_note = "O link não retornou um PDF válido para download local."
            return candidate

        total = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    candidate.full_text_verified = False
                    candidate.full_text_note = "PDF excede o tamanho máximo permitido para download local."
                    try:
                        fh.close()
                    except Exception:
                        pass
                    try:
                        dest.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return candidate
                fh.write(chunk)
    finally:
        try:
            resp.close()
        except Exception:
            pass

    try:
        head = dest.read_bytes()[:8]
    except Exception as exc:
        candidate.full_text_verified = False
        candidate.full_text_note = f"PDF baixado, mas não pôde ser lido localmente: {exc}"
        return candidate

    if not head.startswith(b"%PDF"):
        candidate.full_text_verified = False
        candidate.full_text_note = "Arquivo baixado não possui assinatura PDF válida."
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return candidate

    candidate.downloaded_pdf_path = str(dest)
    candidate.downloaded_pdf_note = "PDF baixado localmente com sucesso."
    candidate.full_text_verified = True
    if not candidate.full_text_note:
        candidate.full_text_note = candidate.downloaded_pdf_note
    return candidate


def ensure_candidate_readable(candidate: CandidatePaper, cache_dir: Path) -> CandidatePaper:
    candidate = resolve_full_text_for_candidate(candidate)
    if candidate.full_text_verified and candidate.pdf_url:
        candidate = download_candidate_pdf(candidate, cache_dir)
    return candidate


def enforce_downloadable_full_text(candidates: list[CandidatePaper], cache_dir: Path) -> tuple[list[CandidatePaper], int, list[str]]:
    verified: list[CandidatePaper] = []
    removed = 0
    warnings: list[str] = []
    for candidate in candidates:
        candidate = resolve_full_text_for_candidate(candidate)
        if candidate.full_text_verified and candidate.pdf_url:
            verified.append(candidate)
        else:
            removed += 1
            warnings.append(
                f"Excluído antes da triagem por falta de texto completo verificável: {candidate.title}."
            )
    return verified, removed, warnings

def maybe_filter_years(candidates: list[CandidatePaper], periodo: str) -> tuple[list[CandidatePaper], int]:
    m = re.fullmatch(r"\s*(\d{4})\s*[-–]\s*(\d{4})\s*", periodo)
    if not m:
        return candidates, 0
    start, end = int(m.group(1)), int(m.group(2))
    filtered = [c for c in candidates if c.year is None or (start <= c.year <= end)]
    removed = max(0, len(candidates) - len(filtered))
    return filtered or candidates, removed if filtered else 0


def collect_candidates(config: Config) -> tuple[list[CandidatePaper], SearchAudit]:
    warnings: list[str] = []
    per_source: list[SourceFetchResult] = []
    all_candidates: list[CandidatePaper] = []

    selected = set(config.bases)
    interactive = not config.nao_interativo

    s2_key = (
        prompt_secret_if_missing("SEMANTIC_SCHOLAR_API_KEY", "Chave do Semantic Scholar", required=False, interactive=interactive)
        if "semantic_scholar" in selected
        else os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )
    scopus_key = (
        prompt_secret_if_missing("SCOPUS_API_KEY", "Chave do Scopus/Elsevier", required=False, interactive=interactive)
        if "scopus" in selected
        else os.getenv("SCOPUS_API_KEY")
    )
    scopus_insttoken = (
        prompt_secret_if_missing("SCOPUS_INSTTOKEN", "Insttoken do Scopus", required=False, interactive=interactive)
        if "scopus" in selected
        else os.getenv("SCOPUS_INSTTOKEN")
    )
    wos_key = (
        prompt_secret_if_missing("WOS_API_KEY", "Chave do Web of Science", required=False, interactive=interactive)
        if "web_of_science" in selected
        else os.getenv("WOS_API_KEY")
    )

    for source in config.bases:
        query = config.source_query(source)
        try:
            if source == "semantic_scholar":
                result = fetch_semantic_scholar_candidates(query, config.quantidade_triagem, s2_key)
            elif source == "scopus":
                if not scopus_key:
                    warnings.append("Scopus selecionado, mas SCOPUS_API_KEY não foi informado; base ignorada.")
                    continue
                result = fetch_scopus_candidates(query, config.quantidade_triagem, scopus_key, scopus_insttoken)
            elif source == "web_of_science":
                if not wos_key:
                    warnings.append("Web of Science selecionado, mas WOS_API_KEY não foi informado; base ignorada.")
                    continue
                result = fetch_wos_candidates(query, config.quantidade_triagem, wos_key)
            else:
                warnings.append(f"Base não suportada e ignorada: {source}")
                continue
        except requests.HTTPError as exc:
            warnings.append(f"Falha ao consultar {SOURCE_LABELS.get(source, source)}: {exc}")
            continue
        per_source.append(result)
        warnings.extend(result.warnings)
        all_candidates.extend(result.candidates)

    if not all_candidates:
        raise RuntimeError("Nenhum candidato foi recuperado das bases configuradas.")

    identified_total = len(all_candidates)
    filtered_by_year, removed_by_year = maybe_filter_years(all_candidates, config.periodo)
    deduped, duplicates_removed = dedupe_candidates(filtered_by_year)
    full_text_cache_dir = config.output_dir / f"{config.prefixo}_fulltext_cache"
    verified_candidates, removed_no_full_text, full_text_warnings = enforce_downloadable_full_text(deduped, full_text_cache_dir)
    warnings.extend(full_text_warnings)
    if not verified_candidates:
        raise RuntimeError(
            "Nenhum candidato com texto completo verificável e link de download acessível foi localizado. "
            "Amplie as bases/queries ou priorize fontes com acesso aberto."
        )
    audit = SearchAudit(
        per_source=per_source,
        identified_total=identified_total,
        duplicates_removed=duplicates_removed,
        other_removed=removed_by_year + removed_no_full_text,
        screened_total=len(verified_candidates),
        warnings=warnings,
    )
    return verified_candidates, audit


def triage_with_openai(client: OpenAI, config: Config, candidates: list[CandidatePaper], model_org_text: str) -> TriageOutput:
    class _TriageOutput(TriageOutput):
        pass

    candidate_payload = [c.short_dict() for c in candidates]
    orientacao_extra = f"\n\nOrientações complementares para o arquivo final:\n{config.texto_orientacao_extra[:12000]}" if config.texto_orientacao_extra else ""

    system_prompt = textwrap.dedent(
        f"""
        Você é um assistente acadêmico especializado em metodologia de revisão e redação em português.
        Sua tarefa é organizar a triagem de resultados de busca para uma atividade acadêmica.

        Regras obrigatórias:
        - Trabalhe apenas com os candidatos fornecidos.
        - Todos os candidatos já foram pré-filtrados para garantir disponibilidade de texto completo verificável e link de download acessível.
        - Selecione exatamente 1 artigo final com stage='incluido'.
        - Os demais devem ser classificados como 'excluido_triagem' ou 'excluido_elegibilidade'.
        - Não invente dados que não estejam sustentados pelos dados fornecidos.
        - Nunca selecione um artigo final sem full_text_verified=true e pdf_url/download_url verificável.
        - Faça a redação em português acadêmico claro.
        - Use como referência de tom e organização o modelo Org abaixo.
        - O tema central é: {config.tema}.
        - O recorte é: {config.recorte}.
        - O tipo priorizado é: {config.tipo_estudo}.
        - As bases consultadas foram: {config.bases_label}.

        Modelo Org de referência:
        {model_org_text[:14000]}
        {orientacao_extra}
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        Dados da atividade:
        - Disciplina: {config.disciplina}
        - Professor(a): {config.professor}
        - Curso: {config.curso}
        - Tema: {config.tema}
        - Recorte: {config.recorte}
        - Objetivo: {config.objetivo}
        - Bases: {config.bases_label}
        - Tipo de estudo priorizado: {config.tipo_estudo}
        - Período desejado: {config.periodo}
        - Idiomas aceitáveis: {', '.join(config.idiomas)}
        - Palavras-chave: {', '.join(config.palavras_chave)}
        - Queries por base:
          {json.dumps({src: config.source_query(src) for src in config.bases}, ensure_ascii=False, indent=2)}

        Candidatos recuperados:
        {json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

        Gere:
        1. pergunta_orientadora
        2. introducao
        3. base_logica_busca
        4. criterios_inclusao
        5. criterios_exclusao
        6. observacao_metodologica
        7. decisions (uma por paper_id)
        8. selected_paper_id
        9. selected_paper_justification
        10. Considere obrigatoriamente a existência de link de download verificável do texto completo ao justificar a seleção final.
        """
    ).strip()

    response = client.responses.parse(
        model=config.model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=_TriageOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("A OpenAI não retornou triagem estruturada.")

    ids_from_model = {d.paper_id for d in parsed.decisions}
    ids_expected = {c.paper_id for c in candidates}
    if ids_expected != ids_from_model:
        missing = ids_expected - ids_from_model
        extra = ids_from_model - ids_expected
        raise RuntimeError(f"Triagem inconsistente. Faltando: {sorted(missing)}. Extras: {sorted(extra)}")

    included = [d for d in parsed.decisions if d.stage == "incluido"]
    if len(included) != 1:
        raise RuntimeError("A triagem precisa marcar exatamente 1 artigo como incluído.")
    if included[0].paper_id != parsed.selected_paper_id:
        raise RuntimeError("selected_paper_id não corresponde ao artigo incluído.")
    return parsed


def analyze_selected_paper(client: OpenAI, config: Config, selected: CandidatePaper, model_org_text: str) -> PaperAnalysisOutput:
    class _PaperAnalysisOutput(PaperAnalysisOutput):
        pass

    orientacao_extra = f"\n\nOrientações complementares para o arquivo final:\n{config.texto_orientacao_extra[:12000]}" if config.texto_orientacao_extra else ""

    if not selected.pdf_url:
        raise RuntimeError("O artigo selecionado não possui link verificável para download do texto completo.")

    local_pdf_path = Path(selected.downloaded_pdf_path) if selected.downloaded_pdf_path else None
    if local_pdf_path is None or not local_pdf_path.exists():
        cache_dir = config.output_dir / f"{config.prefixo}_fulltext_cache"
        selected = download_candidate_pdf(selected, cache_dir)
        local_pdf_path = Path(selected.downloaded_pdf_path) if selected.downloaded_pdf_path else None
    if local_pdf_path is None or not local_pdf_path.exists():
        raise RuntimeError(selected.full_text_note or "O artigo selecionado não pôde ser baixado localmente para leitura.")

    try:
        extracted_pdf_text = read_orientation_file(local_pdf_path, max_chars=120000)
    except Exception as exc:
        raise RuntimeError(f"O artigo selecionado foi baixado, mas não pôde ser lido localmente: {exc}") from exc

    content: list[dict] = [
        {
            "type": "input_text",
            "text": textwrap.dedent(
                f"""
                Analise o artigo selecionado para compor uma atividade acadêmica em português.
                Tema central: {config.tema}
                Recorte: {config.recorte}
                Objetivo: {config.objetivo}
                Bases consultadas: {config.bases_label}
                Estilo de citações e referências bibliográficas: {config.estilo_citacao}

                Metadados do artigo:
                {json.dumps(selected.short_dict(), ensure_ascii=False, indent=2)}

                Use como referência de tom e organização este modelo Org:
                {model_org_text[:14000]}
                {orientacao_extra}

                Gere os campos estruturados solicitados, mantendo rigor metodológico.
                O campo referencia_completa deve vir formatado estritamente no estilo {config.estilo_citacao}.
                O artigo selecionado só pode chegar a esta etapa se houver link verificável para download do texto completo.

                Texto extraído do PDF baixado localmente (trecho):
                {extracted_pdf_text[:100000]}
                """
            ).strip(),
        }
    ]

    response = client.responses.parse(
        model=config.model,
        input=[{"role": "user", "content": content}],
        text_format=_PaperAnalysisOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("A OpenAI não retornou análise estruturada do artigo.")
    return parsed


def compute_prisma_counts(audit: SearchAudit, triage: TriageOutput) -> PrismaCounts:
    excluded_screening = sum(1 for d in triage.decisions if d.stage == "excluido_triagem")
    included = sum(1 for d in triage.decisions if d.stage == "incluido")
    excluded_full = sum(1 for d in triage.decisions if d.stage == "excluido_elegibilidade")
    full_text_assessed = included + excluded_full
    full_text_sought = full_text_assessed
    not_retrieved = 0
    return PrismaCounts(
        identified=audit.identified_total,
        removed_pre=audit.duplicates_removed + audit.other_removed,
        duplicates_removed=audit.duplicates_removed,
        other_removed=audit.other_removed,
        screened=audit.screened_total,
        excluded_screening=excluded_screening,
        full_text_sought=full_text_sought,
        not_retrieved=not_retrieved,
        full_text_assessed=full_text_assessed,
        excluded_full_text=excluded_full,
        included_qualitative=included,
    )


def author_list(authors: list[str]) -> str:
    return "; ".join(authors) if authors else "Autor(es) não informado(s)"


def latex_escape(s: str) -> str:
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in s)


def build_source_summary(audit: SearchAudit) -> str:
    lines = []
    for item in audit.per_source:
        lines.append(f"- {SOURCE_LABELS.get(item.source, item.source)}: *n = {item.retrieved}* registros recuperados")
        lines.append(f"  - Query usada: ={item.query}=")
    return "\n".join(lines)


def build_prisma_flow_text(config: Config, audit: SearchAudit, triage: TriageOutput, counts: PrismaCounts) -> str:
    excl_triagem = [d for d in triage.decisions if d.stage == "excluido_triagem"]
    excl_eleg = [d for d in triage.decisions if d.stage == "excluido_elegibilidade"]
    lines = []
    lines.append("*** Identificação")
    lines.append(f"- Registros identificados nas bases consultadas ({config.bases_label}): *n = {counts.identified}*")
    lines.append(build_source_summary(audit))
    lines.append(f"- Registros removidos antes da triagem: *n = {counts.removed_pre}*")
    lines.append(f"  - Duplicatas removidas: *n = {counts.duplicates_removed}*")
    lines.append(f"  - Removidos por outros motivos: *n = {counts.other_removed}*")
    lines.append("")
    lines.append("*** Triagem")
    lines.append(f"- Registros triados por título, resumo e metadados: *n = {counts.screened}*")
    lines.append(f"- Registros excluídos na triagem: *n = {counts.excluded_screening}*")
    for d in excl_triagem:
        lines.append(f"  - {d.motivo}: *n = 1*")
    lines.append("")
    lines.append("*** Elegibilidade")
    lines.append(f"- Relatórios buscados para recuperação em texto completo: *n = {counts.full_text_sought}*")
    lines.append(f"- Relatórios não recuperados: *n = {counts.not_retrieved}*")
    lines.append(f"- Relatórios avaliados para elegibilidade: *n = {counts.full_text_assessed}*")
    lines.append(f"- Relatórios excluídos após leitura do texto completo: *n = {counts.excluded_full_text}*")
    for d in excl_eleg:
        lines.append(f"  - {d.motivo}: *n = 1*")
    lines.append("")
    lines.append("*** Inclusão")
    lines.append(f"- Estudos incluídos na síntese qualitativa final: *n = {counts.included_qualitative}*")
    return "\n".join(lines)


def build_prisma_table_latex(counts: PrismaCounts) -> str:
    rows = [
        ("Registros identificados", counts.identified),
        ("Registros removidos antes da triagem", counts.removed_pre),
        ("Registros triados", counts.screened),
        ("Registros excluídos na triagem", counts.excluded_screening),
        ("Relatórios buscados para recuperação", counts.full_text_sought),
        ("Relatórios não recuperados", counts.not_retrieved),
        ("Relatórios avaliados para elegibilidade", counts.full_text_assessed),
        ("Relatórios excluídos após leitura completa", counts.excluded_full_text),
        ("Estudos incluídos na síntese qualitativa", counts.included_qualitative),
    ]
    body = "\n".join(f"#+LATEX: {latex_escape(label)} & {value} \\\\" for label, value in rows)
    return textwrap.dedent(
        fr"""
        #+NAME: tab:prisma2020
        #+CAPTION: Síntese do fluxo PRISMA 2020 aplicado à seleção do artigo.
        #+ATTR_LATEX: :center nil :float nil
        #+LATEX: \begingroup
        #+LATEX: \centering
        #+LATEX: \scriptsize
        #+LATEX: \setlength{{\tabcolsep}}{{4pt}}
        #+LATEX: \renewcommand{{\arraystretch}}{{1.15}}
        #+LATEX: \captionof{{table}}{{Síntese do fluxo PRISMA 2020 aplicado à seleção do artigo.}}
        #+LATEX: \label{{tab:prisma2020}}
        #+LATEX: \begin{{tabularx}}{{\textwidth}}{{>{{\raggedright\arraybackslash}}X >{{\raggedleft\arraybackslash}}m{{0.18\textwidth}}}}
        #+LATEX: \toprule
        #+LATEX: Etapa PRISMA 2020 & n \\
        #+LATEX: \midrule
        {body}
        #+LATEX: \bottomrule
        #+LATEX: \end{{tabularx}}
        #+LATEX: \normalsize
        #+LATEX: \endgroup
        """
    ).strip()


def condense_reason(texto: str, max_chars: int = 140) -> str:
    texto = re.sub(r"\s+", " ", (texto or "").strip())
    if not texto:
        return "—"
    primeira_frase = texto.split('. ')[0].strip()
    if primeira_frase and len(primeira_frase) <= max_chars:
        return primeira_frase.rstrip('.') + '.'
    if len(texto) <= max_chars:
        return texto
    return textwrap.shorten(texto, width=max_chars, placeholder="…")


def build_studies_table(candidates: list[CandidatePaper], triage: TriageOutput) -> str:
    decisions = {d.paper_id: d for d in triage.decisions}
    order_map = {"incluido": 0, "excluido_elegibilidade": 1, "excluido_triagem": 2}
    rows = []
    for c in sorted(candidates, key=lambda x: (order_map[decisions[x.paper_id].stage], -(x.year or 0), x.title)):
        d = decisions[c.paper_id]
        situacao = {
            "incluido": "Incluído",
            "excluido_elegibilidade": "Excluído na elegibilidade",
            "excluido_triagem": "Excluído na triagem",
        }[d.stage]
        title = latex_escape(c.title)
        authors = latex_escape(author_list(c.authors))
        year = c.year or "s.d."
        tipo = latex_escape(d.tipo_estudo)
        motivo = latex_escape(condense_reason(d.motivo, max_chars=150))
        bases = latex_escape(", ".join(SOURCE_LABELS.get(src, src) for src in c.sources))
        estudo = f"{authors} ({year}), \\textit{{{title}}}\\\\ \\emph{{Base(s): {bases}}}"
        rows.append(f"#+LATEX: {estudo} & {situacao} & {tipo} & {motivo} \\")
    body = "\n".join(rows)
    return textwrap.dedent(
        fr"""
        #+NAME: tab:estudos_fluxo
        #+CAPTION: Classificação sintética dos estudos identificados no fluxo PRISMA 2020.
        #+ATTR_LATEX: :center nil :float nil
        #+LATEX: \begingroup
        #+LATEX: \scriptsize
        #+LATEX: \setlength{{\tabcolsep}}{{3pt}}
        #+LATEX: \renewcommand{{\arraystretch}}{{1.15}}
        #+LATEX: \begin{{longtable}}{{>{{\raggedright\arraybackslash}}p{{0.36\textwidth}} >{{\raggedright\arraybackslash}}p{{0.18\textwidth}} >{{\raggedright\arraybackslash}}p{{0.14\textwidth}} >{{\raggedright\arraybackslash}}p{{0.24\textwidth}}}}
        #+LATEX: \caption{{Classificação sintética dos estudos identificados no fluxo PRISMA 2020.}}\label{{tab:estudos_fluxo}}\\
        #+LATEX: \toprule
        #+LATEX: Estudo & Situação no fluxo & Tipo & Motivo \\ 
        #+LATEX: \midrule
        #+LATEX: \endfirsthead
        #+LATEX: \toprule
        #+LATEX: Estudo & Situação no fluxo & Tipo & Motivo \\ 
        #+LATEX: \midrule
        #+LATEX: \endhead
        #+LATEX: \midrule
        #+LATEX: \multicolumn{{4}}{{r}}{{Continua na próxima página}} \\ 
        #+LATEX: \endfoot
        #+LATEX: \bottomrule
        #+LATEX: \endlastfoot
        {body}
        #+LATEX: \end{{longtable}}
        #+LATEX: \normalsize
        #+LATEX: \endgroup
        """
    ).strip()


def build_ficha_tecnica(config: Config, titulo_trabalho: str) -> str:
    data_documento = today_pt_br()
    linhas = [
        "\\textbf{Curso}                   & " + latex_escape(config.curso) + r" \\",
        "\\textbf{Turma}                   & " + latex_escape(config.turma) + r" \\",
        "\\textbf{Pólo}                    & " + latex_escape(config.polo) + r" \\",
        "\\textbf{Disciplina}              & " + latex_escape(config.disciplina) + r" \\",
        "\\textbf{Professor}               & " + latex_escape(config.professor) + r" \\",
        "\\textbf{Aluno(s)}                & " + latex_escape(config.aluno) + r" \\",
        "\\textbf{Data}                    & " + latex_escape(data_documento) + r" \\",
        "\\textbf{Título do trabalho}      & " + latex_escape(titulo_trabalho) + r" \\",
    ]
    tabela = "\n".join(linhas)
    bloco = [
        "#+begin_export latex",
        r"\begingroup",
        r"\linespread{1}\selectfont",
        r"\begin{tcolorbox}[title=Ficha Técnica,",
        r"  colback=gray!5,colframe=gray!40,boxrule=0.4pt,sharp corners]",
        r"\begin{tblr}{rowsep=1pt,stretch=1, rows={t}, colspec={Q[l,2.8cm] X[l]}}",
        tabela,
        r"\end{tblr}",
        r"\end{tcolorbox}",
        r"\endgroup",
        "#+end_export",
    ]
    return "\n".join(bloco)


def extract_org_header_from_model(model_text: str) -> str:
    lines = []
    for line in model_text.splitlines():
        if line.startswith("* Introdução"):
            break
        if line.startswith("#+"):
            lines.append(line)

    if not lines:
        lines = [
            "#+OPTIONS: toc:nil num:t H:3 ^:nil",
            "#+STARTUP: overview",
            "#+LATEX_HEADER: \\usepackage{fgv-header}",
            "#+LATEX_HEADER: \\usepackage{float}",
            "#+LATEX_HEADER: \\usepackage{graphicx}",
            "#+LATEX_HEADER: \\usepackage{caption}",
        ]

    required_headers = [
        "#+LATEX_HEADER: \\usepackage{graphicx}",
        "#+LATEX_HEADER: \\usepackage{float}",
        "#+LATEX_HEADER: \\usepackage{caption}",
        "#+LATEX_HEADER: \\usepackage{booktabs}",
        "#+LATEX_HEADER: \\usepackage{tabularx}",
        "#+LATEX_HEADER: \\usepackage{array}",
        "#+LATEX_HEADER: \\usepackage{longtable}",
        "#+LATEX_HEADER: \\usepackage{ltablex}",
        "#+LATEX_HEADER: \\keepXColumns",
        "#+LATEX_HEADER: \\setlength{\\emergencystretch}{3em}",
        "#+LATEX_HEADER: \\sloppy",
    ]
    existing = set(lines)
    for header in required_headers:
        if header not in existing:
            lines.append(header)
    return "\n".join(lines).strip()


def citation_style_to_cite_export(style: str) -> str:
    style = normalize_citation_style(style)
    mapping = {
        "ABNT": "#+CITE_EXPORT: biblatex backend=biber,style=abnt,sortcites=true,sorting=nyt",
        "APA": "#+CITE_EXPORT: biblatex backend=biber,style=apa,sortcites=true,sorting=nyt,giveninits=true,maxcitenames=2,maxbibnames=20,uniquelist=minyear",
        "Chicago": "#+CITE_EXPORT: biblatex backend=biber,style=chicago-authordate,sortcites=true,sorting=nyt",
        "MLA": "#+CITE_EXPORT: biblatex backend=biber,style=mla,sortcites=true,sorting=nyt",
        "Vancouver": "#+CITE_EXPORT: biblatex backend=biber,style=vancouver,sortcites=true,sorting=none",
    }
    return mapping.get(style, mapping["ABNT"])


def extract_bibliography_line(model_text: str) -> str:
    for line in model_text.splitlines():
        if line.strip().lower().startswith("#+bibliography:"):
            return line.strip()
    return "#+BIBLIOGRAPHY: referencias.bib"


def sanitize_model_header(header: str) -> str:
    cleaned: list[str] = []
    blocked_substrings = [
        r"\usepapercover",
        r"\usepackage{fgv-header}",
        r"\usepackage{svg}",
        r"\institution{",
        r"\programname{",
        r"\coursename{",
        r"\disciplinename{",
        r"\professorname{",
        r"\cityname{",
        r"\papertype{",
        r"\covernote{",
    ]
    for line in header.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("#+title:") or lowered.startswith("#+author:") or lowered.startswith("#+date:"):
            continue
        if any(item in stripped for item in blocked_substrings):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def build_fgv_report_header_lines(logo_path: Path | str) -> list[str]:
    logo_file = Path(logo_path).expanduser() if str(logo_path).strip() else Path(DEFAULT_FGV_LOGO_PATH).expanduser()
    logo_exists = logo_file.exists()
    logo_str = logo_file.as_posix()
    logo_include = rf"\includegraphics[height=0.95cm]{{{logo_str}}}" if logo_exists else r"\rule{0pt}{0.95cm}"
    return [
        r"#+LATEX_HEADER: \definecolor{FGVHeaderBlueDark}{HTML}{003A70}",
        r"#+LATEX_HEADER: \definecolor{FGVHeaderBlueMid}{HTML}{005AA9}",
        r"#+LATEX_HEADER: \definecolor{FGVHeaderBlueLight}{HTML}{66A9E0}",
        r"#+LATEX_HEADER: \setlength{\headheight}{48pt}",
        r"#+LATEX_HEADER: \setlength{\headsep}{12pt}",
        rf"#+LATEX_HEADER: \newcommand{{\FGVHeaderLogoInclude}}{{{logo_include}}}",
        r"#+LATEX_HEADER: \newcommand{\FGVHeaderGradientRule}{\begin{tikzpicture}[baseline=(current bounding box.center)]\shade[left color=FGVHeaderBlueDark,middle color=FGVHeaderBlueMid,right color=FGVHeaderBlueLight] (0,0) rectangle (\headwidth,1.2pt);\end{tikzpicture}}",
        r"#+LATEX_HEADER: \newcommand{\FGVHeaderBlock}{\begin{minipage}[b]{\headwidth}\raggedright \FGVHeaderLogoInclude\par\vspace{0.18em}\noindent\FGVHeaderGradientRule\end{minipage}}",
        r"#+LATEX_HEADER: \fancypagestyle{fgvreportstyle}{%",
        r"#+LATEX_HEADER: \fancyhf{}%",
        r"#+LATEX_HEADER: \fancyhead[L]{\makebox[\headwidth][l]{\FGVHeaderBlock}}%",
        r"#+LATEX_HEADER: \fancyhead[C]{}%",
        r"#+LATEX_HEADER: \fancyhead[R]{}%",
        r"#+LATEX_HEADER: \renewcommand{\headrulewidth}{0pt}%",
        r"#+LATEX_HEADER: \fancyfoot[C]{\thepage}%",
        r"#+LATEX_HEADER: }",
        r"#+LATEX_HEADER: \fancypagestyle{plain}{%",
        r"#+LATEX_HEADER: \fancyhf{}%",
        r"#+LATEX_HEADER: \fancyhead[L]{\makebox[\headwidth][l]{\FGVHeaderBlock}}%",
        r"#+LATEX_HEADER: \fancyhead[C]{}%",
        r"#+LATEX_HEADER: \fancyhead[R]{}%",
        r"#+LATEX_HEADER: \renewcommand{\headrulewidth}{0pt}%",
        r"#+LATEX_HEADER: \fancyfoot[C]{\thepage}%",
        r"#+LATEX_HEADER: }",
        r"#+LATEX_HEADER: \AtBeginDocument{\pagestyle{fgvreportstyle}\thispagestyle{fgvreportstyle}}",
    ]


def prepare_model_header(model_org_text: str, estilo_citacao: str, bib_filename: str, fgv_logo_path: Path | str) -> str:
    header = sanitize_model_header(extract_org_header_from_model(model_org_text))
    parts: list[str] = []
    for p in header.splitlines():
        lowered = p.strip().lower()
        if lowered.startswith("#+cite_export:") or lowered.startswith("#+bibliography:") or lowered.startswith("#+print_bibliography:"):
            continue
        if p.strip():
            parts.append(p)
    parts.extend(build_fgv_report_header_lines(fgv_logo_path))
    parts.append(citation_style_to_cite_export(estilo_citacao))
    parts.append(f"#+BIBLIOGRAPHY: {bib_filename}")
    return "\n".join(parts).strip()


def bibtex_escape(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace('\\', r'\textbackslash{}')
    for old, new in [("{", r"\{"), ("}", r"\}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        value = value.replace(old, new)
    return value


def make_bibtex_key(candidate: CandidatePaper) -> str:
    surname = "anon"
    if candidate.authors:
        first = candidate.authors[0].strip()
        if first:
            surname = slugify(first.split()[-1]) or "anon"
    year = str(candidate.year or "nodate")
    first_word = slugify((candidate.title or "paper").split()[0]) or "paper"
    return f"{surname}{year}{first_word}"


def build_bibtex_for_candidate(candidate: CandidatePaper) -> tuple[str, str]:
    key = make_bibtex_key(candidate)
    authors = " and ".join(a.strip() for a in candidate.authors if a and a.strip()) or "Autor desconhecido"
    entry_type = "article" if candidate.venue else "misc"
    fields: list[tuple[str, str]] = [
        ("author", authors),
        ("title", candidate.title or "Sem título"),
    ]
    if candidate.venue:
        fields.append(("journaltitle", candidate.venue))
    if candidate.year:
        fields.append(("year", str(candidate.year)))
    if candidate.publication_date:
        fields.append(("date", candidate.publication_date))
    if candidate.doi:
        fields.append(("doi", candidate.doi))
    if candidate.url:
        fields.append(("url", candidate.url))
    if candidate.pdf_url and candidate.pdf_url != candidate.url:
        fields.append(("url", candidate.pdf_url))
    body = "\n".join(f"  {name} = {{{bibtex_escape(value)}}}," for name, value in fields if value)
    return key, f"@{entry_type}{{{key},\n{body}\n}}"


def build_static_prisma_bibtex() -> tuple[str, str]:
    key = "prisma2020statement"
    entry = """@online{prisma2020statement,
  author = {Page, Matthew J. and McKenzie, Joanne E. and Bossuyt, Patrick M. and Boutron, Isabelle and Hoffmann, Tammy C. and Mulrow, Cynthia D. and Shamseer, Larissa and Tetzlaff, Jennifer M. and Akl, Elie A. and Brennan, Sue E. and Chou, Roger and Glanville, Julie and Grimshaw, Jeremy M. and Hróbjartsson, Asbjørn and Lalu, Manoj M. and Li, Tianjing and Loder, Elizabeth W. and Mayo-Wilson, Evan and McDonald, Steve and McGuinness, Luke A. and Stewart, Lesley A. and Thomas, James and Tricco, Andrea C. and Welch, Vivian A. and Whiting, Penny and Moher, David},
  title = {The PRISMA 2020 statement: An updated guideline for reporting systematic reviews},
  year = {2021},
  url = {https://www.prisma-statement.org/prisma-2020},
  note = {Acesso em: %s}
}""" % bibtex_escape(today_pt_br())
    return key, entry


def save_bib_file(output_dir: Path, prefixo: str, selected: CandidatePaper) -> tuple[Path, str, str]:
    bib_path = output_dir / f"{prefixo}.bib"
    selected_key, selected_entry = build_bibtex_for_candidate(selected)
    prisma_key, prisma_entry = build_static_prisma_bibtex()
    write_text(bib_path, selected_entry.strip() + "\n\n" + prisma_entry.strip() + "\n")
    return bib_path, selected_key, prisma_key


def normalize_org_text(text: str) -> str:
    fixed_lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#+") or stripped.startswith("*"):
            fixed_lines.append(stripped)
        else:
            fixed_lines.append(raw.rstrip())
    fixed = "\n".join(fixed_lines)
    fixed = fixed.replace("#+begin_export latex\n#+end_export\n", "")
    return fixed.strip() + "\n"


def build_org_document(config: Config, model_org_text: str, triage: TriageOutput, analysis: PaperAnalysisOutput,
                       candidates: list[CandidatePaper], audit: SearchAudit, counts: PrismaCounts,
                       figure_pdf_filename: str, bib_filename: str, selected_bib_key: str, prisma_bib_key: str) -> str:
    selected = next(c for c in candidates if c.paper_id == triage.selected_paper_id)
    titulo_trabalho = (config.titulo_trabalho or fallback_work_title(config)).strip()
    header = prepare_model_header(model_org_text, config.estilo_citacao, bib_filename, config.fgv_logo_path)
    ficha = build_ficha_tecnica(config, titulo_trabalho)
    prisma_text = build_prisma_flow_text(config, audit, triage, counts)
    prisma_table = build_prisma_table_latex(counts)
    studies_table = build_studies_table(candidates, triage)

    resumo_section = []
    if config.incluir_resumo_artigo_ia:
        resumo_section = [
            "* Resumo do artigo selecionado",
            "** Referência do estudo",
            f"A referência do estudo selecionado corresponde a [cite:@{selected_bib_key}].",
            analysis.referencia_completa.strip(),
            "** Link para download do texto completo",
            f"[[{selected.pdf_url}][Download do PDF]]" if selected.pdf_url else "Link de download não disponível.",
            "** Problema e objetivo de pesquisa",
            analysis.problema_objetivo.strip(),
            "** Argumento central",
            analysis.argumento_central.strip(),
            "** Desenho de pesquisa",
            analysis.desenho_pesquisa.strip(),
            "** Principais achados",
            analysis.principais_achados.strip(),
            "** Justificativa da seleção final",
            analysis.justificativa_selecao_final.strip(),
            "** Contribuição do estudo para o tema",
            analysis.contribuicao_estudo.strip(),
        ]

    warnings_block = ""
    if audit.warnings:
        warnings_block = "\n".join(f"- {item}" for item in audit.warnings)
        warnings_block = f"** Observações de execução\n\n{warnings_block}"

    sections = [
        header,
        ficha,
        "#+LATEX: \\clearpage",
        "* Introdução",
        triage.introducao.strip(),
        f'O artigo selecionado ao final do processo foi *“{selected.title}”*. A escolha final se justifica porque {triage.selected_paper_justification}'.strip(),
        f"Também é importante registrar uma distinção metodológica. Neste trabalho, o *PRISMA 2020* foi utilizado para organizar o *fluxo da busca e da seleção* nas bases consultadas [cite:@{prisma_bib_key}]. O artigo selecionado foi analisado a partir de seus metadados e, obrigatoriamente, de um *texto completo com link verificável para download* [cite:@{selected_bib_key}].",
        "* Estratégia de busca",
        "** Pergunta orientadora",
        "#+begin_quote\n" + triage.pergunta_orientadora.strip() + "\n#+end_quote",
        "** Bases consultadas",
        f"- {config.bases_label}",
        "** Base e lógica de busca",
        triage.base_logica_busca.strip(),
        "** Queries por base",
        "\n".join(f'- *{SOURCE_LABELS.get(src, src)}*: ={config.source_query(src)}=' for src in config.bases),
        "** Critérios de inclusão",
        "\n".join(f"- {item}" for item in triage.criterios_inclusao),
        "** Critérios de exclusão",
        "\n".join(f"- {item}" for item in triage.criterios_exclusao),
    ]
    if warnings_block:
        sections.append(warnings_block)
    sections.extend([
        "* PRISMA 2020",
        "** Observação metodológica",
        triage.observacao_metodologica.strip(),
        "** Fluxo textual dos estudos",
        prisma_text,
        "** Tabela-resumo do fluxo PRISMA 2020",
        prisma_table,
        "** Estudos classificados no fluxo",
        studies_table,
        "#+LATEX: \\clearpage",
        "** Representação esquemática do fluxo PRISMA 2020",
        "O fluxograma PRISMA 2020 é gerado como *SVG externo* em layout compacto, exibindo apenas as contagens de cada etapa. Os motivos detalhados de exclusão permanecem no fluxo textual e nas tabelas, para evitar sobrecarga visual na figura.",
        "\n".join([
            "#+begin_export latex",
            "\\begin{center}",
            f"\\includegraphics[width=0.78\\textwidth]{{{figure_pdf_filename}}}",
            f"\\captionof{{figure}}{{Fluxograma PRISMA 2020 da seleção do artigo de revisão sobre {config.tema}.}}",
            "\\label{fig:prisma2020_fluxo}",
            "\\end{center}",
            "#+end_export",
        ]),
        *resumo_section,
        "* Texto corrido para entrega",
        analysis.texto_corrido_entrega.strip(),
        "#+PRINT_BIBLIOGRAPHY:",
    ])
    doc = "\n\n".join(s for s in sections if s and s.strip())
    return normalize_org_text(doc)


# =========================
# SVG PRISMA
# =========================

def wrap_lines(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)


@dataclass
class BoxSpec:
    x: int
    y: int
    width: int
    title_lines: list[str]
    body_lines: list[str]
    font_size: int = 24
    line_height: int = 30
    pad_x: int = 22
    pad_top: int = 22
    pad_bottom: int = 22

    @property
    def height(self) -> int:
        total_lines = len(self.title_lines) + len(self.body_lines)
        return self.pad_top + self.pad_bottom + total_lines * self.line_height

    @property
    def bottom(self) -> int:
        return self.y + self.height


def tspan_block(lines: list[str], x_center: float, line_height: int, bold_count: int) -> str:
    out = []
    for idx, line in enumerate(lines):
        weight = " font-weight='700'" if idx < bold_count else ""
        dy = 0 if idx == 0 else line_height
        out.append(f"<tspan x='{x_center:.1f}' dy='{dy}'{weight}>{escape(line)}</tspan>")
    return "".join(out)


def render_box(spec: BoxSpec) -> str:
    x_center = spec.x + spec.width / 2
    start_y = spec.y + spec.pad_top + spec.font_size
    lines = spec.title_lines + spec.body_lines
    text_block = tspan_block(lines, x_center, spec.line_height, len(spec.title_lines))
    return (
        f"<rect x='{spec.x}' y='{spec.y}' width='{spec.width}' height='{spec.height}' "
        f"rx='6' ry='6' fill='white' stroke='#1f1f1f' stroke-width='2.2'/>"
        f"<text x='{x_center:.1f}' y='{start_y:.1f}' text-anchor='middle' "
        f"font-family='Arial, Helvetica, sans-serif' font-size='{spec.font_size}' fill='#111'>{text_block}</text>"
    )


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
        f"stroke='#1f1f1f' stroke-width='2.4' marker-end='url(#arrowhead)'/>"
    )


def build_side_reason_box(title: str, items: list[str], x: int, y: int, width: int) -> BoxSpec:
    title_lines = wrap_lines(title, 32)
    body_lines: list[str] = []
    for item in items:
        wrapped = wrap_lines(item, 34)
        body_lines.extend(wrapped)
    return BoxSpec(x=x, y=y, width=width, title_lines=title_lines, body_lines=body_lines)


def build_main_box(title: str, n: int, x: int, y: int, width: int) -> BoxSpec:
    lines = wrap_lines(title, 34)
    body = [f"(n = {n})"]
    return BoxSpec(x=x, y=y, width=width, title_lines=lines, body_lines=body)


def build_svg_prisma(config: Config, audit: SearchAudit, triage: TriageOutput, counts: PrismaCounts, output_svg: Path) -> None:
    del config, audit, triage

    left_x = 120
    right_x = 760
    box_w = 420
    gap_y = 44
    start_y = 88

    def compact_box(title: str, n: int, x: int, y: int) -> BoxSpec:
        title_lines = wrap_lines(title, 28)
        return BoxSpec(
            x=x,
            y=y,
            width=box_w,
            title_lines=title_lines,
            body_lines=[f"(n = {n})"],
            font_size=22,
            line_height=28,
            pad_x=18,
            pad_top=18,
            pad_bottom=18,
        )

    identified = compact_box("Registros identificados nas bases de dados", counts.identified, left_x, start_y)
    removed = compact_box("Registros removidos antes da triagem", counts.removed_pre, right_x, start_y)

    y2 = max(identified.bottom, removed.bottom) + gap_y
    screened = compact_box("Registros triados", counts.screened, left_x, y2)
    excluded_screen = compact_box("Registros excluídos na triagem", counts.excluded_screening, right_x, y2)

    y3 = max(screened.bottom, excluded_screen.bottom) + gap_y
    sought = compact_box("Relatórios buscados para recuperação", counts.full_text_sought, left_x, y3)
    not_retrieved = compact_box("Relatórios não recuperados", counts.not_retrieved, right_x, y3)

    y4 = max(sought.bottom, not_retrieved.bottom) + gap_y
    assessed = compact_box("Relatórios avaliados para elegibilidade", counts.full_text_assessed, left_x, y4)
    excluded_full = compact_box("Relatórios excluídos após leitura completa", counts.excluded_full_text, right_x, y4)

    y5 = max(assessed.bottom, excluded_full.bottom) + gap_y
    included = compact_box("Estudos incluídos na síntese qualitativa", counts.included_qualitative, left_x, y5)

    boxes = [identified, removed, screened, excluded_screen, sought, not_retrieved, assessed, excluded_full, included]

    svg_width = 1320
    svg_height = included.bottom + 90

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{svg_width}' height='{svg_height}' viewBox='0 0 {svg_width} {svg_height}'>",
        "<defs><marker id='arrowhead' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto'><polygon points='0 0, 10 4, 0 8' fill='#1f1f1f'/></marker></defs>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        "<text x='120' y='42' font-family='Arial, Helvetica, sans-serif' font-size='28' font-weight='700' fill='#111'>Fluxograma PRISMA 2020</text>",
    ]

    for spec in boxes:
        svg_parts.append(render_box(spec))

    svg_parts.append(arrow(identified.x + identified.width, identified.y + identified.height / 2, removed.x, removed.y + removed.height / 2))
    svg_parts.append(arrow(identified.x + identified.width / 2, identified.bottom, screened.x + screened.width / 2, screened.y))
    svg_parts.append(arrow(screened.x + screened.width, screened.y + screened.height / 2, excluded_screen.x, excluded_screen.y + excluded_screen.height / 2))
    svg_parts.append(arrow(screened.x + screened.width / 2, screened.bottom, sought.x + sought.width / 2, sought.y))
    svg_parts.append(arrow(sought.x + sought.width, sought.y + sought.height / 2, not_retrieved.x, not_retrieved.y + not_retrieved.height / 2))
    svg_parts.append(arrow(sought.x + sought.width / 2, sought.bottom, assessed.x + assessed.width / 2, assessed.y))
    svg_parts.append(arrow(assessed.x + assessed.width, assessed.y + assessed.height / 2, excluded_full.x, excluded_full.y + excluded_full.height / 2))
    svg_parts.append(arrow(assessed.x + assessed.width / 2, assessed.bottom, included.x + included.width / 2, included.y))

    svg_parts.append("</svg>")
    write_text(output_svg, "\n".join(svg_parts))


def save_debug_json(output_dir: Path, prefixo: str, data: dict) -> Path:
    path = output_dir / f"{prefixo}_debug.json"
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
    return path


def main() -> int:
    env_path = load_local_env_file()
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logging(config.output_dir, config.prefixo)
    if env_path:
        logger.info("Arquivo .env carregado com prioridade: %s", env_path)
    else:
        logger.info("Nenhum .env encontrado ao lado do script; usando ambiente atual.")
    logger.info("Iniciando geração da atividade.")
    logger.info("Bases selecionadas: %s", config.bases_label)
    logger.info("Tema: %s | Recorte: %s", config.tema, config.recorte)
    logger.info("Queries bilíngues: %s", "sim" if config.query_bilingue else "não")
    if config.arquivo_orientacao:
        logger.info("Arquivo de orientação complementar: %s", config.arquivo_orientacao)

    if config.gerar_env_example:
        env_example_path = config.output_dir / ".env.example"
        write_env_example(env_example_path)
        logger.info("Arquivo .env.example gerado em %s", env_example_path)
    else:
        env_example_path = None

    model_org_text = load_model_org_text(config.org_modelo)
    client = make_openai_client()
    config.titulo_trabalho = suggest_and_confirm_work_title(client, config, model_org_text, interactive=not config.nao_interativo)
    logger.info("Título do trabalho: %s", config.titulo_trabalho)
    candidates, audit = collect_candidates(config)
    logger.info("Registros identificados: %s | Após filtros/deduplicação: %s", audit.identified_total, audit.screened_total)
    for item in audit.per_source:
        logger.info("Fonte %s | query=%s | recuperados=%s | warnings=%s", SOURCE_LABELS.get(item.source, item.source), item.query, item.retrieved, '; '.join(item.warnings) if item.warnings else 'nenhum')
    if not candidates:
        raise RuntimeError("Nenhum candidato permaneceu após filtros e deduplicação.")

    source_log_paths = save_source_logs(config.output_dir, config.prefixo, audit)
    raw_json_paths = save_raw_search_jsons(config.output_dir, config.prefixo, audit) if config.salvar_busca_bruta_json else []
    if raw_json_paths:
        logger.info("Busca bruta salva em JSON por fonte.")

    working_candidates = list(candidates)
    triage = None
    selected = None
    analysis = None
    while working_candidates:
        triage = triage_with_openai(client, config, working_candidates, model_org_text)
        selected = next(c for c in working_candidates if c.paper_id == triage.selected_paper_id)
        logger.info("Artigo selecionado pela triagem: %s", selected.title)
        logger.info("Link verificável para download do texto completo: %s", selected.pdf_url)
        try:
            analysis = analyze_selected_paper(client, config, selected, model_org_text)
            break
        except Exception as exc:
            msg = str(exc)
            logger.warning("O artigo selecionado falhou na leitura local/análise e será excluído da seleção final: %s | motivo: %s", selected.title, msg)
            audit.warnings.append(f"Excluído após tentativa de leitura local do PDF: {selected.title}. Motivo: {msg}")
            audit.screened_total = max(0, audit.screened_total - 1)
            audit.other_removed += 1
            working_candidates = [c for c in working_candidates if c.paper_id != selected.paper_id]
    if triage is None or selected is None or analysis is None:
        raise RuntimeError("Nenhum artigo com texto completo realmente baixável e lível permaneceu após as tentativas de seleção.")
    counts = compute_prisma_counts(audit, triage)

    bib_path, selected_bib_key, prisma_bib_key = save_bib_file(config.output_dir, config.prefixo, selected)
    logger.info("Arquivo BibTeX gerado em %s", bib_path)

    svg_filename = f"{config.prefixo}_prisma.svg"
    figure_pdf_filename = f"{config.prefixo}_prisma.pdf"
    org_filename = f"{config.prefixo}.org"
    svg_path = config.output_dir / svg_filename
    figure_pdf_path = config.output_dir / figure_pdf_filename
    org_path = config.output_dir / org_filename

    build_svg_prisma(config, audit, triage, counts, svg_path)
    logger.info("SVG PRISMA gerado em %s", svg_path)
    convert_svg_to_pdf(svg_path, logger)
    org_content = build_org_document(
        config=config,
        model_org_text=model_org_text,
        triage=triage,
        analysis=analysis,
        candidates=candidates,
        audit=audit,
        counts=counts,
        figure_pdf_filename=figure_pdf_filename,
        bib_filename=bib_path.name,
        selected_bib_key=selected_bib_key,
        prisma_bib_key=prisma_bib_key,
    )
    write_text(org_path, org_content)
    logger.info("Arquivo Org gerado em %s", org_path)

    pdf_path = export_org_to_pdf(org_path, bib_path, logger, config.org_latex_class_init, config.latex_extra_path, config.comando_exportacao_pdf) if config.exportar_pdf else None

    debug_payload = {
        "bib_file": str(bib_path),
        "config": {
            "disciplina": config.disciplina,
            "professor": config.professor,
            "curso": config.curso,
            "tema": config.tema,
            "recorte": config.recorte,
            "objetivo": config.objetivo,
            "bases": config.bases,
            "bases_label": config.bases_label,
            "tipo_estudo": config.tipo_estudo,
            "estilo_citacao": config.estilo_citacao,
            "periodo": config.periodo,
            "idiomas": config.idiomas,
            "palavras_chave": config.palavras_chave,
            "query_bilingue": config.query_bilingue,
            "queries": {src: config.source_query(src) for src in config.bases},
            "arquivo_orientacao": str(config.arquivo_orientacao) if config.arquivo_orientacao else None,
            "model": config.model,
            "titulo_trabalho": config.titulo_trabalho,
            "incluir_resumo_artigo_ia": config.incluir_resumo_artigo_ia,
            "comando_exportacao_pdf": config.comando_exportacao_pdf,
        },
        "audit": {
            "identified_total": audit.identified_total,
            "duplicates_removed": audit.duplicates_removed,
            "other_removed": audit.other_removed,
            "screened_total": audit.screened_total,
            "per_source": [
                {
                    "source": item.source,
                    "label": SOURCE_LABELS.get(item.source, item.source),
                    "query": item.query,
                    "retrieved": item.retrieved,
                    "warnings": item.warnings,
                }
                for item in audit.per_source
            ],
            "warnings": audit.warnings,
        },
        "counts": counts.__dict__,
        "selected": selected.short_dict(),
        "triage": triage.model_dump(),
        "analysis": analysis.model_dump(),
        "candidates": [c.short_dict() for c in candidates],
    }
    debug_path = save_debug_json(config.output_dir, config.prefixo, debug_payload)

    removed_paths: list[Path] = []
    if config.remover_auxiliares:
        removed_paths = cleanup_generated_files(config.output_dir, config.prefixo, logger)
        if removed_paths:
            logger.info("Arquivos auxiliares removidos: %s", ", ".join(p.name for p in removed_paths))

    print("\nArquivos gerados com sucesso:")
    print(f"- TÍTULO: {config.titulo_trabalho}")
    print(f"- ORG:   {org_path}")
    print(f"- SVG:   {svg_path}")
    print(f"- FIGPDF:{figure_pdf_path}")
    print(f"- BIB:   {bib_path}")
    print(f"- DEBUG: {debug_path}")
    if pdf_path:
        print(f"- PDF:   {pdf_path}")
    if audit.warnings:
        print("\nAvisos:")
        for item in audit.warnings:
            print(f"- {item}")
    print("\nObservações:")
    print("- O fluxograma é gerado em SVG e convertido automaticamente para PDF antes da inserção no documento final.")
    print("- A inserção da figura no .org final é feita com \\includegraphics, evitando a reinterpretação do texto do SVG pelo LaTeX.")
    print("- Se você selecionar Scopus ou Web of Science, as respectivas chaves precisam estar no ambiente ou serem informadas no terminal.")
    if config.query_bilingue:
        print("- As queries foram montadas/sugeridas em lógica bilíngue (português + inglês) quando possível.")
    if config.arquivo_orientacao:
        print(f"- Arquivo de orientação complementar aplicado: {config.arquivo_orientacao}")
    if config.remover_auxiliares:
        if removed_paths:
            print(f"- Arquivos auxiliares removidos: {len(removed_paths)}")
        else:
            print("- Nenhum arquivo auxiliar precisou ser removido.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
    except (requests.HTTPError, requests.RequestException) as exc:
        print(f"Erro de rede/API externa: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except ValidationError as exc:
        print(f"Erro de validação do retorno estruturado: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1)

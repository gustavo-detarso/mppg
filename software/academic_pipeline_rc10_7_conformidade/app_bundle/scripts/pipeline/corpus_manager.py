#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore

try:
    import docx  # type: ignore
except Exception:
    docx = None  # type: ignore

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import shorten_text, resolve_path, slugify
else:
    from utils import shorten_text, resolve_path, slugify

TEXT_SUFFIXES = {".txt", ".md", ".org", ".rst", ".tex", ".json", ".csv", ".yaml", ".yml", ".xml"}
BINARY_SUFFIXES = {".pdf", ".docx"}
READABLE_SUFFIXES = TEXT_SUFFIXES | BINARY_SUFFIXES


@dataclass
class SourceDoc:
    path: str
    kind: str
    label: str
    extracted_text: str
    bib_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    out: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.filename or info.filename.endswith("/"):
                continue
            target = (dest_dir / info.filename).resolve()
            if not (str(target).startswith(str(root) + os.sep) or target == root):
                raise RuntimeError(f"Entrada insegura no ZIP: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


class _CaptureLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if msg:
            self.messages.append(str(msg))


def _dedupe_messages(messages: list[str], limit: int = 60) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for msg in messages:
        msg = str(msg or "").strip()
        if not msg:
            continue
        if msg not in seen:
            seen.add(msg)
            out.append(msg)
        if len(out) >= limit:
            break
    return out


@contextlib.contextmanager
def capture_pdf_parser_warnings() -> Any:
    """Captura warnings/logs ruidosos do pypdf por arquivo.

    Alguns PDFs válidos têm tabela xref ou objetos indiretos irregulares. O pypdf
    costuma emitir mensagens como "Ignoring wrong pointing object ..." no stderr
    ou no logger `pypdf`. A captura evita poluir o console e permite reportar o
    problema com o nome exato do arquivo processado.
    """
    logger = logging.getLogger("pypdf")
    old_level = logger.level
    old_propagate = logger.propagate
    handler = _CaptureLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    stderr = io.StringIO()
    caught: list[warnings.WarningMessage] = []
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stderr(stderr):
            warnings.simplefilter("always")
            yield handler.messages, stderr, caught
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def read_text_file_with_diagnostics(path: Path, max_chars: int = 40000) -> tuple[str, list[str]]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return shorten_text(path.read_text(encoding="utf-8", errors="ignore"), max_chars), []
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf não está disponível.")
        captured_messages: list[str] = []
        with capture_pdf_parser_warnings() as (log_messages, stderr_buffer, warning_records):
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
            captured_messages.extend(log_messages)
            captured_messages.extend(str(w.message) for w in warning_records)
            captured_messages.extend(line.strip() for line in stderr_buffer.getvalue().splitlines())
        return shorten_text("\n".join(chunks), max_chars), _dedupe_messages(captured_messages)
    if suffix == ".docx":
        if docx is None:
            raise RuntimeError("python-docx não está disponível.")
        d = docx.Document(str(path))
        return shorten_text("\n".join(p.text for p in d.paragraphs if p.text), max_chars), []
    raise RuntimeError(f"Extensão não suportada: {path.suffix}")


def read_text_file(path: Path, max_chars: int = 40000) -> str:
    text, _diag = read_text_file_with_diagnostics(path, max_chars)
    return text


def _print_pdf_warning_summary(label: str, messages: list[str]) -> None:
    if not messages:
        return
    print(f"[WARN] PDF com estrutura irregular, mas leitura prosseguiu: {label}", flush=True)
    for msg in messages[:5]:
        print(f"       - {msg}", flush=True)
    if len(messages) > 5:
        print(f"       - ... mais {len(messages) - 5} aviso(s) suprimido(s)", flush=True)


def discover_local_documents(cfg: dict[str, Any], work_dir: Path) -> tuple[list[SourceDoc], dict[str, Any]]:
    local = cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    types_raw = local.get("tipos") or ["pdf", "docx", "txt", "md", "org"]
    suffixes = {("." + str(t).lower().lstrip(".")) for t in types_raw}
    suffixes &= READABLE_SUFFIXES
    suffixes = suffixes or {".pdf", ".docx", ".txt", ".md", ".org"}

    files: list[Path] = []
    source_info: dict[str, Any] = {"suffixes": sorted(suffixes), "warnings": []}
    input_zip = resolve_path(local.get("input_zip"), config_dir)
    input_dir = resolve_path(local.get("input_dir"), config_dir)
    extracted_dir = work_dir / "extracted"

    if input_zip and input_zip.exists():
        target = extracted_dir / slugify(input_zip.stem)
        if target.exists() and bool(local.get("limpar_extracao_anterior", True)):
            shutil.rmtree(target)
        files = [p for p in safe_extract_zip(input_zip, target) if p.is_file() and p.suffix.lower() in suffixes]
        source_info["input_zip"] = str(input_zip)
        source_info["extracted_dir"] = str(target)
    elif input_dir and input_dir.exists():
        recursive = bool(local.get("recursive", True))
        iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
        files = [p for p in iterator if p.is_file() and p.suffix.lower() in suffixes]
        source_info["input_dir"] = str(input_dir)
    else:
        raise RuntimeError("[documentos_locais] precisa de input_zip ou input_dir válido.")

    files = sorted({p.resolve() for p in files})
    if not files:
        raise RuntimeError("Nenhum documento local legível encontrado.")

    max_chars = int(local.get("max_caracteres_por_doc") or 45000)
    docs: list[SourceDoc] = []
    for p in files:
        try:
            text, pdf_warnings = read_text_file_with_diagnostics(p, max_chars)
            metadata: dict[str, Any] = {}
            if pdf_warnings:
                metadata["warnings"] = pdf_warnings
                metadata["warning_count"] = len(pdf_warnings)
                source_info.setdefault("warnings", []).append({"file": p.name, "warnings": pdf_warnings})
                _print_pdf_warning_summary(p.name, pdf_warnings)
            docs.append(SourceDoc(path=str(p), kind="documento_base", label=p.name, extracted_text=text, metadata=metadata))
        except Exception as exc:
            docs.append(SourceDoc(path=str(p), kind="documento_base_erro", label=p.name, extracted_text="", metadata={"error": str(exc)}))
    return docs, source_info


def collect_orientation_docs(cfg: dict[str, Any], work_dir: Path) -> list[SourceDoc]:
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    paths: list[Any] = []
    orient = cfg.get("orientacoes", {}) if isinstance(cfg.get("orientacoes"), dict) else {}
    raw = orient.get("paths") or []
    if isinstance(raw, str):
        raw = [raw]
    paths.extend(raw)
    docs: list[SourceDoc] = []
    orient_dir = work_dir / "orientacoes_extraidas"
    for raw in paths:
        p = resolve_path(raw, config_dir)
        if not p or not p.exists():
            continue
        files: list[Path] = []
        if p.is_file() and p.suffix.lower() == ".zip":
            files = [x for x in safe_extract_zip(p, orient_dir / slugify(p.stem)) if x.is_file() and x.suffix.lower() in READABLE_SUFFIXES]
        elif p.is_file() and p.suffix.lower() in READABLE_SUFFIXES:
            files = [p]
        elif p.is_dir():
            files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in READABLE_SUFFIXES]
        for f in sorted(files):
            try:
                text, pdf_warnings = read_text_file_with_diagnostics(f, 30000)
                metadata: dict[str, Any] = {}
                if pdf_warnings:
                    metadata["warnings"] = pdf_warnings
                    metadata["warning_count"] = len(pdf_warnings)
                    _print_pdf_warning_summary(f.name, pdf_warnings)
                docs.append(SourceDoc(path=str(f), kind="orientacao", label=f.name, extracted_text=text, metadata=metadata))
            except Exception as exc:
                docs.append(SourceDoc(path=str(f), kind="orientacao_erro", label=f.name, extracted_text="", metadata={"error": str(exc)}))
    inline = str(orient.get("inline") or "").strip()
    if inline:
        docs.append(SourceDoc(path="inline:orientacoes", kind="orientacao_inline", label="Orientação inline", extracted_text=inline))
    return docs


def copy_documents_to_fulltext_cache(docs: list[SourceDoc], cache: Path, clean: bool = True) -> list[Path]:
    """Copia documentos para o diretório exato de cache resolvido em [paths].cache_dir."""
    if clean and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: set[str] = set()
    for doc in docs:
        p = Path(doc.path)
        if not p.exists() or not p.is_file():
            continue
        target = cache / p.name
        if target.name in seen or target.exists():
            stem, suffix = p.stem, p.suffix
            i = 2
            while (cache / f"{stem}_{i}{suffix}").exists():
                i += 1
            target = cache / f"{stem}_{i}{suffix}"
        shutil.copy2(p, target)
        seen.add(target.name)
        doc.metadata["fulltext_cache_path"] = str(target)
        copied.append(target)
    return copied

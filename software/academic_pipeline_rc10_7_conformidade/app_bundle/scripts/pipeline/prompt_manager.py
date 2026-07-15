#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gerenciamento de diretivas e prompt bank para academic_pipeline rc10.7.

Esta camada permite reutilizar prompts gerais, institucionais, de pesquisa,
triagem, documento, bibliografia, mapa mental e relatório PRISMA sem copiar texto
para cada TOML. Os arquivos são carregados em ordem previsível, saneados e
registrados nos relatórios de execução.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .institution_profiles import find_app_bundle
else:
    from institution_profiles import find_app_bundle
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import shorten_text, write_text
else:
    from utils import shorten_text, write_text


@dataclass
class PromptSource:
    category: str
    path: str
    label: str
    chars: int
    sha256: str
    sanitized: bool = False


@dataclass
class PromptBundle:
    task: str
    text: str
    sources: list[PromptSource]

    def report(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "total_chars": len(self.text),
            "sources": [asdict(s) for s in self.sources],
        }


def _cfg_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = cfg.get(name, {})
    return sec if isinstance(sec, dict) else {}


def _config_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()


def _profile_dir(cfg: dict[str, Any]) -> Path | None:
    raw = cfg.get("__institution_profile_path__")
    if not raw:
        return None
    p = Path(str(raw)).expanduser().resolve()
    return p.parent if p.exists() else None


def resolve_prompt_path(raw: Any, cfg: dict[str, Any]) -> Path | None:
    """Resolve paths de prompts.

    Suporta:
    - app://prompts/...
    - config://arquivo.txt
    - profile://prompts/...
    - caminhos absolutos
    - caminhos relativos ao diretório do TOML, app_bundle, raiz do projeto ou cwd.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    config_dir = _config_dir(cfg)
    app_bundle = find_app_bundle(config_dir)
    profile_dir = _profile_dir(cfg)

    if s.startswith("app://"):
        return (app_bundle / s.removeprefix("app://")).resolve()
    if s.startswith("config://"):
        return (config_dir / s.removeprefix("config://")).resolve()
    if s.startswith("profile://"):
        if not profile_dir:
            return None
        return (profile_dir / s.removeprefix("profile://")).resolve()

    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()

    candidates = [
        config_dir / p,
        app_bundle / p,
        app_bundle.parent / p,
        Path.cwd() / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Retorna a primeira localização esperada para diagnóstico claro.
    return candidates[0].resolve()


def _iter_values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)] if str(value).strip() else []


def sanitize_general_execution_prompt(text: str) -> tuple[str, bool]:
    """Remove exigência inadequada de exposição de chain-of-thought.

    O objetivo é preservar a orientação de planejamento rigoroso, mas trocar pedidos
    de cadeia de pensamento literal por justificativas sintéticas, verificáveis e
    orientadas à decisão.
    """
    original = text or ""
    cleaned = original

    replacement = (
        "- Profundidade e planejamento interno rigoroso\n"
        "Recuse-se a dar respostas superficiais. Planeje internamente antes de responder. "
        "Quando a solicitação for complexa, quebre a solução em etapas verificáveis. "
        "Quando útil, apresente ao usuário uma justificativa sintética, auditável e orientada à decisão, "
        "sem expor raciocínio interno oculto. Se uma resposta direta não resolver o problema raiz, "
        "solicite os dados necessários ou proponha um caminho mais robusto.\n"
    )

    # Substitui o bloco inteiro quando ele aparece com o título original.
    cleaned2 = re.sub(
        r"(?ms)^-\s*Profundidade\s+e\s+Cadeia\s+de\s+Pensamento.*?(?=^\s*-\s+Eleva[cç][aã]o\s+de\s+N[ií]vel|^\s*-\s+Obsess[aã]o|\Z)",
        replacement + "\n",
        cleaned,
    )
    cleaned = cleaned2

    # Segurança adicional: troca expressões residuais que peçam CoT explícito.
    residual_patterns = [
        r"(?i)chain\s+of\s+thought\s*[-–—]?\s*CoT",
        r"(?i)cadeia\s+de\s+pensamento\s*\([^)]*\)",
        r"(?i)utilize\s+o\s+tempo\s+de\s+processamento\s+para\s+planejar\."
    ]
    for pat in residual_patterns:
        cleaned = re.sub(pat, "planejamento interno rigoroso", cleaned)

    return cleaned.strip() + "\n", cleaned != original


def _read_prompt(path: Path, *, category: str) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sanitized = False
    if category == "global" or "orientacao_geral_execucao" in path.name.lower():
        text, sanitized = sanitize_general_execution_prompt(text)
    return text.strip(), sanitized


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _add_prompt(
    cfg: dict[str, Any],
    out: list[tuple[str, Path, str]],
    category: str,
    values: Any,
) -> None:
    for raw in _iter_values(values):
        p = resolve_prompt_path(raw, cfg)
        if p:
            out.append((category, p, raw))


def _task_path_keys(task: str, document_type: str = "") -> list[tuple[str, str]]:
    task = (task or "document").strip().lower()
    document_type = (document_type or "").strip().lower()
    keys: list[tuple[str, str]] = []
    if task in {"research", "triagem", "screening", "ranking"}:
        keys.extend([("research", "research_paths"), ("research", "triagem_paths"), ("research", "screening_paths")])
    elif task in {"prisma", "relatorio_prisma", "relatorio_pesquisa"}:
        keys.extend([("prisma", "prisma_paths"), ("research", "research_paths")])
    elif task in {"bibliography", "bibliografia", "bib"}:
        keys.extend([("bibliography", "bibliography_paths"), ("bibliography", "bib_paths")])
    elif task in {"mindmap", "mapa_mental"}:
        keys.extend([("mindmap", "mindmap_paths"), ("document", "document_paths")])
    else:
        keys.extend([("document", "document_paths")])
        if document_type:
            keys.append(("document_type", f"{document_type}_paths"))
    return keys


def load_prompt_bundle(cfg: dict[str, Any], task: str, *, document_type: str = "") -> PromptBundle:
    prompts = _cfg_section(cfg, "prompts")
    if not prompts:
        return PromptBundle(task=task, text="", sources=[])
    active = prompts.get("ativos")
    if active is False:
        return PromptBundle(task=task, text="", sources=[])

    items: list[tuple[str, Path, str]] = []
    _add_prompt(cfg, items, "global", prompts.get("global_paths"))
    _add_prompt(cfg, items, "institution", prompts.get("institution_paths"))

    for category, key in _task_path_keys(task, document_type):
        _add_prompt(cfg, items, category, prompts.get(key))

    # Também permite blocos inline; eles entram depois dos arquivos, na mesma ordem de especificidade.
    inline_entries: list[tuple[str, str]] = []
    for key in ["global_inline", "institution_inline"]:
        if str(prompts.get(key) or "").strip():
            inline_entries.append((key, str(prompts.get(key))))
    for _, key in _task_path_keys(task, document_type):
        inline_key = key.replace("_paths", "_inline")
        if str(prompts.get(inline_key) or "").strip():
            inline_entries.append((inline_key, str(prompts.get(inline_key))))

    chunks: list[str] = []
    sources: list[PromptSource] = []
    seen: set[Path] = set()
    for category, path, raw_label in items:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            # Falha será capturada no check-config; aqui ignoramos para não quebrar execuções parciais.
            continue
        text, sanitized = _read_prompt(path, category=category)
        if not text:
            continue
        chunks.append(f"## Diretiva: {category} — {path.name}\n{text}")
        sources.append(PromptSource(category=category, path=str(path), label=str(raw_label), chars=len(text), sha256=_sha(text), sanitized=sanitized))

    for label, text in inline_entries:
        cleaned = text.strip()
        if not cleaned:
            continue
        sanitized = False
        if label.startswith("global"):
            cleaned, sanitized = sanitize_general_execution_prompt(cleaned)
            cleaned = cleaned.strip()
        chunks.append(f"## Diretiva inline: {label}\n{cleaned}")
        sources.append(PromptSource(category="inline", path=f"inline:{label}", label=label, chars=len(cleaned), sha256=_sha(cleaned), sanitized=sanitized))

    return PromptBundle(task=task, text="\n\n".join(chunks).strip(), sources=sources)


def prompt_report_for_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    doc = _cfg_section(cfg, "documento")
    doc_type = str(doc.get("tipo_documento") or "paper")
    tasks = ["research", "bibliography", "document", "mindmap", "prisma"]
    return {task: load_prompt_bundle(cfg, task, document_type=doc_type).report() for task in tasks}


def validate_prompt_paths(cfg: dict[str, Any]) -> list[str]:
    prompts = _cfg_section(cfg, "prompts")
    if not prompts or prompts.get("ativos") is False:
        return []
    errors: list[str] = []
    path_keys = [
        "global_paths", "institution_paths", "research_paths", "triagem_paths", "screening_paths",
        "document_paths", "paper_paths", "atividade_paths", "dissertacao_paths", "bibliography_paths",
        "bib_paths", "mindmap_paths", "prisma_paths",
    ]
    for key in path_keys:
        for raw in _iter_values(prompts.get(key)):
            p = resolve_prompt_path(raw, cfg)
            if not p or not p.exists():
                errors.append(f"[prompts].{key} não encontrado: {raw} -> {p}")
    return errors


def install_default_prompt_files(app_bundle: Path, source_dir: Path | None = None) -> dict[str, str]:
    """Cria a estrutura padrão de prompts. Útil para empacotamento/testes."""
    base = app_bundle / "prompts"
    (base / "global").mkdir(parents=True, exist_ok=True)
    (base / "research").mkdir(parents=True, exist_ok=True)
    (base / "document").mkdir(parents=True, exist_ok=True)
    (base / "prisma").mkdir(parents=True, exist_ok=True)
    return {"base": str(base)}

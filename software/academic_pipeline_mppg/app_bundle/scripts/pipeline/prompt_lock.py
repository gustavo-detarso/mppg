#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt lock para academic_pipeline rc10.7.

Registra, de forma reprodutível, quais diretivas/prompt bank foram carregados
na execução. O run_report já inclui um sumário dos prompts; este módulo gera um
arquivo dedicado <prefixo>.prompt_lock.json para auditoria e comparação entre
execuções.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .diagnostics import now_iso, PIPELINE_VERSION
else:
    from diagnostics import now_iso, PIPELINE_VERSION
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .prompt_manager import prompt_report_for_cfg
else:
    from prompt_manager import prompt_report_for_cfg
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import write_json, write_text
else:
    from utils import write_json, write_text


def build_prompt_lock(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": PIPELINE_VERSION,
        "generated_at": now_iso(),
        "config_path": cfg.get("__config_path__"),
        "institution_profile": cfg.get("__institution_profile_name__"),
        "institution_profile_path": cfg.get("__institution_profile_path__"),
        "prompts": prompt_report_for_cfg(cfg),
    }


def write_prompt_lock(cfg: dict[str, Any], output_path: Path) -> dict[str, Any]:
    lock = build_prompt_lock(cfg)
    write_json(output_path, lock)
    return lock


def render_prompt_lock_markdown(lock: dict[str, Any]) -> str:
    lines = [
        "# Prompt lock",
        "",
        f"- Versão do pipeline: `{lock.get('version')}`",
        f"- Gerado em: `{lock.get('generated_at')}`",
        f"- Config: `{lock.get('config_path')}`",
        f"- Perfil institucional: `{lock.get('institution_profile') or 'nenhum'}`",
        "",
    ]
    prompts = lock.get("prompts") or {}
    for task, report in prompts.items():
        lines.append(f"## {task}")
        lines.append("")
        lines.append(f"- Total de caracteres: {report.get('total_chars', 0)}")
        sources = report.get("sources") or []
        if not sources:
            lines.append("- Nenhum prompt carregado.")
            lines.append("")
            continue
        for src in sources:
            sanitized = " sim" if src.get("sanitized") else " não"
            lines.append(f"- `{src.get('category')}` — `{src.get('path')}`")
            lines.append(f"  - sha256: `{src.get('sha256')}`")
            lines.append(f"  - caracteres: {src.get('chars')}")
            lines.append(f"  - saneado: {sanitized}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_prompt_lock_markdown(lock: dict[str, Any], output_path: Path) -> None:
    write_text(output_path, render_prompt_lock_markdown(lock))

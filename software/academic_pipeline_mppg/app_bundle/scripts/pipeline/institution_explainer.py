#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicador de perfis institucionais para academic_pipeline rc10.7."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .institution_profiles import available_institution_profiles, find_app_bundle
else:
    from institution_profiles import available_institution_profiles, find_app_bundle


def _fmt_list(values: list[str] | tuple[str, ...] | None) -> str:
    if not values:
        return "nenhum"
    return ", ".join(str(v) for v in values)


def explain_profile(profile_name: str = "fgv", app_bundle: Path | None = None) -> str:
    app = app_bundle or find_app_bundle()
    profile_name = (profile_name or "fgv").strip().lower()
    profile_path = app / "institutions" / profile_name / "institution_profile.toml"
    if not profile_path.exists():
        available = ", ".join(available_institution_profiles(app)) or "nenhum"
        raise FileNotFoundError(f"Perfil institucional não encontrado: {profile_name}. Disponíveis: {available}")
    with profile_path.open("rb") as f:
        profile: dict[str, Any] = tomllib.load(f)

    meta = profile.get("instituicao", {}) if isinstance(profile.get("instituicao"), dict) else {}
    defaults = profile.get("defaults", {}) if isinstance(profile.get("defaults"), dict) else {}
    document_types = profile.get("document_types", {}) if isinstance(profile.get("document_types"), dict) else {}
    fmt = profile.get("formatacao", {}) if isinstance(profile.get("formatacao"), dict) else {}
    valid = profile.get("validacao", {}) if isinstance(profile.get("validacao"), dict) else {}
    bib = defaults.get("bibliografia", {}) if isinstance(defaults.get("bibliografia"), dict) else {}
    latex = defaults.get("latex", {}) if isinstance(defaults.get("latex"), dict) else {}
    docx = defaults.get("docx", {}) if isinstance(defaults.get("docx"), dict) else {}

    lines: list[str] = [
        f"Perfil institucional: {profile_name}",
        f"Instituição: {meta.get('nome', profile_name)}",
        f"Sigla: {meta.get('sigla', '')}",
        f"Manual: {meta.get('manual', '')} {meta.get('versao_manual', '')}".strip(),
        "",
        "Documentos suportados:",
    ]
    for dt, cfg in sorted(document_types.items()):
        doc = cfg.get("documento", {}) if isinstance(cfg, dict) and isinstance(cfg.get("documento"), dict) else {}
        lines.append(f"- {dt}: template={doc.get('template_org') or doc.get('template_path') or 'n/d'}")
    lines.extend([
        "",
        "Bibliografia:",
        f"- Estilo padrão: {bib.get('estilo_citacao') or bib.get('latex_style') or 'n/d'}",
        f"- CSL DOCX: {bib.get('docx_csl') or docx.get('csl_path') or 'n/d'}",
        "",
        "LaTeX/PDF:",
        f"- Engine: {latex.get('pdf_engine', 'lualatex')}",
        f"- Classe Org/Emacs: {latex.get('org_latex_class_init', 'n/d')}",
        f"- latex_extra_path: {latex.get('latex_extra_path', 'n/d')}",
        "",
        "DOCX:",
        f"- reference_docx: {docx.get('reference_docx', 'n/d')}",
        f"- usar_pandoc: {docx.get('usar_pandoc', False)}",
        "",
        "Regras principais:",
        f"- Papel: {fmt.get('papel', 'n/d')}",
        f"- Margens cm: superior={fmt.get('margem_superior_cm', 'n/d')}; esquerda={fmt.get('margem_esquerda_cm', 'n/d')}; direita={fmt.get('margem_direita_cm', 'n/d')}; inferior={fmt.get('margem_inferior_cm', 'n/d')}",
        f"- Fonte principal: {fmt.get('fonte_principal', 'n/d')} ou {fmt.get('fonte_alternativa', 'n/d')}",
        f"- Fonte texto: {fmt.get('fonte_texto_pt', 'n/d')} pt",
        f"- Fonte auxiliar: {fmt.get('fonte_elementos_auxiliares_pt', 'n/d')} pt",
        f"- Espaçamento texto: {fmt.get('espacamento_texto', 'n/d')}",
        f"- Espaçamento referências: {fmt.get('espacamento_referencias', 'n/d')}",
        f"- Sistema de citação: {fmt.get('sistema_citacao', valid.get('exigir_sistema_autor_data', 'n/d'))}",
        "",
        "Validação:",
        f"- Exigir sistema autor-data: {valid.get('exigir_sistema_autor_data', 'n/d')}",
        f"- Notas de referência: {valid.get('notas_referencia', 'n/d')}",
        f"- Seções primárias iniciam nova página: {valid.get('secoes_primarias_iniciam_nova_pagina', 'n/d')}",
    ])
    return "\n".join(lines).strip() + "\n"

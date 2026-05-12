#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Roteamento institucional de layouts para academic_pipeline rc10.7.31.

A ideia desta camada é separar:
- tipo_conteudo: o que a IA deve produzir;
- genero_academico: atividade, paper, dissertacao, relatorio etc.;
- layout: família visual/institucional;
- classe_latex/front_matter/template/validador: detalhes de renderização.

O perfil institucional pode declarar, por exemplo:

[layouts.atividade_fgv]
genero_academico = "atividade"
front_matter = "atividade_fgv"
classe_latex = "fgv-paper"
template = "profile://templates/template_atividade.org"

O TOML do projeto pode então escolher explicitamente:

[documento]
tipo_conteudo = "resumo_artigos"
genero_academico = "atividade"
layout = "atividade_fgv"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayoutSpec:
    id: str
    institution: str
    genero_academico: str
    tipo_conteudo: str
    front_matter: str
    classe_latex: str
    template: str
    validator: str
    raw: dict[str, Any]


def _section(obj: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    sec = obj.get(name, {})
    return sec if isinstance(sec, dict) else {}


def _doctype_alias(value: str) -> str:
    v = str(value or "").strip().lower()
    aliases = {
        "dissertação": "dissertacao",
        "dissertation": "dissertacao",
        "atividade_academica": "atividade",
        "atividade acadêmica": "atividade",
        "article_summary": "resumo_artigos",
        "resumo_artigos_local": "resumo_artigos",
    }
    return aliases.get(v, v)


def available_layouts(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profile = cfg.get("__institution_profile__", {}) if isinstance(cfg.get("__institution_profile__"), dict) else {}
    layouts = profile.get("layouts", {}) if isinstance(profile.get("layouts"), dict) else {}
    return {str(k): v for k, v in layouts.items() if isinstance(v, dict)}


def _default_layout_for_genero(cfg: dict[str, Any], genero: str, tipo_conteudo: str = "") -> str:
    profile = cfg.get("__institution_profile__", {}) if isinstance(cfg.get("__institution_profile__"), dict) else {}
    content_types = profile.get("document_content_types", {}) if isinstance(profile.get("document_content_types"), dict) else {}
    tipo = _doctype_alias(tipo_conteudo)
    if tipo and isinstance(content_types.get(tipo), dict):
        default_layout = str(content_types[tipo].get("default_layout") or "").strip()
        if default_layout:
            return default_layout

    doc_types = profile.get("document_types", {}) if isinstance(profile.get("document_types"), dict) else {}
    genero = _doctype_alias(genero)
    if genero and isinstance(doc_types.get(genero), dict):
        default_layout = str(doc_types[genero].get("default_layout") or "").strip()
        if default_layout:
            return default_layout

    layouts = available_layouts(cfg)
    for layout_id, spec in layouts.items():
        if _doctype_alias(str(spec.get("genero_academico") or spec.get("genero") or "")) == genero:
            return layout_id

    # Fallbacks convencionais para FGV/nomes históricos.
    if genero == "atividade":
        return "atividade_fgv" if "atividade_fgv" in layouts else "atividade"
    if genero == "dissertacao":
        return "dissertacao_fgv" if "dissertacao_fgv" in layouts else "dissertacao"
    if genero == "paper":
        return "paper_fgv" if "paper_fgv" in layouts else "paper"
    return genero or "paper"


def resolve_layout_spec(cfg: dict[str, Any], doc: Any | None = None) -> LayoutSpec:
    documento = _section(cfg, "documento")
    inst = _section(cfg, "instituicao")
    institution = str(inst.get("perfil") or cfg.get("__institution_profile_name__") or "").strip() or "default"

    meta = getattr(doc, "metadata", None)
    tipo_documento_meta = getattr(meta, "tipo_documento", "") if meta is not None else ""

    tipo_conteudo = _doctype_alias(str(documento.get("tipo_conteudo") or documento.get("content_type") or ""))
    genero = _doctype_alias(str(documento.get("genero_academico") or documento.get("genero") or documento.get("tipo_documento") or tipo_documento_meta or "paper"))
    if not tipo_conteudo:
        tipo_conteudo = _doctype_alias(str(documento.get("tipo_documento") or tipo_documento_meta or genero))

    layout_id = str(documento.get("layout") or documento.get("layout_id") or "").strip()
    if not layout_id:
        layout_id = _default_layout_for_genero(cfg, genero, tipo_conteudo)

    layouts = available_layouts(cfg)
    raw = dict(layouts.get(layout_id, {}))

    # Compatibilidade com layouts não declarados: usa convenções históricas.
    if not raw:
        raw = {
            "genero_academico": genero,
            "front_matter": layout_id,
            "classe_latex": documento.get("classe_latex") or ("fgv-dissertacao" if genero == "dissertacao" else "fgv-paper"),
            "template": documento.get("template_org") or documento.get("template_path") or "",
            "validator": "",
        }

    classe = str(documento.get("classe_latex") or raw.get("classe_latex") or raw.get("latex_class") or "").strip()
    if not classe:
        classe = "fgv-dissertacao" if genero == "dissertacao" else "fgv-paper"
    front = str(raw.get("front_matter") or raw.get("front_matter_renderer") or layout_id or genero).strip()
    template = str(documento.get("template_org") or documento.get("template_path") or raw.get("template") or raw.get("template_org") or "").strip()
    validator = str(raw.get("validator") or raw.get("validator_rules") or "").strip()

    return LayoutSpec(
        id=layout_id,
        institution=institution,
        genero_academico=str(raw.get("genero_academico") or genero),
        tipo_conteudo=tipo_conteudo,
        front_matter=front,
        classe_latex=classe,
        template=template,
        validator=validator,
        raw=raw,
    )

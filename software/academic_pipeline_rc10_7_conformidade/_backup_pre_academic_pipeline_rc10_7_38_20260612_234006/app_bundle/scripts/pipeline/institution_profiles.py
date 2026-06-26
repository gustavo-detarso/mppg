#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perfis institucionais para academic_pipeline rc10.7.

A ideia é deixar as especificações de cada instituição (FGV, ENAP, UnB etc.)
em uma camada própria. O TOML do trabalho pode informar apenas:

    [instituicao]
    perfil = "fgv"

O pipeline então carrega defaults de templates, LaTeX, DOCX, bibliografia e
regras de validação sem exigir que o usuário repita todos os caminhos em cada
arquivo de configuração.
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any


def _cfg_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = cfg.get(name, {})
    return sec if isinstance(sec, dict) else {}


def _config_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()


def find_app_bundle(config_dir: Path | None = None) -> Path:
    """Localiza app_bundle a partir do TOML, do cwd ou do próprio script."""
    candidates: list[Path] = []
    if config_dir:
        config_dir = config_dir.resolve()
        candidates.extend([config_dir, *config_dir.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.name == "app_bundle":
            return candidate
        if (candidate / "app_bundle").exists():
            return (candidate / "app_bundle").resolve()
    raise RuntimeError("Não foi possível localizar app_bundle para carregar perfis institucionais.")


def available_institution_profiles(app_bundle: Path | None = None) -> list[str]:
    app = app_bundle or find_app_bundle()
    base = app / "institutions"
    if not base.exists():
        return []
    return sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and (p / "institution_profile.toml").exists()
    )


def _deep_merge_missing(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    """Merge recursivo que só preenche chaves ausentes/vazias."""
    for key, value in defaults.items():
        if isinstance(value, dict):
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
                target[key] = existing
            _deep_merge_missing(existing, value)
        else:
            current = target.get(key)
            if current is None or current == "" or current == []:
                target[key] = value


def _doctype_alias(value: str) -> str:
    v = str(value or "paper").strip().lower()
    aliases = {
        "dissertação": "dissertacao",
        "dissertation": "dissertacao",
        "tese": "dissertacao",  # mesmo renderer estrutural, se usado futuramente
        "atividade_academica": "atividade",
        "atividade acadêmica": "atividade",
    }
    return aliases.get(v, v)


def _expand_refs(value: Any, *, app_bundle: Path, profile_dir: Path, config_dir: Path) -> Any:
    if isinstance(value, dict):
        return {k: _expand_refs(v, app_bundle=app_bundle, profile_dir=profile_dir, config_dir=config_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_refs(v, app_bundle=app_bundle, profile_dir=profile_dir, config_dir=config_dir) for v in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("profile://"):
        return str((profile_dir / text.removeprefix("profile://")).resolve())
    if text.startswith("app://"):
        return str((app_bundle / text.removeprefix("app://")).resolve())
    if text.startswith("config://"):
        return str((config_dir / text.removeprefix("config://")).resolve())
    return value


def load_institution_profile(cfg: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, Path | None]:
    inst = _cfg_section(cfg, "instituicao")
    profile_name = str(inst.get("perfil") or inst.get("profile") or "").strip().lower()
    if not profile_name:
        return None, None, None
    config_dir = _config_dir(cfg)
    app_bundle = find_app_bundle(config_dir)
    profile_path = app_bundle / "institutions" / profile_name / "institution_profile.toml"
    if not profile_path.exists():
        available = ", ".join(available_institution_profiles(app_bundle)) or "nenhum"
        raise FileNotFoundError(f"Perfil institucional não encontrado: {profile_name}. Disponíveis: {available}")
    with profile_path.open("rb") as f:
        profile = tomllib.load(f)
    profile = _expand_refs(profile, app_bundle=app_bundle, profile_dir=profile_path.parent, config_dir=config_dir)
    return profile_name, profile, profile_path


def apply_institution_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    """Aplica defaults do perfil institucional ao TOML já carregado.

    A função altera uma cópia do cfg e preserva valores explícitos do TOML. Assim,
    a instituição fornece padrões, mas o usuário ainda pode sobrescrever qualquer
    campo quando necessário.
    """
    out = copy.deepcopy(cfg)
    profile_name, profile, profile_path = load_institution_profile(out)
    if not profile:
        return out
    defaults = profile.get("defaults", {}) if isinstance(profile.get("defaults"), dict) else {}
    _deep_merge_missing(out, defaults)

    doc = _cfg_section(out, "documento")
    doc_type = _doctype_alias(str(doc.get("genero_academico") or doc.get("tipo_documento") or "paper"))
    content_type = _doctype_alias(str(doc.get("tipo_conteudo") or doc.get("content_type") or doc.get("tipo_documento") or doc_type))

    # 1) Defaults por gênero acadêmico/document_type histórico.
    per_doc = profile.get("document_types", {}) if isinstance(profile.get("document_types"), dict) else {}
    if isinstance(per_doc.get(doc_type), dict):
        _deep_merge_missing(out, per_doc[doc_type])

    # 2) Defaults por tipo de conteúdo semântico, quando existir.
    content_types = profile.get("document_content_types", {}) if isinstance(profile.get("document_content_types"), dict) else {}
    if isinstance(content_types.get(content_type), dict):
        _deep_merge_missing(out, content_types[content_type])

    # 3) Roteamento de layout institucional. O TOML explícito preserva prioridade.
    doc = out.setdefault("documento", {})
    layouts = profile.get("layouts", {}) if isinstance(profile.get("layouts"), dict) else {}
    layout_id = str(doc.get("layout") or "").strip()
    if not layout_id:
        ct_spec = content_types.get(content_type, {}) if isinstance(content_types.get(content_type), dict) else {}
        dt_spec = per_doc.get(doc_type, {}) if isinstance(per_doc.get(doc_type), dict) else {}
        layout_id = str(ct_spec.get("default_layout") or dt_spec.get("default_layout") or "").strip()
    if not layout_id and isinstance(layouts, dict):
        for candidate_id, candidate in layouts.items():
            if isinstance(candidate, dict) and _doctype_alias(str(candidate.get("genero_academico") or candidate.get("genero") or "")) == doc_type:
                layout_id = str(candidate_id)
                break
    if layout_id:
        doc.setdefault("layout", layout_id)
        layout_spec = layouts.get(layout_id, {}) if isinstance(layouts.get(layout_id), dict) else {}
        if layout_spec:
            # Copia apenas defaults seguros para o bloco documento.
            if not doc.get("classe_latex") and layout_spec.get("classe_latex"):
                doc["classe_latex"] = layout_spec.get("classe_latex")
            if not doc.get("template_org") and layout_spec.get("template"):
                doc["template_org"] = layout_spec.get("template")
            if not doc.get("template_path") and layout_spec.get("template"):
                doc["template_path"] = layout_spec.get("template")
            if not doc.get("genero_academico") and layout_spec.get("genero_academico"):
                doc["genero_academico"] = layout_spec.get("genero_academico")
            out["__layout_spec__"] = {"id": layout_id, **layout_spec}

    inst = out.setdefault("instituicao", {})
    meta = profile.get("instituicao", {}) if isinstance(profile.get("instituicao"), dict) else {}
    for key in ("nome", "sigla", "manual", "versao_manual"):
        if key in meta and not inst.get(key):
            inst[key] = meta[key]
    inst["perfil"] = profile_name
    out["__institution_profile_name__"] = profile_name
    out["__institution_profile_path__"] = str(profile_path)
    out["__institution_profile__"] = profile
    return out


def describe_institution_profiles(app_bundle: Path | None = None) -> str:
    app = app_bundle or find_app_bundle()
    names = available_institution_profiles(app)
    if not names:
        return "Nenhum perfil institucional encontrado."
    lines = ["Perfis institucionais disponíveis:"]
    for name in names:
        path = app / "institutions" / name / "institution_profile.toml"
        try:
            with path.open("rb") as f:
                profile = tomllib.load(f)
            meta = profile.get("instituicao", {}) if isinstance(profile.get("instituicao"), dict) else {}
            label = meta.get("nome") or name
            manual = meta.get("manual") or ""
            lines.append(f"- {name}: {label}" + (f" ({manual})" if manual else ""))
        except Exception:
            lines.append(f"- {name}: {path}")
    return "\n".join(lines)

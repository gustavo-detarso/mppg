#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tradução auditável de ``AcademicDocument`` para versões adicionais.

A tradução é sempre aplicada ao ``document.json`` canônico já produzido em
português. Referências bibliográficas, chaves de citação, URLs, DOI, números,
siglas e campos técnicos são preservados. O módulo não escolhe conteúdo, não
altera o corpus e não cria uma segunda análise: apenas traduz a redação do
modelo do documento, em lotes rastreáveis.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Iterable


class TranslationError(RuntimeError):
    """Indica falha na tradução automática de uma versão adicional."""


_LANGUAGE_ALIASES: dict[str, tuple[str, str]] = {
    "en": ("en", "English"),
    "english": ("en", "English"),
    "ingles": ("en", "English"),
    "inglês": ("en", "English"),
    "es": ("es", "Español"),
    "spanish": ("es", "Español"),
    "espanhol": ("es", "Español"),
    "español": ("es", "Español"),
    "fr": ("fr", "Français"),
    "french": ("fr", "Français"),
    "frances": ("fr", "Français"),
    "francês": ("fr", "Français"),
    "it": ("it", "Italiano"),
    "italian": ("it", "Italiano"),
    "italiano": ("it", "Italiano"),
    "de": ("de", "Deutsch"),
    "german": ("de", "Deutsch"),
    "alemao": ("de", "Deutsch"),
    "alemão": ("de", "Deutsch"),
}

# Ramificações cujo conteúdo nunca deve passar por tradução. O objetivo é
# preservar rastreabilidade bibliográfica, caminhos, identificadores, códigos
# LaTeX/Org e diagnósticos internos do pipeline.
_PROTECTED_PATH_TOKENS = {
    "bibliography", "bibliografia", "diagnostics", "diagnosticos",
    "citation", "citations", "citacao", "citacoes", "references",
    "referencias", "entries_used", "bib_path", "doi", "url", "uri",
    "href", "link", "path", "paths", "arquivo", "filename", "file",
    "image", "imagem", "figure_path", "source_info", "mindmap",
    "latex", "org", "json", "id", "identifier", "key", "keys",
    "slug", "created_at", "updated_at", "timestamp", "model", "version",
}
_PROTECTED_LEAF_KEYS = {
    "autor", "author", "authors", "aluno", "student", "professor",
    "professor_name", "orientador", "advisor", "coorientador", "institution_name",
    "institution", "program_name", "course_name", "discipline_name", "city_name",
    "ano", "year", "data", "date", "issn", "isbn", "volume", "issue", "pages",
}


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, dict) else {}


def normalize_language(value: Any) -> tuple[str, str]:
    """Normaliza idioma para código estável de diretório e nome de exibição."""
    raw = str(value or "").strip()
    if not raw:
        raise TranslationError("Idioma adicional vazio.")
    key = raw.casefold().replace("_", "-")
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,4})?", key):
        return key, raw
    safe = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
    if not safe:
        raise TranslationError(f"Idioma adicional inválido: {raw!r}")
    return safe, raw


def requested_translation_languages(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Retorna idiomas adicionais únicos configurados no TOML."""
    section = _section(cfg, "idiomas_saida")
    if not bool(section.get("gerar_traducao_ia", False)):
        return []
    values = section.get("idiomas_adicionais", [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    principal_code, _ = normalize_language(section.get("principal") or "pt-BR")
    seen: set[str] = {principal_code}
    result: list[tuple[str, str]] = []
    for raw in values:
        code, label = normalize_language(raw)
        if code not in seen:
            result.append((code, label))
            seen.add(code)
    return result


def translation_batch_size(cfg: dict[str, Any]) -> int:
    section = _section(cfg, "idiomas_saida")
    try:
        value = int(section.get("max_chars_por_lote", 12000))
    except (TypeError, ValueError):
        value = 12000
    return max(2500, min(value, 24000))


def _path_is_protected(path: tuple[str | int, ...]) -> bool:
    tokens = {str(item).casefold() for item in path if isinstance(item, str)}
    if tokens & _PROTECTED_PATH_TOKENS:
        return True
    leaf = str(path[-1]).casefold() if path else ""
    return leaf in _PROTECTED_LEAF_KEYS


def _looks_like_prose(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 2:
        return False
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", stripped):
        return False
    if re.fullmatch(r"[A-Za-z0-9_.:/?=&%#@+\-]+", stripped):
        return False
    return True


def collect_translatable_strings(payload: Any) -> list[tuple[tuple[str | int, ...], str]]:
    """Extrai folhas textuais do modelo, preservando campos técnicos."""
    items: list[tuple[tuple[str | int, ...], str]] = []

    def walk(value: Any, path: tuple[str | int, ...]) -> None:
        if _path_is_protected(path):
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                walk(nested, (*path, str(key)))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, (*path, index))
        elif isinstance(value, str) and _looks_like_prose(value):
            items.append((path, value))

    walk(payload, ())
    return items


def _set_at_path(payload: Any, path: tuple[str | int, ...], value: str) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _chunk_items(items: Iterable[tuple[tuple[str | int, ...], str]], max_chars: int) -> list[list[tuple[str, tuple[str | int, ...], str]]]:
    chunks: list[list[tuple[str, tuple[str | int, ...], str]]] = []
    current: list[tuple[str, tuple[str | int, ...], str]] = []
    current_size = 0
    for index, (path, text) in enumerate(items, start=1):
        item_size = len(text) + 120
        if current and current_size + item_size > max_chars:
            chunks.append(current)
            current, current_size = [], 0
        current.append((f"t{index:05d}", path, text))
        current_size += item_size
    if current:
        chunks.append(current)
    return chunks


def _extract_response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception as exc:  # pragma: no cover - depende do cliente remoto
        raise TranslationError("A API de IA não devolveu conteúdo de tradução utilizável.") from exc
    if isinstance(content, list):
        content = "".join(
            str(getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else ""))
            for item in content
        )
    return str(content or "").strip()


def _request_translation_batch(client: Any, model: str, language_label: str, items: list[tuple[str, tuple[str | int, ...], str]]) -> dict[str, str]:
    source = {item_id: text for item_id, _path, text in items}
    instructions = (
        "Você é um tradutor acadêmico. Traduza cada texto do português para " + language_label + ". "
        "Não resuma, não reordene e não acrescente conteúdo. Preserve exatamente citações e comandos "
        "LaTeX/Org/Markdown, chaves BibTeX, URLs, DOI, números, percentuais, fórmulas, siglas, nomes próprios "
        "e marcadores de tabela. Devolva JSON estrito no formato {\"translations\": [{\"id\": \"...\", \"text\": \"...\"}]}. "
        "Cada id recebido deve aparecer uma única vez."
    )
    payload = json.dumps({"items": [{"id": item_id, "text": text} for item_id, text in source.items()]}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": payload},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Alguns gateways compatíveis não aceitam response_format. Mantém o
        # mesmo contrato e valida o JSON explicitamente na sequência.
        response = client.chat.completions.create(model=model, messages=messages)
    content = _extract_response_content(response)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TranslationError("A IA devolveu tradução fora do formato JSON esperado.") from exc
    values = parsed.get("translations", []) if isinstance(parsed, dict) else []
    if not isinstance(values, list):
        raise TranslationError("A IA não devolveu a lista translations esperada.")
    translated: dict[str, str] = {}
    for row in values:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id") or "").strip()
        text = row.get("text")
        if item_id and isinstance(text, str) and text.strip():
            translated[item_id] = text
    missing = [item_id for item_id in source if item_id not in translated]
    if missing:
        raise TranslationError("A IA não devolveu tradução para todos os campos: " + ", ".join(missing[:8]))
    return {item_id: translated[item_id] for item_id in source}


def translate_document_model(client: Any, model: str, document: Any, target_language: str, *, max_chars: int = 12000) -> tuple[Any, dict[str, Any]]:
    """Cria uma cópia traduzida do document model e um registro de auditoria."""
    code, label = normalize_language(target_language)
    try:
        raw = document.model_dump(mode="python")
    except TypeError:
        raw = document.model_dump()
    payload = copy.deepcopy(raw)
    items = collect_translatable_strings(payload)
    if not items:
        raise TranslationError("Nenhum campo textual elegível foi encontrado no document.json.")
    chunks = _chunk_items(items, max_chars=max_chars)
    translated_count = 0
    for chunk in chunks:
        result = _request_translation_batch(client, model, label, chunk)
        for item_id, path, _source in chunk:
            _set_at_path(payload, path, result[item_id])
            translated_count += 1
    model_class = type(document)
    try:
        translated_document = model_class.model_validate(payload)
    except Exception as exc:
        raise TranslationError("A tradução não passou na validação do document model.") from exc
    audit = {
        "idioma_codigo": code,
        "idioma_destino": label,
        "modelo": model,
        "campos_traduzidos": translated_count,
        "lotes": len(chunks),
        "caracteres_origem": sum(len(text) for _path, text in items),
        "referencias_preservadas": True,
        "metodo": "tradução do document.json canônico; sem nova análise do corpus",
    }
    return translated_document, audit

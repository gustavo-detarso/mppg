#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Iterable
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .document_model import Citation, Inline, TextSpan
else:
    from document_model import Citation, Inline, TextSpan


def render_latex_citation(citation: Citation) -> str:
    keys = [k.strip().lstrip("@") for k in citation.keys if k.strip()]
    if not keys:
        return ""
    joined = ",".join(keys)
    if citation.mode == "narrative":
        if len(keys) == 1:
            body = rf"\textcite{{{keys[0]}}}"
        elif len(keys) == 2:
            body = rf"\textcite{{{keys[0]}}} e \textcite{{{keys[1]}}}"
        else:
            body = ", ".join(rf"\textcite{{{k}}}" for k in keys[:-1]) + " e " + rf"\textcite{{{keys[-1]}}}"
    elif citation.mode == "author":
        body = rf"\citeauthor{{{joined}}}"
    elif citation.mode == "year":
        body = rf"\citeyear{{{joined}}}"
    else:
        body = rf"\parencite{{{joined}}}"
    return f"{citation.prefix}{body}{citation.suffix}"


def render_latex_inlines(content: list[Inline]) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, Citation) or getattr(item, "type", None) == "citation":
            parts.append(render_latex_citation(item))
        else:
            text = getattr(item, "text", "")
            if getattr(item, "bold", False):
                text = rf"\textbf{{{text}}}"
            if getattr(item, "italic", False):
                text = rf"\textit{{{text}}}"
            parts.append(text)
    return "".join(parts)


def extract_cited_keys_from_model_inline(content: list[Inline]) -> list[str]:
    keys: list[str] = []
    for item in content:
        if isinstance(item, Citation) or getattr(item, "type", None) == "citation":
            for k in item.keys:
                k = str(k).strip().lstrip("@")
                if k and k not in keys:
                    keys.append(k)
    return keys


def extract_latex_cited_keys(text: str) -> list[str]:
    out: list[str] = []
    for group in re.findall(r"\\(?:paren|text|auto|foot|smart|cite|citeauthor|citeyear)cite\{([^}]+)\}", text or ""):
        for key in group.split(","):
            k = key.strip().lstrip("@")
            if k and k not in out:
                out.append(k)
    return out

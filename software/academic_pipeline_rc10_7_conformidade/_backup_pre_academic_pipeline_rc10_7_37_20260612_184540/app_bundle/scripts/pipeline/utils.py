#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    text = str(text or "").strip().lower()
    repl = {"á":"a","à":"a","â":"a","ã":"a","é":"e","ê":"e","í":"i","ó":"o","ô":"o","õ":"o","ú":"u","ü":"u","ç":"c"}
    for a,b in repl.items(): text = text.replace(a,b)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "item"


def normalize_title_loose(text: str) -> str:
    text = str(text or "").strip().lower()
    repl = {"á":"a","à":"a","â":"a","ã":"a","é":"e","ê":"e","í":"i","ó":"o","ô":"o","õ":"o","ú":"u","ü":"u","ç":"c"}
    for a,b in repl.items(): text = text.replace(a,b)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def shorten_text(text: str, limit: int = 12000) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def resolve_path(raw: Any, base_dir: Path | None = None) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(os.path.expanduser(s))
    if not p.is_absolute():
        p = (base_dir or Path.cwd()) / p
    return p.resolve()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(value or ""))


def normalize_snippet_placeholders(text: str) -> str:
    text = re.sub(r"\$\{\d+:([^{}]*)\}", lambda m: m.group(1).strip(), text or "")
    text = re.sub(r"\$\{\d+\}", "", text)
    return text


def safe_filename(name: str) -> str:
    return slugify(Path(name).stem) + Path(name).suffix.lower()

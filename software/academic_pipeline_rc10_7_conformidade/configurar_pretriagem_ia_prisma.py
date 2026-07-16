#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configura pré-triagem assistida por IA em um TOML PRISMA existente.

O script não lê nem grava credenciais. Ele apenas adiciona campos públicos à
seção [busca_prisma], valida o TOML proposto e cria backup antes da escrita.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
from datetime import datetime
from pathlib import Path

SETTINGS: list[tuple[str, str]] = [
    ("pre_triagem_ia", "true"),
    ("pre_triagem_ia_modelo", '""'),
    ("pre_triagem_ia_lote", "20"),
    ("pre_triagem_ia_max_registros", "1500"),
    ("pre_triagem_ia_reserva_incertos", "40"),
    ("pre_triagem_ia_min_confianca", "55"),
    ("pre_triagem_ia_max_chars_resumo", "700"),
    ("semantic_scholar_min_interval", "1.05"),
]
COMMENT = "# A IA apenas prioriza a leitura; inclusão/exclusão final permanece humana."


def _section_bounds(lines: list[str]) -> tuple[int, int]:
    start = -1
    for index, line in enumerate(lines):
        if line.strip() == "[busca_prisma]":
            start = index
            break
    if start < 0:
        raise RuntimeError("A seção [busca_prisma] não foi encontrada no TOML.")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].lstrip()
        if stripped.startswith("["):
            end = index
            break
    return start, end


def _render_updated(text: str) -> str:
    had_newline = text.endswith("\n")
    lines = text.splitlines()
    start, end = _section_bounds(lines)
    values = dict(SETTINGS)
    seen: set[str] = set()
    pattern = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=).*$", flags=re.ASCII)
    for index in range(start + 1, end):
        match = pattern.match(lines[index])
        if not match:
            continue
        key = match.group(2)
        if key not in values:
            continue
        lines[index] = f"{match.group(1)}{key}{match.group(3)} {values[key]}"
        seen.add(key)
    insertion = []
    if COMMENT not in lines[start + 1:end]:
        insertion.append(COMMENT)
    insertion.extend(f"{key} = {value}" for key, value in SETTINGS if key not in seen)
    if insertion:
        lines[end:end] = insertion + [""]
    rendered = "\n".join(lines)
    return rendered + ("\n" if had_newline or rendered else "")


def _validate_toml(text: str) -> None:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"A atualização proposta gerou TOML inválido: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Caminho para o TOML PRISMA")
    parser.add_argument("--dry-run", action="store_true", help="Mostra a estratégia sem alterar o arquivo")
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TOML não encontrado: {path}")
    original = path.read_text(encoding="utf-8")
    rendered = _render_updated(original)
    _validate_toml(rendered)

    print("Pré-triagem assistida por IA proposta:")
    print("- ativa antes do corte da planilha de triagem")
    print("- lote: 20 registros; máximo: 1500 registros deduplicados")
    print("- reserva humana: 40 itens incertos/falhos; confiança mínima: 55/100")
    print("- resumo enviado à IA: até 700 caracteres por registro")
    print("- Semantic Scholar: intervalo mínimo de 1,05 s por requisição")
    print("- inclusão/exclusão final: decisão humana obrigatória")
    if args.dry_run:
        print("Dry-run concluído: nenhum arquivo foi alterado.")
        return 0

    backup = path.with_name(path.name + ".bak_pretriagem_ia_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(path, backup)
    try:
        path.write_text(rendered, encoding="utf-8")
        with path.open("rb") as handle:
            tomllib.load(handle)
    except Exception:
        shutil.copy2(backup, path)
        raise
    print(f"TOML atualizado: {path}")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(1)

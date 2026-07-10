#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI de diagnóstico/estado do fluxo de artigo PRISMA."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from article_workflow import ArticleWorkflow


def build_workflow(args: argparse.Namespace) -> ArticleWorkflow:
    return ArticleWorkflow(
        Path(args.art_dir),
        cfg_art=Path(args.cfg_art).expanduser().resolve() if args.cfg_art else None,
        prisma_cfg=Path(args.prisma_cfg).expanduser().resolve() if args.prisma_cfg else None,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estado e validação do fluxo Artigo PRISMA.")
    parser.add_argument("action", choices=["status", "validate", "mark-reviewed"], help="Ação a executar.")
    parser.add_argument("--art-dir", required=True, help="Pasta do artigo.")
    parser.add_argument("--cfg-art", default="", help="TOML do artigo final.")
    parser.add_argument("--prisma-cfg", default="", help="TOML PRISMA ativo/preliminar.")
    parser.add_argument("--xlsx", default="", help="XLSX revisado para mark-reviewed.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    wf = build_workflow(args)
    if args.action in {"status", "validate"}:
        print(wf.format_status())
        if args.action == "validate":
            results = wf.validations()
            return 0 if all(item.ok for item in results) else 1
        return 0

    if args.action == "mark-reviewed":
        if not args.xlsx:
            raise SystemExit("ERRO: informe --xlsx para confirmar a revisão humana.")
        wf.mark_human_review(Path(args.xlsx))
        print("[OK] Revisão humana do XLSX confirmada.")
        print(wf.format_status())
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

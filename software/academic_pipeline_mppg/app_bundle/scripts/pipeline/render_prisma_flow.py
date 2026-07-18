#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import html
from pathlib import Path
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .prisma_model import PrismaFlow, PrismaReport
else:
    from prisma_model import PrismaFlow, PrismaReport


def flow_items(flow: PrismaFlow) -> list[tuple[str, int]]:
    return [
        ("Registros identificados", flow.identificados),
        ("Duplicados removidos", flow.duplicados_removidos),
        ("Registros após deduplicação", flow.apos_deduplicacao),
        ("Registros triados por título/resumo", flow.triados_titulo_resumo),
        ("Excluídos na triagem", flow.excluidos_titulo_resumo),
        ("Textos completos avaliados", flow.avaliados_texto_completo),
        ("Excluídos após texto completo", flow.excluidos_texto_completo),
        ("Estudos incluídos", flow.incluidos),
    ]


def render_prisma_flow_latex(report: PrismaReport) -> str:
    """Fluxograma em LaTeX simples, sem depender de TikZ/Graphviz."""
    blocks: list[str] = [r"\begin{center}", r"\setlength{\fboxsep}{8pt}"]
    for idx, (label, count) in enumerate(flow_items(report.flow)):
        blocks.append(r"\fbox{\begin{minipage}{0.78\textwidth}\centering")
        blocks.append(r"\textbf{" + label.replace("_", r"\_") + r"}\\")
        blocks.append(str(count))
        blocks.append(r"\end{minipage}}")
        if idx < len(flow_items(report.flow)) - 1:
            blocks.append(r"\\[0.45em]$\downarrow$\\[0.45em]")
    blocks.append(r"\end{center}")
    return "\n".join(blocks)


def render_prisma_flow_svg(report: PrismaReport, output_path: Path) -> Path:
    items = flow_items(report.flow)
    width = 760
    box_w = 520
    box_h = 64
    gap = 34
    top = 32
    height = top * 2 + len(items) * box_h + (len(items) - 1) * gap
    x = (width - box_w) // 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif;} .label{font-size:17px;font-weight:700;} .count{font-size:16px;} .arrow{font-size:24px;}</style>',
    ]
    y = top
    for idx, (label, count) in enumerate(items):
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="10" ry="10" fill="#F8FAFC" stroke="#1E3A8A" stroke-width="1.5"/>')
        parts.append(f'<text class="label" x="{width/2}" y="{y+27}" text-anchor="middle" fill="#111827">{html.escape(label)}</text>')
        parts.append(f'<text class="count" x="{width/2}" y="{y+50}" text-anchor="middle" fill="#111827">{count}</text>')
        if idx < len(items) - 1:
            parts.append(f'<text class="arrow" x="{width/2}" y="{y+box_h+25}" text-anchor="middle" fill="#1E3A8A">↓</text>')
        y += box_h + gap
    parts.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path

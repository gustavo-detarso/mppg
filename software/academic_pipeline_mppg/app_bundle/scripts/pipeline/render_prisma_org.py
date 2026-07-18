#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .prisma_model import PrismaReport, StudyRecord
else:
    from prisma_model import PrismaReport, StudyRecord
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .render_prisma_flow import render_prisma_flow_latex
else:
    from render_prisma_flow import render_prisma_flow_latex
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import latex_escape, write_text
else:
    from utils import latex_escape, write_text
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .render_org_latex import (
        bibliography_style_from_cfg,
        render_bibliography_preamble,
        strip_org_cite_export_lines,
    )
else:
    from render_org_latex import (
        bibliography_style_from_cfg,
        render_bibliography_preamble,
        strip_org_cite_export_lines,
    )


def _table(headers: list[str], rows: list[list[str]], name: str = "", caption: str = "") -> str:
    lines: list[str] = []
    if caption:
        lines.append(f"#+CAPTION: {caption}")
    if name:
        lines.append(f"#+NAME: {name}")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for row in rows:
        clean = [str(cell or "").replace("\n", " ").replace("|", "/") for cell in row]
        lines.append("| " + " | ".join(clean) + " |")
    return "\n".join(lines)


def _study_rows(studies: list[StudyRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for idx, study in enumerate(studies, start=1):
        rows.append([
            str(idx),
            study.titulo,
            "; ".join(study.autores),
            study.ano,
            study.doi,
            study.bib_key,
            study.justificativa or study.motivo,
        ])
    return rows


def render_prisma_org(report: PrismaReport, output_path: Path, bib_filename: str | None = None, cfg: dict | None = None) -> str:
    meta = report.metadata
    bib_name = Path(bib_filename).name if bib_filename else ""
    style = bibliography_style_from_cfg(cfg, None)
    bib_lines = render_bibliography_preamble(bib_name, style, cfg) if bib_name else []
    lines: list[str] = [
        f"#+TITLE: {meta.titulo}",
        f"#+AUTHOR: {meta.responsavel}",
        "#+LANGUAGE: pt_BR",
        "#+OPTIONS: toc:nil num:t ^:nil",
        "#+LATEX_CLASS: fgv-paper",
        *bib_lines,
        "#+LATEX_HEADER: \\usepapercover",
        f"#+LATEX_HEADER: \\institution{{{latex_escape(meta.instituicao)}}}",
        "#+LATEX_HEADER: \\programname{}",
        f"#+LATEX_HEADER: \\coursename{{{latex_escape(meta.curso)}}}",
        f"#+LATEX_HEADER: \\disciplinename{{{latex_escape(meta.disciplina)}}}",
        f"#+LATEX_HEADER: \\professorname{{{latex_escape(meta.professor)}}}",
        f"#+LATEX_HEADER: \\cityname{{{latex_escape(meta.cidade)}}}",
        "#+LATEX_HEADER: \\papertype{Relatório de pesquisa}",
        "#+LATEX_HEADER: \\covernote{Relatório metodológico de busca, triagem e seleção de estudos.}",
        "",
        "#+LATEX: \\makemytitle",
        "",
        "* Introdução do relatório",
        "Este relatório apresenta a trilha metodológica de identificação, triagem, elegibilidade e inclusão dos textos utilizados na pesquisa. O objetivo é registrar de forma auditável as bases ou fontes consultadas, os critérios de inclusão e exclusão, os totais do fluxo e os estudos efetivamente incorporados ao corpus.",
        "",
        "* Tema, recorte e objetivo da busca",
        f"- Tema: {meta.tema or 'Não informado'}",
        f"- Recorte: {meta.recorte or 'Não informado'}",
        f"- Objetivo: {meta.objetivo or 'Não informado'}",
        f"- Pergunta de pesquisa: {meta.pergunta_pesquisa or 'Não informada'}",
        "",
        "* Estratégia de busca",
        f"- Bases/fontes: {'; '.join(report.search_strategy.bases) if report.search_strategy.bases else 'Não informado'}",
        f"- Idiomas: {'; '.join(report.search_strategy.idiomas) if report.search_strategy.idiomas else 'Não informado'}",
        f"- Período: {report.search_strategy.periodo or 'Não informado'}",
        "",
    ]
    if report.search_strategy.queries:
        lines.append(_table(
            ["Base", "Consulta", "Resultados brutos", "Observações"],
            [[q.base, q.query, str(q.resultados_brutos), q.observacoes] for q in report.search_strategy.queries],
            name="tab:queries_prisma",
            caption="Consultas e fontes utilizadas na etapa de busca.",
        ))
        lines.append("")
    lines += [
        "* Critérios de inclusão e exclusão",
        "** Critérios de inclusão",
        *[f"- {item}" for item in report.criteria.inclusao],
        "",
        "** Critérios de exclusão",
        *[f"- {item}" for item in report.criteria.exclusao],
        "",
        "* Resultados da busca e da triagem",
        _table(
            ["Etapa", "Total"],
            [
                ["Registros identificados", str(report.flow.identificados)],
                ["Duplicados removidos", str(report.flow.duplicados_removidos)],
                ["Registros após deduplicação", str(report.flow.apos_deduplicacao)],
                ["Registros triados por título/resumo", str(report.flow.triados_titulo_resumo)],
                ["Excluídos na triagem", str(report.flow.excluidos_titulo_resumo)],
                ["Textos completos avaliados", str(report.flow.avaliados_texto_completo)],
                ["Excluídos após texto completo", str(report.flow.excluidos_texto_completo)],
                ["Estudos incluídos", str(report.flow.incluidos)],
            ],
            name="tab:fluxo_prisma_totais",
            caption="Totais consolidados do fluxo de identificação, triagem, elegibilidade e inclusão.",
        ),
        "",
        "* Fluxograma PRISMA",
        "#+begin_export latex",
        render_prisma_flow_latex(report),
        "#+end_export",
        "",
        "* Estudos incluídos",
    ]
    lines.append(_table(
        ["#", "Título", "Autores", "Ano", "DOI", "Chave", "Justificativa"],
        _study_rows(report.included_studies),
        name="tab:estudos_incluidos",
        caption="Estudos incluídos no corpus final.",
    ) if report.included_studies else "Nenhum estudo incluído registrado.")
    lines += ["", "* Estudos excluídos"]
    lines.append(_table(
        ["#", "Título", "Autores", "Ano", "DOI", "Chave", "Motivo"],
        _study_rows(report.excluded_studies),
        name="tab:estudos_excluidos",
        caption="Estudos excluídos e justificativas de exclusão.",
    ) if report.excluded_studies else "Nenhum estudo excluído registrado.")
    lines += [
        "",
        "* Diagnóstico metodológico",
        f"- Fontes de artefatos: {'; '.join(report.diagnostics.fontes_artefatos) if report.diagnostics.fontes_artefatos else 'Não informado'}",
        f"- Bases com erro: {'; '.join(report.diagnostics.bases_com_erro) if report.diagnostics.bases_com_erro else 'Nenhuma registrada'}",
        f"- Avisos: {'; '.join(report.diagnostics.avisos) if report.diagnostics.avisos else 'Nenhum'}",
        "",
    ]
    if bib_name:
        lines += ["* Referências", "#+LATEX: \\printbibliography"]
    org = "\n".join(ln for ln in lines if ln is not None)
    org = org.replace("\n\n\n", "\n\n").strip() + "\n"
    org = strip_org_cite_export_lines(org)
    write_text(output_path, org)
    return org

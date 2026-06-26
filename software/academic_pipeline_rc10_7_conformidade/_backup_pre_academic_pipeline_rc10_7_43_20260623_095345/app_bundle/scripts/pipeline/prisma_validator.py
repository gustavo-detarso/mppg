#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from prisma_model import PrismaReport


def validate_prisma_report(report: PrismaReport, *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    if not report.metadata.titulo.strip():
        errors.append("Relatório sem título.")
    if report.flow.incluidos != len(report.included_studies):
        warnings.append(
            f"Total de incluídos no fluxo ({report.flow.incluidos}) difere da lista de incluídos ({len(report.included_studies)})."
        )
    if report.flow.identificados < report.flow.incluidos:
        errors.append("Total de identificados menor que total de incluídos.")
    if report.flow.apos_deduplicacao and report.flow.apos_deduplicacao > report.flow.identificados:
        errors.append("Total após deduplicação maior que total identificado.")
    for idx, study in enumerate(report.included_studies, start=1):
        if not study.titulo.strip():
            errors.append(f"Estudo incluído #{idx} sem título.")
        if not study.justificativa.strip():
            warnings.append(f"Estudo incluído sem justificativa: {study.titulo[:80]}")
        if study.doi and not re.match(r"^10\.\d{4,9}/\S+$", study.doi):
            warnings.append(f"DOI possivelmente malformado em estudo incluído: {study.titulo[:80]} → {study.doi}")
    for idx, study in enumerate(report.excluded_studies, start=1):
        if not study.titulo.strip():
            errors.append(f"Estudo excluído #{idx} sem título.")
        if not study.motivo.strip():
            warnings.append(f"Estudo excluído sem motivo: {study.titulo[:80]}")
    if not report.search_strategy.bases:
        warnings.append("Relatório sem bases/ fontes de busca registradas.")
    if strict and errors:
        return errors + ["Aviso: " + w for w in warnings]
    return errors + ["Aviso: " + w for w in warnings]


def raise_if_prisma_errors(messages: list[str], title: str = "Validação do relatório PRISMA falhou") -> None:
    hard = [m for m in messages if not m.startswith("Aviso:")]
    if hard:
        raise RuntimeError(title + ":\n- " + "\n- ".join(messages))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modelo canônico do relatório de pesquisa/PRISMA — academic_pipeline rc10.

O relatório de pesquisa é um artefato autônomo: registra estratégia de busca,
triagem, elegibilidade, incluídos, excluídos, diagnósticos e totais do fluxo.
Ele não substitui o documento acadêmico final; é a trilha metodológica
auditável que pode ser exportada para JSON/ORG/PDF/DOCX/XLSX.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class PrismaMetadata(BaseModel):
    titulo: str = "Relatório de Pesquisa PRISMA"
    tema: str = ""
    recorte: str = ""
    objetivo: str = ""
    pergunta_pesquisa: str = ""
    responsavel: str = "Gustavo M. Mendes de Tarso"
    instituicao: str = "Fundação Getúlio Vargas"
    curso: str = "Mestrado Acadêmico em Políticas Públicas e Governo"
    disciplina: str = ""
    professor: str = ""
    cidade: str = "Brasília"
    data_execucao: str = ""
    tipo_relatorio: str = "prisma"


class QueryRecord(BaseModel):
    base: str
    query: str
    resultados_brutos: int = 0
    filtros: dict[str, Any] = Field(default_factory=dict)
    observacoes: str = ""


class SearchStrategy(BaseModel):
    bases: list[str] = Field(default_factory=list)
    idiomas: list[str] = Field(default_factory=list)
    periodo: str = ""
    queries: list[QueryRecord] = Field(default_factory=list)
    notas: str = ""


class ScreeningCriteria(BaseModel):
    inclusao: list[str] = Field(default_factory=list)
    exclusao: list[str] = Field(default_factory=list)


class PrismaFlow(BaseModel):
    identificados: int = 0
    duplicados_removidos: int = 0
    apos_deduplicacao: int = 0
    triados_titulo_resumo: int = 0
    excluidos_titulo_resumo: int = 0
    avaliados_texto_completo: int = 0
    excluidos_texto_completo: int = 0
    incluidos: int = 0

    @model_validator(mode="after")
    def normalize_counts(self) -> "PrismaFlow":
        if self.apos_deduplicacao == 0 and self.identificados:
            self.apos_deduplicacao = max(0, self.identificados - self.duplicados_removidos)
        if self.triados_titulo_resumo == 0:
            self.triados_titulo_resumo = self.apos_deduplicacao
        if self.excluidos_titulo_resumo == 0 and self.triados_titulo_resumo and self.avaliados_texto_completo:
            self.excluidos_titulo_resumo = max(0, self.triados_titulo_resumo - self.avaliados_texto_completo)
        if self.avaliados_texto_completo == 0:
            self.avaliados_texto_completo = self.incluidos + self.excluidos_texto_completo
        return self


class StudyRecord(BaseModel):
    bib_key: str = ""
    titulo: str = ""
    autores: list[str] = Field(default_factory=list)
    ano: str = ""
    doi: str = ""
    url: str = ""
    base: str = ""
    fonte: str = ""
    arquivo_local: str = ""
    score_aderencia: float | None = None
    decisao: Literal["incluido", "elegivel", "excluido", "duplicado", "triado", "identificado"] = "incluido"
    motivo: str = ""
    justificativa: str = ""
    resumo: str = ""
    metadados: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bib_key")
    @classmethod
    def normalize_key(cls, value: str) -> str:
        return str(value or "").strip().lstrip("@")

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        doi = str(value or "").strip()
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
        doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
        return doi.strip().strip(".")


class Diagnostics(BaseModel):
    bases_com_erro: list[str] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
    fontes_artefatos: list[str] = Field(default_factory=list)
    parametros: dict[str, Any] = Field(default_factory=dict)


class PrismaReport(BaseModel):
    metadata: PrismaMetadata
    search_strategy: SearchStrategy = Field(default_factory=SearchStrategy)
    criteria: ScreeningCriteria = Field(default_factory=ScreeningCriteria)
    flow: PrismaFlow = Field(default_factory=PrismaFlow)
    included_studies: list[StudyRecord] = Field(default_factory=list)
    excluded_studies: list[StudyRecord] = Field(default_factory=list)
    duplicate_studies: list[StudyRecord] = Field(default_factory=list)
    all_records: list[StudyRecord] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    @model_validator(mode="after")
    def normalize_report(self) -> "PrismaReport":
        if self.flow.incluidos == 0 and self.included_studies:
            self.flow.incluidos = len(self.included_studies)
        if self.flow.identificados == 0:
            self.flow.identificados = len(self.all_records) or len(self.included_studies) + len(self.excluded_studies) + len(self.duplicate_studies)
        if self.flow.duplicados_removidos == 0 and self.duplicate_studies:
            self.flow.duplicados_removidos = len(self.duplicate_studies)
        if self.flow.excluidos_texto_completo == 0 and self.excluded_studies:
            self.flow.excluidos_texto_completo = len([s for s in self.excluded_studies if s.decisao == "excluido"])
        self.flow = PrismaFlow.model_validate(self.flow.model_dump())
        return self

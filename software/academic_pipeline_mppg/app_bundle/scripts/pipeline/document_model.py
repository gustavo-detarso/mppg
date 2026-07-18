#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modelo canônico do documento acadêmico — academic_pipeline rc10.

A IA deve gerar este modelo estruturado. Renderizadores determinísticos
transformam o mesmo modelo em ORG/LaTeX/PDF, DOCX e outros formatos.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import re


class StrictBaseModel(BaseModel):
    """Base estrita compatível com OpenAI Structured Outputs.

    O endpoint Responses exige additionalProperties=false em todos os
    objetos do schema. Por isso, evitamos dict[str, Any] e proibimos
    propriedades extras nos modelos que a IA deve retornar.
    """
    model_config = ConfigDict(extra="forbid")


DocumentType = Literal["paper", "atividade", "dissertacao", "pesquisa", "relatorio_pesquisa", "resumo", "fichamento", "ensaio", "resposta_discursiva"]
CitationMode = Literal["parenthetical", "narrative", "author", "year"]
BlockType = Literal[
    "paragraph",
    "heading",
    "quote",
    "bullet_list",
    "numbered_list",
    "table",
    "figure",
    "page_break",
]
FigurePlacement = Literal["inline", "after_references", "before_references", "appendix"]


class Citation(StrictBaseModel):
    type: Literal["citation"] = "citation"
    mode: CitationMode = "parenthetical"
    keys: list[str] = Field(default_factory=list)
    prefix: str = ""
    suffix: str = ""

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for key in value or []:
            k = str(key).strip().lstrip("@").strip()
            if k and k not in cleaned:
                cleaned.append(k)
        return cleaned


class TextSpan(StrictBaseModel):
    type: Literal["text"] = "text"
    text: str = ""
    italic: bool = False
    bold: bool = False


Inline = TextSpan | Citation


class TableData(StrictBaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str = ""


class Block(StrictBaseModel):
    type: BlockType
    text: str = ""
    content: list[Inline] = Field(default_factory=list)
    level: int = 1
    items: list[str] = Field(default_factory=list)
    table: TableData | None = None
    id: str = ""
    title: str = ""
    path: str = ""
    placement: FigurePlacement = "inline"
    page_break_before: bool = False
    page_break_after: bool = False

    @model_validator(mode="after")
    def normalize_block(self) -> "Block":
        if self.type == "paragraph" and not self.content and self.text:
            self.content = [TextSpan(text=self.text)]
        if self.type == "heading":
            self.level = max(1, min(int(self.level or 1), 6))
        return self


class Section(StrictBaseModel):
    id: str = ""
    level: int = 1
    title: str
    blocks: list[Block] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_section(self) -> "Section":
        if not self.id:
            self.id = slugify(self.title)
        self.level = max(1, min(int(self.level or 1), 6))
        return self


class DocumentMetadata(StrictBaseModel):
    tipo_documento: DocumentType = "paper"
    titulo: str
    subtitulo: str = ""
    autor: str = "Gustavo M. Mendes de Tarso"
    instituicao: str = "Fundação Getúlio Vargas"
    programa: str = ""
    curso: str = "Mestrado Acadêmico em Políticas Públicas e Governo"
    turma: str = ""
    polo: str = "Brasília"
    disciplina: str = ""
    professor: str = ""
    cidade: str = "Brasília"
    ano: str = ""
    data: str = ""
    tipo_trabalho: str = "Paper acadêmico"
    nota_capa: str = "Trabalho acadêmico elaborado para a disciplina."
    idioma: str = "pt_BR"


class AbstractBlock(StrictBaseModel):
    titulo: str = "Resumo"
    texto: str = ""
    palavras_chave: list[str] = Field(default_factory=list)


class FigureSpec(StrictBaseModel):
    id: str
    title: str
    path: str
    placement: FigurePlacement = "inline"
    page_break_before: bool = False
    page_break_after: bool = False


class BibliographyInfo(StrictBaseModel):
    bib_path: str = ""
    style: str = "apa"
    entries_used: list[str] = Field(default_factory=list)


class DiagnosticsInfo(StrictBaseModel):
    """Diagnósticos serializados após a geração.

    Não usamos dict livre aqui porque isso quebra o schema estrito exigido
    pelo Responses API para text_format=AcademicDocument.
    Campos JSON ficam como string e são preenchidos pelo pipeline depois
    que a IA retorna o documento canônico.
    """
    prompts_json: str = ""
    mindmap_json: str = ""
    source_info_json: str = ""
    relatorio_pesquisa_json: str = ""
    warnings: list[str] = Field(default_factory=list)


class AcademicDocument(StrictBaseModel):
    metadata: DocumentMetadata
    abstract: AbstractBlock | None = None
    sections: list[Section] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    bibliography: BibliographyInfo = Field(default_factory=BibliographyInfo)
    diagnostics: DiagnosticsInfo = Field(default_factory=DiagnosticsInfo)

    @model_validator(mode="after")
    def validate_document(self) -> "AcademicDocument":
        if not self.sections:
            raise ValueError("O documento canônico precisa conter ao menos uma seção.")
        return self


def slugify(text: str) -> str:
    text = str(text or "").strip().lower()
    replacements = {
        "á": "a", "à": "a", "â": "a", "ã": "a", "é": "e", "ê": "e", "í": "i",
        "ó": "o", "ô": "o", "õ": "o", "ú": "u", "ü": "u", "ç": "c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "secao"

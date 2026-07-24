# Adaptador nativo isolado de --list-profiles (AP-007D.2).
from __future__ import annotations
from collections.abc import Sequence

import textwrap
from dataclasses import dataclass

@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    document_type: str
    local_corpus: bool
    prisma_report: bool
    executar_documento: bool
    executar_pesquisa: bool
    render_only: bool = False
    default_toml: str = 'config.toml'

PRESETS: list[Preset] = [Preset(key='atividade_local_fgv', label='Atividade local FGV', description='Gera atividade acadêmica com Ficha Técnica FGV a partir de corpus local. Não faz busca externa nem relatório PRISMA.', document_type='atividade', local_corpus=True, prisma_report=False, executar_documento=True, executar_pesquisa=False, default_toml='atividade_config.toml'), Preset(key='resumo_artigos_local_fgv', label='Resumo analítico de artigos locais FGV', description='Gera documento FGV com Ficha Técnica, introdução, resumos individuais, comparação, síntese analítica, considerações finais, referências e mapa mental opcional a partir de corpus local. Não faz busca externa nem relatório PRISMA.', document_type='atividade', local_corpus=True, prisma_report=False, executar_documento=True, executar_pesquisa=False, default_toml='resumo_artigos_config.toml'), Preset(key='paper_local_fgv', label='Paper local FGV', description='Gera paper acadêmico FGV a partir de PDFs/DOCX/TXT locais, com ORG/PDF/DOCX, conformidade e qualidade.', document_type='paper', local_corpus=True, prisma_report=False, executar_documento=True, executar_pesquisa=False, default_toml='paper_config.toml'), Preset(key='paper_prisma_fgv', label='Paper + relatório PRISMA FGV', description='Gera paper e saída própria de relatório PRISMA. O relatório usa corpus local, prisma_report.json ou diretório de pesquisa/triagem existente.', document_type='paper', local_corpus=True, prisma_report=True, executar_documento=True, executar_pesquisa=True, default_toml='paper_prisma_config.toml'), Preset(key='dissertacao_local_fgv', label='Dissertação local FGV', description='Gera dissertação FGV a partir de corpus local, com campos de orientador, área, linha e pré-textuais básicos.', document_type='dissertacao', local_corpus=True, prisma_report=False, executar_documento=True, executar_pesquisa=False, default_toml='dissertacao_config.toml'), Preset(key='dissertacao_prisma_fgv', label='Dissertação + relatório PRISMA FGV', description='Gera dissertação e relatório PRISMA auditável a partir de corpus local ou dados de pesquisa existentes.', document_type='dissertacao', local_corpus=True, prisma_report=True, executar_documento=True, executar_pesquisa=True, default_toml='dissertacao_prisma_config.toml'), Preset(key='relatorio_prisma_fgv', label='Relatório PRISMA autônomo FGV', description='Gera apenas o relatório de pesquisa/PRISMA. Observação: a versão atual ainda exige corpus local, prisma_report.json ou diretório de pesquisa existente como insumo.', document_type='relatorio_prisma', local_corpus=True, prisma_report=True, executar_documento=False, executar_pesquisa=True, default_toml='relatorio_prisma_config.toml'), Preset(key='relatorio_prisma_busca_orientada_fgv', label='Relatório PRISMA com busca orientada FGV', description='Executa busca bibliográfica externa rastreável nas fontes configuráveis, deduplica registros, gera triagem humana e produz relatório PRISMA preliminar/final em ORG e PDF no layout institucional escolhido. O fluxo começa pela descoberta externa e não exige corpus local.', document_type='relatorio_prisma', local_corpus=False, prisma_report=True, executar_documento=False, executar_pesquisa=True, default_toml='relatorio_prisma_busca_orientada_config.toml'), Preset(key='somente_renderizar_fgv', label='Somente renderizar document.json FGV', description='Cria TOML para renderizar um document.json existente em ORG/PDF/DOCX, sem recriar conteúdo pela IA.', document_type='paper', local_corpus=False, prisma_report=False, executar_documento=True, executar_pesquisa=False, render_only=True, default_toml='render_config.toml')]

def print_profiles() -> None:
    for p in PRESETS:
        print(f'{p.key}: {p.label}')
        print(textwrap.fill('  ' + p.description, width=92, subsequent_indent='  '))
        print()

__all__ = ["run_list_profiles_command"]

def run_list_profiles_command(argv: Sequence[str] | None = None) -> int:
    explicit_argv = list(argv or [])
    unexpected = [item for item in explicit_argv if item != "--list-profiles"]
    if unexpected:
        raise ValueError(f"argumentos não suportados: {unexpected!r}")
    print_profiles()
    return 0
    return 0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
academic_pipeline_toml_generator.py

Gerador, validador e assistente interativo de arquivos TOML para o software
academic_pipeline / gerador acadêmico MPPG.

Recursos:
- modo interativo guiado;
- geração rápida de modelos comentados por perfil;
- validação estrutural de TOML existente;
- autocomplete de caminhos no terminal quando o usuário informa arquivos/pastas;
- salvamento perguntado ao final quando --output não é informado.

Exemplos:
  python academic_pipeline_toml_generator.py --interativo
  python academic_pipeline_toml_generator.py --modelo-comentado --perfil somente_mapa_mental
  python academic_pipeline_toml_generator.py --perfil derivacao_dissertacao --output derivacao_atestmed.toml
  python academic_pipeline_toml_generator.py --validar atividade_aula_2.toml

Compatibilidade: Python 3.11+.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import glob
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

try:
    import readline  # type: ignore
except Exception:  # pragma: no cover
    readline = None  # type: ignore

APP_NAME = "academic_pipeline_toml_generator"
APP_VERSION = "0.2.1"

PROFILE_INFO: dict[str, dict[str, str]] = {
    "atividade_local": {
        "titulo": "Atividade/documento a partir de ZIP ou pasta local",
        "fluxo": "Lê ZIP ou pasta local, monta corpus local, gera .bib quando necessário, cria .org acadêmico e exporta PDF.",
        "quando_usar": "Use para atividades de aula, fichamentos, respostas discursivas ou textos curtos baseados em PDFs/DOCX locais.",
    },
    "pesquisa_prisma_documento": {
        "titulo": "Pesquisa PRISMA + geração de documento",
        "fluxo": "Executa ou reaproveita a etapa de pesquisa, usa triagem/bibliografia e gera um documento acadêmico final.",
        "quando_usar": "Use quando o documento depende de busca estruturada, triagem e fontes selecionadas pela pesquisa.",
    },
    "pesquisa_dissertacao_excelencia": {
        "titulo": "Pesquisa PRISMA robusta + dissertação FGV com camada de excelência",
        "fluxo": "Configura pesquisa PRISMA ampliada, bases múltiplas, triagem orientada, template FGV, metas longas de dissertação e exportação LaTeX/Org.",
        "quando_usar": "Use para dissertação longa e formal do MPPG/FGV baseada em revisão robusta, com alto controle de busca, saída e formatação.",
    },
    "paper": {
        "titulo": "Paper/artigo acadêmico",
        "fluxo": "Gera artigo/paper a partir de pesquisa existente, corpus local ou fontes orientadas, com menor extensão e estrutura mais enxuta.",
        "quando_usar": "Use para artigos, papers de disciplina, ensaios acadêmicos ou texto publicável mais curto.",
    },
    "dissertacao": {
        "titulo": "Dissertação completa a partir de pesquisa/fontes",
        "fluxo": "Gera dissertação a partir de fontes/pesquisa, com opção de escrita em etapas, metas de extensão e exportação PDF.",
        "quando_usar": "Use para dissertação sem a camada PRISMA/excelência completa ou para partir de corpus já preparado.",
    },
    "somente_mapa_mental": {
        "titulo": "Inserir/atualizar mapa mental em .org existente",
        "fluxo": "Lê um .org já gerado, cria mapa PlantUML, renderiza imagem, insere em *_com_mapa.org e recompila sem sobrescrever o original.",
        "quando_usar": "Use quando o texto já está pronto e você quer apenas acrescentar ou atualizar o mapa mental.",
    },
    "derivacao_dissertacao": {
        "titulo": "Derivar/reorientar dissertação para novo objeto empírico",
        "fluxo": "Usa uma dissertação existente como matriz intelectual, preserva o original, incorpora novas orientações/dados/fontes e gera outro .org/.bib/PDF.",
        "quando_usar": "Use para transformar, por exemplo, uma dissertação sobre IA/ESG em outra aplicada ao ATESTMED ou a novo objeto empírico.",
    },
    "recompilar_pdf": {
        "titulo": "Recompilar PDF a partir de .org existente",
        "fluxo": "Aponta para um .org existente e executa a exportação PDF com a configuração LaTeX/Emacs informada.",
        "quando_usar": "Use quando só precisa recompilar, sem chamar IA nem alterar conteúdo.",
    },
    "completo_comentado": {
        "titulo": "Modelo completo comentado",
        "fluxo": "Gera um TOML amplo com as principais seções e comentários, servindo como referência geral de configuração.",
        "quando_usar": "Use como manual de consulta ou ponto de partida para montar configurações muito customizadas.",
    },
}

PROFILES = {key: info["titulo"] for key, info in PROFILE_INFO.items()}

PATH_KEYS = {
    "input_zip",
    "input_dir",
    "output_dir",
    "template_path",
    "template_org",
    "org_modelo",
    "script_pesquisa",
    "bundle_dir",
    "fgv_logo_path",
    "documento_org_existente",
    "contexto_json_existente",
    "fulltext_cache_dir",
    "plantuml_jar_path",
    "documento_org_base",
    "bib_base",
    "latex_extra_path",
    "org_latex_class_init",
    "pesquisa_dir_existente",
    "documento_org_para_recompilar",
}

MAPA_MENTAL_MAIN_KEYS = {
    "plantuml_jar_path",
    "gerar",
    "somente_mapa_mental",
    "documento_org_existente",
    "contexto_json_existente",
    "fulltext_cache_dir",
    "linguagem",
    "formato",
    "renderizar",
    "inserir_no_org",
    "recompilar_pdf",
    "titulo",
    "arquivo",
    "posicao",
    "max_niveis",
    "max_nos",
    "incluir_codigo_fonte",
    "dpi",
    "plantuml_limit_size",
    "falhar_se_nao_renderizar",
    "diretorio_imagens",
    "colorir_niveis",
    "cores_niveis",
    "fonte",
    "font_name",
    "sobrescrever_org_existente",
}

COLOR_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$|^[A-Za-z][A-Za-z0-9_ -]*$")

# ---------------------------------------------------------------------------
# Utilidades de terminal / autocomplete
# ---------------------------------------------------------------------------

def _home() -> str:
    return str(Path.home())


def _display_path_for_completion(path: str, original_text: str) -> str:
    """Preserva ~ quando o usuário começou com ~."""
    path = path.replace(os.sep, "/") if os.sep != "/" else path
    if original_text.startswith("~"):
        home = _home()
        if path.startswith(home):
            return "~" + path[len(home):]
    return path


def _path_completer(text: str, state: int) -> str | None:
    try:
        expanded = os.path.expanduser(text)
        matches = glob.glob(expanded + "*")
        results: list[str] = []
        for match in sorted(matches):
            if os.path.isdir(match):
                match = match.rstrip(os.sep) + os.sep
            results.append(_display_path_for_completion(match, text))
        if state < len(results):
            return results[state]
    except Exception:
        return None
    return None


@contextlib.contextmanager
def path_autocomplete() -> Iterable[None]:
    """Ativa autocomplete de caminho durante um input()."""
    if readline is None:
        yield
        return
    old_completer = readline.get_completer()
    try:
        readline.set_completer(_path_completer)
        # Não quebrar caminhos em /, ., -, _. Evita comportamento ruim para paths.
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
        yield
    finally:
        readline.set_completer(old_completer)


def ask_text(prompt: str, default: str | None = None, *, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            value = default
        if value or not required:
            return value
        print("Valor obrigatório.")


def ask_bool(prompt: str, default: bool = False) -> bool:
    d = "S/n" if default else "s/N"
    while True:
        value = input(f"{prompt} [{d}]: ").strip().lower()
        if not value:
            return default
        if value in {"s", "sim", "y", "yes", "1", "true"}:
            return True
        if value in {"n", "nao", "não", "no", "0", "false"}:
            return False
        print("Responda com sim ou não.")


def ask_path(prompt: str, default: str | None = None, *, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        with path_autocomplete():
            value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            value = default
        if value or not required:
            return value
        print("Caminho obrigatório.")


def ask_path_list(prompt: str) -> list[str]:
    print(prompt)
    print("Informe um caminho por vez. Pressione Enter vazio para encerrar.")
    items: list[str] = []
    while True:
        item = ask_path(f"  caminho {len(items) + 1}")
        if not item:
            break
        items.append(item)
    return items


def print_profile_explanation(profile: str, *, compact: bool = False) -> None:
    info = PROFILE_INFO.get(profile, {})
    titulo = info.get("titulo", profile)
    fluxo = info.get("fluxo", "")
    quando = info.get("quando_usar", "")
    if compact:
        print(f"{profile}: {titulo}")
        print(f"  Fluxo: {fluxo}")
        print(f"  Uso: {quando}")
    else:
        print(f"\n{profile} — {titulo}")
        print(f"Fluxo: {fluxo}")
        print(f"Quando usar: {quando}")


def choose_profile() -> str:
    print("\nPerfis disponíveis:")
    keys = list(PROFILES.keys())
    for i, key in enumerate(keys, start=1):
        print(f"  {i}. {key} — {PROFILES[key]}")
        print(f"     Fluxo: {PROFILE_INFO[key]['fluxo']}")
    while True:
        raw = input("Escolha o perfil: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            chosen = keys[int(raw) - 1]
            print_profile_explanation(chosen)
            return chosen
        if raw in PROFILES:
            print_profile_explanation(raw)
            return raw
        print("Perfil inválido.")

# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------

def toml_quote(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def toml_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def toml_array(values: list[str], *, indent: str = "  ") -> str:
    if not values:
        return "[]"
    lines = ["["]
    for value in values:
        lines.append(f"{indent}{toml_quote(value)},")
    lines.append("]")
    return "\n".join(lines)


def render_key_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {toml_bool(value)}"
    if isinstance(value, int):
        return f"{key} = {value}"
    if isinstance(value, float):
        return f"{key} = {value}"
    if isinstance(value, list):
        return f"{key} = {toml_array([str(v) for v in value])}"
    return f"{key} = {toml_quote(value)}"


def header(profile: str) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = PROFILE_INFO.get(profile, {})
    titulo = info.get("titulo", profile)
    fluxo = info.get("fluxo", "")
    quando = info.get("quando_usar", "")
    return textwrap.dedent(f"""
    # =============================================================================
    # academic_pipeline — arquivo TOML gerado por {APP_NAME} {APP_VERSION}
    # Perfil: {profile} — {titulo}
    # Gerado em: {now}
    #
    # Fluxo deste perfil:
    # - {fluxo}
    #
    # Quando usar:
    # - {quando}
    #
    # Observações:
    # - Caminhos podem ser absolutos ou relativos ao diretório do TOML.
    # - Evite mover blocos aninhados, como [mapa_mental.cores_por_nivel], para o
    #   meio de uma seção principal. Em TOML, tudo abaixo do cabeçalho pertence
    #   àquela seção até aparecer outro cabeçalho.
    # - Para mapa mental em modo seguro, prefira gerar *_com_mapa.org e nunca
    #   sobrescrever o .org base.
    # =============================================================================
    """).strip() + "\n\n"

# ---------------------------------------------------------------------------
# Templates por perfil
# ---------------------------------------------------------------------------

def section_atividade(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [atividade]
    # Metadados acadêmicos usados em capa/ficha técnica.
    curso = {toml_quote(v.get('curso', 'Mestrado Profissional em Políticas Públicas e Governo'))}
    turma = {toml_quote(v.get('turma', '2026.1'))}
    polo = {toml_quote(v.get('polo', 'Brasília'))}
    disciplina = {toml_quote(v.get('disciplina', 'Políticas Públicas: Teorias e Análises'))}
    professor = {toml_quote(v.get('professor', ''))}
    aluno = {toml_quote(v.get('aluno', 'Gustavo M. Mendes de Tarso'))}
    data = {toml_quote(v.get('data', 'Abril de 2026'))}
    titulo_trabalho = {toml_quote(v.get('titulo_trabalho', 'Atividade Aula 2'))}
    modo = {toml_quote(v.get('modo', 'documentos_locais'))}
    """).strip() + "\n\n"


def section_openai(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [openai]
    # Modelo usado nas etapas de IA. Também pode ser definido por OPENAI_MODEL.
    model = {toml_quote(v.get('model', 'gpt-5.4'))}
    """).strip() + "\n\n"


def section_pipeline(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [pipeline]
    # Modos usuais: "documentos_locais", "pesquisa", "recompilar".
    modo_entrada = {toml_quote(v.get('modo_entrada', 'documentos_locais'))}
    executar_pesquisa = {toml_bool(v.get('executar_pesquisa', False))}
    executar_documento = {toml_bool(v.get('executar_documento', True))}

    # Quando você já tem uma pesquisa pronta, aponte para o diretório dela.
    pesquisa_dir_existente = {toml_quote(v.get('pesquisa_dir_existente', ''))}
    """).strip() + "\n\n"


def section_saida(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [saida]
    # Diretório-base de saída do corpus/pesquisa.
    output_dir = {toml_quote(v.get('output_dir', './output/corpus_local'))}
    prefixo = {toml_quote(v.get('prefixo', 'atividade_aula_2'))}
    criar_subdiretorio = {toml_bool(v.get('criar_subdiretorio', True))}

    # Orientações gerais da pesquisa/documento. Pode ser lista de caminhos ou texto inline.
    orientacoes_paths = {toml_array(v.get('orientacoes_paths', []))}
    orientacao_inline = {toml_quote(v.get('orientacao_inline', ''))}
    """).strip() + "\n\n"


def section_documentos_locais(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [documentos_locais]
    # Use este bloco para gerar atividade/documento a partir de ZIP ou pasta local.
    ativos = {toml_bool(v.get('ativos', True))}
    modo_entrada = {toml_quote('documentos_locais')}

    # Informe UM destes dois caminhos.
    input_zip = {toml_quote(v.get('input_zip', ''))}
    input_dir = {toml_quote(v.get('input_dir', ''))}

    # Tipos de arquivos aceitos no corpus local.
    tipos = ["pdf", "docx", "txt", "md", "org"]
    recursive = true
    limpar_extracao_anterior = true
    copiar_para_fulltext_cache = true
    limpar_cache_anterior = false

    # Se houver .bib ao lado dos arquivos, o gerador tenta detectá-lo.
    auto_detect_bib = true
    gerar_bib_revisado_ia = false
    autor_padrao = "Material fornecido pelo professor"
    ano_padrao = "s.d."
    """).strip() + "\n\n"


def section_pesquisa(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [pesquisa]
    tema = {toml_quote(v.get('tema', 'Tema da pesquisa'))}
    recorte = {toml_quote(v.get('recorte', 'Recorte analítico'))}
    objetivo = {toml_quote(v.get('objetivo', 'Objetivo geral'))}
    pergunta_pesquisa = {toml_quote(v.get('pergunta_pesquisa', ''))}
    hipotese = {toml_quote(v.get('hipotese', ''))}
    palavras_chave = {toml_array(v.get('palavras_chave', []))}
    idiomas = ["português", "inglês"]
    tipo_estudo = {toml_quote(v.get('tipo_estudo', 'revisao_sistematica'))}
    """).strip() + "\n\n"


def section_busca_triagem() -> str:
    return textwrap.dedent("""
    [busca]
    # Configure aqui bases, limites e parâmetros de busca quando for usar pesquisa PRISMA.
    max_resultados_por_base = 50
    usar_semantic_scholar = true
    usar_crossref = true
    usar_openalex = true
    usar_pubmed = false
    usar_core = false

    [triagem]
    # Rigor: "baixo", "moderado", "alto".
    rigor = "moderado"
    max_elegiveis = 30
    max_incluidos = 8
    orientacoes_paths = []
    orientacao_inline = ""

    [queries]
    # Consultas manuais opcionais. Se vazio, o gerador pode criar consultas a partir do tema.
    consultas = []
    """).strip() + "\n\n"


def section_bibliografia() -> str:
    return textwrap.dedent("""
    [bibliografia]
    estilo_citacao = "apa"
    # Caminho opcional para .bib externo.
    bib_path = ""
    """).strip() + "\n\n"


def section_documento(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [documento]
    tipo_documento = {toml_quote(v.get('tipo_documento', 'atividade'))}
    output_dir = {toml_quote(v.get('output_dir', './output/documento'))}
    prefixo = {toml_quote(v.get('prefixo', 'atividade_aula_2'))}
    criar_subdiretorio = true

    # Template Org opcional. Se vazio, o gerador usa fallback interno.
    template_path = {toml_quote(v.get('template_path', ''))}

    titulo_trabalho = {toml_quote(v.get('titulo_trabalho', 'Atividade Aula 2'))}
    estilo_citacao = {toml_quote(v.get('estilo_citacao', 'apa'))}

    # Estratégia de escrita: "novo", "reescrever" ou "expandir".
    modo_escrita = {toml_quote(v.get('modo_escrita', 'novo'))}
    perfil_redacao = {toml_quote(v.get('perfil_redacao', 'academico_equilibrado'))}

    # Para reaproveitar um .org anterior como orientação.
    reescrever_a_partir_do_org_atual = {toml_bool(v.get('reescrever_a_partir_do_org_atual', False))}
    documento_org_existente = {toml_quote(v.get('documento_org_existente', ''))}
    preservar_estrutura_do_org_anterior = {toml_bool(v.get('preservar_estrutura_do_org_anterior', True))}

    usar_bib_da_pesquisa = true
    usar_artigos_selecionados_pesquisa = true
    citar_todos_fulltext_cache = true
    priorizar_citacoes_dos_selecionados = true

    # Fontes acadêmicas extras a serem adicionadas ao documento/.bib.
    artigos_extras_paths = {toml_array(v.get('artigos_extras_paths', []))}
    incluir_artigos_extras_no_bib = {toml_bool(v.get('incluir_artigos_extras_no_bib', True))}
    extras_so_complementam = true

    # Orientações específicas do documento.
    orientacoes_paths = {toml_array(v.get('orientacoes_paths', []))}
    orientacao_inline = {toml_quote(v.get('orientacao_inline', ''))}

    # Dissertação: geração em etapas é boa para criar do zero; desligue para preservar texto prévio.
    geracao_em_etapas = {toml_bool(v.get('geracao_em_etapas', False))}

    exportar_pdf = {toml_bool(v.get('exportar_pdf', True))}
    """).strip() + "\n\n"


def section_mapa_mental(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [mapa_mental]
    # Geração de mapa mental em PlantUML.
    gerar = {toml_bool(v.get('gerar', False))}

    # true = usa um .org existente e cria/atualiza apenas o mapa mental.
    # false = roda o fluxo normal e insere o mapa ao final.
    somente_mapa_mental = {toml_bool(v.get('somente_mapa_mental', False))}

    # Modo seguro: quando somente_mapa_mental=true, preserva o .org base e cria *_com_mapa.org.
    sobrescrever_org_existente = false

    documento_org_existente = {toml_quote(v.get('documento_org_existente', ''))}
    contexto_json_existente = {toml_quote(v.get('contexto_json_existente', ''))}
    fulltext_cache_dir = {toml_quote(v.get('fulltext_cache_dir', ''))}

    linguagem = "plantuml"
    formato = {toml_quote(v.get('formato', 'png'))}
    renderizar = true
    inserir_no_org = true
    recompilar_pdf = {toml_bool(v.get('recompilar_pdf', True))}

    titulo = {toml_quote(v.get('titulo', 'Mapa mental dos textos analisados'))}
    arquivo = {toml_quote(v.get('arquivo', 'mapa_mental'))}
    posicao = {toml_quote(v.get('posicao', 'apos_referencias'))}

    max_niveis = {int(v.get('max_niveis', 4))}
    max_nos = {int(v.get('max_nos', 45))}
    incluir_codigo_fonte = false

    dpi = {int(v.get('dpi', 300))}
    plantuml_limit_size = {int(v.get('plantuml_limit_size', 8192))}
    falhar_se_nao_renderizar = {toml_bool(v.get('falhar_se_nao_renderizar', True))}
    diretorio_imagens = {toml_quote(v.get('diretorio_imagens', 'images'))}

    # Caminho para o JAR do PlantUML. Na versão validada, este campo funciona aqui.
    plantuml_jar_path = {toml_quote(v.get('plantuml_jar_path', ''))}

    # Cores por nível no mapa mental.
    colorir_niveis = {toml_bool(v.get('colorir_niveis', True))}
    cores_niveis = ["#DCEBFF", "#DCFCE7", "#FEF3C7", "#FEE2E2", "#F3E8FF"]

    # ATENÇÃO: deixe o bloco aninhado no final. Tudo abaixo dele pertence a cores_por_nivel.
    [mapa_mental.cores_por_nivel]
    1 = "#DCEBFF"
    2 = "#DCFCE7"
    3 = "#FEF3C7"
    4 = "#FEE2E2"
    5 = "#F3E8FF"
    """).strip() + "\n\n"


def section_derivacao(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [derivacao]
    # Modo de derivação/reorientação de dissertação.
    # Usa uma dissertação existente como matriz intelectual e gera outro documento.
    ativo = {toml_bool(v.get('ativo', False))}
    tipo = {toml_quote(v.get('tipo', 'reorientacao_tematico_empirica'))}

    documento_org_base = {toml_quote(v.get('documento_org_base', ''))}
    bib_base = {toml_quote(v.get('bib_base', ''))}
    sufixo_saida = {toml_quote(v.get('sufixo_saida', '_derivada'))}
    preservar_original = true

    novo_tema = {toml_quote(v.get('novo_tema', ''))}
    novo_recorte = {toml_quote(v.get('novo_recorte', ''))}
    novo_objetivo = {toml_quote(v.get('novo_objetivo', ''))}
    nova_pergunta_pesquisa = {toml_quote(v.get('nova_pergunta_pesquisa', ''))}

    estrategia = "preservar_estrutura_e_adaptar_conteudo"
    preservar_estrutura = true
    exportar_pdf = {toml_bool(v.get('exportar_pdf', True))}

    # Orientações novas de reorientação.
    orientacoes_paths = {toml_array(v.get('orientacoes_paths', []))}
    orientacao_inline = {toml_quote(v.get('orientacao_inline', ''))}

    # Dados locais/contexto empírico: relatórios, descrições, tabelas convertidas etc.
    dados_locais_paths = {toml_array(v.get('dados_locais_paths', []))}
    dados_locais_inline = {toml_quote(v.get('dados_locais_inline', ''))}

    # Fontes acadêmicas novas.
    artigos_extras_paths = {toml_array(v.get('artigos_extras_paths', []))}
    incluir_artigos_extras_no_bib = {toml_bool(v.get('incluir_artigos_extras_no_bib', True))}

    # Proteção: aborta se a derivação ficar pequena demais em relação ao original.
    proporcao_minima_palavras = 0.55
    """).strip() + "\n\n"


def section_latex(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [latex]
    # Configuração de exportação Org/LaTeX.
    pdf_engine = {toml_quote(v.get('pdf_engine', 'lualatex'))}

    # Arquivo .el que registra classes como fgv-paper/fgv-dissertacao no Emacs batch.
    org_latex_class_init = {toml_quote(v.get('org_latex_class_init', ''))}

    # Diretório extra com .cls/.sty, quando necessário.
    latex_extra_path = {toml_quote(v.get('latex_extra_path', ''))}
    """).strip() + "\n\n"


def section_recompilar(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [recompilar]
    ativo = {toml_bool(v.get('ativo', False))}
    documento_org_para_recompilar = {toml_quote(v.get('documento_org_para_recompilar', ''))}
    exportar_pdf = true
    """).strip() + "\n\n"



def section_pipeline_excelencia(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [pipeline]
    # Fluxo robusto: pesquisa PRISMA + geração posterior/opcional da dissertação.
    executar_pesquisa = {toml_bool(v.get('executar_pesquisa', True))}
    executar_bundle = {toml_bool(v.get('executar_bundle', False))}
    executar_documento = {toml_bool(v.get('executar_documento', False))}
    pesquisa_dir_existente = {toml_quote(v.get('pesquisa_dir_existente', ''))}
    script_pesquisa = {toml_quote(v.get('script_pesquisa', './scripts/research/gerador_pesquisa_rc_2.py'))}
    bundle_dir = {toml_quote(v.get('bundle_dir', ''))}
    """).strip() + "\n\n"


def section_atividade_excelencia(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    curso = v.get('curso', 'Mestrado Profissional em Políticas Públicas e Governo')
    turma = v.get('turma', '2026.01')
    polo = v.get('polo', 'Brasília')
    disciplina = v.get('disciplina', 'Teorias da Administração Pública')
    professor = v.get('professor', 'Bernardo Buta')
    aluno = v.get('aluno', 'Gustavo M. Mendes de Tarso')
    data = v.get('data', '2026')
    # Mantém campos no topo para o academic_pipeline_rc.py e também a subtable metadados
    # usada por TOMLs antigos da etapa de pesquisa.
    return textwrap.dedent(f"""
    [atividade]
    modo = "revisao_sistematica"
    curso = {toml_quote(curso)}
    turma = {toml_quote(turma)}
    polo = {toml_quote(polo)}
    disciplina = {toml_quote(disciplina)}
    professor = {toml_quote(professor)}
    aluno = {toml_quote(aluno)}
    data = {toml_quote(data)}
    titulo_trabalho = {toml_quote(v.get('titulo_trabalho', 'Dissertação'))}

    [atividade.metadados]
    disciplina = {toml_quote(disciplina)}
    professor = {toml_quote(professor)}
    curso = {toml_quote(curso)}
    turma = {toml_quote(turma)}
    polo = {toml_quote(polo)}
    aluno = {toml_quote(aluno)}
    """).strip() + "\n\n"


def section_pesquisa_excelencia(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [pesquisa]
    tema = {toml_quote(v.get('tema', 'Uso de inteligência artificial no governo público federal como instrumento de governança, ética, compliance e gestão de riscos.'))}
    recorte = {toml_quote(v.get('recorte', 'Análise do uso de sistemas e aplicações de inteligência artificial por órgãos e entidades da administração pública federal brasileira, com ênfase em governança, ética, compliance e gestão de riscos, no período de 2019 a 2026.'))}
    objetivo = {toml_quote(v.get('objetivo', 'Investigar como a literatura acadêmica e técnico-institucional trata a adoção de inteligência artificial no governo público federal brasileiro, identificando aplicações, modelos de governança, dilemas éticos, exigências de compliance, práticas de gestão de riscos e diretrizes para implementação responsável.'))}
    pergunta_pesquisa = {toml_quote(v.get('pergunta_pesquisa', ''))}
    hipotese = {toml_quote(v.get('hipotese', ''))}
    trabalho = "Dissertação"
    tipo_estudo = "Revisão de literatura"
    periodo = {toml_quote(v.get('periodo', '2019-2026'))}
    idiomas = ["inglês", "português"]
    palavras_chave = {toml_array(v.get('palavras_chave', []))}
    bases = ["semantic_scholar", "scopus", "pubmed", "openalex", "crossref", "europe_pmc", "core"]
    """).strip() + "\n\n"


def section_busca_triagem_excelencia(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [busca]
    sugerir_palavras_chave_ia = true
    query_bilingue = true
    quantidade_triagem = {int(v.get('quantidade_triagem', 10))}
    quantidade_selecionados = {int(v.get('quantidade_selecionados', 1))}
    salvar_busca_bruta_json = true
    incluir_analise_detalhada_ia = true
    incluir_sintese_integradora_ia = true

    [triagem]
    rigor = {toml_quote(v.get('rigor', 'moderado'))}
    usar_score_hibrido = true
    orientacoes_paths = {toml_array(v.get('orientacoes_paths', []))}
    orientacao_inline = {toml_quote(v.get('orientacao_inline', 'A triagem deve priorizar estudos substantivamente conectados ao tema, evitando textos apenas genericamente relacionados.'))}
    permitir_textos_nao_publicos = false

    [queries]
    query_geral = ""
    query_semantic = ""
    query_scopus = ""
    query_wos = ""
    query_pubmed = ""
    query_openalex = ""
    query_crossref = ""
    query_europepmc = ""
    query_core = ""
    """).strip() + "\n\n"


def section_saida_excelencia(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [saida]
    prefixo = {toml_quote(v.get('prefixo', 'dissertacao_ia_governo_publico_federal'))}
    output_dir = {toml_quote(v.get('output_dir', './output/pesquisa'))}
    criar_subdiretorio = true
    org_modelo = {toml_quote(v.get('org_modelo', './templates/template_research.org'))}
    orientacoes_paths = {toml_array(v.get('orientacoes_paths', []))}
    orientacao_inline = {toml_quote(v.get('orientacao_inline', 'Na etapa de pesquisa, priorize a construção de corpus conceitualmente robusto, bem delimitado e aderente ao recorte.'))}
    exportar_pdf = true
    gerar_env_example = false
    remover_auxiliares = true
    """).strip() + "\n\n"


def section_documento_dissertacao_excelencia(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [documento]
    tipo_documento = "dissertacao"
    prefixo = {toml_quote(v.get('prefixo', 'dissertacao_ia_governo_publico_federal'))}
    output_dir = {toml_quote(v.get('output_dir', './output/documento'))}
    criar_subdiretorio = true
    template_org = {toml_quote(v.get('template_org', './templates/template_dissertacao_fgv_apa_v2.org'))}
    template_path = {toml_quote(v.get('template_path', v.get('template_org', './templates/template_dissertacao_fgv_apa_v2.org')))}
    exportar_pdf = true

    institution_name = "Fundação Getúlio Vargas"
    school_name = ""
    program_name = "Mestrado Profissional em Políticas Públicas e Governo"
    area_de_concentracao = "Políticas Públicas e Governo"
    ano = {toml_quote(v.get('ano', '2026'))}
    linha_pesquisa = {toml_quote(v.get('linha_pesquisa', 'Governança, Estado e Políticas Públicas'))}
    coorientador = ""
    data_aprovacao = "A definir"
    banca = []

    # Metas de extensão para dissertação longa.
    min_palavras_total = {int(v.get('min_palavras_total', 22000))}
    alvo_palavras_total = {int(v.get('alvo_palavras_total', 28000))}
    min_palavras_introducao = {int(v.get('min_palavras_introducao', 1800))}
    min_palavras_referencial = {int(v.get('min_palavras_referencial', 7000))}
    min_palavras_metodologia = {int(v.get('min_palavras_metodologia', 3000))}
    min_palavras_resultados = {int(v.get('min_palavras_resultados', 8500))}
    min_palavras_conclusao = {int(v.get('min_palavras_conclusao', 1800))}

    orientacoes_paths = {toml_array(v.get('orientacoes_paths', []))}
    orientacao_inline = {toml_quote(v.get('orientacao_inline', 'Produza uma DISSERTAÇÃO, não um paper curto. Use os estudos selecionados na etapa de pesquisa como base principal; preserve densidade analítica, estrutura FGV e discussão robusta.'))}

    usar_artigos_selecionados_pesquisa = true
    artigos_extras_paths = {toml_array(v.get('artigos_extras_paths', []))}
    modo_escrita = "novo"
    perfil_redacao = "academico_equilibrado"
    usar_contexto_consolidado_da_pesquisa = true
    reformular_tema_recorte_objetivo = false
    permitir_busca_correlata_extra = false
    priorizar_citacoes_dos_selecionados = true
    extras_so_complementam = true
    usar_bib_da_pesquisa = true
    incluir_artigos_extras_no_bib = true
    reescrever_a_partir_do_org_atual = false
    documento_org_existente = ""
    preservar_estrutura_do_org_anterior = false
    geracao_em_etapas = true
    """).strip() + "\n\n"


def section_latex_excelencia(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [latex]
    org_latex_class_init = {toml_quote(v.get('org_latex_class_init', './misc/academic-writing.el'))}
    # Pode apontar para diretório com .cls/.sty ou para um arquivo .sty específico.
    latex_extra_path = {toml_quote(v.get('latex_extra_path', './misc/fgv/fgv-dissertacao.sty'))}
    comando_exportacao_pdf = {toml_quote(v.get('comando_exportacao_pdf', ''))}
    fgv_logo_path = {toml_quote(v.get('fgv_logo_path', ''))}
    """).strip() + "\n\n"


def section_controle(values: dict[str, Any] | None = None) -> str:
    v = values or {}
    return textwrap.dedent(f"""
    [controle]
    nao_interativo = {toml_bool(v.get('nao_interativo', True))}
    dry_run = {toml_bool(v.get('dry_run', False))}
    mock_run = {toml_bool(v.get('mock_run', False))}
    """).strip() + "\n\n"

def build_template(profile: str, values: dict[str, dict[str, Any]] | None = None) -> str:
    values = values or {}
    doc = header(profile)

    if profile == "atividade_local":
        doc += section_atividade(values.get("atividade"))
        doc += section_openai(values.get("openai"))
        doc += section_pipeline({"modo_entrada": "documentos_locais", "executar_pesquisa": False, "executar_documento": True})
        doc += section_documentos_locais(values.get("documentos_locais"))
        doc += section_saida(values.get("saida"))
        doc += section_pesquisa(values.get("pesquisa"))
        doc += section_documento(values.get("documento"))
        doc += section_mapa_mental(values.get("mapa_mental"))
        doc += section_latex(values.get("latex"))
        return doc

    if profile == "pesquisa_prisma_documento":
        doc += section_atividade(values.get("atividade"))
        doc += section_openai(values.get("openai"))
        doc += section_pipeline({"modo_entrada": "pesquisa", "executar_pesquisa": True, "executar_documento": True})
        doc += section_saida(values.get("saida"))
        doc += section_pesquisa(values.get("pesquisa"))
        doc += section_busca_triagem()
        doc += section_bibliografia()
        doc += section_documento(values.get("documento"))
        doc += section_mapa_mental(values.get("mapa_mental"))
        doc += section_latex(values.get("latex"))
        return doc

    if profile == "pesquisa_dissertacao_excelencia":
        doc += section_pipeline_excelencia(values.get("pipeline"))
        doc += section_atividade_excelencia(values.get("atividade"))
        doc += section_pesquisa_excelencia(values.get("pesquisa"))
        doc += section_bibliografia()
        doc += section_busca_triagem_excelencia(values.get("busca_triagem"))
        doc += section_saida_excelencia(values.get("saida"))
        doc += section_documento_dissertacao_excelencia(values.get("documento"))
        doc += section_latex_excelencia(values.get("latex"))
        doc += section_openai(values.get("openai"))
        doc += section_controle(values.get("controle"))
        return doc

    if profile == "paper":
        doc += section_atividade(values.get("atividade"))
        doc += section_openai(values.get("openai"))
        doc += section_pipeline(values.get("pipeline") or {"modo_entrada": "pesquisa", "executar_pesquisa": False, "executar_documento": True})
        doc += section_saida(values.get("saida"))
        doc += section_pesquisa(values.get("pesquisa"))
        doc += section_bibliografia()
        pv = {"tipo_documento": "paper", "prefixo": "paper", "geracao_em_etapas": False, **values.get("documento", {})}
        doc += section_documento(pv)
        doc += section_mapa_mental(values.get("mapa_mental"))
        doc += section_latex(values.get("latex"))
        return doc

    if profile == "dissertacao":
        doc += section_atividade(values.get("atividade"))
        doc += section_openai(values.get("openai"))
        doc += section_pipeline(values.get("pipeline") or {"modo_entrada": "pesquisa", "executar_pesquisa": False, "executar_documento": True})
        doc += section_saida(values.get("saida"))
        doc += section_pesquisa(values.get("pesquisa"))
        dv = {"tipo_documento": "dissertacao", "prefixo": "dissertacao", "geracao_em_etapas": True, **values.get("documento", {})}
        doc += section_documento(dv)
        doc += section_mapa_mental(values.get("mapa_mental"))
        doc += section_latex(values.get("latex"))
        return doc

    if profile == "somente_mapa_mental":
        doc += section_openai(values.get("openai"))
        mv = {"gerar": True, "somente_mapa_mental": True, **values.get("mapa_mental", {})}
        doc += section_mapa_mental(mv)
        doc += section_latex(values.get("latex"))
        return doc

    if profile == "derivacao_dissertacao":
        doc += section_openai(values.get("openai"))
        doc += section_atividade(values.get("atividade"))
        doc += section_derivacao({"ativo": True, **values.get("derivacao", {})})
        doc += section_latex(values.get("latex"))
        return doc

    if profile == "recompilar_pdf":
        doc += section_latex(values.get("latex"))
        doc += section_recompilar({"ativo": True, **values.get("recompilar", {})})
        return doc

    if profile == "completo_comentado":
        doc += section_atividade(values.get("atividade"))
        doc += section_openai(values.get("openai"))
        doc += section_pipeline(values.get("pipeline"))
        doc += section_documentos_locais(values.get("documentos_locais"))
        doc += section_saida(values.get("saida"))
        doc += section_pesquisa(values.get("pesquisa"))
        doc += section_busca_triagem()
        doc += section_bibliografia()
        doc += section_documento(values.get("documento"))
        doc += section_mapa_mental(values.get("mapa_mental"))
        doc += section_derivacao(values.get("derivacao"))
        doc += section_latex(values.get("latex"))
        doc += section_recompilar(values.get("recompilar"))
        return doc

    raise ValueError(f"Perfil desconhecido: {profile}")

# ---------------------------------------------------------------------------
# Modo interativo
# ---------------------------------------------------------------------------

def interactive_values(profile: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}

    if profile in {"atividade_local", "pesquisa_prisma_documento", "pesquisa_dissertacao_excelencia", "paper", "dissertacao", "derivacao_dissertacao"}:
        print("\n== Metadados acadêmicos ==")
        values["atividade"] = {
            "curso": ask_text("Curso", "Mestrado Profissional em Políticas Públicas e Governo"),
            "turma": ask_text("Turma", "2026.1"),
            "polo": ask_text("Pólo", "Brasília"),
            "disciplina": ask_text("Disciplina", "Políticas Públicas: Teorias e Análises"),
            "professor": ask_text("Professor", ""),
            "aluno": ask_text("Aluno(s)", "Gustavo M. Mendes de Tarso"),
            "data": ask_text("Data", "Abril de 2026"),
            "titulo_trabalho": ask_text("Título do trabalho", "Atividade Aula 2"),
        }

    if profile == "atividade_local":
        print("\n== Entrada local ==")
        use_zip = ask_bool("Usar arquivo ZIP?", True)
        local: dict[str, Any] = {"ativos": True}
        if use_zip:
            local["input_zip"] = ask_path("Caminho do ZIP", required=True)
            local["input_dir"] = ""
        else:
            local["input_zip"] = ""
            local["input_dir"] = ask_path("Caminho da pasta", required=True)
        values["documentos_locais"] = local
        values["saida"] = {
            "output_dir": ask_path("Diretório de saída do corpus", "./output/corpus_local"),
            "prefixo": ask_text("Prefixo dos arquivos", "atividade_aula_2"),
        }
        values["pesquisa"] = {
            "tema": ask_text("Tema", "Implementação de políticas públicas"),
            "recorte": ask_text("Recorte", "Análise de textos selecionados para atividade"),
            "objetivo": ask_text("Objetivo", "Resumir, comparar e sintetizar analiticamente os textos fornecidos"),
        }
        values["documento"] = {
            "tipo_documento": ask_text("Tipo de documento", "atividade"),
            "output_dir": ask_path("Diretório de saída do documento", "./output/documento"),
            "prefixo": ask_text("Prefixo do documento", "atividade_aula_2"),
            "template_path": ask_path("Template .org opcional", ""),
            "exportar_pdf": ask_bool("Exportar PDF?", True),
        }

    elif profile == "pesquisa_dissertacao_excelencia":
        print("\n== Pesquisa PRISMA + Dissertação FGV Excelência ==")
        bundle_dir = ask_path("Diretório do bundle academic_pipeline", "./bundle_projeto_pesquisa_documento_rc_20")
        b = bundle_dir.rstrip("/")
        prefixo = ask_text("Prefixo", "dissertacao_ia_governo_publico_federal")
        values["pipeline"] = {
            "executar_pesquisa": ask_bool("Executar pesquisa PRISMA agora?", True),
            "executar_bundle": False,
            "executar_documento": ask_bool("Gerar dissertação agora?", False),
            "pesquisa_dir_existente": ask_path("Diretório de pesquisa existente, se executar_pesquisa=false", ""),
            "script_pesquisa": ask_path("Script de pesquisa", f"{b}/scripts/research/gerador_pesquisa_rc_2.py"),
            "bundle_dir": b,
        }
        values["pesquisa"] = {
            "tema": ask_text("Tema", "Uso de inteligência artificial no governo público federal como instrumento de governança, ética, compliance e gestão de riscos."),
            "recorte": ask_text("Recorte", "Administração pública federal brasileira, governança, ética, compliance e gestão de riscos associados à IA, 2019-2026."),
            "objetivo": ask_text("Objetivo", "Investigar aplicações, modelos de governança, dilemas éticos, exigências de compliance, práticas de gestão de riscos e diretrizes para implementação responsável de IA no governo público federal."),
            "pergunta_pesquisa": ask_text("Pergunta de pesquisa opcional", ""),
            "hipotese": ask_text("Hipótese opcional", ""),
            "periodo": ask_text("Período da revisão", "2019-2026"),
            "palavras_chave": [p.strip() for p in ask_text("Palavras-chave separadas por vírgula", "").split(",") if p.strip()],
        }
        values["busca_triagem"] = {
            "orientacoes_paths": [
                f"{b}/prompts/triagem_prompt.txt",
                f"{b}/prompts/diretivas_extras.txt",
            ],
            "quantidade_triagem": int(ask_text("Quantidade para triagem", "10") or 10),
            "quantidade_selecionados": int(ask_text("Quantidade de selecionados", "1") or 1),
            "rigor": ask_text("Rigor da triagem", "moderado"),
        }
        values["saida"] = {
            "prefixo": prefixo,
            "output_dir": ask_path("Diretório de saída da pesquisa", f"{b}/output/pesquisa"),
            "org_modelo": ask_path("Template de pesquisa .org", f"{b}/templates/template_research.org"),
            "orientacoes_paths": [
                f"{b}/prompts/orientacao_geral_execucao.txt",
                f"{b}/prompts/diretivas_extras.txt",
            ],
        }
        values["documento"] = {
            "prefixo": prefixo,
            "output_dir": ask_path("Diretório de saída da dissertação", f"{b}/output/documento"),
            "template_org": ask_path("Template de dissertação .org", f"{b}/templates/template_dissertacao_fgv_apa_v2.org"),
            "orientacoes_paths": ask_path_list("Arquivos de orientação da dissertação, além dos modelos FGV") or [
                f"{b}/prompts/orientacao_geral_execucao.txt",
                f"{b}/docs/modelos/fgv/formatacao-de-trabalhos-academicos-manual-fgv-impressao-2025.pdf",
                f"{b}/docs/modelos/fgv/modelo-de-dissertacao-2025.docx",
            ],
            "ano": ask_text("Ano", "2026"),
            "linha_pesquisa": ask_text("Linha de pesquisa", "Governança, Estado e Políticas Públicas"),
        }
        values["latex"] = {
            "org_latex_class_init": ask_path("Arquivo academic-writing.el", f"{b}/misc/academic-writing.el"),
            "latex_extra_path": ask_path("Diretório/arquivo LaTeX extra", f"{b}/misc/fgv/fgv-dissertacao.sty"),
            "fgv_logo_path": ask_path("Logo FGV opcional", ""),
        }
        values["openai"] = {"model": ask_text("Modelo OpenAI", "gpt-5.4")}
        values["controle"] = {"nao_interativo": True, "dry_run": False, "mock_run": False}

    elif profile in {"pesquisa_prisma_documento", "paper", "dissertacao"}:
        print("\n== Pesquisa / Documento ==")
        if profile == "paper":
            executar_pesquisa = ask_bool("Executar pesquisa agora?", False)
            values["pipeline"] = {
                "modo_entrada": "pesquisa",
                "executar_pesquisa": executar_pesquisa,
                "executar_documento": True,
                "pesquisa_dir_existente": ask_path("Diretório de pesquisa/corpus existente opcional", "") if not executar_pesquisa else "",
            }
        values["saida"] = {
            "output_dir": ask_path("Diretório de saída da pesquisa/corpus", "./output/pesquisa"),
            "prefixo": ask_text("Prefixo", "pesquisa"),
        }
        values["pesquisa"] = {
            "tema": ask_text("Tema", required=True),
            "recorte": ask_text("Recorte", required=True),
            "objetivo": ask_text("Objetivo", required=True),
            "pergunta_pesquisa": ask_text("Pergunta de pesquisa", ""),
            "palavras_chave": [p.strip() for p in ask_text("Palavras-chave separadas por vírgula", "").split(",") if p.strip()],
        }
        default_tipo = "dissertacao" if profile == "dissertacao" else "paper"
        default_prefixo = "dissertacao" if profile == "dissertacao" else "paper"
        values["documento"] = {
            "tipo_documento": default_tipo if profile in {"paper", "dissertacao"} else ask_text("Tipo de documento", "paper"),
            "output_dir": ask_path("Diretório de saída do documento", "./output/documento"),
            "prefixo": ask_text("Prefixo do documento", default_prefixo),
            "template_path": ask_path("Template .org opcional", ""),
            "geracao_em_etapas": ask_bool("Gerar em etapas?", profile == "dissertacao"),
            "exportar_pdf": ask_bool("Exportar PDF?", True),
        }

    elif profile == "somente_mapa_mental":
        print("\n== Somente mapa mental ==")
        values["mapa_mental"] = {
            "gerar": True,
            "somente_mapa_mental": True,
            "documento_org_existente": ask_path("Caminho do .org base", required=True),
            "contexto_json_existente": ask_path("Contexto JSON existente opcional", ""),
            "fulltext_cache_dir": ask_path("Diretório fulltext_cache opcional", ""),
            "plantuml_jar_path": ask_path("Caminho do plantuml.jar", ""),
            "titulo": ask_text("Título do mapa", "Mapa mental dos textos analisados"),
            "arquivo": ask_text("Nome-base do arquivo do mapa", "mapa_mental"),
            "recompilar_pdf": ask_bool("Recompilar PDF?", True),
            "falhar_se_nao_renderizar": ask_bool("Falhar se não renderizar?", True),
            "colorir_niveis": ask_bool("Colorir níveis?", True),
        }

    elif profile == "derivacao_dissertacao":
        print("\n== Derivação de dissertação ==")
        values["derivacao"] = {
            "ativo": True,
            "documento_org_base": ask_path("Dissertação .org base", required=True),
            "bib_base": ask_path(".bib base opcional", ""),
            "sufixo_saida": ask_text("Sufixo da derivação", "_derivada"),
            "novo_tema": ask_text("Novo tema", required=True),
            "novo_recorte": ask_text("Novo recorte", required=True),
            "novo_objetivo": ask_text("Novo objetivo", required=True),
            "nova_pergunta_pesquisa": ask_text("Nova pergunta de pesquisa", ""),
            "orientacoes_paths": ask_path_list("Orientações novas"),
            "dados_locais_paths": ask_path_list("Dados locais/contexto empírico"),
            "artigos_extras_paths": ask_path_list("Novas fontes acadêmicas"),
            "exportar_pdf": ask_bool("Exportar PDF derivado?", True),
        }

    elif profile == "recompilar_pdf":
        print("\n== Recompilar PDF ==")
        values["recompilar"] = {
            "ativo": True,
            "documento_org_para_recompilar": ask_path("Caminho do .org", required=True),
        }

    # Opções comuns opcionais
    if profile in {"atividade_local", "pesquisa_prisma_documento", "pesquisa_dissertacao_excelencia", "paper", "dissertacao"}:
        if ask_bool("Gerar mapa mental também?", False):
            values.setdefault("mapa_mental", {})
            values["mapa_mental"].update({
                "gerar": True,
                "somente_mapa_mental": False,
                "plantuml_jar_path": ask_path("Caminho do plantuml.jar", ""),
                "colorir_niveis": ask_bool("Colorir níveis?", True),
            })

    if profile != "completo_comentado":
        if ask_bool("Configurar caminhos LaTeX/Emacs agora?", False):
            values["latex"] = {
                "org_latex_class_init": ask_path("Arquivo .el que registra classes FGV", ""),
                "latex_extra_path": ask_path("Diretório extra de .cls/.sty", ""),
                "pdf_engine": ask_text("PDF engine", "lualatex"),
            }

    return values


# ---------------------------------------------------------------------------
# Assistente inteligente
# ---------------------------------------------------------------------------

def choose_profile_by_answers() -> str:
    """Escolhe o perfil adequado a partir de perguntas de alto nível.

    Os perfis continuam existindo como presets auditáveis e explicáveis; este
    assistente apenas escolhe o preset mais coerente conforme as respostas.
    """
    print("\nAssistente inteligente do academic_pipeline")
    print("Responda algumas perguntas e eu escolherei o fluxo TOML mais adequado.\n")

    if ask_bool("Você quer apenas recompilar um PDF a partir de um .org existente?", False):
        return "recompilar_pdf"
    if ask_bool("Você quer apenas inserir/atualizar mapa mental em um .org já existente?", False):
        return "somente_mapa_mental"
    if ask_bool("Você quer derivar/reorientar uma dissertação existente para outro objeto/tema?", False):
        return "derivacao_dissertacao"

    doc_type = ask_text("Tipo principal de documento (atividade/paper/dissertacao)", "dissertacao").strip().lower()
    if doc_type in {"paper", "artigo", "artigo acadêmico"}:
        return "paper"
    if doc_type in {"atividade", "resposta", "resposta discursiva", "fichamento", "resumo", "ensaio"}:
        if ask_bool("Você usará ZIP/pasta de documentos locais?", True):
            return "atividade_local"
        return "pesquisa_prisma_documento"

    # Dissertação
    if ask_bool("A dissertação deve usar pesquisa PRISMA robusta + template FGV + camada de excelência?", True):
        return "pesquisa_dissertacao_excelencia"
    return "dissertacao"


def run_smart_assistant(output: str | None = None) -> int:
    profile = choose_profile_by_answers()
    print_profile_explanation(profile)
    values = interactive_values(profile)
    content = build_template(profile, values)
    default_name = f"academic_pipeline_{profile}.toml"
    out = save_toml(content, output, default_name=default_name)
    print(f"\nArquivo TOML salvo em: {out}")
    errors, warnings = validate_config(out)
    print_validation_result(errors, warnings)
    return 1 if errors else 0

# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("tomllib não disponível. Use Python 3.11+.")
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(os.path.expanduser(str(value)))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def is_nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def iter_paths_in_value(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            if is_nonempty(item):
                yield str(item)
    elif is_nonempty(value):
        yield str(value)


def validate_path(label: str, raw: Any, base_dir: Path, errors: list[str], warnings: list[str], *, required: bool = False, must_be_file: bool | None = None, must_be_dir: bool | None = None) -> None:
    if not is_nonempty(raw):
        if required:
            errors.append(f"{label}: caminho obrigatório ausente.")
        return
    p = resolve_path(str(raw), base_dir)
    if not p.exists():
        errors.append(f"{label}: caminho não existe: {p}")
        return
    if must_be_file and not p.is_file():
        errors.append(f"{label}: esperado arquivo, mas não é arquivo: {p}")
    if must_be_dir and not p.is_dir():
        errors.append(f"{label}: esperado diretório, mas não é diretório: {p}")


def validate_color(value: Any, label: str, warnings: list[str]) -> None:
    text = str(value).strip()
    if not text:
        return
    if not COLOR_RE.match(text):
        warnings.append(f"{label}: cor em formato possivelmente inválido: {text}")


def validate_config(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    base_dir = path.parent.resolve()
    try:
        cfg = load_toml(path)
    except Exception as exc:
        return [f"Falha ao ler TOML: {exc}"], []

    # documentos_locais
    local = cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}
    if local.get("ativos") or local.get("modo_entrada") == "documentos_locais":
        input_zip = local.get("input_zip", "")
        input_dir = local.get("input_dir", "")
        if not is_nonempty(input_zip) and not is_nonempty(input_dir):
            errors.append("[documentos_locais]: informe input_zip ou input_dir.")
        if is_nonempty(input_zip):
            validate_path("[documentos_locais].input_zip", input_zip, base_dir, errors, warnings, must_be_file=True)
        if is_nonempty(input_dir):
            validate_path("[documentos_locais].input_dir", input_dir, base_dir, errors, warnings, must_be_dir=True)

    # documento
    documento = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
    if documento:
        validate_path("[documento].template_path", documento.get("template_path", ""), base_dir, errors, warnings, must_be_file=True)
        if documento.get("reescrever_a_partir_do_org_atual"):
            validate_path("[documento].documento_org_existente", documento.get("documento_org_existente", ""), base_dir, errors, warnings, required=True, must_be_file=True)
        for raw in iter_paths_in_value(documento.get("artigos_extras_paths", [])):
            validate_path("[documento].artigos_extras_paths", raw, base_dir, errors, warnings, must_be_file=True)
        for raw in iter_paths_in_value(documento.get("orientacoes_paths", [])):
            validate_path("[documento].orientacoes_paths", raw, base_dir, errors, warnings, must_be_file=True)

    # mapa mental
    mm = cfg.get("mapa_mental", {}) if isinstance(cfg.get("mapa_mental"), dict) else {}
    if mm.get("gerar") or mm.get("somente_mapa_mental"):
        if mm.get("somente_mapa_mental"):
            validate_path("[mapa_mental].documento_org_existente", mm.get("documento_org_existente", ""), base_dir, errors, warnings, required=True, must_be_file=True)
        formato = str(mm.get("formato", "png")).lower().lstrip(".")
        if formato not in {"png", "svg"}:
            errors.append("[mapa_mental].formato deve ser 'png' ou 'svg'.")
        if mm.get("renderizar", True):
            jar = mm.get("plantuml_jar_path") or documento.get("plantuml_jar_path") or os.getenv("PLANTUML_JAR")
            plantuml_cmd = shutil.which("plantuml")
            if is_nonempty(jar):
                validate_path("plantuml_jar_path", jar, base_dir, errors, warnings, must_be_file=True)
                if not shutil.which("java"):
                    errors.append("Java não encontrado no PATH; necessário para executar plantuml.jar.")
            elif not plantuml_cmd:
                errors.append("Mapa mental com renderizar=true, mas não há plantuml no PATH nem plantuml_jar_path/PLANTUML_JAR.")
        if mm.get("colorir_niveis", False):
            for idx, color in enumerate(mm.get("cores_niveis", []) or [], start=1):
                validate_color(color, f"[mapa_mental].cores_niveis[{idx}]", warnings)
        nested = mm.get("cores_por_nivel", {})
        if isinstance(nested, dict):
            for key, value in nested.items():
                if not str(key).isdigit():
                    warnings.append(f"[mapa_mental.cores_por_nivel]: chave esperada numérica, encontrada: {key}")
                validate_color(value, f"[mapa_mental.cores_por_nivel].{key}", warnings)
                # Detecta erro comum: colocar opções do mapa depois do bloco aninhado.
                if str(key) in MAPA_MENTAL_MAIN_KEYS:
                    errors.append(f"A chave '{key}' parece estar dentro de [mapa_mental.cores_por_nivel]. Mova-a para [mapa_mental].")
            for bad_key in MAPA_MENTAL_MAIN_KEYS:
                if bad_key in nested:
                    errors.append(f"[mapa_mental.cores_por_nivel] contém '{bad_key}', que pertence a [mapa_mental].")

    # derivação
    deriv = cfg.get("derivacao", {}) if isinstance(cfg.get("derivacao"), dict) else {}
    if deriv.get("ativo"):
        validate_path("[derivacao].documento_org_base", deriv.get("documento_org_base", ""), base_dir, errors, warnings, required=True, must_be_file=True)
        validate_path("[derivacao].bib_base", deriv.get("bib_base", ""), base_dir, errors, warnings, must_be_file=True)
        if not is_nonempty(deriv.get("novo_tema")):
            errors.append("[derivacao].novo_tema é obrigatório.")
        if not is_nonempty(deriv.get("novo_objetivo")):
            errors.append("[derivacao].novo_objetivo é obrigatório.")
        for raw in iter_paths_in_value(deriv.get("orientacoes_paths", [])):
            validate_path("[derivacao].orientacoes_paths", raw, base_dir, errors, warnings, must_be_file=True)
        for raw in iter_paths_in_value(deriv.get("dados_locais_paths", [])):
            validate_path("[derivacao].dados_locais_paths", raw, base_dir, errors, warnings, must_be_file=True)
        for raw in iter_paths_in_value(deriv.get("artigos_extras_paths", [])):
            validate_path("[derivacao].artigos_extras_paths", raw, base_dir, errors, warnings, must_be_file=True)

    # latex
    latex = cfg.get("latex", {}) if isinstance(cfg.get("latex"), dict) else {}
    if latex:
        validate_path("[latex].org_latex_class_init", latex.get("org_latex_class_init", ""), base_dir, errors, warnings, must_be_file=True)
        latex_extra = latex.get("latex_extra_path", "")
        if is_nonempty(latex_extra):
            p_extra = resolve_path(str(latex_extra), base_dir)
            if not p_extra.exists():
                errors.append(f"[latex].latex_extra_path: caminho não existe: {p_extra}")
            elif not (p_extra.is_dir() or p_extra.is_file()):
                errors.append(f"[latex].latex_extra_path: esperado arquivo .sty/.cls ou diretório: {p_extra}")

    # recompilar
    recomp = cfg.get("recompilar", {}) if isinstance(cfg.get("recompilar"), dict) else {}
    if recomp.get("ativo"):
        validate_path("[recompilar].documento_org_para_recompilar", recomp.get("documento_org_para_recompilar", ""), base_dir, errors, warnings, required=True, must_be_file=True)

    return errors, warnings


def print_validation_result(errors: list[str], warnings: list[str]) -> None:
    if errors:
        print("\nERROS:")
        for item in errors:
            print(f"  - {item}")
    if warnings:
        print("\nAVISOS:")
        for item in warnings:
            print(f"  - {item}")
    if not errors and not warnings:
        print("\nValidação concluída: nenhum problema encontrado.")
    elif not errors:
        print("\nValidação concluída com avisos, sem erros bloqueantes.")
    else:
        print("\nValidação concluída com erros.")

# ---------------------------------------------------------------------------
# Salvamento
# ---------------------------------------------------------------------------

def save_toml(content: str, output: str | None, *, default_name: str) -> Path:
    if output:
        raw = output
    else:
        raw = ask_path("Onde salvar o arquivo TOML?", default_name, required=True)
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not ask_bool(f"O arquivo já existe: {path}. Sobrescrever?", False):
        raise SystemExit("Operação cancelada pelo usuário.")
    path.write_text(content, encoding="utf-8")
    return path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Gerador/validador de TOML para academic_pipeline.",
    )
    parser.add_argument("--interativo", action="store_true", help="Abrir modo guiado interativo por perfil.")
    parser.add_argument("--assistente", "--inteligente", action="store_true", help="Abrir assistente inteligente que escolhe o perfil por perguntas.")
    parser.add_argument("--modelo-comentado", action="store_true", help="Gerar modelo comentado rapidamente.")
    parser.add_argument("--perfil", choices=sorted(PROFILES), help="Perfil do TOML a gerar.")
    parser.add_argument("--output", "-o", help="Caminho de saída do TOML.")
    parser.add_argument("--validar", help="Validar um arquivo TOML existente.")
    parser.add_argument("--listar-perfis", action="store_true", help="Listar perfis disponíveis.")
    parser.add_argument("--versao", action="store_true", help="Mostrar versão.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.versao:
        print(f"{APP_NAME} {APP_VERSION}")
        return 0

    if args.listar_perfis:
        for key in PROFILES:
            print_profile_explanation(key, compact=True)
        return 0

    if args.validar:
        errors, warnings = validate_config(Path(args.validar).expanduser().resolve())
        print_validation_result(errors, warnings)
        return 1 if errors else 0

    if args.assistente:
        return run_smart_assistant(args.output)

    # Sem argumentos: abre assistente inteligente.
    if not argv:
        args.assistente = True
        return run_smart_assistant(args.output)

    if args.interativo:
        profile = args.perfil or choose_profile()
        values = interactive_values(profile)
        content = build_template(profile, values)
        default_name = f"academic_pipeline_{profile}.toml"
        out = save_toml(content, args.output, default_name=default_name)
        print(f"\nArquivo TOML salvo em: {out}")
        errors, warnings = validate_config(out)
        print_validation_result(errors, warnings)
        return 1 if errors else 0

    if args.modelo_comentado or args.perfil:
        profile = args.perfil or choose_profile()
        content = build_template(profile)
        default_name = f"academic_pipeline_{profile}.toml"
        out = save_toml(content, args.output, default_name=default_name)
        print(f"\nArquivo TOML salvo em: {out}")
        errors, warnings = validate_config(out)
        print_validation_result(errors, warnings)
        return 1 if errors else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

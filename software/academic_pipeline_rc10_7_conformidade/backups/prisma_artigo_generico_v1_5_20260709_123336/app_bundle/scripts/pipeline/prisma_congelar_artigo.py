#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_ARTIGO_DIR = Path("/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo")
DEFAULT_TITULO = "ATESTMED, saúde digital e decisão baseada em evidências: revisão estruturada e proposta de redesenho do fluxo pericial"
DEFAULT_AUTOR = "Gustavo M. Mendes de Tarso"
DEFAULT_PROFESSOR = "Marcos Aurélio Pereira Valadão"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path, copied_from: str | None = None) -> dict:
    st = path.stat()
    return {
        "arquivo": path.name,
        "caminho": str(path),
        "origem": copied_from,
        "tamanho_bytes": st.st_size,
        "mtime_iso": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
    }


def copy_one(src: Path, dest_dir: Path, required: bool, copied: list[dict], missing: list[str]) -> None:
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {src}")
        missing.append(str(src))
        return
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    copied.append(file_info(dest, copied_from=str(src)))


def toml_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def generate_article_toml(
    artigo_dir: Path,
    dados_dir: Path,
    out_dir: Path,
    toml_output: Path,
    root_dir: Path,
    csl_path: Path,
    titulo: str,
    autor: str,
    professor: str = DEFAULT_PROFESSOR,
    openai_model: str = "gpt-4.1-mini",
) -> Path:
    artigo_dir = artigo_dir.resolve()
    dados_dir = dados_dir.resolve()
    out_dir = out_dir.resolve()
    toml_output = toml_output.resolve()
    root_dir = root_dir.resolve()
    csl_path = csl_path.resolve()

    bib_path = dados_dir / "relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas.bib"
    csv_path = dados_dir / "relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas_seminario.csv"
    prisma_pdf = dados_dir / "relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.pdf"

    missing_required = [p for p in [bib_path, csv_path, prisma_pdf] if not p.exists()]
    if missing_required:
        joined = "\n".join(f"- {p}" for p in missing_required)
        raise FileNotFoundError(f"Não é possível gerar TOML do artigo; arquivos ausentes:\n{joined}")

    if not csl_path.exists():
        print(f"[WARN] CSL ABNT não encontrado em: {csl_path}")
        print("[WARN] O TOML será gerado, mas o DOCX em ABNT autor-data pode falhar ou perder conformidade.")

    out_dir.mkdir(parents=True, exist_ok=True)
    toml_output.parent.mkdir(parents=True, exist_ok=True)

    t = toml_escape(titulo)
    a = toml_escape(autor)
    p = toml_escape(professor)

    work_dir = toml_escape(str(artigo_dir / ".academic_pipeline" / "work"))
    cache_dir = toml_escape(str(artigo_dir / ".academic_pipeline" / "cache"))
    dados_s = toml_escape(str(dados_dir))
    out_s = toml_escape(str(out_dir))
    bib_s = toml_escape(str(bib_path))
    csl_s = toml_escape(str(csl_path))
    root_academic = toml_escape(str(root_dir / "app_bundle" / "misc" / "academic-writing.el"))
    root_fgv = toml_escape(str(root_dir / "app_bundle" / "misc" / "fgv"))
    root_logo = toml_escape(str(root_dir / "app_bundle" / "misc" / "fgv.png"))
    prompt_global = toml_escape(str(root_dir / "app_bundle" / "prompts" / "global" / "orientacao_geral_execucao.txt"))
    prompt_paper = toml_escape(str(root_dir / "app_bundle" / "prompts" / "document" / "paper.txt"))

    content = f"""[projeto]
nome = "artigo_final_atestmed_abnt"
descricao = "Artigo final gerado automaticamente a partir dos estudos selecionados no PRISMA final."
preset = "paper_local_fgv"

[instituicao]
perfil = "fgv"

[openai]
model = "{toml_escape(openai_model)}"

[pipeline]
modo_entrada = "documentos_locais"
executar_pesquisa = false
executar_documento = true
executar_bundle = false

[paths]
document_output_dir = "{out_s}"
research_output_dir = "output_pesquisa"
work_dir = "{work_dir}"
cache_dir = "{cache_dir}"
document_prefix = "artigo_final_atestmed_abnt"
research_prefix = "relatorio_prisma_artigo_final_atestmed"
create_document_subdir = false
create_research_subdir = true
create_work_subdir = true
create_cache_subdir = true

[orientacoes]
paths = []
inline = \"\"\"
Produza um paper acadêmico em português, no layout paper_fgv, com citações autor-data e referências finais em ABNT.

O artigo deve usar exclusivamente os estudos selecionados no PRISMA final, registrados em:
- referencias_incluidas.bib;
- referencias_incluidas_seminario.csv;
- triagem_humana.csv;
- relatorio_prisma_final.pdf;
- manifestos de congelamento.

Não invente referências, autores, DOI, periódicos, dados empíricos ou conclusões que não estejam nos insumos.

Estrutura obrigatória:
1. Resumo;
2. Palavras-chave;
3. Introdução;
4. Método;
5. Resultados da revisão estruturada;
6. Discussão;
7. Proposta de redesenho do fluxo decisório do ATESTMED;
8. Limitações;
9. Conclusão.

O método deve explicitar que o artigo deriva de revisão estruturada com fluxo PRISMA, curadoria assistida por IA e revisão humana final.

A discussão deve conectar as evidências selecionadas ao problema da Perícia Médica Federal: análise documental, teleperícia, perícia presencial, gestão de filas, alocação de capacidade, qualidade decisória, auditabilidade e equidade territorial.

Não crie seção manual de Referências. O renderizador do programa deve inserir a bibliografia automaticamente.
\"\"\"

[documentos_locais]
ativos = true
modo_entrada = "documentos_locais"
input_zip = ""
input_dir = "{dados_s}"
tipos = ["pdf", "csv", "json", "txt", "md", "org"]
recursive = false
limpar_extracao_anterior = true
copiar_para_fulltext_cache = true
limpar_cache_anterior = true
max_caracteres_por_doc = 80000
auto_detect_bib = true
gerar_bib_revisado_ia = false
enriquecer_metadados_buscadores = false
fontes_metadados = []
min_score_match_metadados = 0.82
extrair_doi_dos_pdfs = false
doi_manifest_path = ""
preferir_doi_manual = true
buscar_metadados_por_doi = false
incluir_notas_metadados_inferidos = false
deduplicar_bib = true
deduplicar_referencias = true
autor_padrao = ""
ano_padrao = "s.d."

[pesquisa]
tema = "ATESTMED, saúde digital e decisão baseada em evidências"
recorte = "Redesenho do fluxo de análise dos benefícios por incapacidade na Perícia Médica Federal, articulando análise documental, teleperícia, perícia presencial, capacidade operacional, auditabilidade e equidade territorial."
objetivo = "Analisar evidências selecionadas por revisão estruturada para propor um fluxo decisório mais eficiente, seguro e auditável para o ATESTMED e para a alocação de requerimentos de benefícios por incapacidade."
pergunta_pesquisa = "Como estruturar um fluxo de alocação de requerimentos entre análise documental, teleperícia e perícia presencial que reduza o tempo de espera sem comprometer a qualidade decisória, a equidade territorial e o controle?"
hipotese = ""
palavras_chave = ["ATESTMED", "saúde digital", "perícia médica", "benefícios por incapacidade", "decisão baseada em evidências", "teleperícia"]
gerar_palavras_chave_ia = false
idiomas = ["português", "inglês", "espanhol"]
tipo_estudo = "paper acadêmico baseado em revisão estruturada de evidências"

[paper]
ativo = true
tese_central = "O ATESTMED deve ser compreendido como componente de um fluxo decisório estratificado por risco, suficiência documental e necessidade de avaliação presencial, e não apenas como substituição digital da perícia presencial."
estrutura_desejada = ["Introdução", "Método", "Resultados", "Discussão", "Proposta de redesenho do fluxo decisório", "Limitações", "Conclusão"]
argumentos_obrigatorios = [
  "Distinguir análise documental, teleperícia e perícia presencial como camadas decisórias complementares.",
  "Relacionar redução de fila com qualidade decisória, auditabilidade e equidade territorial.",
  "Apresentar critérios de elegibilidade documental e de escalonamento para avaliação remota ou presencial.",
  "Discutir o papel de dados e inteligência artificial como apoio, não como substituição da decisão pericial."
]
orientacoes_metodologicas = "Explicitar revisão estruturada com fluxo PRISMA, curadoria assistida por IA e decisão humana final sobre estudos incluídos."
limites_do_escopo = "Não avaliar causalmente o impacto do ATESTMED; não extrapolar além das evidências selecionadas."
tom_de_redacao = "acadêmico analítico"
instrucoes_adicionais = "Usar citações frequentes, em autor-data, exclusivamente com as chaves BibTeX permitidas."

[resumo_artigos]
ativo = false
geracao_em_etapas = false
modo = "analitico_comparativo"
usar_apenas_corpus_local = false
incluir_referencias = true
incluir_mapa_mental = false

[atividade]
curso = "Mestrado Acadêmico em Políticas Públicas e Governo"
turma = "2026.1"
polo = "Brasília"
disciplina = "Decisões Baseadas em Evidência"
professor = "{p}"
aluno = "{a}"
data = "2026"
titulo_trabalho = "{t}"

[documento]
tipo_documento = "paper"
tipo_conteudo = "paper"
genero_academico = "paper"
layout = "paper_fgv"
classe_latex = "fgv-paper"
titulo_trabalho = "{t}"
autor = "{a}"
inferir_campos_vazios_ia = true
institution_name = "Fundação Getúlio Vargas"
program_name = "Mestrado Acadêmico em Políticas Públicas e Governo"
course_name = "Mestrado Acadêmico em Políticas Públicas e Governo"
discipline_name = "Decisões Baseadas em Evidência"
professor_name = "{p}"
city_name = "Brasília"
ano = "2026"
data = "2026"
papertype = "Paper acadêmico"
perfil_redacao = "academico_analitico"
covernote = "Trabalho acadêmico elaborado para a disciplina Decisões Baseadas em Evidência."
estilo_citacao = "abnt"
sistema_citacao = "autor-data"
referencias_formais = true
exportar_org = true
exportar_pdf = true
exportar_docx = true
gerar_documento_json = true
modo_renderizacao = "document_model"
usar_citacoes_latex_diretas = true
validar_org_final = true
falhar_se_org_tiver_chave_crua = true
falhar_se_org_tiver_empty_citation = true
falhar_se_org_tiver_mencao_tecnica = true

[idiomas_saida]
principal = "pt-BR"
gerar_traducao_ia = false
idiomas_adicionais = []
preservar_referencias_originais = true
max_chars_por_lote = 12000

[resumos_paper]
ativo = true
principal = "pt-BR"
gerar_resumo_principal = true
gerar_resumo_adicional = false
idiomas_adicionais = []
gerar_palavras_chave_adicionais = true
max_palavras = 250

[bibliografia]
ativo = true
bib_path = "{bib_s}"
referencias_bib = "{bib_s}"
estilo_citacao = "abnt"
latex_style = "abnt"
latex_options = "backend=biber,style=abnt,sorting=nty,giveninits=true"
docx_csl = "{csl_s}"
sistema_citacao = "autor-data"
backend = "biber"
gerar_arquivo_bib = true
buscar_metadados_por_doi = false
enriquecer_metadados_buscadores = false

[relatorio_pesquisa]
ativo = false
tipo = "prisma"
titulo = ""
exportar_json = false
exportar_org = false
exportar_pdf = false
exportar_docx = false
exportar_xlsx = false
exportar_fluxograma = false
validar = false
falhar_se_invalido = false
prisma_json_path = ""
pesquisa_dir_existente = "{dados_s}"

[docx]
ativo = true
reference_docx = ""
usar_pandoc = true
csl_path = "{csl_s}"
falhar_se_pandoc_falhar = true
incluir_capa = true
incluir_referencias = true
incluir_mapa_mental = false

[latex]
pdf_engine = "lualatex"
org_latex_class_init = "{root_academic}"
latex_extra_path = "{root_fgv}"
fgv_logo_path = "{root_logo}"

[prompts]
ativos = true
global_paths = ["{prompt_global}"]
institution_paths = ["profile://prompts/fgv_geral.txt"]
research_paths = []
paper_paths = ["{prompt_paper}"]
atividade_paths = []
resumo_artigos_paths = []
dissertacao_paths = []
prisma_paths = []
document_paths = []

[mapa_mental]
gerar = false
ativo = false
posicao = "apos_referencias"
titulo = "Mapa mental dos textos analisados"
arquivo = "mapa_mental"
formato = "png"
renderizar = false
inserir_no_org = false
plantuml_jar_path = ""
plantuml_limit_size = 8192
colorido = true
falhar_se_nao_renderizar = false

[conformidade]
ativo = true
gerar_relatorio = true

[qualidade]
ativo = true
gerar_relatorio = true

[controle]
nao_interativo = true
dry_run = false
mock_run = false
"""
    toml_output.write_text(content, encoding="utf-8")
    print(f"[OK] TOML do artigo gerado: {toml_output}")
    return toml_output


def freeze_inputs(
    out_dir: Path,
    artigo_dir: Path | None = None,
    dest_dir: Path | None = None,
    prefix: str | None = None,
    gerar_toml_artigo: bool = False,
    toml_output: Path | None = None,
    root_dir: Path | None = None,
    csl_path: Path | None = None,
    titulo: str = DEFAULT_TITULO,
    autor: str = DEFAULT_AUTOR,
    professor: str = DEFAULT_PROFESSOR,
    openai_model: str = "gpt-4.1-mini",
) -> Path:
    out_dir = out_dir.resolve()
    prefix = prefix or out_dir.name
    artigo_dir = (artigo_dir or DEFAULT_ARTIGO_DIR).resolve()
    dest_dir = (dest_dir.resolve() if dest_dir else artigo_dir / "dados_prisma")
    dest_dir.mkdir(parents=True, exist_ok=True)

    required_names = [
        f"{prefix}.referencias_incluidas.bib",
        f"{prefix}.referencias_incluidas_seminario.csv",
        f"{prefix}.triagem_humana.csv",
        f"{prefix}.relatorio_prisma_final.pdf",
    ]
    optional_names = [
        f"{prefix}.curadoria_ia_referencias.xlsx",
        f"{prefix}.relatorio_prisma_preliminar.pdf",
        f"{prefix}.diagrama_prisma.png",
        f"{prefix}.diagrama_prisma_contagens.json",
        f"{prefix}.busca_prisma_log.json",
        f"{prefix}.triagem_titulo_resumo.csv",
        f"{prefix}.triagem_titulo_resumo.xlsx",
        f"{prefix}.curadoria_ia_resumo.txt",
        f"{prefix}.curadoria_ia_log.json",
    ]

    copied: list[dict] = []
    missing_optional: list[str] = []

    for name in required_names:
        copy_one(out_dir / name, dest_dir, required=True, copied=copied, missing=missing_optional)
    for name in optional_names:
        copy_one(out_dir / name, dest_dir, required=False, copied=copied, missing=missing_optional)

    manifest_lines = [f"{item['sha256']}  {item['arquivo']}" for item in sorted(copied, key=lambda x: x["arquivo"])]
    (dest_dir / "MANIFESTO_SHA256.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    arquivo_lines = [
        "Arquivo\tTamanho_bytes\tModificado_em\tSHA256",
        *[
            f"{item['arquivo']}\t{item['tamanho_bytes']}\t{item['mtime_iso']}\t{item['sha256']}"
            for item in sorted(copied, key=lambda x: x["arquivo"])
        ],
    ]
    (dest_dir / "ARQUIVOS_CONGELADOS.txt").write_text("\n".join(arquivo_lines) + "\n", encoding="utf-8")

    generated_toml = None
    if gerar_toml_artigo:
        root_dir = (root_dir or Path.cwd()).resolve()
        csl_path = (csl_path or root_dir / "app_bundle" / "templates" / "csl" / "associacao-brasileira-de-normas-tecnicas.csl").resolve()
        toml_output = (toml_output or artigo_dir / "artigo_final_atestmed_abnt.toml").resolve()
        generated_toml = generate_article_toml(
            artigo_dir=artigo_dir,
            dados_dir=dest_dir,
            out_dir=artigo_dir / "output",
            toml_output=toml_output,
            root_dir=root_dir,
            csl_path=csl_path,
            titulo=titulo,
            autor=autor,
            professor=professor,
            openai_model=openai_model,
        )

    manifest_json = {
        "gerado_em": dt.datetime.now().isoformat(timespec="seconds"),
        "out_dir_origem": str(out_dir),
        "artigo_dir": str(artigo_dir),
        "destino": str(dest_dir),
        "prefixo": prefix,
        "arquivos": sorted(copied, key=lambda x: x["arquivo"]),
        "opcionais_ausentes": missing_optional,
        "toml_artigo_gerado": str(generated_toml) if generated_toml else None,
    }
    (dest_dir / "MANIFESTO_ARTIGO.json").write_text(
        json.dumps(manifest_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Insumos congelados em: {dest_dir}")
    print(f"[OK] Arquivos copiados: {len(copied)}")
    print(f"[OK] Manifesto SHA256: {dest_dir / 'MANIFESTO_SHA256.txt'}")
    print(f"[OK] Manifesto JSON: {dest_dir / 'MANIFESTO_ARTIGO.json'}")
    if generated_toml:
        print(f"[OK] TOML do artigo: {generated_toml}")
    if missing_optional:
        print(f"[INFO] Arquivos opcionais ausentes: {len(missing_optional)}")
    return dest_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Congela insumos finais do artigo, cria manifestos e opcionalmente gera TOML do artigo.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--artigo-dir", default=str(DEFAULT_ARTIGO_DIR))
    p.add_argument("--dest-dir", default=None)
    p.add_argument("--prefix", default=None)
    p.add_argument("--gerar-toml-artigo", action="store_true")
    p.add_argument("--toml-output", default=None)
    p.add_argument("--root-dir", default=None)
    p.add_argument("--csl-path", default=None)
    p.add_argument("--titulo", default=DEFAULT_TITULO)
    p.add_argument("--autor", default=DEFAULT_AUTOR)
    p.add_argument("--professor", default=DEFAULT_PROFESSOR)
    p.add_argument("--openai-model", default="gpt-4.1-mini")
    a = p.parse_args(argv)

    out_dir = Path(a.out_dir)
    artigo_dir = Path(a.artigo_dir) if a.artigo_dir else DEFAULT_ARTIGO_DIR
    dest_dir = Path(a.dest_dir) if a.dest_dir else None
    prefix = a.prefix or out_dir.name
    freeze_inputs(
        out_dir=out_dir,
        artigo_dir=artigo_dir,
        dest_dir=dest_dir,
        prefix=prefix,
        gerar_toml_artigo=bool(a.gerar_toml_artigo),
        toml_output=Path(a.toml_output) if a.toml_output else None,
        root_dir=Path(a.root_dir) if a.root_dir else None,
        csl_path=Path(a.csl_path) if a.csl_path else None,
        titulo=a.titulo,
        autor=a.autor,
        professor=a.professor,
        openai_model=a.openai_model,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

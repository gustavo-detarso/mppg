#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .document_model import (
        AcademicDocument,
        Section,
        Block,
        TextSpan,
        StrictBaseModel,
    )
else:
    from document_model import (
        AcademicDocument,
        Section,
        Block,
        TextSpan,
        StrictBaseModel,
    )
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .corpus_manager import SourceDoc
else:
    from corpus_manager import SourceDoc
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import shorten_text, write_json, slugify
else:
    from utils import shorten_text, write_json, slugify
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .prompt_manager import load_prompt_bundle
else:
    from prompt_manager import load_prompt_bundle


def _is_resumo_artigos_profile(cfg: dict[str, Any]) -> bool:
    projeto = cfg.get("projeto", {}) if isinstance(cfg.get("projeto"), dict) else {}
    resumo = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
    return bool(resumo.get("ativo")) or str(projeto.get("preset") or "").strip() == "resumo_artigos_local_fgv"


FICHAMENTO_SECTION_TITLES: tuple[str, ...] = (
    "REFERÊNCIAS BIBLIOGRÁFICAS",
    "SÍNTESE DOS TEXTOS",
    "PRINCIPAIS CONCEITOS E ARGUMENTOS",
    "ANÁLISE CRÍTICA E REFLEXÕES PESSOAIS",
    "CONEXÕES E DIÁLOGOS ENTRE OS TEXTOS",
    "APLICAÇÕES EM POLÍTICAS PÚBLICAS E GOVERNO",
    "QUESTÕES PARA APROFUNDAMENTO",
)


def _is_fichamento_profile(cfg: dict[str, Any]) -> bool:
    projeto = cfg.get("projeto", {}) if isinstance(cfg.get("projeto"), dict) else {}
    documento = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
    fichamento = cfg.get("fichamento", {}) if isinstance(cfg.get("fichamento"), dict) else {}
    return bool(fichamento.get("ativo")) or str(projeto.get("preset") or "").strip() == "fichamento_fgv" or str(documento.get("tipo_conteudo") or "").strip() == "fichamento"


def _fichamento_minimum(cfg: dict[str, Any]) -> int:
    selection = cfg.get("selecao_corpus", {}) if isinstance(cfg.get("selecao_corpus"), dict) else {}
    try:
        value = int(selection.get("quantidade_minima_textos", 3))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("[selecao_corpus].quantidade_minima_textos deve ser inteiro.") from exc
    if value < 3:
        raise RuntimeError("O perfil fichamento_fgv exige quantidade_minima_textos >= 3.")
    return value


def _validate_fichamento_corpus_contract(cfg: dict[str, Any], docs: list[SourceDoc]) -> None:
    if not _is_fichamento_profile(cfg):
        return
    minimum = _fichamento_minimum(cfg)
    usable = [d for d in docs if str(getattr(d, "extracted_text", "") or "").strip() and not str(getattr(d, "kind", "") or "").endswith("_erro")]
    if len(usable) < minimum:
        raise RuntimeError(f"O fichamento exige ao menos {minimum} textos substantivos utilizáveis; foram admitidos {len(usable)}.")


def _normalize_fichamento_title(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _validate_fichamento_document_contract(document: AcademicDocument) -> None:
    titles = [_normalize_fichamento_title(getattr(section, "title", "")) for section in document.sections]
    expected = [_normalize_fichamento_title(title) for title in FICHAMENTO_SECTION_TITLES]
    if titles != expected:
        raise RuntimeError("O document.json de fichamento não respeitou as sete seções canônicas em ordem. " + f"Esperado={list(FICHAMENTO_SECTION_TITLES)!r}; observado={[getattr(section, 'title', '') for section in document.sections]!r}")


def _safe_int(value: Any, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _resumo_artigos_depth_directives(cfg: dict[str, Any], docs_count: int) -> str:
    if not _is_resumo_artigos_profile(cfg):
        return ""
    resumo = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
    nivel = str(resumo.get("nivel_detalhamento") or resumo.get("profundidade_analitica") or "profundo").strip().lower()
    if nivel == "médio":
        nivel = "medio"
    if docs_count <= 4:
        default_words = 1500 if nivel == "exaustivo" else 1200 if nivel in {"profundo", "alto"} else 850 if nivel == "medio" else 650
        default_paragraphs = 11 if nivel == "exaustivo" else 9 if nivel in {"profundo", "alto"} else 6
    elif docs_count <= 8:
        default_words = 1100 if nivel == "exaustivo" else 900 if nivel in {"profundo", "alto"} else 650 if nivel == "medio" else 450
        default_paragraphs = 9 if nivel == "exaustivo" else 7 if nivel in {"profundo", "alto"} else 5
    else:
        default_words = 800 if nivel == "exaustivo" else 650 if nivel in {"profundo", "alto"} else 450 if nivel == "medio" else 300
        default_paragraphs = 7 if nivel == "exaustivo" else 5 if nivel in {"profundo", "alto"} else 4
    min_words = _safe_int(resumo.get("min_palavras_por_artigo"), default_words, minimum=250, maximum=2500)
    min_paragraphs = _safe_int(resumo.get("min_paragrafos_por_artigo"), default_paragraphs, minimum=3, maximum=18)
    min_comparison = _safe_int(resumo.get("min_palavras_comparacao"), 1100 if docs_count <= 5 else 850, minimum=300, maximum=2500)
    min_synthesis = _safe_int(resumo.get("min_palavras_sintese"), 900 if docs_count <= 5 else 650, minimum=300, maximum=2200)
    eixos = resumo.get("eixos_analise") or [
        "problema e questão central",
        "objetivo e escopo do texto",
        "argumento/tese principal",
        "conceitos e categorias analíticas",
        "método, desenho de pesquisa ou tipo de evidência",
        "achados e contribuições",
        "limites, tensões e lacunas",
        "diálogo com os demais textos do corpus",
    ]
    if not isinstance(eixos, list):
        eixos = [str(eixos)]
    eixos_txt = "; ".join(str(e).strip() for e in eixos if str(e).strip())
    exigir_matriz = bool(resumo.get("exigir_matriz_analitica_por_texto", True))
    incluir_tabela = bool(resumo.get("incluir_tabela_comparativa", True))
    dialogo = bool(resumo.get("exigir_dialogo_entre_textos", True))
    evitar_sinoptico = bool(resumo.get("evitar_resumo_sinoptico", True))
    matriz_line = "- Em cada subseção individual, organize a análise por eixos, mesmo em texto corrido: " + eixos_txt + "." if exigir_matriz else "- Em cada subseção individual, cubra os eixos analíticos relevantes sem transformar a seção em ficha mecânica."
    tabela_line = "- Inclua uma tabela comparativa sintética antes ou no início da seção de comparação, com colunas como texto, problema, abordagem/método, conceito central, contribuição e limite/tensão." if incluir_tabela and docs_count <= 8 else "- A comparação pode ser feita em texto corrido quando a tabela não for adequada."
    dialogo_line = "- Ao final de cada subseção individual, inclua pelo menos um parágrafo de diálogo analítico com outro(s) texto(s) do corpus, sem forçar convergências artificiais." if dialogo and docs_count > 1 else ""
    sinoptico_line = "- Não aceite respostas sinópticas. Evite parágrafos genéricos; cada afirmação analítica importante deve estar ancorada em elementos do texto-base." if evitar_sinoptico else ""
    return f"""

Diretrizes específicas de profundidade para o perfil resumo_artigos_local_fgv:
- Profundidade analítica solicitada: {nivel}. O documento deve ser substancial, interpretativo e comparativo; não deve parecer ementa, sinopse ou resumo escolar curto.
- Para cada artigo/texto, produza pelo menos {min_paragraphs} parágrafos analíticos e aproximadamente {min_words} palavras, salvo se o texto-base for muito curto.
{matriz_line}
- Em cada resumo individual, diferencie claramente: exposição fiel do argumento do autor; análise crítica/interpretativa; contribuição para a disciplina; e limites/tensões.
{dialogo_line}
{sinoptico_line}
{tabela_line}
- A comparação entre os textos deve ter pelo menos {min_comparison} palavras e discutir convergências, divergências, diferenças de escala, método, objeto, pressupostos teóricos, arquitetura conceitual e contribuição para a disciplina.
- A síntese analítica deve ter pelo menos {min_synthesis} palavras e apresentar interpretação própria do corpus, não apenas repetir os resumos.
- Use citações ao longo das análises individuais e da comparação, mas nunca escreva chaves BibTeX cruas no texto visível.
- Não numere manualmente títulos de seções ou subseções. Use títulos como "Introdução", "Resumo individual dos textos" e o nome de cada artigo; o Org/LaTeX fará a numeração automaticamente.
""".strip()


def _meta_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    atividade = cfg.get("atividade", {}) if isinstance(cfg.get("atividade"), dict) else {}
    documento = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
    pesquisa = cfg.get("pesquisa", {}) if isinstance(cfg.get("pesquisa"), dict) else {}
    return {
        "tipo_documento": documento.get("tipo_documento") or "paper",
        "titulo": documento.get("titulo_trabalho") or atividade.get("titulo_trabalho") or pesquisa.get("titulo_sugerido") or "",
        "autor": documento.get("autor") or atividade.get("aluno") or "Gustavo M. Mendes de Tarso",
        "instituicao": documento.get("institution_name") or "Fundação Getúlio Vargas",
        "programa": documento.get("program_name") or "",
        "curso": documento.get("course_name") or atividade.get("curso") or "Mestrado Acadêmico em Políticas Públicas e Governo",
        "turma": atividade.get("turma") or "",
        "polo": atividade.get("polo") or "Brasília",
        "disciplina": documento.get("discipline_name") or atividade.get("disciplina") or "",
        "professor": documento.get("professor_name") or atividade.get("professor") or "",
        "cidade": documento.get("city_name") or atividade.get("polo") or "Brasília",
        "ano": documento.get("ano") or atividade.get("data") or "",
        "data": atividade.get("data") or documento.get("data") or "",
        "tipo_trabalho": documento.get("papertype") or ("Paper acadêmico" if str(documento.get("tipo_documento") or "paper") == "paper" else "Atividade acadêmica"),
        "nota_capa": documento.get("covernote") or "Trabalho acadêmico elaborado para a disciplina.",
    }


def build_prompt(
    cfg: dict[str, Any],
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    bib_keys: list[str],
    bib_path: Path,
) -> str:
    pesquisa = cfg.get("pesquisa", {}) if isinstance(cfg.get("pesquisa"), dict) else {}
    documento = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
    meta = _meta_from_cfg(cfg)
    prompt_bundle = load_prompt_bundle(cfg, "document", document_type=str(meta.get("tipo_documento") or "paper"))
    resumo_cfg = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
    if _is_resumo_artigos_profile(cfg):
        per_doc_chars = _safe_int(resumo_cfg.get("max_chars_por_documento"), 14000, minimum=4000, maximum=30000)
        total_corpus_chars = _safe_int(resumo_cfg.get("max_chars_total_corpus"), 95000, minimum=30000, maximum=180000)
        if docs:
            per_doc_chars = min(per_doc_chars, max(4000, total_corpus_chars // max(1, len(docs))))
    else:
        per_doc_chars = 5000
        total_corpus_chars = 50000

    doc_summaries = []
    for d in docs:
        doc_summaries.append({
            "label": d.label,
            "bib_key": d.bib_key,
            "excerpt": shorten_text(d.extracted_text, per_doc_chars),
        })
    orientation_summaries = []
    for d in orientations:
        orientation_summaries.append({"label": d.label, "excerpt": shorten_text(d.extracted_text, 4000)})

    prompt_extras = prompt_bundle.text or "Nenhuma diretiva complementar carregada."
    depth_directives = _resumo_artigos_depth_directives(cfg, len(docs))

    return textwrap.dedent(f"""
    Você é um gerador acadêmico estruturado. Gere um AcademicDocument JSON canônico, seguindo estritamente o schema.

    Regras obrigatórias:
    1. Não gere ORG, LaTeX, DOCX, Markdown ou PDF. Gere apenas o objeto estruturado AcademicDocument.
    2. Use citações como objetos do tipo citation, nunca como texto cru.
    3. Toda citação deve usar exclusivamente chaves BibTeX da lista permitida.
    3.1. Nunca escreva a chave BibTeX literal no texto, em títulos, listas, tabelas ou parágrafos. Exemplo proibido: geet2019_policy_design. Para citar, use objeto citation com keys=["geet2019_policy_design"].
    3.2. Nunca use citações numéricas em colchetes como [1], [2] ou [1, 2, 3]. Para documentos FGV/ABNT/APA, a saída deve ficar em autor-data, gerada pelo renderizador a partir de objetos citation.
    3.3. Se quiser citar vários textos no mesmo ponto, use um único objeto citation com várias chaves, por exemplo keys=["chave1", "chave2"], e nunca texto como [1, 2].
    4. Não mencione pipeline, cache, OCR, fulltext_cache, metadados incompletos, metadados inferidos ou limitações técnicas de extração.
    5. Se houver mapa mental, não o crie como seção textual comum; ele será renderizado por outro módulo.
    6. O conteúdo deve seguir as orientações e o tipo do documento.
    7. Mantenha coerência acadêmica: pergunta, objetivo, tese/hipótese, desenvolvimento e conclusão.
    8. Para paper, produza estrutura acadêmica enxuta; para atividade, responda ao roteiro e sintetize/comparare os textos conforme orientação.
    9. Não invente referência fora do .bib.
    9.1. Não crie seção final chamada REFERÊNCIAS, BIBLIOGRAFIA ou similar; o renderizador insere a bibliografia automaticamente com base no .bib.
    10. Inclua metadata conforme os dados fornecidos.
    10.1. Se tema, recorte, objetivo ou pergunta orientadora estiverem vazios no TOML, mas estiverem presentes nas orientações carregadas, infira-os fielmente dessas orientações; não invente um novo enunciado.
    10.2. Se pesquisa.gerar_palavras_chave_ia=true e pesquisa.palavras_chave estiver vazio, infira de 4 a 6 palavras-chave acadêmicas a partir do tema, da pergunta orientadora, do enunciado, das orientações e do corpus local. Use-as no campo de palavras-chave do documento quando esse campo for aplicável, sem criar uma seção artificial apenas para listá-las.

    Diretivas complementares carregadas pelo prompt bank:
    {prompt_extras}

    {depth_directives}

    Metadados sugeridos:
    {json.dumps(meta, ensure_ascii=False, indent=2)}

    Pesquisa/TOML:
    {json.dumps(pesquisa, ensure_ascii=False, indent=2)}

    Configuração do documento:
    {json.dumps(documento, ensure_ascii=False, indent=2)}

    Chaves BibTeX permitidas:
    {json.dumps(bib_keys, ensure_ascii=False)}

    Caminho do .bib:
    {bib_path.name}

    Orientações:
    {shorten_text(json.dumps(orientation_summaries, ensure_ascii=False, indent=2), 25000)}

    Documentos-base:
    {shorten_text(json.dumps(doc_summaries, ensure_ascii=False, indent=2), total_corpus_chars)}
    """).strip()



class SectionBundle(StrictBaseModel):
    """Pacote estruturado usado na geração incremental por etapas."""
    sections: list[Section] = Field(default_factory=list)
    entries_used: list[str] = Field(default_factory=list)


def _is_substantive_text(value: Any) -> bool:
    text = str(value or "").strip()
    # evita tratar apenas pontuação, numeração ou fragmentos mínimos como parágrafo real
    return len(text) >= 30 and bool(re.search(r"[A-Za-zÀ-ÿ]{4,}", text))


def _content_has_substantive_text(content: list[Any] | None) -> bool:
    for item in content or []:
        if getattr(item, "type", None) == "text" and _is_substantive_text(getattr(item, "text", "")):
            return True
    return False


def _content_citations(content: list[Any] | None) -> list[Any]:
    out: list[Any] = []
    for item in content or []:
        if getattr(item, "type", None) == "citation" and getattr(item, "keys", None):
            out.append(item)
    return out


def _strip_numeric_citation_markers(text: str) -> str:
    # Remove citações numéricas cruas que a IA às vezes deixa no campo text quando
    # as citações estruturadas já vieram separadas em block.content.
    cleaned = re.sub(r"\s*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", "", str(text or ""))
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def repair_section_blocks(section: Section) -> Section:
    """Repara parágrafos retornados pela IA em Structured Outputs.

    Em algumas chamadas por etapa, a IA preencheu o texto principal em
    ``block.text`` e colocou apenas a citação em ``block.content``. Como o
    renderizador prioriza ``content`` quando existe, o PDF acabava mostrando
    somente linhas como [1], [2] etc. A correção normaliza esses blocos para
    ``content=[TextSpan(text=...), Citation(...)]`` antes de qualquer etapa
    posterior, checkpoint ou renderização.
    """
    for block in section.blocks or []:
        if block.type == "paragraph":
            raw_text = _strip_numeric_citation_markers(block.text or "")
            if raw_text and not _content_has_substantive_text(block.content):
                citations = _content_citations(block.content)
                block.content = [TextSpan(text=raw_text), *citations]
                block.text = ""
            elif raw_text and not block.content:
                block.content = [TextSpan(text=raw_text)]
                block.text = ""
        elif block.type in {"bullet_list", "numbered_list"}:
            block.items = [_strip_numeric_citation_markers(x) for x in (block.items or [])]
        elif block.type == "table" and block.table:
            block.table.headers = [_strip_numeric_citation_markers(x) for x in (block.table.headers or [])]
            block.table.rows = [[_strip_numeric_citation_markers(x) for x in row] for row in (block.table.rows or [])]
    return section


def repair_section_bundle(bundle: SectionBundle) -> SectionBundle:
    for section in bundle.sections or []:
        repair_section_blocks(section)
    return bundle


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def _section_plain_text(section: Section, limit: int = 12000) -> str:
    parts: list[str] = [str(section.title or "").strip()]
    for block in section.blocks or []:
        if block.type == "paragraph":
            # Usa block.text quando content contém apenas citação, caso típico que
            # gerava resumos individuais visualmente vazios/citação-only.
            if block.text and not _content_has_substantive_text(block.content):
                raw = _strip_numeric_citation_markers(block.text)
                citations = []
                for item in _content_citations(block.content):
                    keys = ",".join(getattr(item, "keys", []) or [])
                    if keys:
                        citations.append(f"[{keys}]")
                parts.append((raw + (" " + " ".join(citations) if citations else "")).strip())
            elif block.content:
                buf = []
                for item in block.content:
                    if getattr(item, "type", None) == "citation":
                        keys = ",".join(getattr(item, "keys", []) or [])
                        if keys:
                            buf.append(f"[{keys}]")
                    else:
                        buf.append(str(getattr(item, "text", "") or ""))
                parts.append("".join(buf))
            elif block.text:
                parts.append(_strip_numeric_citation_markers(block.text))
        elif block.type in {"bullet_list", "numbered_list"}:
            parts.extend(str(x) for x in block.items or [])
        elif block.type == "table" and block.table:
            if block.table.caption:
                parts.append(block.table.caption)
            if block.table.headers:
                parts.append(" | ".join(block.table.headers))
            for row in block.table.rows or []:
                parts.append(" | ".join(str(x) for x in row))
        elif block.text:
            parts.append(block.text)
    return shorten_text("\n".join(x for x in parts if x), limit)


def _model_dump_for_checkpoint(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _write_checkpoint(checkpoint_dir: Path | None, prefix: str, name: str, payload: Any) -> None:
    if not checkpoint_dir:
        return
    safe = slugify(name)
    path = checkpoint_dir / f"{prefix}.checkpoint_{safe}.json"
    write_json(path, _model_dump_for_checkpoint(payload))


def _resumo_limits(cfg: dict[str, Any], docs_count: int) -> tuple[int, int]:
    resumo_cfg = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
    per_doc_chars = _safe_int(resumo_cfg.get("max_chars_por_documento"), 14000, minimum=4000, maximum=30000)
    total_corpus_chars = _safe_int(resumo_cfg.get("max_chars_total_corpus"), 95000, minimum=30000, maximum=180000)
    if docs_count:
        per_doc_chars = min(per_doc_chars, max(4000, total_corpus_chars // max(1, docs_count)))
    return per_doc_chars, total_corpus_chars


def _common_generation_rules(cfg: dict[str, Any], bib_keys: list[str], bib_path: Path, docs_count: int) -> str:
    prompt_bundle = load_prompt_bundle(
        cfg,
        "document",
        document_type=str((cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}).get("tipo_documento") or "paper"),
    )
    prompt_extras = prompt_bundle.text or "Nenhuma diretiva complementar carregada."
    depth_directives = _resumo_artigos_depth_directives(cfg, docs_count)
    return textwrap.dedent(f"""
    Regras obrigatórias para a geração estruturada:
    1. Gere apenas o objeto estruturado solicitado. Não gere Markdown, ORG, LaTeX ou texto fora do JSON estruturado.
    2. Use citações como objetos do tipo citation, nunca como texto cru.
    3. Toda citação deve usar exclusivamente chaves BibTeX da lista permitida.
    4. Nunca escreva a chave BibTeX literal no texto visível, em títulos, listas, tabelas ou parágrafos.
    5. Nunca use citações numéricas em colchetes, como [1], [2] ou [1, 2, 3]. A saída final deve ser autor-data pelo renderizador.
    6. Não crie seção final chamada REFERÊNCIAS, BIBLIOGRAFIA ou similar; o renderizador insere a bibliografia automaticamente com base no .bib.
    7. Não numere manualmente títulos de seções ou subseções. O Org/LaTeX fará a numeração automaticamente.
    8. Não mencione pipeline, cache, OCR, fulltext_cache, metadados incompletos, metadados inferidos ou limitações técnicas de extração.
    9. Não invente referência fora do .bib.

    Diretivas complementares carregadas pelo prompt bank:
    {prompt_extras}

    {depth_directives}

    Chaves BibTeX permitidas:
    {json.dumps(bib_keys, ensure_ascii=False)}

    Caminho do .bib:
    {bib_path.name}
    """).strip()


def _orientation_context(orientations: list[SourceDoc], limit: int = 18000) -> str:
    orientation_summaries = []
    for d in orientations:
        orientation_summaries.append({"label": d.label, "excerpt": shorten_text(d.extracted_text, 4000)})
    return shorten_text(json.dumps(orientation_summaries, ensure_ascii=False, indent=2), limit)


def _docs_index_context(docs: list[SourceDoc], current: SourceDoc | None = None, limit: int = 18000) -> str:
    rows = []
    for idx, d in enumerate(docs, start=1):
        rows.append({
            "indice": idx,
            "label": d.label,
            "bib_key": d.bib_key,
            "documento_atual": bool(current is not None and d is current),
            "amostra": shorten_text(d.extracted_text, 1200 if current is not d else 2200),
        })
    return shorten_text(json.dumps(rows, ensure_ascii=False, indent=2), limit)


def _parse_section(client: Any, model: str, prompt: str, *, section_name: str) -> Section:
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=Section,
    )
    if resp.output_parsed is None:
        raise RuntimeError(f"A IA não retornou Section estruturada para {section_name}.")
    return repair_section_blocks(resp.output_parsed)


def _parse_section_bundle(client: Any, model: str, prompt: str, *, bundle_name: str) -> SectionBundle:
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=SectionBundle,
    )
    if resp.output_parsed is None:
        raise RuntimeError(f"A IA não retornou SectionBundle estruturado para {bundle_name}.")
    return repair_section_bundle(resp.output_parsed)


def _generate_article_section(
    client: Any,
    model: str,
    cfg: dict[str, Any],
    doc: SourceDoc,
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    bib_keys: list[str],
    bib_path: Path,
    idx: int,
    total: int,
) -> Section:
    per_doc_chars, _ = _resumo_limits(cfg, len(docs))
    pesquisa = cfg.get("pesquisa", {}) if isinstance(cfg.get("pesquisa"), dict) else {}
    common_rules = _common_generation_rules(cfg, bib_keys, bib_path, len(docs))
    prompt = textwrap.dedent(f"""
    Você está gerando, em etapas, a seção individual de análise do artigo {idx}/{total}.

    Retorne exatamente uma Section estruturada:
    - level = 2
    - title = título do artigo/texto analisado, sem numeração manual
    - blocks = parágrafos analíticos densos, e tabela apenas se realmente útil

    A seção deve reconstruir profundamente o texto analisado, cobrindo:
    - problema/questão central;
    - objetivo e escopo;
    - argumento ou tese principal;
    - conceitos e categorias mobilizados;
    - método, evidências ou estratégia argumentativa;
    - achados, contribuições e implicações para a disciplina;
    - limites, tensões e lacunas;
    - diálogo com o corpus, sem forçar unidade artificial.

    Use preferencialmente a chave do próprio texto analisado: {doc.bib_key}
    Pode usar outras chaves apenas para diálogo comparativo real.

    {common_rules}

    Pesquisa/TOML:
    {json.dumps(pesquisa, ensure_ascii=False, indent=2)}

    Orientações do projeto:
    {_orientation_context(orientations)}

    Índice dos documentos do corpus:
    {_docs_index_context(docs, current=doc)}

    Texto-base integral/excerto ampliado do artigo atual:
    {json.dumps({"label": doc.label, "bib_key": doc.bib_key, "excerpt": shorten_text(doc.extracted_text, per_doc_chars)}, ensure_ascii=False, indent=2)}
    """).strip()
    section = _parse_section(client, model, prompt, section_name=f"artigo_{idx}")
    section.level = 2
    return section


def _generate_comparison_section(
    client: Any,
    model: str,
    cfg: dict[str, Any],
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    article_sections: list[Section],
    bib_keys: list[str],
    bib_path: Path,
) -> Section:
    pesquisa = cfg.get("pesquisa", {}) if isinstance(cfg.get("pesquisa"), dict) else {}
    common_rules = _common_generation_rules(cfg, bib_keys, bib_path, len(docs))
    analyses = [{"title": s.title, "analysis": _section_plain_text(s, 12000)} for s in article_sections]
    prompt = textwrap.dedent(f"""
    Você está gerando a seção de comparação transversal entre os textos do corpus.

    Retorne exatamente uma Section estruturada:
    - level = 1
    - title = "Comparação entre os textos"
    - blocks = tabela comparativa sintética, se útil, seguida de análise em prosa densa.

    A comparação deve discutir, quando aplicável:
    - convergências e divergências reais;
    - diferença de objeto, escala analítica e método;
    - conceitos e categorias de cada texto;
    - pressupostos teóricos;
    - contribuição para a disciplina;
    - limites comparativos;
    - articulações possíveis sem forçar unidade temática artificial.

    {common_rules}

    Pesquisa/TOML:
    {json.dumps(pesquisa, ensure_ascii=False, indent=2)}

    Orientações do projeto:
    {_orientation_context(orientations)}

    Índice dos documentos do corpus:
    {_docs_index_context(docs)}

    Análises individuais já geradas:
    {shorten_text(json.dumps(analyses, ensure_ascii=False, indent=2), 50000)}
    """).strip()
    section = _parse_section(client, model, prompt, section_name="comparacao")
    section.level = 1
    section.title = section.title or "Comparação entre os textos"
    return section


def _generate_intro_synthesis_bundle(
    client: Any,
    model: str,
    cfg: dict[str, Any],
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    article_sections: list[Section],
    comparison_section: Section,
    bib_keys: list[str],
    bib_path: Path,
) -> SectionBundle:
    pesquisa = cfg.get("pesquisa", {}) if isinstance(cfg.get("pesquisa"), dict) else {}
    common_rules = _common_generation_rules(cfg, bib_keys, bib_path, len(docs))
    analyses = [{"title": s.title, "analysis": _section_plain_text(s, 9000)} for s in article_sections]
    comparison = _section_plain_text(comparison_section, 14000)
    prompt = textwrap.dedent(f"""
    Você está gerando as seções de abertura e fechamento analítico do documento.

    Retorne um SectionBundle com exatamente três seções, nesta ordem:
    1. Section level=1 title="Introdução"
    2. Section level=1 title="Síntese analítica"
    3. Section level=1 title="Considerações finais"

    A Introdução deve apresentar o corpus, o foco de leitura, a heterogeneidade ou unidade temática dos textos, o objetivo do documento e o caminho analítico adotado.
    A Síntese analítica deve produzir uma interpretação própria do corpus, articulando os textos sem repetir mecanicamente a comparação.
    As Considerações finais devem consolidar os ganhos analíticos e as principais contribuições para a disciplina.

    {common_rules}

    Pesquisa/TOML:
    {json.dumps(pesquisa, ensure_ascii=False, indent=2)}

    Orientações do projeto:
    {_orientation_context(orientations)}

    Índice dos documentos do corpus:
    {_docs_index_context(docs)}

    Análises individuais já geradas:
    {shorten_text(json.dumps(analyses, ensure_ascii=False, indent=2), 42000)}

    Comparação já gerada:
    {comparison}
    """).strip()
    bundle = _parse_section_bundle(client, model, prompt, bundle_name="sintese")
    for s in bundle.sections:
        s.level = 1
    return bundle


def _build_resumo_artigos_document_model_staged(
    client: Any,
    model: str,
    cfg: dict[str, Any],
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    bib_keys: list[str],
    bib_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    checkpoint_dir: Path | None = None,
    prefix: str = "documento",
) -> AcademicDocument:
    if not docs:
        raise RuntimeError("O perfil resumo_artigos_local_fgv exige ao menos um documento no corpus local.")
    meta_cfg = _meta_from_cfg(cfg)
    article_sections: list[Section] = []
    total = len(docs)
    for idx, d in enumerate(docs, start=1):
        _progress(progress, f"Gerando análise do artigo {idx}/{total}")
        section = _generate_article_section(client, model, cfg, d, docs, orientations, bib_keys, bib_path, idx, total)
        article_sections.append(section)
        _write_checkpoint(checkpoint_dir, prefix, f"artigo_{idx:02d}", section)

    _progress(progress, "Gerando comparação")
    comparison_section = _generate_comparison_section(client, model, cfg, docs, orientations, article_sections, bib_keys, bib_path)
    _write_checkpoint(checkpoint_dir, prefix, "comparacao", comparison_section)

    _progress(progress, "Gerando síntese")
    synthesis_bundle = _generate_intro_synthesis_bundle(client, model, cfg, docs, orientations, article_sections, comparison_section, bib_keys, bib_path)
    _write_checkpoint(checkpoint_dir, prefix, "sintese", synthesis_bundle)

    intro_sections = [s for s in synthesis_bundle.sections if "introdu" in str(s.title).lower()]
    synthesis_sections = [s for s in synthesis_bundle.sections if "sint" in str(s.title).lower()]
    final_sections = [s for s in synthesis_bundle.sections if "consider" in str(s.title).lower() or "conclus" in str(s.title).lower()]
    used_ids = {id(s) for s in intro_sections + synthesis_sections + final_sections}
    other_synthesis_sections = [s for s in synthesis_bundle.sections if id(s) not in used_ids]

    sections: list[Section] = []
    fallback_intro = Section(
        level=1,
        title="Introdução",
        blocks=[Block(type="paragraph", content=[TextSpan(text="Este documento apresenta um resumo analítico dos textos do corpus local, articulando análise individual, comparação transversal e síntese interpretativa.")])],
    )
    fallback_final = Section(
        level=1,
        title="Considerações finais",
        blocks=[Block(type="paragraph", content=[TextSpan(text="O conjunto analisado permite consolidar os principais argumentos, contribuições e limites dos textos, preservando a especificidade de cada obra e suas conexões possíveis.")])],
    )
    sections.extend(intro_sections or [fallback_intro])
    sections.append(Section(level=1, title="Resumo individual dos textos", blocks=[]))
    sections.extend(article_sections)
    sections.append(comparison_section)
    sections.extend(synthesis_sections or other_synthesis_sections)
    sections.extend(final_sections or [fallback_final])

    bib_style = str((cfg.get("bibliografia", {}) if isinstance(cfg.get("bibliografia"), dict) else {}).get("latex_style") or "apa")
    document = AcademicDocument(
        metadata=meta_cfg,
        sections=sections,
        bibliography={"bib_path": bib_path.name, "style": bib_style, "entries_used": sorted(set(bib_keys))},
    )
    prompt_bundle = load_prompt_bundle(cfg, "document", document_type=str(meta_cfg.get("tipo_documento") or "atividade"))
    document.diagnostics.prompts_json = json.dumps({
        "document": prompt_bundle.report(),
        "generation_mode": "staged_article_analysis",
        "stages": ["article_sections", "comparison", "synthesis"],
        "checkpoints": bool(checkpoint_dir),
    }, ensure_ascii=False)
    return document

def build_document_model(
    client: Any,
    model: str,
    cfg: dict[str, Any],
    docs: list[SourceDoc],
    orientations: list[SourceDoc],
    bib_keys: list[str],
    bib_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    checkpoint_dir: Path | None = None,
    prefix: str = "documento",
) -> AcademicDocument:
    _validate_fichamento_corpus_contract(cfg, docs)
    resumo_cfg = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
    if _is_resumo_artigos_profile(cfg) and bool(resumo_cfg.get("geracao_em_etapas", True)):
        return _build_resumo_artigos_document_model_staged(
            client,
            model,
            cfg,
            docs,
            orientations,
            bib_keys,
            bib_path,
            progress=progress,
            checkpoint_dir=checkpoint_dir,
            prefix=prefix,
        )

    prompt = build_prompt(cfg, docs, orientations, bib_keys, bib_path)
    prompt_bundle = load_prompt_bundle(cfg, "document", document_type=str((cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}).get("tipo_documento") or "paper"))
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=AcademicDocument,
    )
    if resp.output_parsed is None:
        raise RuntimeError("A IA não retornou AcademicDocument estruturado.")
    doc = resp.output_parsed
    # Garante valores de capa vindos do TOML, sem deixar a IA sobrescrever indevidamente.
    meta_cfg = _meta_from_cfg(cfg)
    for key, value in meta_cfg.items():
        if value or not getattr(doc.metadata, key, None):
            setattr(doc.metadata, key, value)
    doc.bibliography.bib_path = bib_path.name
    doc.bibliography.entries_used = sorted(set(doc.bibliography.entries_used or []))
    doc.diagnostics.prompts_json = json.dumps({"document": prompt_bundle.report()}, ensure_ascii=False)
    if _is_fichamento_profile(cfg):
        _validate_fichamento_document_contract(doc)
    return doc

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except Exception:
    tomllib = None


def backup(path: Path) -> None:
    if path.exists():
        b = path.with_suffix(path.suffix + ".bak_artigo_longo_v1_12_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(path, b)
        print(f"[OK] Backup: {b}")


def sniff_csv(path: Path):
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:30000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        return csv.excel


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, dialect=sniff_csv(path)))


def norm_key(v: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v or "").lower()).strip("_")


def pick(row: dict, *names: str) -> str:
    d = {norm_key(k): v for k, v in row.items()}
    for name in names:
        v = d.get(norm_key(name))
        if v not in (None, ""):
            return str(v).strip()
    return ""


def extract_pdf_text(path: Path, max_chars: int) -> tuple[str, int]:
    if not path.exists():
        return "[PDF NÃO ENCONTRADO]", 0
    try:
        try:
            from pypdf import PdfReader
        except Exception:
            from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
            if sum(len(p) for p in parts) >= max_chars:
                break
        text = "\n".join(parts).strip()
        return text[:max_chars], len(reader.pages)
    except Exception as e:
        return f"[ERRO AO EXTRAIR PDF: {e}]", 0


def read_toml_dict(path: Path) -> dict:
    if tomllib is None or not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def find_section(lines: list[str], section: str) -> tuple[int | None, int | None]:
    header = f"[{section}]"
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == header:
            start = i
            break
    if start is None:
        return None, None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if s.startswith("[") and s.endswith("]"):
            end = j
            break
    return start, end


def is_key_line(line: str, key: str) -> bool:
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if not s.startswith(key):
        return False
    rest = s[len(key):].lstrip()
    return rest.startswith("=")


def skip_value_block(lines: list[str], i: int, section_end: int) -> int:
    line = lines[i]
    after = line.split("=", 1)[1] if "=" in line else ""
    if after.count('"""') % 2 == 1:
        j = i + 1
        while j < section_end:
            if '"""' in lines[j]:
                return j + 1
            j += 1
        return section_end
    balance = after.count("[") - after.count("]")
    if balance > 0:
        j = i + 1
        while j < section_end:
            balance += lines[j].count("[") - lines[j].count("]")
            if balance <= 0:
                return j + 1
            j += 1
        return section_end
    return i + 1


def remove_key(lines: list[str], section: str, key: str) -> list[str]:
    start, end = find_section(lines, section)
    if start is None:
        return lines
    out = lines[:start + 1]
    i = start + 1
    while i < end:
        if is_key_line(lines[i], key):
            i = skip_value_block(lines, i, end)
            continue
        out.append(lines[i])
        i += 1
    out.extend(lines[end:])
    return out


def ensure_section(lines: list[str], section: str) -> list[str]:
    start, _ = find_section(lines, section)
    if start is not None:
        return lines
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(f"\n[{section}]\n")
    return lines


def insert_after_header(lines: list[str], section: str, value_line: str) -> list[str]:
    lines = ensure_section(lines, section)
    start, _ = find_section(lines, section)
    assert start is not None
    return lines[:start + 1] + [value_line if value_line.endswith("\n") else value_line + "\n"] + lines[start + 1:]


def set_scalar(lines: list[str], section: str, key: str, value: str) -> list[str]:
    lines = remove_key(lines, section, key)
    return insert_after_header(lines, section, f"{key} = {value}")


def set_array(lines: list[str], section: str, key: str, values: list[str]) -> list[str]:
    lines = remove_key(lines, section, key)
    arr = json.dumps(values, ensure_ascii=False, indent=2)
    return insert_after_header(lines, section, f"{key} = {arr}")


def set_multiline_string(lines: list[str], section: str, key: str, value: str) -> list[str]:
    lines = remove_key(lines, section, key)
    return insert_after_header(lines, section, f'{key} = """{value.strip()}\n"""')


def patch_toml(cfg: Path, art_dir: Path, prompt_path: Path, corpus_dir: Path, target_n: int, min_words: int) -> None:
    backup(cfg)
    cfg_data = read_toml_dict(cfg)
    text = cfg.read_text(encoding="utf-8")

    text = re.sub(r"\b14 estudos\b", f"{target_n} estudos com texto completo recuperado", text, flags=re.I)
    text = re.sub(r"\bselecionados 14\b", f"selecionados {target_n}", text, flags=re.I)

    lines = text.splitlines(keepends=True)

    current_docs = []
    if isinstance(cfg_data.get("documentos_locais"), dict):
        current_docs = list(cfg_data.get("documentos_locais", {}).get("paths") or [])
    for p in [str((art_dir / "dados_prisma").resolve()), str(corpus_dir.resolve())]:
        if p not in current_docs:
            current_docs.append(p)

    current_orient = []
    if isinstance(cfg_data.get("orientacoes"), dict):
        current_orient = list(cfg_data.get("orientacoes", {}).get("paths") or [])
    for p in [str(prompt_path.resolve()), str((art_dir / "dados_prisma" / "orientacoes_fulltext_garantido_artigo.md").resolve())]:
        if p not in current_orient:
            current_orient.append(p)

    linhas_orientacao = f"""
ARTIGO CIENTÍFICO LONGO COM 20 ESTUDOS — INSTRUÇÃO OBRIGATÓRIA

O artigo deve usar o corpus final de {target_n} estudos com texto completo recuperado, não 14 estudos.

Arquivos de leitura obrigatória:
- {prompt_path}
- {corpus_dir / "referencias_incluidas_20.md"}
- {corpus_dir / "estatisticas_prisma_fulltext.md"}
- {corpus_dir / "corpus_fulltext_compilado.md"}

Requisitos:
- mínimo de {min_words} palavras no corpo textual;
- padrão de artigo científico completo, aproximadamente 18 a 25 páginas;
- seção REFERÊNCIAS obrigatória;
- citação de todos os {target_n} estudos incluídos;
- matriz de evidências por estudo;
- matriz de aderência dos textos ao tema, objetivo, recorte e pergunta;
- discussão substantiva, conectando os textos completos ao redesenho do ATESTMED;
- destaque ao estudo diretamente relacionado ao ATESTMED.
"""

    lines = set_array(lines, "documentos_locais", "paths", current_docs)
    lines = set_scalar(lines, "documentos_locais", "recursive", "true")
    lines = set_scalar(lines, "documentos_locais", "tipos", '["pdf", "csv", "json", "txt", "md", "org", "png", "bib"]')
    lines = set_scalar(lines, "documentos_locais", "max_caracteres_por_doc", "650000")

    lines = set_array(lines, "orientacoes", "paths", current_orient)
    lines = set_multiline_string(lines, "orientacoes", "inline", linhas_orientacao)

    lines = set_scalar(lines, "qualidade", "min_palavras", str(min_words))
    lines = set_scalar(lines, "qualidade", "min_referencias", str(target_n))
    lines = set_scalar(lines, "qualidade", "min_paginas_estimadas", "18")
    lines = set_scalar(lines, "qualidade", "falhar_se_curto", "true")

    cfg.write_text("".join(lines), encoding="utf-8")

    if tomllib:
        with cfg.open("rb") as f:
            tomllib.load(f)
    print(f"[OK] TOML atualizado e validado: {cfg}")


def build_prompt_and_corpus(art_dir: Path, target_n: int, min_words: int, chars_por_pdf: int) -> dict:
    dados = art_dir / "dados_prisma"
    full_dir = dados / "fulltext_garantido"
    csv_path = full_dir / "relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas_fulltext_garantido.csv"
    counts_path = full_dir / "relatorio_prisma_prisma_fluxo_pmf.contagens_prisma_fulltext_garantido.json"
    matrix_path = full_dir / "relatorio_prisma_prisma_fluxo_pmf.matriz_prisma_fulltext_garantido.csv"
    exclusions_path = full_dir / "relatorio_prisma_prisma_fluxo_pmf.exclusoes_fulltext_nao_recuperado.csv"

    if not csv_path.exists():
        raise SystemExit(f"ERRO: CSV de referências fulltext não encontrado: {csv_path}")

    rows = read_csv(csv_path)[:target_n]
    if len(rows) < target_n:
        raise SystemExit(f"ERRO: só há {len(rows)} referências fulltext; alvo solicitado: {target_n}")

    corpus_dir = dados / "artigo_longo_fulltext"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    if counts_path.exists():
        try:
            counts = json.loads(counts_path.read_text(encoding="utf-8"))
        except Exception:
            counts = {}

    refs_md = corpus_dir / "referencias_incluidas_20.md"
    corpus_md = corpus_dir / "corpus_fulltext_compilado.md"
    prompt_md = corpus_dir / "prompt_artigo_cientifico_longo.md"
    stats_md = corpus_dir / "estatisticas_prisma_fulltext.md"

    with refs_md.open("w", encoding="utf-8") as f:
        f.write(f"# Referências incluídas no corpus final ({len(rows)} estudos)\n\n")
        for i, r in enumerate(rows, start=1):
            f.write(f"## {i}. {pick(r, 'title', 'titulo')}\n\n")
            f.write(f"- Autores: {pick(r, 'authors', 'autores')}\n")
            f.write(f"- Ano: {pick(r, 'year', 'ano')}\n")
            f.write(f"- Periódico/Fonte: {pick(r, 'journal', 'periodico')}\n")
            f.write(f"- DOI: {pick(r, 'doi')}\n")
            f.write(f"- PDF: {pick(r, 'pdf_path')}\n")
            f.write(f"- Score de aderência: {pick(r, 'score_aderencia')}\n")
            f.write(f"- Prioritário: {pick(r, 'prioridade_manual')}\n\n")

    with stats_md.open("w", encoding="utf-8") as f:
        f.write("# Estatísticas PRISMA/full text para uso obrigatório no artigo\n\n")
        for k, v in counts.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n- estudos finais esperados no artigo: {len(rows)}\n")
        f.write(f"- mínimo de palavras exigido: {min_words}\n")
        f.write(f"- matriz full text: {matrix_path}\n")
        f.write(f"- exclusões por texto completo não recuperado: {exclusions_path}\n")

    with corpus_md.open("w", encoding="utf-8") as f:
        f.write("# Corpus full text compilado para artigo científico longo\n\n")
        f.write("Este arquivo consolida trechos extraídos dos PDFs originais recuperados. A redação do artigo deve usar este corpus e os PDFs locais como base de evidência.\n\n")
        f.write(f"Total de estudos no corpus final: {len(rows)}.\n\n")
        for i, r in enumerate(rows, start=1):
            title = pick(r, "title", "titulo")
            pdf_raw = pick(r, "pdf_path")
            pdf_path = Path(pdf_raw) if pdf_raw else Path("")
            text, pages = extract_pdf_text(pdf_path, chars_por_pdf)
            f.write("\n\n" + "=" * 100 + "\n\n")
            f.write(f"## ESTUDO {i}: {title}\n\n")
            f.write(f"- Autores: {pick(r, 'authors', 'autores')}\n")
            f.write(f"- Ano: {pick(r, 'year', 'ano')}\n")
            f.write(f"- DOI: {pick(r, 'doi')}\n")
            f.write(f"- PDF: {pdf_path}\n")
            f.write(f"- Páginas detectadas: {pages}\n")
            f.write(f"- Score de aderência: {pick(r, 'score_aderencia')}\n")
            f.write(f"- Prioritário: {pick(r, 'prioridade_manual')}\n\n")
            f.write("### Texto extraído do PDF\n\n")
            f.write(text if text else "[SEM TEXTO EXTRAÍDO]\n")

    prompt = f"""# Prompt obrigatório — artigo científico longo baseado em full text

Você está redigindo um artigo científico ABNT/FGV sobre ATESTMED, saúde digital e decisão baseada em evidências no contexto da Perícia Médica Federal.

Use o corpus final de {len(rows)} estudos com texto completo recuperado. Não afirmar que são 14 estudos. O número correto é {len(rows)} estudos.

Arquivos obrigatórios:
- {refs_md}
- {stats_md}
- {corpus_md}
- {matrix_path}
- {exclusions_path}

O artigo deve ter pelo menos {min_words} palavras no corpo textual, equivalendo aproximadamente a 18–25 páginas, com tabelas e referências.

Estrutura obrigatória:
1. INTRODUÇÃO.
2. MÉTODO.
3. RESULTADOS.
4. DISCUSSÃO.
5. PROPOSTA DE REDESENHO DO FLUXO DECISÓRIO DO ATESTMED.
6. INDICADORES E MONITORAMENTO.
7. LIMITAÇÕES.
8. CONCLUSÃO.
9. REFERÊNCIAS.

Incluir tabela com: Estudo; Tipo de evidência; Achado relevante; Aderência ao ATESTMED; Implicação para análise documental; Implicação para teleperícia; Implicação para perícia presencial; Risco/limitação; Uso no redesenho do fluxo.

Citar todos os {len(rows)} estudos incluídos ao longo do texto. A referência sobre “A sistemática do ATESTMED...” deve receber destaque.

É proibido gerar texto curto, mencionar 14 estudos ou deixar a bibliografia ausente.
"""
    prompt_md.write_text(prompt, encoding="utf-8")

    return {
        "rows": len(rows),
        "corpus_dir": corpus_dir,
        "prompt_md": prompt_md,
        "refs_md": refs_md,
        "corpus_md": corpus_md,
        "stats_md": stats_md,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--cfg-art", required=True)
    ap.add_argument("--target-n", type=int, default=20)
    ap.add_argument("--min-palavras", type=int, default=8500)
    ap.add_argument("--chars-por-pdf", type=int, default=18000)
    args = ap.parse_args()

    art_dir = Path(args.art_dir).resolve()
    cfg = Path(args.cfg_art).resolve()

    info = build_prompt_and_corpus(art_dir, args.target_n, args.min_palavras, args.chars_por_pdf)
    patch_toml(cfg, art_dir, info["prompt_md"], info["corpus_dir"], info["rows"], args.min_palavras)

    print("[OK] Corpus e prompt de artigo longo preparados:")
    for k in ["prompt_md", "refs_md", "corpus_md", "stats_md"]:
        print(f"- {info[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

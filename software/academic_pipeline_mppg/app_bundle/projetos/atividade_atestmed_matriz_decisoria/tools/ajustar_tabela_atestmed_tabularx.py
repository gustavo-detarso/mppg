#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ajusta a tabela da matriz decisória do ATESTMED no ORG para caber no PDF.

Uso:
  python ajustar_tabela_atestmed_tabularx.py --org caminho/arquivo.org

O script:
- cria backup .bak_tabela_atestmed_<timestamp>;
- adiciona pacotes LaTeX necessários ao cabeçalho ORG;
- localiza a tabela cujo caption contém "Síntese de dimensões";
- substitui a tabela ORG por um bloco LaTeX tabularx com quebra automática de linhas.
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

DEFAULT_CAPTION = "Síntese de dimensões para uma matriz decisória de triagem no ATESTMED"
DEFAULT_ROWS = [
    ["Dimensão analítica", "Critério observado", "Implicação para o fluxo"],
    ["Qualidade documental", "Completude, legibilidade e consistência mínima", "Permite triagem inicial entre caso apto, caso pendente e caso crítico"],
    ["Qualidade clínica da evidência", "Coerência entre informações e suficiência para juízo administrativo", "Distingue casos resolvíveis documentalmente de casos que exigem exame adicional"],
    ["Risco decisório", "Potencial de erro com impacto relevante", "Eleva necessidade de revisão técnica, auditoria ou perícia presencial"],
    ["Complexidade do caso", "Maior ambiguidade ou necessidade interpretativa", "Reduz adequação de tratamento padronizado e favorece encaminhamento especializado"],
    ["Prioridade administrativa", "Necessidade de resposta tempestiva combinada com segurança", "Orienta ordenação da fila e uso racional da capacidade pericial"],
]

REQUIRED_HEADERS = [
    r"#+LATEX_HEADER: \\usepackage{tabularx}",
    r"#+LATEX_HEADER: \\usepackage{array}",
    r"#+LATEX_HEADER: \\usepackage{ragged2e}",
]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def add_headers(text: str) -> str:
    existing = set(line.strip() for line in text.splitlines())
    missing = [h for h in REQUIRED_HEADERS if h not in existing]
    if not missing:
        return text
    lines = text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#+"):
            insert_at = i + 1
    lines[insert_at:insert_at] = missing
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def split_org_row(line: str) -> list[str]:
    raw = line.strip()
    if not raw.startswith("|"):
        return []
    if re.fullmatch(r"[|+\-:\s]+", raw):
        return []
    parts = [p.strip() for p in raw.strip("|").split("|")]
    return parts


def parse_table_rows(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            break
        row = split_org_row(lines[i])
        if row:
            rows.append(row)
        i += 1
    rows = [row[:3] for row in rows if len(row) >= 3]
    return rows, i


def build_latex_table(rows: list[list[str]], caption: str) -> str:
    if not rows or len(rows[0]) < 3:
        rows = DEFAULT_ROWS
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else DEFAULT_ROWS[1:]

    out = [
        "#+BEGIN_EXPORT latex",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.31\textwidth}>{\raggedright\arraybackslash}X}",
        r"\hline",
        " & ".join(rf"\textbf{{{latex_escape(cell)}}}" for cell in header[:3]) + r" \\",
        r"\hline",
    ]
    for row in body:
        row = (row + [""] * 3)[:3]
        out.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    out.extend([
        r"\hline",
        r"\end{tabularx}",
        r"\normalsize",
        r"\end{table}",
        "#+END_EXPORT",
    ])
    return "\n".join(out)


def find_and_replace(text: str, caption_token: str) -> tuple[str, bool]:
    lines = text.splitlines()
    # Caso padrão: caption Org + tabela pipe logo depois.
    for idx, line in enumerate(lines):
        if "#+CAPTION:" in line and caption_token.casefold() in line.casefold():
            caption = line.split(":", 1)[1].strip()
            # Remove eventual prefixo "Tabela 1 –" do caption para LaTeX numerar sozinho.
            caption = re.sub(r"^Tabela\s+\d+\s*[–-]\s*", "", caption).strip() or DEFAULT_CAPTION
            table_start = idx + 1
            while table_start < len(lines) and (
                not lines[table_start].lstrip().startswith("|")
            ):
                # mantém/descarta linhas ATTR/NAME junto da substituição
                if lines[table_start].strip() and not lines[table_start].startswith("#+"):
                    break
                table_start += 1
            if table_start >= len(lines) or not lines[table_start].lstrip().startswith("|"):
                continue
            rows, table_end = parse_table_rows(lines, table_start)
            latex = build_latex_table(rows, caption)
            new_lines = lines[:idx] + [latex] + lines[table_end:]
            return "\n".join(new_lines) + "\n", True

    # Fallback: encontra a primeira tabela pipe que contenha cabeçalhos esperados.
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("|") and "Dimensão analítica" in line and "Critério observado" in line:
            rows, table_end = parse_table_rows(lines, idx)
            latex = build_latex_table(rows, DEFAULT_CAPTION)
            # Remove caption textual imediatamente anterior, se houver.
            start = idx
            if idx > 0 and "Síntese de dimensões" in lines[idx - 1]:
                start = idx - 1
            new_lines = lines[:start] + [latex] + lines[table_end:]
            return "\n".join(new_lines) + "\n", True

    return text, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True, help="Caminho do ORG gerado pelo pipeline")
    parser.add_argument("--caption-token", default="Síntese de dimensões", help="Trecho do caption para localizar a tabela")
    args = parser.parse_args()

    path = Path(args.org).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"ORG não encontrado: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    backup = path.with_suffix(path.suffix + f".bak_tabela_atestmed_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(path, backup)

    text = add_headers(text)
    text, changed = find_and_replace(text, args.caption_token)
    if not changed:
        shutil.copy2(backup, path)
        raise SystemExit(
            "Não encontrei a tabela da matriz decisória no ORG. "
            f"Backup preservado em: {backup}"
        )
    path.write_text(text, encoding="utf-8")
    print(f"Tabela ajustada com tabularx: {path}")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gerador simples de TOML rc10 document_model.

Perfis:
- paper: documento acadêmico final sem relatório PRISMA ativado por padrão.
- atividade: atividade acadêmica sem relatório PRISMA ativado por padrão.
- prisma: relatório de pesquisa/PRISMA como saída própria + documento de atividade.
- atividade_prisma: alias de prisma.
- paper_prisma: paper + bloco [relatorio_pesquisa] ativado.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "config/examples"


def source_for_profile(profile: str) -> Path:
    if profile == "paper":
        return EXAMPLES / "paper_rc10_exemplo.toml"
    if profile == "atividade":
        return EXAMPLES / "atividade_rc10_exemplo.toml"
    if profile in {"prisma", "atividade_prisma"}:
        return EXAMPLES / "relatorio_prisma_rc10_exemplo.toml"
    if profile == "paper_prisma":
        base = EXAMPLES / "paper_rc10_exemplo.toml"
        if base.exists():
            return base
    raise ValueError(f"Perfil desconhecido: {profile}")


def append_prisma_block_if_needed(content: str, profile: str) -> str:
    if profile != "paper_prisma" or "[relatorio_pesquisa]" in content:
        return content
    return content.rstrip() + '''

[relatorio_pesquisa]
ativo = true
tipo = "prisma"
output_dir = "./output/pesquisa"
prefixo = "relatorio_prisma_paper"
criar_subdiretorio = true
exportar_json = true
exportar_org = true
exportar_pdf = true
exportar_docx = true
exportar_xlsx = true
exportar_fluxograma = true
validar = true
falhar_se_invalido = false
prisma_json_path = ""
pesquisa_dir_existente = ""
criterios_inclusao = [
  "Aderência substantiva ao tema, recorte e objetivo.",
  "Disponibilidade de metadados mínimos ou DOI para identificação bibliográfica.",
  "Texto acadêmico ou técnico-científico relacionado ao problema de pesquisa."
]
criterios_exclusao = [
  "Fora do tema ou do recorte.",
  "Duplicado.",
  "Ausência de relação substantiva com a pergunta de pesquisa."
]
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera TOML rc10 por perfil.")
    parser.add_argument("--perfil", choices=["paper", "atividade", "prisma", "atividade_prisma", "paper_prisma"], default="paper")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    src = source_for_profile(args.perfil)
    content = src.read_text(encoding="utf-8")
    content = append_prisma_block_if_needed(content, args.perfil)
    if args.output:
        out = Path(args.output).expanduser().resolve()
    else:
        default_name = {
            "paper": "paper_config_rc10.toml",
            "atividade": "atividade_config_rc10.toml",
            "prisma": "relatorio_prisma_config_rc10.toml",
            "atividade_prisma": "atividade_prisma_config_rc10.toml",
            "paper_prisma": "paper_prisma_config_rc10.toml",
        }[args.perfil]
        out = Path.cwd() / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"TOML gerado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

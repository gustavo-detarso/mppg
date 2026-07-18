#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executa diagnóstico isolado de uma fonte bibliográfica PRISMA.

Esta versão é compatível com a estratégia por blocos e com a pré-triagem por
IA. O diagnóstico nunca dispara Unpaywall nem pré-triagem. Quando ``--query``
é informado, a estratégia do TOML é substituída temporariamente por uma única
consulta, para que o teste de uma fonte seja realmente isolado.

Uso, na raiz do Academic Pipeline:
    pipenv run python diagnosticar_fontes_prisma.py \
      app_bundle/projetos/prisma_fluxo_pmf/prisma_fluxo_pmf.toml semantic_scholar \
      --query "telemedicine" --limite 5
"""
from __future__ import annotations

import argparse
import copy
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    """Localiza a raiz quando o script foi copiado para ela."""
    candidates = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for candidate in candidates:
        if (candidate / "app_bundle" / "scripts" / "pipeline" / "prisma_busca_externa.py").is_file():
            return candidate
    raise RuntimeError(
        "Não encontrei app_bundle/scripts/pipeline/prisma_busca_externa.py. "
        "Copie este script para a raiz do projeto e execute-o a partir dela."
    )


def _load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuração não encontrada: {path}")
    with path.open("rb") as handle:
        cfg = tomllib.load(handle)
    cfg["__config_path__"] = str(path)
    return cfg


def _format_status(status: dict[str, Any] | None) -> str:
    if not status:
        return "status indisponível"
    detected = ", ".join(str(item) for item in status.get("credentials_detected", []) if str(item))
    missing = ", ".join(str(item) for item in status.get("credentials_required_missing", []) if str(item))
    suffixes: list[str] = []
    if detected:
        suffixes.append(f"variáveis detectadas: {detected}")
    if missing:
        suffixes.append(f"variáveis obrigatórias ausentes: {missing}")
    suffix = f" ({'; '.join(suffixes)})" if suffixes else ""
    return str(status.get("status") or "status não informado") + suffix


def _query_label(item: dict[str, Any]) -> str:
    query = str(item.get("consulta") or item.get("query") or "").strip()
    block = str(item.get("bloco_rotulo") or item.get("rotulo") or "").strip()
    if block and query:
        return f"{block}: {query}"
    return query or block or "consulta não identificada"


def _print_provider_result(
    item: dict[str, Any],
    *,
    labels: dict[str, str],
) -> int:
    provider = str(item.get("provider") or "?")
    label = labels.get(provider, provider)
    retrieved = int(item.get("retrieved") or 0)
    error = str(item.get("error") or "").strip()
    print(f"{label}: {retrieved} registro(s)")
    if error:
        print(f"  erro consolidado: {error}")
    else:
        print("  erro consolidado: nenhum")

    failures = 1 if error else 0
    queries = item.get("consultas")
    if isinstance(queries, list):
        for row in queries:
            if not isinstance(row, dict):
                continue
            query_error = str(row.get("error") or "").strip()
            query_count = int(row.get("retrieved") or 0)
            print(f"  - {_query_label(row)} → {query_count} registro(s)")
            if query_error:
                print(f"    erro: {query_error}")
    else:
        url = str(item.get("url") or "").strip()
        if url:
            print(f"  URL registrada: {url}")
    return failures


def main() -> int:
    root = _project_root()
    pipeline_dir = root / "app_bundle" / "scripts" / "pipeline"
    sys.path.insert(0, str(pipeline_dir))

    from prisma_busca_externa import (  # type: ignore[import-not-found]
        PROVIDER_LABELS,
        PROVIDER_ORDER,
        provider_statuses,
        run_external_prisma_search,
    )

    parser = argparse.ArgumentParser(
        description="Testa uma fonte PRISMA isoladamente, sem alterar o TOML original."
    )
    parser.add_argument("config", type=Path, help="TOML do projeto PRISMA")
    parser.add_argument(
        "fonte",
        choices=[*PROVIDER_ORDER, "todas"],
        help="Fonte a testar; use 'todas' apenas para uma varredura rápida.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Consulta temporária. Quando informada, substitui os blocos do TOML só neste diagnóstico.",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=5,
        help="Máximo de registros por consulta no teste (padrão: 5; intervalo: 1–25).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Pasta de saída do diagnóstico. O padrão fica no diretório do TOML.",
    )
    args = parser.parse_args()

    limit = max(1, min(int(args.limite), 25))
    cfg_path = args.config.expanduser().resolve()
    cfg = copy.deepcopy(_load_config(cfg_path))

    sources = list(PROVIDER_ORDER) if args.fonte == "todas" else [args.fonte]
    search = cfg.get("busca_prisma")
    if not isinstance(search, dict):
        search = {}
        cfg["busca_prisma"] = search

    # Diagnóstico deve isolar conectividade e recuperação. Nunca pode acionar
    # pré-triagem por IA ou Unpaywall, pois ambos adicionam dependências e custo.
    search["ativo"] = True
    search["modo"] = "busca_externa"
    search["bases"] = sources
    search["limite_por_base"] = limit
    search["limite_scopus_por_consulta"] = min(limit, 10)
    search["limite_triagem_inicial"] = limit
    search["enriquecer_unpaywall"] = False
    search["limite_unpaywall"] = 0
    search["pre_triagem_ia"] = False

    isolated_query = str(args.query or "").strip()
    if args.query is not None:
        if not isolated_query:
            parser.error("--query não pode ser vazia.")
        # O pipeline v15+ lê [[busca_prisma.estrategias]] antes de consulta_geral.
        # Limpamos essas tabelas apenas na cópia em memória, para garantir que
        # um teste de 'telemedicine' não dispare todos os blocos temáticos.
        search["estrategia"] = "consulta_unica"
        search["estrategias"] = []
        search["consulta_geral"] = isolated_query
        strategy_display = "consulta única temporária"
    else:
        strategy_display = str(search.get("estrategia") or "configuração original").strip()

    query_display = isolated_query or str(search.get("consulta_geral") or "").strip()
    if not query_display:
        research = cfg.get("pesquisa") if isinstance(cfg.get("pesquisa"), dict) else {}
        query_display = " ".join(str(item) for item in research.get("palavras_chave", []) if str(item)).strip()
        if not query_display:
            query_display = str(research.get("tema") or "").strip()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_slug = "_".join(sources)
    out_dir = (
        args.saida.expanduser().resolve()
        if args.saida is not None
        else cfg_path.parent / "diagnosticos_fontes_prisma" / f"{stamp}_{source_slug}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"diagnostico_{source_slug}"

    print("Diagnóstico PRISMA isolado")
    print("=" * 72)
    print(f"Configuração-base: {cfg_path}")
    print(f"Fontes em teste: {', '.join(PROVIDER_LABELS.get(item, item) for item in sources)}")
    print(f"Limite por consulta: {limit}")
    print("Unpaywall: desativado neste diagnóstico")
    print("Pré-triagem por IA: desativada neste diagnóstico")
    print(f"Estratégia efetiva: {strategy_display}")
    print(f"Consulta de teste: {query_display or '[vazia]'}")
    print("\nStatus de configuração (nomes de variáveis, nunca valores):")
    statuses = provider_statuses()
    for source in sources:
        print(f"- {PROVIDER_LABELS.get(source, source)}: {_format_status(statuses.get(source))}")

    def progress(message: str) -> None:
        print(f"[ETAPA] {message}")

    try:
        result = run_external_prisma_search(cfg, out_dir, prefix, progress=progress)
    except Exception as exc:
        print(f"\nFALHA GLOBAL: {type(exc).__name__}: {exc}")
        print(f"Saída parcial: {out_dir}")
        return 2

    logs = result.get("fontes") if isinstance(result, dict) else []
    logs = logs if isinstance(logs, list) else []
    print("\nResultado por fonte")
    print("-" * 72)
    failures = 0
    for item in logs:
        if isinstance(item, dict):
            failures += _print_provider_result(item, labels=PROVIDER_LABELS)

    artifacts = result.get("artefatos") if isinstance(result, dict) else {}
    print(f"\nDiretório do diagnóstico: {out_dir}")
    if isinstance(artifacts, dict):
        for name in ("candidatos_brutos", "candidatos_deduplicados", "log", "prisma_report_json"):
            path = artifacts.get(name)
            if path:
                print(f"- {name}: {path}")

    if failures:
        print("\nResultado: uma ou mais consultas retornaram erro; use as mensagens acima para corrigir credenciais, quota ou adaptador.")
        return 2
    print("\nResultado: diagnóstico concluído sem erros da fonte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

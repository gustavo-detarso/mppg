#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repara a regressão de importação introduzida pela v15 da busca PRISMA.

A v15 removeu inadvertidamente o exportador ORG do módulo
``prisma_busca_externa.py``, embora ``academic_pipeline_rc10.py`` ainda o
importe e o utilize. Este reparo restaura um renderizador compatível tanto com
relatórios PRISMA preliminares quanto com relatórios consolidados, incluindo a
nova estratégia por blocos temáticos.

Uso (na raiz do projeto):
    pipenv run python reparar_renderer_prisma_v15_2.py
"""
from __future__ import annotations

import py_compile
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TARGET_RELATIVE = Path("app_bundle/scripts/pipeline/prisma_busca_externa.py")
MARKER = "\ndef _decision("

RENDERER_BLOCK = r'''
# ---------------------------------------------------------------------------
# Relatório PRISMA em ORG (prévio e consolidado)
# Restaurado pela correção v15.2.
# ---------------------------------------------------------------------------


def _prisma_org_text(value: Any) -> str:
    """Normaliza texto de metadados para uso seguro em linhas ORG."""
    raw = "" if value is None else str(value)
    text = " ".join(raw.replace("\u00a0", " ").split())
    return text.replace("|", "/")


def _prisma_org_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Monta tabela ORG compacta, sem depender do renderizador do paper."""
    lines = ["| " + " | ".join(_prisma_org_text(item) for item in headers) + " |"]
    lines.append("|-" + "-+-".join("-" * max(3, len(_prisma_org_text(item))) for item in headers) + "-|")
    for row in rows:
        lines.append("| " + " | ".join(_prisma_org_text(item) for item in row) + " |")
    return lines


def _prisma_doi(value: Any) -> str:
    raw = _prisma_org_text(value).strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if raw.casefold().startswith(prefix):
            raw = raw[len(prefix):]
    return raw.strip()


def _prisma_reference(item: dict[str, Any]) -> str:
    authors = _prisma_org_text(item.get("autores") or "Autor não informado")
    year = _prisma_org_text(item.get("ano") or "s.d.")
    title = _prisma_org_text(item.get("titulo") or "Título não informado")
    venue = _prisma_org_text(item.get("periodico") or "")
    doi = _prisma_doi(item.get("doi"))
    parts = [f"{authors} ({year}). {title}."]
    if venue:
        parts.append(venue + ".")
    if doi:
        parts.append(f"DOI: https://doi.org/{doi}.")
    return " ".join(parts)


def _prisma_matrix_cell(item: dict[str, Any], key: str) -> str:
    value = _prisma_org_text(item.get(key) or "")
    return value if value else "—"


def _prisma_matrix_study_label(item: dict[str, Any]) -> str:
    return (
        f"{_prisma_matrix_cell(item, 'autores')} "
        f"({_prisma_matrix_cell(item, 'ano')}) — "
        f"{_prisma_matrix_cell(item, 'titulo')}"
    )


def _prisma_append_matrix_annex_org(lines: list[str], included: list[dict[str, Any]]) -> None:
    """Acrescenta matriz documental em duas longtables no anexo do relatório."""
    lines += [
        "",
        "* Anexo A — Matriz dos estudos incluídos",
        "A matriz abaixo é a versão documental da planilha de triagem. Os campos analíticos devem ser preenchidos após a leitura do texto completo dos estudos incluídos. A versão editável permanece disponível em CSV/XLSX.",
        "",
        "** A.1 Identificação, contexto e método",
        "#+BEGIN_EXPORT latex",
        "\\begin{landscape}",
        "\\footnotesize",
        "#+END_EXPORT",
        "#+ATTR_LATEX: :environment longtable :align p{0.22\\linewidth}p{0.11\\linewidth}p{0.17\\linewidth}p{0.17\\linewidth}p{0.16\\linewidth}",
    ]
    part_one = [
        [
            _prisma_matrix_study_label(item),
            _prisma_matrix_cell(item, "pais_contexto"),
            _prisma_matrix_cell(item, "objetivo_estudo"),
            _prisma_matrix_cell(item, "desenho_metodo"),
            _prisma_matrix_cell(item, "amostra_base"),
        ]
        for item in included
        if isinstance(item, dict)
    ]
    lines.extend(
        _prisma_org_table(
            ["Estudo", "Contexto/país", "Objetivo", "Desenho/método", "Amostra/base"],
            part_one,
        )
    )
    lines += [
        "#+BEGIN_EXPORT latex",
        "\\normalsize",
        "\\end{landscape}",
        "#+END_EXPORT",
        "",
        "** A.2 Achados, limitações e contribuição",
        "#+BEGIN_EXPORT latex",
        "\\begin{landscape}",
        "\\footnotesize",
        "#+END_EXPORT",
        "#+ATTR_LATEX: :environment longtable :align p{0.20\\linewidth}p{0.26\\linewidth}p{0.19\\linewidth}p{0.19\\linewidth}",
    ]
    part_two = [
        [
            _prisma_matrix_study_label(item),
            _prisma_matrix_cell(item, "achados_principais"),
            _prisma_matrix_cell(item, "limitacoes"),
            _prisma_matrix_cell(item, "contribuicao_pergunta"),
        ]
        for item in included
        if isinstance(item, dict)
    ]
    lines.extend(
        _prisma_org_table(
            ["Estudo", "Achados principais", "Limitações", "Contribuição para a pergunta"],
            part_two,
        )
    )
    lines += [
        "#+BEGIN_EXPORT latex",
        "\\normalsize",
        "\\end{landscape}",
        "#+END_EXPORT",
    ]


def _prisma_source_summary(source: dict[str, Any]) -> tuple[Any, str]:
    """Resume logs de fonte dos formatos antigo e novo."""
    queries = source.get("consultas") or source.get("queries") or []
    if not isinstance(queries, list):
        queries = []
    retrieved = source.get("retrieved")
    if retrieved is None:
        retrieved = sum(
            int(item.get("retrieved") or 0)
            for item in queries
            if isinstance(item, dict)
        )
    errors = []
    top_error = _prisma_org_text(source.get("error") or "")
    if top_error:
        errors.append(top_error)
    for item in queries:
        if isinstance(item, dict) and _prisma_org_text(item.get("error") or ""):
            errors.append(_prisma_org_text(item.get("error")))
    if errors and len(errors) >= max(1, len(queries)):
        return retrieved, "erro"
    if errors:
        return retrieved, "parcial"
    return retrieved, "ok"


def _prisma_strategy_org_lines(payload: dict[str, Any]) -> list[str]:
    strategy = payload.get("estrategia_busca")
    if not isinstance(strategy, dict):
        query = _prisma_org_text(payload.get("consulta_geral") or "")
        return [f"- Consulta geral: ={query}="]
    mode = _prisma_org_text(strategy.get("modo") or "consulta_unica")
    rows = strategy.get("blocos") or []
    lines = [f"- Estratégia: ={mode}=."]
    if not isinstance(rows, list) or not rows:
        query = _prisma_org_text(payload.get("consulta_geral") or "")
        lines.append(f"- Consulta geral: ={query}=")
        return lines
    lines.append("** Blocos e consultas executadas")
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = _prisma_org_text(row.get("bloco_rotulo") or row.get("rotulo") or row.get("id") or "Bloco")
        query = _prisma_org_text(row.get("consulta") or "")
        if query:
            lines.append(f"- {label}: ={query}=")
    return lines


def render_external_prisma_org_report(
    cfg: dict[str, Any],
    out_dir: Path,
    prefix: str,
    payload: dict[str, Any],
    *,
    phase: str,
) -> Path:
    """Gera ORG FGV para a busca inicial ou a consolidação após triagem."""
    phase_norm = str(phase or "").strip().lower()
    if phase_norm not in {"preliminar", "final"}:
        raise ValueError("phase deve ser 'preliminar' ou 'final'.")

    def section(name: str) -> dict[str, Any]:
        value = cfg.get(name, {})
        return value if isinstance(value, dict) else {}

    project = section("projeto")
    research = section("pesquisa")
    report = section("relatorio_pesquisa")
    document = section("documento")
    activity = section("atividade")
    title = _prisma_org_text(report.get("titulo") or document.get("titulo_trabalho") or "Relatório de Pesquisa PRISMA")
    author = _prisma_org_text(document.get("autor") or activity.get("aluno") or "")
    date_value = _prisma_org_text(document.get("data") or activity.get("data") or "")
    latex_class = _prisma_org_text(document.get("classe_latex") or "fgv-paper")
    layout = _prisma_org_text(document.get("layout") or "")
    suffix = "relatorio_prisma_preliminar" if phase_norm == "preliminar" else "relatorio_prisma_final"
    target = out_dir / f"{prefix}.{suffix}.org"

    counts = payload.get("contagens") if isinstance(payload.get("contagens"), dict) else {}
    criteria_inclusion = payload.get("criterios_inclusao") or report.get("criterios_inclusao") or []
    criteria_exclusion = payload.get("criterios_exclusao") or report.get("criterios_exclusao") or []
    providers = payload.get("bases") or []
    source_logs = payload.get("fontes") or []
    artifacts = payload.get("artefatos") if isinstance(payload.get("artefatos"), dict) else {}
    labels = globals().get("PROVIDER_LABELS", {})

    lines = [
        f"#+TITLE: {title}",
        f"#+AUTHOR: {author}",
        f"#+DATE: {date_value}",
        "#+LANGUAGE: pt-BR",
        "#+OPTIONS: toc:t num:t",
        f"#+LATEX_CLASS: {latex_class}",
        "#+LATEX_HEADER: \\usepackage{longtable}",
        "#+LATEX_HEADER: \\usepackage{booktabs}",
        "#+LATEX_HEADER: \\usepackage{pdflscape}",
        "#+LATEX_HEADER: \\usepackage{array}",
        "",
        "* Identificação",
        "- Perfil: =relatorio_prisma_busca_orientada_fgv=.",
        f"- Etapa: {'busca e triagem pendente' if phase_norm == 'preliminar' else 'consolidação após triagem humana'}.",
        f"- Layout institucional: {layout or 'não informado'}.",
        f"- Projeto: {_prisma_org_text(project.get('nome') or '')}.",
        "",
        "* Questão e escopo da revisão",
        f"- Tema: {_prisma_org_text(research.get('tema') or '')}",
        f"- Recorte: {_prisma_org_text(research.get('recorte') or '')}",
        f"- Objetivo: {_prisma_org_text(research.get('objetivo') or '')}",
        f"- Pergunta de pesquisa: {_prisma_org_text(research.get('pergunta_pesquisa') or '')}",
        f"- Tipo de estudo: {_prisma_org_text(research.get('tipo_estudo') or '')}",
        "",
        "* Protocolo de busca",
    ]
    lines.extend(_prisma_strategy_org_lines(payload))
    keywords = payload.get("palavras_chave") or research.get("palavras_chave") or []
    languages = research.get("idiomas") or []
    provider_labels = [labels.get(str(item), str(item)) for item in providers]
    lines += [
        f"- Palavras-chave: {_prisma_org_text('; '.join(str(item) for item in keywords))}",
        f"- Idiomas: {_prisma_org_text('; '.join(str(item) for item in languages))}",
        f"- Bases selecionadas: {_prisma_org_text('; '.join(provider_labels))}",
        "",
        "** Critérios de inclusão",
    ]
    inclusion_values = criteria_inclusion if isinstance(criteria_inclusion, list) else [criteria_inclusion]
    lines.extend(f"- {_prisma_org_text(item)}" for item in inclusion_values if _prisma_org_text(item))
    if not any(_prisma_org_text(item) for item in inclusion_values):
        lines.append("- Não informado.")
    lines += ["", "** Critérios de exclusão"]
    exclusion_values = criteria_exclusion if isinstance(criteria_exclusion, list) else [criteria_exclusion]
    lines.extend(f"- {_prisma_org_text(item)}" for item in exclusion_values if _prisma_org_text(item))
    if not any(_prisma_org_text(item) for item in exclusion_values):
        lines.append("- Não informado.")

    lines += ["", "* Registro por fonte"]
    if isinstance(source_logs, list) and source_logs:
        rows: list[list[Any]] = []
        for source in source_logs:
            if not isinstance(source, dict):
                continue
            provider = str(source.get("provider") or "")
            retrieved, status = _prisma_source_summary(source)
            rows.append([labels.get(provider, provider or "fonte"), retrieved, status])
        lines.extend(_prisma_org_table(["Base", "Registros recuperados", "Situação"], rows) if rows else ["- Nenhum registro de fonte disponível."])
    else:
        lines.append("- Nenhum registro de fonte disponível.")

    lines += ["", "* Fluxo de seleção"]
    if counts:
        rows = [[str(key).replace("_", " "), value] for key, value in counts.items()]
        lines.extend(_prisma_org_table(["Etapa", "Quantidade"], rows))
    else:
        lines.append("- As contagens ainda não foram registradas.")

    if phase_norm == "preliminar":
        lines += [
            "",
            "* Situação da triagem",
            "A busca foi concluída, mas a inclusão e a exclusão de estudos permanecem pendentes de decisão humana na planilha de triagem.",
        ]
        triage_path = _prisma_org_text(artifacts.get("planilha_triagem_xlsx") or artifacts.get("planilha_triagem_csv") or "")
        if triage_path:
            lines.append(f"- Planilha de triagem: ={triage_path}=")
        lines += [
            "- Após preencher a planilha, importe-a com =--prisma-importar-triagem=. O pipeline então produzirá a versão final deste relatório no mesmo layout e com a mesma engine PDF configurada no TOML.",
            "",
            "* Nota metodológica",
            "Este documento registra a estratégia de descoberta e a situação da triagem. Ele não apresenta a busca como revisão concluída nem substitui a avaliação humana dos títulos, resumos e textos completos.",
        ]
    else:
        included = payload.get("estudos_incluidos") or []
        lines += ["", "* Estudos incluídos"]
        if isinstance(included, list) and included:
            for index, item in enumerate(included, start=1):
                if isinstance(item, dict):
                    lines.append(f"{index}. {_prisma_reference(item)}")
        else:
            lines.append("Nenhum estudo foi marcado como incluído. Revise a planilha de triagem antes de interpretar o relatório como concluído.")
        matrix_path = _prisma_org_text(artifacts.get("matriz_estudos_incluidos_xlsx") or artifacts.get("matriz_estudos_incluidos_csv") or "")
        if matrix_path:
            lines += ["", f"- Matriz editável (CSV/XLSX): ={matrix_path}="]
        if isinstance(included, list) and included:
            _prisma_append_matrix_annex_org(lines, [item for item in included if isinstance(item, dict)])
        lines += [
            "",
            "* Nota metodológica",
            "As contagens e os estudos incluídos foram consolidados exclusivamente a partir da planilha de triagem preenchida pela pessoa responsável. O sistema não inferiu decisões de elegibilidade.",
        ]

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target
'''.strip()


def run_import_check(module_path: Path) -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(module_path.parent)!r}); "
        "import prisma_busca_externa as module; "
        "assert callable(getattr(module, 'render_external_prisma_org_report', None)); "
        "print('Importação do renderizador PRISMA: OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(module_path.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stdout + "\n" + proc.stderr).strip()
        raise RuntimeError(detail or "A importação do módulo falhou sem detalhe.")
    print(proc.stdout.strip())


def main() -> int:
    root = Path.cwd().resolve()
    target = root / TARGET_RELATIVE
    if not target.is_file():
        raise FileNotFoundError(
            "Arquivo-alvo não encontrado. Execute a partir da raiz do projeto:\n"
            f"- esperado: {target}"
        )

    original = target.read_text(encoding="utf-8")
    if "def render_external_prisma_org_report(" in original:
        print("O renderizador PRISMA já está presente; nenhuma alteração foi necessária.")
        py_compile.compile(str(target), doraise=True)
        run_import_check(target)
        return 0

    insertion_at = original.find(MARKER)
    if insertion_at < 0:
        raise RuntimeError(
            "Não foi encontrado o ponto seguro de inserção antes de def _decision(...). "
            "Nenhuma alteração foi feita."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_renderer_prisma_v15_2_{stamp}")
    shutil.copy2(target, backup)
    updated = original[:insertion_at].rstrip() + "\n\n" + RENDERER_BLOCK + "\n\n" + original[insertion_at:]

    try:
        target.write_text(updated, encoding="utf-8")
        py_compile.compile(str(target), doraise=True)
        run_import_check(target)
    except Exception as exc:
        shutil.copy2(backup, target)
        raise RuntimeError(
            "A validação falhou; o arquivo original foi restaurado. "
            f"Detalhe: {exc}"
        ) from exc

    print("Reparo v15.2 concluído.")
    print(f"Arquivo atualizado: {target}")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


PREFIX_DEFAULT = "relatorio_prisma_prisma_fluxo_pmf"
OUT_DEFAULT = "app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf"


ALIASES = {
    "identificados": [
        "registros identificados",
        "registros_identificados",
        "identified",
        "records_identified",
        "total_identificados",
    ],
    "duplicatas_removidas": [
        "duplicatas removidas",
        "duplicatas_removidas",
        "duplicates_removed",
        "duplicates removidas",
        "registros_duplicados_removidos",
    ],
    "apos_deduplicacao": [
        "registros apos deduplicacao",
        "registros após deduplicação",
        "registros_apos_deduplicacao",
        "after_deduplication",
        "records_after_duplicates_removed",
    ],
    "pre_triagem_ia": [
        "registros pre triagem ia avaliados",
        "registros pré triagem ia avaliados",
        "registros_pre_triagem_ia_avaliados",
        "pre_triagem_ia_avaliados",
        "ai_prescreened",
    ],
    "enviados_triagem": [
        "registros enviados para triagem",
        "registros_enviados_para_triagem",
        "sent_to_screening",
    ],
    "triagem_concluida": [
        "triagem titulo resumo concluida",
        "triagem título resumo concluída",
        "triagem_titulo_resumo_concluida",
        "registros com triagem titulo resumo concluida",
        "registros com triagem título resumo concluída",
    ],
    "excluidos_titulo_resumo": [
        "registros excluidos titulo resumo",
        "registros excluídos título resumo",
        "registros_excluidos_titulo_resumo",
        "excluded_title_abstract",
    ],
    "textos_completos_avaliados": [
        "textos completos avaliados",
        "textos_completos_avaliados",
        "full_text_assessed",
    ],
    "textos_completos_excluidos": [
        "textos completos excluidos",
        "textos completos excluídos",
        "textos_completos_excluidos",
        "full_text_excluded",
    ],
    "incluidos": [
        "estudos incluidos",
        "estudos incluídos",
        "estudos_incluidos",
        "included",
        "studies_included",
    ],
    "registros_planilha": [
        "registros na planilha",
        "registros_na_planilha",
    ],
}


def log(kind: str, msg: str) -> None:
    print(f"[{kind}] {msg}")


def norm(s: object) -> str:
    txt = str(s or "").strip()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"[_\-]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip().lower()
    return txt


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    txt = str(value).strip()
    if not txt:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", txt)
    if not m:
        return None
    try:
        return int(float(m.group(0).replace(",", ".")))
    except Exception:
        return None


def set_count(counts: dict[str, int], key: str, value: Any, *, overwrite: bool = False) -> None:
    n = to_int(value)
    if n is None or n < 0:
        return
    if overwrite or key not in counts:
        counts[key] = n


def canonical_key(label: str) -> str | None:
    nlabel = norm(label)
    for key, aliases in ALIASES.items():
        for alias in aliases:
            if norm(alias) == nlabel:
                return key
    # fallback por contains
    if "identificado" in nlabel and "registro" in nlabel:
        return "identificados"
    if "duplicata" in nlabel:
        return "duplicatas_removidas"
    if "deduplic" in nlabel:
        return "apos_deduplicacao"
    if "pre triagem" in nlabel or "pré triagem" in nlabel:
        return "pre_triagem_ia"
    if "enviado" in nlabel and "triagem" in nlabel:
        return "enviados_triagem"
    if "excluido" in nlabel and "titulo" in nlabel:
        return "excluidos_titulo_resumo"
    if "texto" in nlabel and "avaliado" in nlabel:
        return "textos_completos_avaliados"
    if "texto" in nlabel and "excluido" in nlabel:
        return "textos_completos_excluidos"
    if "incluido" in nlabel and "estudo" in nlabel:
        return "incluidos"
    if "planilha" in nlabel:
        return "registros_planilha"
    return None


def recursive_scan_json(obj: Any, counts: dict[str, int]) -> None:
    if isinstance(obj, dict):
        # direct key-value
        for k, v in obj.items():
            ck = canonical_key(str(k))
            if ck:
                set_count(counts, ck, v)

        # table/list row patterns
        possible_label = None
        possible_value = None
        for lk in ["etapa", "label", "nome", "name", "indicador", "metric"]:
            if lk in obj:
                possible_label = obj[lk]
                break
        for vk in ["quantidade", "valor", "value", "n", "count", "total"]:
            if vk in obj:
                possible_value = obj[vk]
                break
        if possible_label is not None:
            ck = canonical_key(str(possible_label))
            if ck:
                set_count(counts, ck, possible_value)

        for v in obj.values():
            recursive_scan_json(v, counts)

    elif isinstance(obj, list):
        for item in obj:
            recursive_scan_json(item, counts)


def scan_json_files(out_dir: Path, prefix: str, counts: dict[str, int]) -> None:
    for p in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        recursive_scan_json(data, counts)


def parse_org_flow_tables(text: str, counts: dict[str, int]) -> None:
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if set(cells[0]) <= {"-", "+"}:
            continue
        ck = canonical_key(cells[0])
        if ck:
            set_count(counts, ck, cells[1])


def scan_org_files(out_dir: Path, prefix: str, counts: dict[str, int], org_path: Path | None = None) -> None:
    candidates = []
    if org_path and org_path.exists():
        candidates.append(org_path)
    candidates.extend(sorted(out_dir.glob("*.org"), key=lambda p: p.stat().st_mtime, reverse=True))
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            parse_org_flow_tables(p.read_text(encoding="utf-8", errors="ignore"), counts)
        except Exception:
            continue


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return [dict(r) for r in reader]
        except UnicodeDecodeError:
            continue
    return []


def find_triage_csv(out_dir: Path, prefix: str) -> Path | None:
    candidates = [
        out_dir / f"{prefix}.triagem_humana.csv",
        out_dir / f"{prefix}.referencias_incluidas_seminario.csv",
        out_dir / f"{prefix}.triagem_titulo_resumo.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted(out_dir.glob("*triagem_humana*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def upper(row: dict[str, str], key: str) -> str:
    return str(row.get(key, "") or "").strip().upper()


def derive_from_triage_csv(out_dir: Path, prefix: str, counts: dict[str, int]) -> None:
    csv_path = find_triage_csv(out_dir, prefix)
    if not csv_path:
        return
    rows = read_csv_rows(csv_path)
    if not rows:
        return

    total = len(rows)
    set_count(counts, "registros_planilha", total, overwrite=True)
    set_count(counts, "enviados_triagem", total, overwrite=True)

    decided_title = 0
    include_final = 0
    exclude_full = 0
    include_title = 0
    exclude_title_marked = 0

    for row in rows:
        dtitulo = upper(row, "decisao_titulo_resumo")
        dtexto = upper(row, "decisao_texto_completo")
        final = upper(row, "incluir_final")

        if dtitulo and dtitulo not in {"PENDENTE", "NAO_INICIADO", "NÃO_INICIADO"}:
            decided_title += 1
        if final == "SIM":
            include_final += 1
        if dtitulo == "INCLUIR":
            include_title += 1
        if dtitulo == "EXCLUIR":
            exclude_title_marked += 1
        if dtexto == "EXCLUIR":
            exclude_full += 1

    # Se a planilha final é a fonte de verdade, usamos contagens reconciliadas.
    # Full-text assessed = incluídos finais + excluídos em texto completo.
    full_assessed = include_final + exclude_full
    title_excluded = total - full_assessed
    if title_excluded < 0:
        # fallback se a coluna de texto completo estiver mal preenchida
        title_excluded = exclude_title_marked
        full_assessed = max(0, total - title_excluded)

    set_count(counts, "triagem_concluida", decided_title or total, overwrite=True)
    set_count(counts, "excluidos_titulo_resumo", title_excluded, overwrite=True)
    set_count(counts, "textos_completos_avaliados", full_assessed, overwrite=True)
    set_count(counts, "textos_completos_excluidos", exclude_full, overwrite=True)
    set_count(counts, "incluidos", include_final, overwrite=True)


def derive_counts(
    out_dir: Path,
    prefix: str = PREFIX_DEFAULT,
    org_path: Path | None = None,
    prisma_payload: dict[str, Any] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    if prisma_payload:
        recursive_scan_json(prisma_payload, counts)

    scan_json_files(out_dir, prefix, counts)
    scan_org_files(out_dir, prefix, counts, org_path=org_path)
    derive_from_triage_csv(out_dir, prefix, counts)

    # Reconciliações de identificação/deduplicação.
    if "identificados" in counts and "duplicatas_removidas" in counts and "apos_deduplicacao" not in counts:
        set_count(counts, "apos_deduplicacao", counts["identificados"] - counts["duplicatas_removidas"])
    if "apos_deduplicacao" in counts and "duplicatas_removidas" in counts and "identificados" not in counts:
        set_count(counts, "identificados", counts["apos_deduplicacao"] + counts["duplicatas_removidas"])
    if "identificados" in counts and "apos_deduplicacao" in counts and "duplicatas_removidas" not in counts:
        set_count(counts, "duplicatas_removidas", counts["identificados"] - counts["apos_deduplicacao"])

    # Se houver pré-triagem, o que não chegou à triagem humana é um descarte operacional.
    if "apos_deduplicacao" in counts and "pre_triagem_ia" not in counts:
        # Nem todo fluxo usa pré-triagem; só não inventa quando não houver sinal.
        pass

    return counts


def fmt_n(counts: dict[str, int], key: str) -> str:
    return str(counts[key]) if key in counts else "—"


def draw_diagram_png(counts: dict[str, int], png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except Exception as exc:
        raise RuntimeError(f"matplotlib não disponível para gerar PNG: {exc}") from exc

    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.2, 11.0), dpi=220)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 13.9)
    ax.axis("off")

    def box(x, y, w, h, title, body="", fontsize=8.1, face="#F8FAFD", edge="#3B5F8A", lw=1.0):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(rect)
        text = title if not body else f"{title}\n{body}"
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize,
            wrap=True,
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.95,
            color="#4A4A4A",
        ))

    identified = fmt_n(counts, "identificados")
    dup = fmt_n(counts, "duplicatas_removidas")
    dedup = fmt_n(counts, "apos_deduplicacao")
    preai = fmt_n(counts, "pre_triagem_ia")
    sent = fmt_n(counts, "enviados_triagem")
    title_exc = fmt_n(counts, "excluidos_titulo_resumo")
    full = fmt_n(counts, "textos_completos_avaliados")
    full_exc = fmt_n(counts, "textos_completos_excluidos")
    inc = fmt_n(counts, "incluidos")

    preai_not_sent = "—"
    if "pre_triagem_ia" in counts and "enviados_triagem" in counts:
        preai_not_sent = str(max(0, counts["pre_triagem_ia"] - counts["enviados_triagem"]))
    dedup_not_preai = "—"
    if "apos_deduplicacao" in counts and "pre_triagem_ia" in counts:
        dedup_not_preai = str(max(0, counts["apos_deduplicacao"] - counts["pre_triagem_ia"]))

    ax.text(5.7, 13.45, "Fluxo PRISMA adaptado", ha="center", va="center", fontsize=13.2, fontweight="bold")
    ax.text(
        5.7, 13.12,
        "Busca, deduplicação, pré-triagem por IA, triagem humana e inclusão final",
        ha="center", va="center", fontsize=8.1
    )

    main_x, main_w, main_h = 1.0, 7.25, 0.72
    side_x, side_w, side_h = 8.75, 2.65, 0.72

    y1, y2, y3, y4, y5, y6 = 12.05, 10.85, 9.65, 8.45, 7.25, 6.05

    box(main_x, y1, main_w, main_h, "Registros identificados nas bases", f"n = {identified}", face="#EEF3F9")
    box(main_x, y2, main_w, main_h, "Registros após deduplicação", f"n = {dedup}", face="#EEF3F9")
    box(main_x, y3, main_w, main_h, "Registros avaliados na pré-triagem por IA", f"n = {preai}", face="#F1F7F1")
    box(main_x, y4, main_w, main_h, "Registros enviados à triagem humana", f"n = {sent}", face="#F1F7F1")
    box(main_x, y5, main_w, main_h, "Textos completos/registros avaliados para inclusão", f"n = {full}", face="#FFF9EC")
    box(main_x, y6, main_w, main_h, "Estudos incluídos na revisão", f"n = {inc}", face="#EDF8F0", edge="#2E8B57", lw=1.05)

    box(side_x, y2, side_w, side_h, "Duplicatas removidas", f"n = {dup}", fontsize=7.0, face="#FDF1F0", edge="#A5443B")
    box(side_x, y3, side_w, side_h, "Não avaliados na\npré-triagem IA", f"n = {dedup_not_preai}", fontsize=6.9, face="#FDF1F0", edge="#A5443B")
    box(side_x, y4, side_w, side_h, "Não enviados à\ntriagem humana", f"n = {preai_not_sent}", fontsize=6.9, face="#FDF1F0", edge="#A5443B")
    box(side_x, y5, side_w, side_h, "Excluídos em\ntítulo/resumo", f"n = {title_exc}", fontsize=6.9, face="#FDF1F0", edge="#A5443B")
    box(side_x, y6, side_w, side_h, "Excluídos após\ntexto completo", f"n = {full_exc}", fontsize=6.9, face="#FDF1F0", edge="#A5443B")

    cx = main_x + main_w / 2
    arrow(cx, y1, cx, y2 + main_h)
    arrow(cx, y2, cx, y3 + main_h)
    arrow(cx, y3, cx, y4 + main_h)
    arrow(cx, y4, cx, y5 + main_h)
    arrow(cx, y5, cx, y6 + main_h)

    rx1 = main_x + main_w
    rx2 = side_x
    arrow(rx1, y2 + main_h / 2, rx2, y2 + side_h / 2)
    arrow(rx1, y3 + main_h / 2, rx2, y3 + side_h / 2)
    arrow(rx1, y4 + main_h / 2, rx2, y4 + side_h / 2)
    arrow(rx1, y5 + main_h / 2, rx2, y5 + side_h / 2)
    arrow(rx1, y6 + main_h / 2, rx2, y6 + side_h / 2)

    ax.text(
        5.7, 5.25,
        "Nota: o diagrama é derivado automaticamente da triagem humana CSV; "
        "quando houver conflito com contagens antigas, prevalece a planilha final.",
        ha="center", va="center", fontsize=7.0, color="#4B4B4B", wrap=True,
    )

    fig.tight_layout(pad=0.6)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

def build_flow_table(counts: dict[str, int]) -> str:
    rows = [
        ("registros identificados", fmt_n(counts, "identificados")),
        ("duplicatas removidas", fmt_n(counts, "duplicatas_removidas")),
        ("registros apos deduplicacao", fmt_n(counts, "apos_deduplicacao")),
        ("registros pre triagem ia avaliados", fmt_n(counts, "pre_triagem_ia")),
        ("registros enviados para triagem", fmt_n(counts, "enviados_triagem")),
        ("triagem titulo resumo concluida", fmt_n(counts, "triagem_concluida")),
        ("registros excluidos titulo resumo", fmt_n(counts, "excluidos_titulo_resumo")),
        ("textos completos avaliados", fmt_n(counts, "textos_completos_avaliados")),
        ("textos completos excluidos", fmt_n(counts, "textos_completos_excluidos")),
        ("estudos incluidos", fmt_n(counts, "incluidos")),
        ("registros na planilha", fmt_n(counts, "registros_planilha")),
    ]
    lines = ["| Etapa | Quantidade |", "|-+-|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def org_prisma_section(counts: dict[str, int], image_filename: str, level: str = "*") -> str:
    return f"""{level} Fluxo de seleção

#+CAPTION: Diagrama PRISMA adaptado ao fluxo de busca, pré-triagem por IA e triagem humana.
#+ATTR_LATEX: :width 0.95\\textwidth
[[file:{image_filename}]]

{build_flow_table(counts)}

A figura apresenta o funil de seleção da revisão. As contagens finais de triagem humana são derivadas do arquivo =triagem_humana.csv=; quando contagens intermediárias antigas entram em conflito com a planilha final, prevalece a reconciliação pela decisão humana final.
"""


def inject_section(org_path: Path, counts: dict[str, int], image_path: Path) -> bool:
    text = org_path.read_text(encoding="utf-8", errors="ignore")
    image_filename = image_path.name

    # Garante graphicx mesmo em classes que não carregam automaticamente.
    if r"\usepackage{graphicx}" not in text and r"\usepackage[" not in text:
        # Não usar esta condição para evitar falsos negativos; abaixo fica mais seguro.
        pass
    if r"\usepackage{graphicx}" not in text and "graphicx" not in text:
        lines = text.splitlines()
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("#+LATEX_HEADER:"):
                insert_at = i + 1
        lines.insert(insert_at, r"#+LATEX_HEADER: \usepackage{graphicx}")
        text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    m = re.search(r"(?m)^(?P<level>\*+)\s+Fluxo de seleção\s*$", text)
    if m:
        level = m.group("level")
        start = m.start()
        next_heading = re.search(rf"(?m)^{re.escape(level)}\s+(?!Fluxo de seleção\b).*$", text[m.end():])
        end = m.end() + next_heading.start() if next_heading else len(text)
        new_section = org_prisma_section(counts, image_filename, level=level)
        new_text = text[:start] + new_section + "\n" + text[end:].lstrip("\n")
    else:
        # Fallback: insere antes de Estudos incluídos.
        m2 = re.search(r"(?m)^(?P<level>\*+)\s+Estudos incluídos\s*$", text)
        if m2:
            level = m2.group("level")
            new_section = org_prisma_section(counts, image_filename, level=level)
            new_text = text[:m2.start()] + new_section + "\n" + text[m2.start():]
        else:
            new_text = text.rstrip() + "\n\n" + org_prisma_section(counts, image_filename, level="*") + "\n"

    changed = new_text != text
    if changed:
        org_path.write_text(new_text, encoding="utf-8")
    return changed


def find_org(out_dir: Path, prefix: str, explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    candidates = sorted(out_dir.glob("*.org"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "Fluxo de seleção" in text or "Estudos incluídos" in text:
            return p
    return candidates[0] if candidates else None


def ensure_prisma_flow_diagram(
    cfg: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = PREFIX_DEFAULT,
    org_path: str | Path | None = None,
    prisma_payload: dict[str, Any] | None = None,
    phase: str | None = None,
) -> Path | None:
    out = Path(out_dir or OUT_DEFAULT)
    org = Path(org_path) if org_path else find_org(out, prefix)
    if org is None or not org.exists():
        log("WARN", f"ORG do relatório PRISMA não encontrado em {out}")
        return None

    counts = derive_counts(out, prefix=prefix, org_path=org, prisma_payload=prisma_payload)
    image = out / f"{prefix}.diagrama_prisma.png"
    draw_diagram_png(counts, image)
    inject_section(org, counts, image)

    counts_path = out / f"{prefix}.diagrama_prisma_contagens.json"
    counts_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    log("OK", f"Diagrama PRISMA gerado: {image}")
    log("OK", f"Relatório ORG atualizado: {org}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DEFAULT)
    parser.add_argument("--prefix", default=PREFIX_DEFAULT)
    parser.add_argument("--org", default="")
    args = parser.parse_args()

    out = Path(args.out_dir)
    org = find_org(out, args.prefix, args.org or None)
    if org is None:
        log("ERRO", f"Nenhum arquivo .org encontrado em {out}")
        return 1
    ensure_prisma_flow_diagram(out_dir=out, prefix=args.prefix, org_path=org)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# PRISMA_DIAGRAMA_TRUNCADO_AJUSTE_V1

# PRISMA_DIAGRAMA_ESTETICO_V2

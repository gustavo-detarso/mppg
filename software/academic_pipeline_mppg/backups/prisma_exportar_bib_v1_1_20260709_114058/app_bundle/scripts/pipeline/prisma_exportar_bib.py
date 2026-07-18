#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, re, unicodedata
from pathlib import Path

TRUTHY = {"sim","s","yes","y","true","1","incluir","incluido","incluído"}

def norm(v):
    s = str(v or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

def get(row, *names):
    d = {norm(k): v for k, v in row.items()}
    for n in names:
        v = d.get(norm(n))
        if v not in (None, ""):
            return str(v).strip()
    return ""

def bib_escape(s):
    s = str(s or "").strip()
    for a,b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a,b)
    return s

def split_authors(s):
    s = str(s or "").strip().strip("[]")
    if not s:
        return ""
    parts = re.split(r"\s*;\s*|\s+\band\b\s+|\s+\be\s+|\s*\|\s*", s, flags=re.I)
    parts = [p.strip(" '\"\t\r\n") for p in parts if p.strip(" '\"\t\r\n")]
    return " and ".join(parts) if parts else s

def year_from(row):
    y = get(row, "ano", "year", "publication_year", "published_year", "published", "data_publicacao")
    m = re.search(r"(19|20)\d{2}", y or " ".join(map(str, row.values())))
    return m.group(0) if m else ""

def included(row):
    inc = norm(get(row, "incluir_final", "incluido", "incluído", "incluir", "include_final"))
    if inc in TRUTHY:
        return True
    dtr = norm(get(row, "decisao_titulo_resumo", "decisão_título_resumo"))
    dtc = norm(get(row, "decisao_texto_completo", "decisão_texto_completo"))
    return dtr == "incluir" and dtc in {"incluir","incluido","incluido_final","sim"}

def dialect(path):
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        return csv.excel

def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, dialect=dialect(path)))

def make_key(authors, year, title, used):
    base = "autor"
    if authors:
        toks = re.findall(r"[A-Za-zÀ-ÿ0-9]+", re.split(r"\s+and\s+|;", authors)[0])
        if toks:
            base = toks[-1]
    word = ""
    for tok in re.findall(r"[A-Za-zÀ-ÿ0-9]+", title):
        if len(tok) > 3 and tok.lower() not in {"the","and","for","with","from","uma","para","com","dos","das","medical","study"}:
            word = tok
            break
    key = norm(f"{base}{year}{word}").replace("_", "") or "ref"
    base_key, i = key, 2
    while key in used:
        key = f"{base_key}{i}"
        i += 1
    used.add(key)
    return key

def choose_input(out_dir, prefix):
    a = out_dir / f"{prefix}.referencias_incluidas_seminario.csv"
    b = out_dir / f"{prefix}.triagem_humana.csv"
    if a.exists():
        return a, False
    if b.exists():
        return b, True
    raise FileNotFoundError(f"não encontrei {a} nem {b}")

def row_entry(row, used):
    title = get(row, "titulo", "título", "title", "article_title", "nome")
    if not title:
        return None
    authors = split_authors(get(row, "autores", "authors", "author", "criadores", "creators"))
    year = year_from(row)
    journal = get(row, "periodico", "periódico", "journal", "journaltitle", "venue", "source", "container_title", "publication")
    doi = get(row, "doi")
    url = get(row, "url", "link", "source_url", "landing_page")
    abstract = get(row, "resumo", "abstract", "summary")
    volume = get(row, "volume")
    number = get(row, "numero", "número", "number", "issue")
    pages = get(row, "paginas", "páginas", "pages")
    entry_type = "article" if journal else "misc"
    fields = [("title", title)]
    if authors: fields.append(("author", authors))
    if year: fields.append(("year", year))
    if journal: fields.append(("journaltitle", journal))
    if volume: fields.append(("volume", volume))
    if number: fields.append(("number", number))
    if pages: fields.append(("pages", pages))
    if doi: fields.append(("doi", doi))
    if url: fields.append(("url", url))
    if abstract: fields.append(("abstract", abstract[:900]))
    key = make_key(authors, year, title, used)
    body = ",\n".join(f"  {k} = {{{bib_escape(v)}}}" for k,v in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"

def export_bib(out_dir, prefix, input_csv=None, output_bib=None, filter_included=None):
    out_dir = Path(out_dir).resolve()
    if input_csv:
        input_csv = Path(input_csv).resolve()
        if filter_included is None:
            filter_included = False
    else:
        input_csv, inferred = choose_input(out_dir, prefix)
        if filter_included is None:
            filter_included = inferred
    output_bib = Path(output_bib).resolve() if output_bib else out_dir / f"{prefix}.referencias_incluidas.bib"
    rows = read_rows(input_csv)
    if filter_included:
        rows = [r for r in rows if included(r)]
    used, entries, skipped = set(), [], 0
    for r in rows:
        e = row_entry(r, used)
        if e: entries.append(e)
        else: skipped += 1
    if not entries:
        raise RuntimeError(f"nenhuma referência incluída encontrada em {input_csv}")
    output_bib.parent.mkdir(parents=True, exist_ok=True)
    output_bib.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"[OK] Entrada CSV: {input_csv}")
    print(f"[OK] Referências exportadas: {len(entries)}")
    if skipped: print(f"[WARN] Registros ignorados por falta de título: {skipped}")
    print(f"[OK] BibLaTeX gerado: {output_bib}")
    return output_bib

def main(argv=None):
    p = argparse.ArgumentParser(description="Exporta BibLaTeX das referências incluídas no PRISMA.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default=None)
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--filtrar-incluidas", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    prefix = a.prefix or out.name
    export_bib(out, prefix, a.input, a.output, True if a.filtrar_incluidas else None)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

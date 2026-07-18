
#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import subprocess
import sys

def count_words(text: str) -> int:
    text = re.sub(r"(?s)#\+begin_src.*?#\+end_src", " ", text)
    text = re.sub(r"(?m)^#\+.*$", " ", text)
    text = re.sub(r"[@\\][A-Za-z0-9:_-]+", " ", text)
    return len(re.findall(r"\b[\wÀ-ÿ-]+\b", text))

def count_bib_entries(text: str) -> int:
    return len(re.findall(r"(?m)^@\w+\s*\{", text))

def count_pdf_pages(pdf: Path) -> int:
    if not pdf.exists():
        return 0
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf)], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"^Pages:\s+(\d+)", out, re.M)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--min-palavras", type=int, default=8500)
    ap.add_argument("--min-referencias", type=int, default=20)
    ap.add_argument("--min-paginas", type=int, default=18)
    args = ap.parse_args()

    art = Path(args.art_dir)
    out = art / "output"

    org = out / "artigo_final_atestmed_abnt.org"
    bib = out / "artigo_final_atestmed_abnt.bib"
    pdf = out / "artigo_final_atestmed_abnt.pdf"
    doc_json = out / "artigo_final_atestmed_abnt.document.json"

    errors = []

    org_txt = org.read_text(encoding="utf-8", errors="ignore") if org.exists() else ""
    bib_txt = bib.read_text(encoding="utf-8", errors="ignore") if bib.exists() else ""
    json_txt = doc_json.read_text(encoding="utf-8", errors="ignore") if doc_json.exists() else ""

    org_words = count_words(org_txt)
    json_words = count_words(json_txt)
    bib_entries = count_bib_entries(bib_txt)
    pages = count_pdf_pages(pdf)

    print("Validação artigo longo full text")
    print("=" * 72)
    print(f"ORG palavras aproximadas: {org_words}")
    print(f"document.json palavras aproximadas: {json_words}")
    print(f"BibTeX entradas: {bib_entries}")
    print(f"PDF páginas: {pages}")

    if org_words < args.min_palavras:
        errors.append(f"ORG curto: {org_words} palavras; mínimo exigido: {args.min_palavras}")

    if bib_entries < args.min_referencias:
        errors.append(f"Referências insuficientes: {bib_entries}; mínimo exigido: {args.min_referencias}")

    if pages and pages < args.min_paginas:
        errors.append(f"PDF curto: {pages} páginas; mínimo exigido: {args.min_paginas}")

    bad_14 = re.search(r"\b14\s+estudos\b|\bquatorze\s+estudos\b", org_txt, re.I)
    if bad_14:
        errors.append("O texto ainda menciona corpus de 14 estudos.")

    required_terms = [
        "20 estudos",
        "PRISMA",
        "full text",
        "matriz de evidências",
        "ATESTMED",
        "Perícia Médica Federal",
    ]

    for term in required_terms:
        if term.lower() not in org_txt.lower():
            errors.append(f"Termo/seção obrigatória ausente no ORG: {term}")

    if errors:
        print("\nERROS:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("\n[OK] Artigo longo validado.")

if __name__ == "__main__":
    main()

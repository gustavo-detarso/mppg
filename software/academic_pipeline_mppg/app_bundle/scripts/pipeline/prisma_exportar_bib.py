#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re, unicodedata
from pathlib import Path

TRUTHY = {"sim","s","yes","y","true","1","incluir","incluido","incluído"}

def norm(v):
    s=str(v or "").strip().lower()
    s=unicodedata.normalize("NFKD",s)
    return re.sub(r"[^a-z0-9]+","_","".join(ch for ch in s if not unicodedata.combining(ch))).strip("_")

def get(row,*names):
    d={norm(k):v for k,v in row.items()}
    for n in names:
        v=d.get(norm(n))
        if v not in (None,""): return str(v).strip()
    return ""

def dialect(path):
    sample=path.read_text(encoding="utf-8-sig",errors="ignore")[:8192]
    try: return csv.Sniffer().sniff(sample,delimiters=",;\t")
    except Exception: return csv.excel

def read_rows(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f,dialect=dialect(path)))

def included(row):
    inc=norm(get(row,"incluir_final","incluido","incluído","incluir","include_final"))
    if inc in TRUTHY: return True
    return norm(get(row,"decisao_titulo_resumo","decisão_título_resumo"))=="incluir" and norm(get(row,"decisao_texto_completo","decisão_texto_completo")) in {"incluir","incluido","sim"}

def year_from(row):
    y=get(row,"ano","year","publication_year","published_year","published")
    m=re.search(r"(19|20)\d{2}", y or " ".join(map(str,row.values())))
    return m.group(0) if m else ""

def bib_escape(s):
    s=str(s or "").strip()
    for a,b in [("\\",r"\textbackslash{}"),("&",r"\&"),("%",r"\%"),("$",r"\$"),("#",r"\#"),("_",r"\_")]:
        s=s.replace(a,b)
    return s

def split_authors(s):
    s=str(s or "").strip().strip("[]")
    if not s: return ""
    parts=re.split(r"\s*;\s*|\s+\band\b\s+|\s+\be\s+|\s*\|\s*",s,flags=re.I)
    parts=[p.strip(" '\"\t\r\n") for p in parts if p.strip(" '\"\t\r\n")]
    return " and ".join(parts) if parts else s

def make_key(authors,year,title,used):
    base="autor"
    if authors:
        toks=re.findall(r"[A-Za-zÀ-ÿ0-9]+",re.split(r"\s+and\s+|;",authors)[0])
        if toks: base=toks[-1]
    word=""
    for tok in re.findall(r"[A-Za-zÀ-ÿ0-9]+",title):
        if len(tok)>3 and tok.lower() not in {"the","and","for","with","from","uma","para","com","dos","das","medical","study"}:
            word=tok; break
    key=norm(f"{base}{year}{word}").replace("_","") or "ref"
    original,i=key,2
    while key in used:
        key=f"{original}{i}"; i+=1
    used.add(key); return key

def choose_input(out_dir,prefix):
    a=out_dir/f"{prefix}.referencias_incluidas_seminario.csv"
    b=out_dir/f"{prefix}.triagem_humana.csv"
    if a.exists(): return a,False
    if b.exists(): return b,True
    raise FileNotFoundError(f"não encontrei {a} nem {b}")

def row_entry(row,used):
    title=get(row,"titulo","título","title","article_title","nome")
    if not title: return None
    authors=split_authors(get(row,"autores","authors","author","criadores","creators"))
    year=year_from(row)
    journal=get(row,"periodico","periódico","journal","journaltitle","venue","source","container_title","publication")
    doi=get(row,"doi"); url=get(row,"url","link","source_url","landing_page")
    abstract=get(row,"resumo","abstract","summary")
    entry_type="article" if journal else "misc"
    fields=[("title",title)]
    if authors: fields.append(("author",authors))
    if year: fields.append(("year",year))
    if journal: fields.append(("journaltitle",journal))
    if doi: fields.append(("doi",doi))
    if url: fields.append(("url",url))
    if abstract: fields.append(("abstract",abstract[:900]))
    key=make_key(authors,year,title,used)
    body=",\n".join(f"  {k} = {{{bib_escape(v)}}}" for k,v in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"

def export_bib(out_dir,prefix,input_csv=None,output_bib=None,filter_included=None):
    out_dir=Path(out_dir).resolve()
    if input_csv:
        input_csv=Path(input_csv).resolve()
        filter_included=False if filter_included is None else filter_included
    else:
        input_csv,inferred=choose_input(out_dir,prefix)
        filter_included=inferred if filter_included is None else filter_included
    output_bib=Path(output_bib).resolve() if output_bib else out_dir/f"{prefix}.referencias_incluidas.bib"
    rows=read_rows(input_csv)
    if filter_included: rows=[r for r in rows if included(r)]
    used=set(); entries=[]
    for r in rows:
        e=row_entry(r,used)
        if e: entries.append(e)
    if not entries: raise RuntimeError(f"nenhuma referência incluída encontrada em {input_csv}")
    output_bib.parent.mkdir(parents=True,exist_ok=True)
    output_bib.write_text("\n\n".join(entries)+"\n",encoding="utf-8")
    print(f"[OK] Entrada CSV: {input_csv}")
    print(f"[OK] Referências exportadas: {len(entries)}")
    print(f"[OK] BibLaTeX gerado: {output_bib}")
    return output_bib

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True); p.add_argument("--prefix",default=None)
    p.add_argument("--input",default=None); p.add_argument("--output",default=None)
    p.add_argument("--filtrar-incluidas",action="store_true")
    a=p.parse_args(argv)
    out=Path(a.out_dir)
    export_bib(out,a.prefix or out.name,a.input,a.output,True if a.filtrar_incluidas else None)
    return 0
if __name__=="__main__": raise SystemExit(main())

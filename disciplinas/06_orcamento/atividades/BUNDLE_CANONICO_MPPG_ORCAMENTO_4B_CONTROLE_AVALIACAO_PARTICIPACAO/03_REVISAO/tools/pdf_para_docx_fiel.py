#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, shutil, subprocess, tempfile, zipfile
from pathlib import Path

EMU_PER_PT = 12700
TWIPS_PER_PT = 20


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise SystemExit(f"ERRO executando {' '.join(map(str,cmd))}:\n{p.stdout}\n{p.stderr}")
    return p.stdout


def pdf_size_points(pdf: Path):
    out = run(["pdfinfo", "-f", "1", "-l", "1", str(pdf)])
    m = re.search(r"^Page(?:\s+\d+)?\s+size:\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts", out, re.M)
    if not m:
        raise SystemExit("ERRO: não consegui obter o tamanho da página via pdfinfo.")
    return float(m.group(1)), float(m.group(2))


def page_count(pdf: Path):
    out = run(["pdfinfo", str(pdf)])
    m = re.search(r"^Pages:\s*(\d+)", out, re.M)
    if not m:
        raise SystemExit("ERRO: não consegui obter o número de páginas via pdfinfo.")
    return int(m.group(1))


def natural_key(p: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def drawing_xml(rid: str, idx: int, cx: int, cy: int):
    return f'''<w:drawing><wp:inline xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{idx}" name="Página {idx}"/>
<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>
<pic:nvPicPr><pic:cNvPr id="0" name="page-{idx}.png"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"/></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>'''


def build_docx(pdf: Path, out: Path, skeleton: Path, dpi: int):
    for cmd in ("pdfinfo", "pdftoppm"):
        if shutil.which(cmd) is None:
            raise SystemExit(f"ERRO: comando obrigatório não encontrado: {cmd}")
    if not skeleton.is_file():
        raise SystemExit(f"ERRO: skeleton DOCX não encontrado: {skeleton}")

    wpt, hpt = pdf_size_points(pdf)
    n = page_count(pdf)
    cx, cy = round(wpt * EMU_PER_PT), round(hpt * EMU_PER_PT)
    tw_w, tw_h = round(wpt * TWIPS_PER_PT), round(hpt * TWIPS_PER_PT)

    with tempfile.TemporaryDirectory(prefix="mppg_pdf2docx_") as td:
        td = Path(td)
        prefix = td / "page"
        run(["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)])
        imgs = sorted(td.glob("page-*.png"), key=natural_key)
        if len(imgs) != n:
            raise SystemExit(f"ERRO: esperava {n} imagens de página, obtive {len(imgs)}.")

        with zipfile.ZipFile(skeleton, 'r') as zin:
            document = zin.read('word/document.xml').decode('utf-8')
            rels = zin.read('word/_rels/document.xml.rels').decode('utf-8')
            ctype = zin.read('[Content_Types].xml').decode('utf-8')
            members = {i.filename: zin.read(i.filename) for i in zin.infolist()}

        # Garantir os namespaces que o DrawingML usa. O skeleton do python-docx já os contém,
        # mas o gate evita gerar arquivo silenciosamente inválido se o skeleton mudar.
        for token in ('xmlns:wp=', 'xmlns:r=', 'xmlns:w='):
            if token not in document:
                raise SystemExit(f'ERRO: skeleton sem namespace obrigatório: {token}')

        paras=[]
        rel_items=[]
        for i in range(1, n+1):
            rid=f'rIdPage{i}'
            rel_items.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/page-{i}.png"/>')
            content=drawing_xml(rid,i,cx,cy)
            if i<n:
                content += '<w:br w:type="page"/>'
            paras.append(f'<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr><w:r>{content}</w:r></w:p>')

        sect=f'<w:sectPr><w:pgSz w:w="{tw_w}" w:h="{tw_h}"/><w:pgMar w:top="0" w:right="0" w:bottom="0" w:left="0" w:header="0" w:footer="0" w:gutter="0"/><w:cols w:space="0"/><w:docGrid w:linePitch="1"/></w:sectPr>'
        body=''.join(paras)+sect
        document2=re.sub(r'<w:body>.*?</w:body>', f'<w:body>{body}</w:body>', document, flags=re.S)
        if document2 == document:
            raise SystemExit('ERRO: não consegui substituir o body do skeleton.')

        # Remove relações de imagem do skeleton e injeta apenas as páginas atuais.
        rels_clean=re.sub(r'<Relationship[^>]+Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"[^>]*/>', '', rels)
        rels2=rels_clean.replace('</Relationships>', ''.join(rel_items)+'</Relationships>')
        if 'Extension="png"' not in ctype and "Extension='png'" not in ctype:
            ctype2=ctype.replace('</Types>', '<Default Extension="png" ContentType="image/png"/></Types>')
        else:
            ctype2=ctype

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp=out.with_suffix(out.suffix+'.tmp')
        if tmp.exists(): tmp.unlink()
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zout:
            for name,data in members.items():
                if name in ('word/document.xml','word/_rels/document.xml.rels','[Content_Types].xml') or name.startswith('word/media/'):
                    continue
                zout.writestr(name,data)
            zout.writestr('[Content_Types].xml',ctype2)
            zout.writestr('word/document.xml',document2)
            zout.writestr('word/_rels/document.xml.rels',rels2)
            for i,img in enumerate(imgs,1):
                zout.write(img,f'word/media/page-{i}.png')
        os.replace(tmp,out)
        return n,wpt,hpt


def main():
    ap=argparse.ArgumentParser(description='Gera DOCX com fidelidade visual ao PDF. Cada página do PDF vira uma página-image no DOCX.')
    ap.add_argument('pdf',type=Path)
    ap.add_argument('docx',type=Path)
    ap.add_argument('--skeleton',type=Path,required=True)
    ap.add_argument('--dpi',type=int,default=180)
    args=ap.parse_args()
    if not args.pdf.is_file(): raise SystemExit(f'ERRO: PDF não encontrado: {args.pdf}')
    n,w,h=build_docx(args.pdf,args.docx,args.skeleton,args.dpi)
    print(f'PASS: {args.docx}')
    print(f'Páginas: {n}; tamanho: {w:.3f} x {h:.3f} pt; raster: {args.dpi} dpi')
    print('DOCX de fidelidade visual: o layout é o do PDF; o conteúdo das páginas não é texto Word editável.')

if __name__=='__main__': main()

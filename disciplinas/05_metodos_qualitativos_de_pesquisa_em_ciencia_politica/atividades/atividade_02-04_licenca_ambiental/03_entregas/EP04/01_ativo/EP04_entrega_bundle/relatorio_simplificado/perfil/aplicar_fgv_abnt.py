#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import re
import tomllib

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip('#')
    if len(value) != 6:
        raise ValueError(f'Cor inválida: {value!r}')
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def load_profile(path: Path) -> dict:
    with path.open('rb') as f:
        return tomllib.load(f)


def set_font(style, name: str, size: float | None = None, bold: bool | None = None, color: str | None = None):
    style.font.name = name
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color:
        style.font.color.rgb = RGBColor(*hex_rgb(color))
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.rFonts
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    for a in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        rFonts.set(qn('w:' + a), name)



def set_run_language(run_or_style, lang: str):
    """Define idioma OOXML para habilitar hifenização/correção em pt-BR."""
    if hasattr(run_or_style, 'element'):
        el = run_or_style.element
        if hasattr(el, 'get_or_add_rPr'):
            rPr = el.get_or_add_rPr()
        else:
            rPr = el.get_or_add_rPr() if hasattr(el, 'get_or_add_rPr') else None
    else:
        rPr = None
    if rPr is None and hasattr(run_or_style, '_r'):
        rPr = run_or_style._r.get_or_add_rPr()
    if rPr is None:
        return
    lang_el = rPr.find(qn('w:lang'))
    if lang_el is None:
        lang_el = OxmlElement('w:lang')
        rPr.append(lang_el)
    lang_el.set(qn('w:val'), lang)
    lang_el.set(qn('w:eastAsia'), lang)
    lang_el.set(qn('w:bidi'), lang)


def configure_language_and_hyphenation(doc, lang: str = 'pt-BR'):
    """Ativa hifenização automática e marca o documento como português brasileiro."""
    settings = doc.settings._element

    def upsert(tag: str, value: str):
        el = settings.find(qn('w:' + tag))
        if el is None:
            el = OxmlElement('w:' + tag)
            settings.append(el)
        el.set(qn('w:val'), value)

    upsert('autoHyphenation', '1')
    upsert('consecutiveHyphenLimit', '2')
    upsert('hyphenationZone', '360')
    upsert('doNotHyphenateCaps', '0')

    # Idioma padrão do documento.
    styles_el = doc.styles.element
    doc_defaults = styles_el.find(qn('w:docDefaults'))
    if doc_defaults is None:
        doc_defaults = OxmlElement('w:docDefaults')
        styles_el.insert(0, doc_defaults)
    rpr_default = doc_defaults.find(qn('w:rPrDefault'))
    if rpr_default is None:
        rpr_default = OxmlElement('w:rPrDefault')
        doc_defaults.append(rpr_default)
    rpr = rpr_default.find(qn('w:rPr'))
    if rpr is None:
        rpr = OxmlElement('w:rPr')
        rpr_default.append(rpr)
    lang_el = rpr.find(qn('w:lang'))
    if lang_el is None:
        lang_el = OxmlElement('w:lang')
        rpr.append(lang_el)
    lang_el.set(qn('w:val'), lang)
    lang_el.set(qn('w:eastAsia'), lang)
    lang_el.set(qn('w:bidi'), lang)

    for style in doc.styles:
        try:
            if getattr(style, 'type', None) is not None:
                set_run_language(style, lang)
        except Exception:
            pass

    for p in doc.paragraphs:
        pPr = p._p.get_or_add_pPr()
        suppress = pPr.find(qn('w:suppressAutoHyphens'))
        if suppress is not None:
            pPr.remove(suppress)
        for run in p.runs:
            set_run_language(run, lang)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    pPr = p._p.get_or_add_pPr()
                    suppress = pPr.find(qn('w:suppressAutoHyphens'))
                    if suppress is not None:
                        pPr.remove(suppress)
                    for run in p.runs:
                        set_run_language(run, lang)

def pborder_bottom(style, color: str, sz='8', space='4'):
    pPr = style.element.get_or_add_pPr()
    pBdr = pPr.find(qn('w:pBdr'))
    if pBdr is None:
        pBdr = OxmlElement('w:pBdr')
        pPr.append(pBdr)
    bottom = pBdr.find(qn('w:bottom'))
    if bottom is None:
        bottom = OxmlElement('w:bottom')
        pBdr.append(bottom)
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), sz)
    bottom.set(qn('w:space'), space)
    bottom.set(qn('w:color'), color)


def replace_paragraph_runs_with_quote(p, speaker: str, quote: str, font_name: str, font_pt: float):
    p_el = p._p
    for child in list(p_el):
        if child.tag != qn('w:pPr'):
            p_el.remove(child)

    r1 = p.add_run(speaker + ':')
    r1.bold = True
    r1.font.size = Pt(font_pt)
    r1.font.name = font_name

    r2 = p.add_run(' ' + quote)
    r2.italic = True
    r2.font.size = Pt(font_pt)
    r2.font.name = font_name


def apply_profile(src: Path, dst: Path, profile_path: Path):
    profile = load_profile(profile_path)
    doc = Document(src)

    page = profile['pagina']
    typo = profile['tipografia']
    colors = profile['cores']
    titles = profile['titulos']
    ficha = profile['ficha']
    excerpts = profile['excertos']
    refs = profile['referencias']
    figreads = profile['leituras_figuras']
    temas = profile['temas']
    figuras = profile['figuras']
    breaks = profile['quebras']
    document = profile['documento']

    font_name = typo['fonte_principal']
    lang = typo.get('idioma', 'pt-BR')

    # Página: fonte canônica FGV/ABNT no TOML.
    for sec in doc.sections:
        sec.top_margin = Cm(page['margem_superior_cm'])
        sec.bottom_margin = Cm(page['margem_inferior_cm'])
        sec.left_margin = Cm(page['margem_esquerda_cm'])
        sec.right_margin = Cm(page['margem_direita_cm'])
        sec.header_distance = Cm(page['distancia_cabecalho_cm'])
        sec.footer_distance = Cm(page['distancia_rodape_cm'])
    sec = doc.sections[0]

    # Estilos-base.
    normal = doc.styles['Normal']
    set_font(normal, font_name, typo['corpo_pt'])
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Cm(typo['corpo_recuo_primeira_linha_cm'])
    normal.paragraph_format.space_after = Pt(typo['corpo_espaco_depois_pt'])
    normal.paragraph_format.line_spacing = typo['corpo_entrelinhas']

    if 'Body Text' in doc.styles:
        body = doc.styles['Body Text']
        set_font(body, font_name, typo['corpo_pt'])
        body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        body.paragraph_format.first_line_indent = Cm(typo['corpo_recuo_primeira_linha_cm'])
        body.paragraph_format.space_after = Pt(typo['corpo_espaco_depois_pt'])
        body.paragraph_format.line_spacing = typo['corpo_entrelinhas']

    title = doc.styles['Title']
    set_font(title, font_name, titles['titulo_pt'], True, colors['titulo'])
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(titles['titulo_espaco_depois_pt'])
    title.paragraph_format.line_spacing = 1.0
    pborder_bottom(title, colors['linha_titulo'])

    heading_spec = {
        'Heading 1': ('heading1_pt', 'heading1_antes_pt', 'heading1_depois_pt', 'heading1'),
        'Heading 2': ('heading2_pt', 'heading2_antes_pt', 'heading2_depois_pt', 'heading2'),
        'Heading 3': ('heading3_pt', 'heading3_antes_pt', 'heading3_depois_pt', 'heading3'),
    }
    for nm, (sz_key, bef_key, aft_key, color_key) in heading_spec.items():
        s = doc.styles[nm]
        set_font(s, font_name, titles[sz_key], True, colors[color_key])
        s.paragraph_format.left_indent = Cm(0)
        s.paragraph_format.right_indent = Cm(0)
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.space_before = Pt(titles[bef_key])
        s.paragraph_format.space_after = Pt(titles[aft_key])
        s.paragraph_format.keep_with_next = True

    # Pandoc frequentemente usa Body Text; convertemos ao estilo canônico.
    for p in doc.paragraphs:
        if p.style.name == 'Body Text':
            p.style = normal

    # Cabeçalho e rodapé.
    header = sec.header
    for p in header.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    hp = header.paragraphs[0]
    hp.clear()
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = hp.add_run(document['cabecalho_texto'])
    r.font.name = font_name
    r.font.size = Pt(typo['cabecalho_pt'])
    r.font.color.rgb = RGBColor(*hex_rgb(colors['cabecalho']))

    footer = sec.footer
    for p in footer.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    fp = footer.paragraphs[0]
    fp.clear()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run(document['rodape_prefixo'])
    r.font.name = font_name
    r.font.size = Pt(typo['rodape_pt'])
    fld = OxmlElement('w:fldSimple')
    fld.set(qn('w:instr'), 'PAGE')
    fp._p.append(fld)

    # Título institucional inserido antes do primeiro parágrafo do Org.
    first = doc.paragraphs[0]
    newp = OxmlElement('w:p')
    first._p.addprevious(newp)
    tp = Paragraph(newp, first._parent)
    tp.style = title
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = tp.add_run(document['titulo'])
    tr.font.name = font_name

    # Primeiro parágrafo do Org = subtítulo da capa.
    first.style = normal
    first.alignment = WD_ALIGN_PARAGRAPH.CENTER
    first.paragraph_format.first_line_indent = Cm(0)
    first.paragraph_format.space_after = Pt(6)
    first.paragraph_format.line_spacing = 1.0
    for r in first.runs:
        r.font.name = font_name
        r.font.size = Pt(typo['subtitulo_capa_pt'])
        r.bold = True
        r.font.color.rgb = RGBColor(70, 70, 70)

    # Tabela de identificação.
    if not doc.tables:
        raise SystemExit('Tabela de identificação não encontrada após Pandoc.')
    t = doc.tables[0]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    pr = t._tbl.tblPr
    layout = pr.find(qn('w:tblLayout'))
    if layout is None:
        layout = OxmlElement('w:tblLayout')
        pr.append(layout)
    layout.set(qn('w:type'), 'fixed')
    widths = [Cm(ficha['coluna_rotulo_cm']), Cm(ficha['coluna_valor_cm'])]
    for row in t.rows:
        for ci, c in enumerate(row.cells):
            c.width = widths[ci]
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr = c._tc.get_or_add_tcPr()
            tcW = tcPr.find(qn('w:tcW'))
            if tcW is None:
                tcW = OxmlElement('w:tcW')
                tcPr.append(tcW)
            tcW.set(qn('w:type'), 'dxa')
            tcW.set(qn('w:w'), str(round(widths[ci].emu / 914400 * 1440 / 2.54)))
            shd = tcPr.find(qn('w:shd'))
            if shd is None:
                shd = OxmlElement('w:shd')
                tcPr.append(shd)
            shd.set(qn('w:fill'), colors['ficha_fundo'])
            borders = tcPr.find(qn('w:tcBorders'))
            if borders is None:
                borders = OxmlElement('w:tcBorders')
                tcPr.append(borders)
            for edge in ('top', 'left', 'bottom', 'right'):
                e = borders.find(qn('w:' + edge))
                if e is None:
                    e = OxmlElement('w:' + edge)
                    borders.append(e)
                e.set(qn('w:val'), 'single')
                e.set(qn('w:sz'), '2')
                e.set(qn('w:color'), colors['ficha_borda'])
            for p in c.paragraphs:
                p.style = normal
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                for r in p.runs:
                    r.font.name = font_name
                    r.font.size = Pt(ficha['fonte_pt'])
                    r.bold = (ci == 0)

    # Tabelas de conteúdo do relatório simplificado: largura útil A4 = 16 cm.
    # Mantém a identidade visual do relatório final, mas evita colunas comprimidas
    # nas três tabelas substantivas do modelo simplificado.
    usable_cm = page['margem_esquerda_cm']
    usable_cm = 21.0 - page['margem_esquerda_cm'] - page['margem_direita_cm']
    content_specs = {
        ('Pergunta', 'Anchor code', 'Códigos', 'Temas'): [2.1, 9.3, 2.3, 2.3],
        ('Trecho selecionado', 'Código', 'Description'): [6.5, 3.5, 6.0],
        ('Anchor code', 'Temas finais'): [4.2, 11.8],
    }
    for ct in doc.tables[1:]:
        headers = tuple(c.text.strip() for c in ct.rows[0].cells) if ct.rows else tuple()
        widths_cm = content_specs.get(headers)
        if widths_cm is None:
            n = len(headers) or len(ct.columns)
            widths_cm = [usable_cm / max(n, 1)] * max(n, 1)
        ct.alignment = WD_TABLE_ALIGNMENT.CENTER
        ct.autofit = False
        tblPr = ct._tbl.tblPr
        layout = tblPr.find(qn('w:tblLayout'))
        if layout is None:
            layout = OxmlElement('w:tblLayout')
            tblPr.append(layout)
        layout.set(qn('w:type'), 'fixed')
        for ri, row in enumerate(ct.rows):
            # Evita dividir uma linha da tabela entre duas páginas. Isso mantém
            # tema e lista de códigos juntos quando a tabela cruza uma página.
            trPr = row._tr.get_or_add_trPr()
            if trPr.find(qn('w:cantSplit')) is None:
                trPr.append(OxmlElement('w:cantSplit'))
            if ri == 0 and trPr.find(qn('w:tblHeader')) is None:
                trPr.append(OxmlElement('w:tblHeader'))
            for ci, cell in enumerate(row.cells):
                width = Cm(widths_cm[min(ci, len(widths_cm)-1)])
                cell.width = width
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
                tcPr = cell._tc.get_or_add_tcPr()
                tcW = tcPr.find(qn('w:tcW'))
                if tcW is None:
                    tcW = OxmlElement('w:tcW')
                    tcPr.append(tcW)
                tcW.set(qn('w:type'), 'dxa')
                tcW.set(qn('w:w'), str(round(width.emu / 914400 * 1440 / 2.54)))
                borders = tcPr.find(qn('w:tcBorders'))
                if borders is None:
                    borders = OxmlElement('w:tcBorders')
                    tcPr.append(borders)
                for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
                    e = borders.find(qn('w:' + edge))
                    if e is None:
                        e = OxmlElement('w:' + edge)
                        borders.append(e)
                    e.set(qn('w:val'), 'single')
                    e.set(qn('w:sz'), '2')
                    e.set(qn('w:color'), colors['ficha_borda'])
                if ri == 0:
                    shd = tcPr.find(qn('w:shd'))
                    if shd is None:
                        shd = OxmlElement('w:shd')
                        tcPr.append(shd)
                    shd.set(qn('w:fill'), 'EAF0F7')
                for cp in cell.paragraphs:
                    cp.style = normal
                    cp.paragraph_format.first_line_indent = Cm(0)
                    cp.paragraph_format.space_before = Pt(0)
                    cp.paragraph_format.space_after = Pt(0)
                    cp.paragraph_format.line_spacing = 1.0
                    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER if (headers in content_specs and ci >= 2 and headers == ('Pergunta','Anchor code','Códigos','Temas')) else WD_ALIGN_PARAGRAPH.LEFT
                    for rr in cp.runs:
                        rr.font.name = font_name
                        rr.font.size = Pt(7.6 if len(headers) >= 3 else 8.0)
                        rr.bold = (ri == 0)

    # Figuras.
    image_sizes = [tuple(x) for x in figuras['tamanhos_cm']]
    page_break_before = set(figuras['quebra_pagina_antes'])
    imgidx = 0
    for p in doc.paragraphs:
        drawings = list(p._p.iter(qn('w:drawing')))
        if not drawings:
            continue
        if imgidx >= len(image_sizes):
            break
        current_idx = imgidx + 1
        wcm, hcm = image_sizes[imgidx]
        imgidx += 1
        if current_idx in page_break_before:
            p.paragraph_format.page_break_before = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(figuras['espaco_antes_pt'])
        p.paragraph_format.space_after = Pt(figuras['espaco_depois_pt'])
        for d in drawings:
            for e in d.iter(qn('wp:extent')):
                e.set('cx', str(Cm(wcm).emu))
                e.set('cy', str(Cm(hcm).emu))
            for e in d.iter(qn('a:ext')):
                if e.get('cx') is not None and e.get('cy') is not None:
                    e.set('cx', str(Cm(wcm).emu))
                    e.set('cy', str(Cm(hcm).emu))
    if imgidx != len(image_sizes):
        raise SystemExit(f'Esperava {len(image_sizes)} figuras; encontrei {imgidx}.')

    # Parágrafos especiais.
    seen_refs = False
    cases = tuple(excerpts['casos'])
    tema_prefixos = tuple(temas['prefixos'])
    break_prefixes = tuple(breaks['heading1_prefixos'])

    for p in doc.paragraphs:
        txt = p.text.strip()

        if p.style.name == 'Heading 1' and txt.startswith(break_prefixes):
            p.paragraph_format.page_break_before = True
        if p.style.name == 'Heading 1' and txt.startswith('5. REFERÊNCIAS'):
            seen_refs = True
            continue
        if p.style.name.startswith('Heading'):
            p.paragraph_format.left_indent = Cm(0)
            p.paragraph_format.right_indent = Cm(0)
            p.paragraph_format.first_line_indent = Cm(0)
            continue

        if seen_refs and txt:
            p.style = normal
            p.paragraph_format.left_indent = Cm(refs['recuo_esquerdo_cm'])
            p.paragraph_format.right_indent = Cm(0)
            p.paragraph_format.first_line_indent = Cm(refs['recuo_primeira_linha_cm'])
            p.paragraph_format.space_after = Pt(refs['espaco_depois_pt'])
            p.paragraph_format.line_spacing = refs['entrelinhas']

        elif txt and ':' in txt and any(
            txt.startswith(nome + ',') or txt.startswith(nome + ':')
            for nome in cases
        ):
            p.style = normal
            p.paragraph_format.left_indent = Cm(excerpts['recuo_esquerdo_cm'])
            p.paragraph_format.right_indent = Cm(excerpts['recuo_direito_cm'])
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(excerpts['espaco_depois_pt'])
            p.paragraph_format.line_spacing = excerpts['entrelinhas']

            speaker, quote = txt.split(':', 1)
            quote = quote.strip()
            quote = re.sub(r'^/(["“])', r'\1', quote)
            quote = re.sub(r'(["”])/$', r'\1', quote)
            replace_paragraph_runs_with_quote(p, speaker, quote, font_name, excerpts['fonte_pt'])

        elif txt.startswith('Leitura da Figura'):
            p.paragraph_format.first_line_indent = Cm(figreads['recuo_primeira_linha_cm'])
            p.paragraph_format.space_after = Pt(figreads['espaco_depois_pt'])

        elif txt.startswith(tema_prefixos):
            p.style = normal
            p.paragraph_format.left_indent = Cm(0)
            p.paragraph_format.right_indent = Cm(0)
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(temas['espaco_depois_pt'])

    # Compatibilidade tipográfica das runs.
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.name is None:
                r.font.name = font_name

    # Idioma e hifenização automáticos em português brasileiro.
    configure_language_and_hyphenation(doc, lang)

    # Metadados + rastreabilidade do perfil.
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    doc.core_properties.title = document['titulo']
    doc.core_properties.author = document['autor']
    doc.core_properties.subject = document['assunto']
    doc.core_properties.comments = f"Perfil FGV/ABNT: {profile_path.name}; SHA-256: {profile_sha}"

    doc.save(dst)
    return profile_sha


def main():
    ap = argparse.ArgumentParser(description='Aplica o perfil FGV/ABNT a um DOCX intermediário.')
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('profile', type=Path)
    args = ap.parse_args()
    sha = apply_profile(args.src, args.dst, args.profile)
    print(f'PASS: perfil aplicado: {args.profile}')
    print(f'PASS: perfil SHA-256: {sha}')
    print(f'PASS: DOCX pós-processado: {args.dst}')


if __name__ == '__main__':
    main()

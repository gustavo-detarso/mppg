#!/usr/bin/env python3
"""EP04 — apresentação a partir do Org usando o PPTX de referência do projeto.

Implementação sem dependências Python externas: usa apenas biblioteca padrão para
validar o Org/PPTX, copia o PPTX de referência como PPTX final e usa
LibreOffice + pdftoppm + pdflatex para gerar o PDF Beamer visualmente fiel ao PPTX.
"""
from pathlib import Path
import json, re, shutil, subprocess, zipfile, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
ORG = ROOT / 'EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.org'
REF_PPTX = ROOT / 'modelo' / 'EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL_REFERENCIA.pptx'
OUT_PPTX = ROOT / 'EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL.pptx'
PDF_FROM_PPTX = ROOT / 'EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL.pdf'
BEAMER_PDF = ROOT / 'EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.pdf'
RENDER_DIR = ROOT / 'render'
BEAMER_DIR = ROOT / 'beamer_render'


def read_json_from_org(path: Path):
    txt = path.read_text(encoding='utf-8')
    m = re.search(r"#\+begin_src\s+json[^\n]*\n(.*?)\n#\+end_src", txt, re.I | re.S)
    if not m:
        raise RuntimeError(f'Bloco JSON não encontrado em {path}')
    return json.loads(m.group(1))


def extract_pptx_texts(path: Path):
    """Extrai textos dos slides do PPTX com zipfile+xml.etree, sem python-pptx."""
    texts = []
    with zipfile.ZipFile(path) as z:
        slide_names = sorted(
            [n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', n)],
            key=lambda n: int(re.search(r'slide(\d+)\.xml$', n).group(1))
        )
        for name in slide_names:
            root = ET.fromstring(z.read(name))
            parts = []
            # DrawingML text nodes
            for node in root.iter():
                if node.tag.endswith('}t') and node.text:
                    parts.append(node.text)
            texts.append('\n'.join(parts))
    return texts


def validate_reference_against_org(data):
    texts = extract_pptx_texts(REF_PPTX)
    if len(texts) != 6:
        raise RuntimeError(f'A apresentação de referência deve ter 6 slides; encontrou {len(texts)}')
    expected_titles = [s.get('title','').split('\n')[0] for s in data['slides']]
    expected_titles[0] = 'Licenciamento'
    misses = []
    for idx, title in enumerate(expected_titles):
        if title and title not in texts[idx]:
            misses.append((idx+1, title))
    if misses:
        raise RuntimeError('PPTX de referência não confere com o Org nos títulos: ' + str(misses))
    required = ['QualCoder 3.8.2', 'temas', 'Pressões sobre ritmo, decisão e legitimidade', 'Uso de IA generativa']
    full = '\n'.join(texts)
    absent = [x for x in required if x not in full]
    if absent:
        raise RuntimeError('PPTX de referência não contém termos esperados: ' + str(absent))
    print('PASS: Org e PPTX de referência conferem por títulos e termos-chave')


def convert_pptx_to_pdf():
    """Converte o PPTX em PDF com perfil LibreOffice isolado.

    O perfil isolado evita o caso comum em desktops Linux no qual uma instância
    gráfica do LibreOffice já aberta recebe a solicitação, retorna código 0 e
    não cria o PDF no diretório solicitado. A conversão é feita em diretório
    temporário e somente depois o PDF validado é copiado para o destino final.
    """
    import tempfile

    if PDF_FROM_PPTX.exists():
        PDF_FROM_PPTX.unlink()

    with tempfile.TemporaryDirectory(prefix='ep04_lo_out_') as out_tmp, \
         tempfile.TemporaryDirectory(prefix='ep04_lo_profile_') as profile_tmp:
        out_dir = Path(out_tmp)
        profile_uri = Path(profile_tmp).resolve().as_uri()
        cmd = [
            'libreoffice',
            f'-env:UserInstallation={profile_uri}',
            '--headless', '--nologo', '--nodefault', '--nofirststartwizard',
            '--convert-to', 'pdf:impress_pdf_Export',
            '--outdir', str(out_dir),
            str(OUT_PPTX),
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        lo_log = ROOT / 'libreoffice_pptx_pdf.log'
        lo_log.write_text(
            'COMMAND: ' + ' '.join(cmd) + '\n\nSTDOUT:\n' + proc.stdout +
            '\nSTDERR:\n' + proc.stderr + f'\nRETURN_CODE: {proc.returncode}\n',
            encoding='utf-8'
        )
        if proc.returncode != 0:
            raise RuntimeError(
                'LibreOffice falhou ao converter o PPTX. '
                f'Veja {lo_log}.\nSTDOUT: {proc.stdout.strip()}\nSTDERR: {proc.stderr.strip()}'
            )

        expected = out_dir / f'{OUT_PPTX.stem}.pdf'
        candidates = sorted(out_dir.glob('*.pdf'))
        if expected.exists():
            generated = expected
        elif len(candidates) == 1:
            generated = candidates[0]
        else:
            listing = ', '.join(x.name for x in out_dir.iterdir()) or '<vazio>'
            raise RuntimeError(
                'LibreOffice retornou sucesso, mas o PDF não apareceu no diretório temporário. '
                f'Conteúdo: {listing}. Veja {lo_log}.'
            )
        if generated.stat().st_size == 0:
            raise RuntimeError(f'PDF convertido está vazio: {generated}')
        shutil.copy2(generated, PDF_FROM_PPTX)

    if not PDF_FROM_PPTX.exists() or PDF_FROM_PPTX.stat().st_size == 0:
        raise RuntimeError('PDF convertido do PPTX não foi materializado no destino final')


def _ppm_bbox_nonwhite(ppm_path: Path, threshold=248):
    """Return (w,h,bbox) for binary P6 PPM; bbox=(minx,miny,maxx,maxy)."""
    data = ppm_path.read_bytes()
    pos = 0
    def read_token():
        nonlocal pos
        while pos < len(data) and chr(data[pos]).isspace():
            pos += 1
        if pos < len(data) and data[pos:pos+1] == b'#':
            while pos < len(data) and data[pos:pos+1] not in (b'\n', b'\r'):
                pos += 1
            return read_token()
        start = pos
        while pos < len(data) and not chr(data[pos]).isspace():
            pos += 1
        return data[start:pos].decode('ascii')
    magic = read_token()
    if magic != 'P6':
        raise RuntimeError(f'PPM inesperado ({magic}) em {ppm_path}')
    w = int(read_token()); h = int(read_token()); maxv = int(read_token())
    if maxv != 255:
        raise RuntimeError(f'PPM com maxval inesperado {maxv} em {ppm_path}')
    # Após o maxval, o PPM P6 usa exatamente um byte de whitespace antes dos pixels.
    # Não pular todos os whitespaces, porque valores binários de pixel podem coincidir
    # com bytes de whitespace (10, 13, 32 etc.).
    if pos < len(data) and chr(data[pos]).isspace():
        pos += 1
    pix = data[pos:]
    if len(pix) < w*h*3:
        raise RuntimeError(f'PPM truncado em {ppm_path}')
    minx, miny = w, h
    maxx = maxy = -1
    for y in range(h):
        row = y*w*3
        for x in range(w):
            i = row + x*3
            r,g,b = pix[i], pix[i+1], pix[i+2]
            if not (r >= threshold and g >= threshold and b >= threshold):
                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y
                if y > maxy: maxy = y
    return w, h, (minx, miny, maxx, maxy)


def _render_pdf_page_ppm(pdf: Path, page: int, out_dir: Path, prefix: str):
    out_prefix = out_dir / prefix
    subprocess.run([
        'pdftoppm', '-r', '72', '-f', str(page), '-l', str(page),
        str(pdf), str(out_prefix)
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    candidates = sorted(out_dir.glob(f'{prefix}-*.ppm'))
    if not candidates:
        candidates = sorted(out_dir.glob(f'{prefix}*.ppm'))
    if not candidates:
        raise RuntimeError(f'pdftoppm não gerou PPM para {pdf} página {page}')
    return candidates[0]


def _assert_full_bleed(pdf: Path):
    """Gate: capa/final não podem ter margem branca lateral gerada pelo Beamer."""
    gate_dir = BEAMER_DIR / 'full_bleed_gate'
    if gate_dir.exists():
        shutil.rmtree(gate_dir)
    gate_dir.mkdir(parents=True)
    failures = []
    for page in (1, 6):
        ppm = _render_pdf_page_ppm(pdf, page, gate_dir, f'page{page}')
        w, h, (minx, miny, maxx, maxy) = _ppm_bbox_nonwhite(ppm)
        # tolerância pequena para antialias/bordas do renderizador.
        if minx > 2 or miny > 2 or maxx < w-3 or maxy < h-3:
            failures.append((page, w, h, (minx, miny, maxx, maxy)))
    if failures:
        raise RuntimeError(
            'PDF Beamer não está em full bleed; provável margem interna de Beamer. '
            f'Falhas: {failures}'
        )
    print('PASS: PDF Beamer sem margem deslocada (full bleed nas páginas 1 e 6)')


def make_beamer_pdf_from_pptx_pdf():
    if RENDER_DIR.exists():
        shutil.rmtree(RENDER_DIR)
    if BEAMER_DIR.exists():
        shutil.rmtree(BEAMER_DIR)
    RENDER_DIR.mkdir(parents=True)
    BEAMER_DIR.mkdir(parents=True)
    subprocess.run(['pdftoppm', '-png', '-r', '144', str(PDF_FROM_PPTX), str(RENDER_DIR / 'slide')], check=True)
    pngs = sorted(RENDER_DIR.glob('slide-*.png'))
    if len(pngs) != 6:
        raise RuntimeError(f'Esperava 6 imagens renderizadas, encontrei {len(pngs)}')

    # Montagem leve opcional sem dependências Python externas; só se ImageMagick estiver disponível.
    magick = shutil.which('montage') or shutil.which('magick')
    if magick:
        try:
            if Path(magick).name == 'magick':
                subprocess.run(['magick', 'montage', *map(str, pngs), '-tile', '2x3', '-geometry', '320x180+8+24', str(RENDER_DIR / 'montage.png')], check=False)
            else:
                subprocess.run([magick, *map(str, pngs), '-tile', '2x3', '-geometry', '320x180+8+24', str(RENDER_DIR / 'montage.png')], check=False)
        except Exception:
            pass

    tex_lines = [
        r'\documentclass[aspectratio=169]{beamer}',
        r'\geometry{paperwidth=10in,paperheight=5.625in,margin=0pt}',
        r'\usepackage{graphicx}',
        r'\setbeamertemplate{navigation symbols}{}',
        r'\setbeamertemplate{footline}{}',
        r'\setbeamertemplate{headline}{}',
        r'\setbeamersize{text margin left=0pt,text margin right=0pt}',
        r'\begin{document}'
    ]
    for idx, p in enumerate(pngs, 1):
        dest = BEAMER_DIR / f'slide_{idx:02d}.png'
        shutil.copy2(p, dest)
        # Usar background template, não includegraphics dentro da área de texto.
        # Isso evita a margem interna padrão do Beamer e garante full bleed.
        tex_lines += [
            rf'\usebackgroundtemplate{{\includegraphics[width=\paperwidth,height=\paperheight]{{slide_{idx:02d}.png}}}}',
            r'\begin{frame}[plain]',
            r'\end{frame}',
            r'\usebackgroundtemplate{}'
        ]
    tex_lines.append(r'\end{document}')
    tex = BEAMER_DIR / 'EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.tex'
    tex.write_text('\n'.join(tex_lines), encoding='utf-8')
    proc = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex.name], cwd=BEAMER_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (ROOT / 'compilacao_beamer.log').write_text(proc.stdout + '\nSTDERR:\n' + proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError('pdflatex falhou. Veja compilacao_beamer.log')
    gen = BEAMER_DIR / 'EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.pdf'
    if not gen.exists():
        raise RuntimeError('PDF Beamer não foi gerado')
    shutil.copy2(gen, BEAMER_PDF)
    _assert_full_bleed(BEAMER_PDF)

def main():
    if not ORG.exists():
        raise FileNotFoundError(ORG)
    if not REF_PPTX.exists():
        raise FileNotFoundError(REF_PPTX)
    data = read_json_from_org(ORG)
    validate_reference_against_org(data)
    shutil.copy2(REF_PPTX, OUT_PPTX)
    print(f'PASS: PPTX gerado a partir da referência do projeto: {OUT_PPTX}')
    convert_pptx_to_pdf()
    print(f'PASS: PDF derivado do PPTX: {PDF_FROM_PPTX}')
    make_beamer_pdf_from_pptx_pdf()
    print(f'PASS: PDF Beamer visualmente fiel ao PPTX: {BEAMER_PDF}')


if __name__ == '__main__':
    main()

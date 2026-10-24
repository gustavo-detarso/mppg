#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_BUNDLE="$(cd "$HERE/.." && pwd)"
OUT_DIR="$ROOT_BUNDLE/05_ENTREGAS"
TOOLS="$HERE/tools"
GEN="$TOOLS/pdf_para_docx_fiel.py"
SKELETON="$TOOLS/DOCX_FIDELIDADE_BASE.docx"
DPI="${MPPG_DOCX_DPI:-200}"

MAIN_STEM="REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO"
GLOSS_STEM="GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO"
MAIN_PDF="$HERE/${MAIN_STEM}.pdf"
MAIN_PDF_DELIVERY="$OUT_DIR/${MAIN_STEM}.pdf"
MAIN_DOCX="$OUT_DIR/${MAIN_STEM}.docx"
GLOSS_PDF="$OUT_DIR/${GLOSS_STEM}.pdf"
GLOSS_DOCX="$OUT_DIR/${GLOSS_STEM}.docx"

mkdir -p "$OUT_DIR"
for cmd in python3 pdfinfo pdftoppm; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERRO: comando obrigatório ausente: $cmd"; exit 2; }
done
[[ -f "$GEN" ]] || { echo "ERRO: gerador DOCX ausente: $GEN"; exit 3; }
[[ -f "$SKELETON" ]] || { echo "ERRO: base DOCX ausente: $SKELETON"; exit 3; }
[[ -s "$MAIN_PDF" ]] || { echo "ERRO: PDF principal não existe. Rode ./compilar.sh antes."; exit 4; }
[[ -s "$GLOSS_PDF" ]] || { echo "ERRO: PDF do glossário não existe. Rode ./glossario/compilar_glossario.sh antes."; exit 4; }

printf '=== DOCX de fidelidade visual ===\n'
printf 'DPI: %s\n' "$DPI"
python3 "$GEN" "$MAIN_PDF" "$MAIN_DOCX" --skeleton "$SKELETON" --dpi "$DPI"
python3 "$GEN" "$GLOSS_PDF" "$GLOSS_DOCX" --skeleton "$SKELETON" --dpi "$DPI"

# Gates estruturais simples e determinísticos.
python3 - "$MAIN_PDF" "$MAIN_DOCX" "$GLOSS_PDF" "$GLOSS_DOCX" <<'PY'
from pathlib import Path
import re, subprocess, sys, zipfile

def pages_pdf(p):
    out=subprocess.check_output(['pdfinfo', str(p)], text=True)
    m=re.search(r'^Pages:\s*(\d+)', out, re.M)
    if not m: raise SystemExit(f'ERRO: sem contagem de páginas em {p}')
    return int(m.group(1))

def pages_docx(p):
    with zipfile.ZipFile(p) as z:
        media=[n for n in z.namelist() if n.startswith('word/media/page-') and n.endswith('.png')]
        bad=z.testzip()
        if bad: raise SystemExit(f'ERRO: ZIP DOCX corrompido em {p}: {bad}')
    return len(media)

for pdf,docx in [(Path(sys.argv[1]),Path(sys.argv[2])),(Path(sys.argv[3]),Path(sys.argv[4]))]:
    a,b=pages_pdf(pdf),pages_docx(docx)
    if a!=b: raise SystemExit(f'ERRO: divergência de páginas PDF/DOCX em {docx.name}: {a} != {b}')
    print(f'GATE_DOCX_PAGE_PARITY=PASS {docx.name}: {a} páginas')
PY

# Materializa também o PDF principal em 05_ENTREGAS.
# A cópia só é atualizada depois de os DOCX passarem pelos gates acima, garantindo
# que o conjunto de entrega corresponda ao mesmo PDF validado.
install -m 0644 "$MAIN_PDF" "$MAIN_PDF_DELIVERY.tmp"
mv -f "$MAIN_PDF_DELIVERY.tmp" "$MAIN_PDF_DELIVERY"
cmp -s "$MAIN_PDF" "$MAIN_PDF_DELIVERY" || { echo "ERRO: cópia do PDF principal em 05_ENTREGAS diverge do PDF canônico."; exit 5; }
echo "GATE_MAIN_PDF_DELIVERY_COPY=PASS $MAIN_PDF_DELIVERY"

printf '\nPASS: pacote de entregas atualizado em %s\n' "$OUT_DIR"
printf '  - %s\n' "$MAIN_PDF_DELIVERY" "$MAIN_DOCX" "$GLOSS_PDF" "$GLOSS_DOCX"
printf 'OBS: são DOCX de fidelidade visual ao PDF; cada página é uma imagem integral e, por isso, o texto não é editável como texto Word.\n'

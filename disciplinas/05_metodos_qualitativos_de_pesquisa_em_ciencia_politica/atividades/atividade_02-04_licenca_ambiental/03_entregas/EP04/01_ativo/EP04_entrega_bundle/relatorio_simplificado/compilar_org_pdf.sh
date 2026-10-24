#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ORG_FILE="${ORG_FILE:-$ROOT/EP04_relatorio_final_simplificado_licenciamento_ambiental.org}"
MEDIA_DIR="${MEDIA_DIR:-$ROOT/media}"
MODEL_DOCX="${MODEL_DOCX:-$ROOT/modelo/FGV_ABNT_EP04_reference.docx}"
PROFILE_FILE="${PROFILE_FILE:-$ROOT/perfil/fgv_abnt.toml}"
APPLY_PROFILE_PY="$ROOT/perfil/aplicar_fgv_abnt.py"
GENERATE_STY_PY="$ROOT/perfil/gerar_fgv_abnt_sty.py"
GENERATED_STY="$ROOT/perfil/fgv_abnt_generated.sty"

STEM="$(basename "${ORG_FILE%.org}")"
OUTPUT_DOCX="$ROOT/$STEM.docx"
OUTPUT_PDF="$ROOT/$STEM.pdf"
LOG_DIR="$ROOT/logs"
BUILD_ROOT="$ROOT/.build_org_pdf"
STAMP="$(date '+%Y%m%d_%H%M%S')"
BUILD_DIR="$BUILD_ROOT/$STAMP"
LOG="$LOG_DIR/compilacao_org_pdf_${STAMP}.log"
CANONICAL_LOG="$LOG_DIR/compilacao_org_pdf.log"
mkdir -p "$LOG_DIR" "$BUILD_DIR"
exec > >(tee -a "$LOG") 2>&1

cleanup() {
  rc=$?
  if [[ $rc -eq 0 ]]; then
    rm -rf "$BUILD_DIR"
    rmdir "$BUILD_ROOT" 2>/dev/null || true
  else
    echo
    echo "A compilação falhou. Diagnóstico preservado em: $BUILD_DIR"
  fi
  exit $rc
}
trap cleanup EXIT

need(){ command -v "$1" >/dev/null 2>&1 || { echo "ERRO: falta $1"; exit 3; }; }
for c in pandoc libreoffice pdfinfo pdftotext sha256sum; do need "$c"; done

# Python + python-docx + tomllib. Em ambientes com pyenv, o python3 do PATH
# pode não enxergar pacotes APT; por isso procuramos explicitamente um Python
# que consiga importar ambas as dependências.
PYTHON_BIN="${PYTHON_BIN:-}"
python_ok() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  "$candidate" - <<'PY' >/dev/null 2>&1
import docx, tomllib, sys
PY
}

if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(readlink -f "$PYTHON_BIN" 2>/dev/null || printf '%s' "$PYTHON_BIN")"
  python_ok "$PYTHON_BIN" || { echo "ERRO: PYTHON_BIN não consegue importar python-docx/tomllib: $PYTHON_BIN"; exit 4; }
else
  PY_CANDIDATES=()
  command -v python3 >/dev/null 2>&1 && PY_CANDIDATES+=("$(command -v python3)")
  [[ -x /usr/bin/python3 ]] && PY_CANDIDATES+=("/usr/bin/python3")
  command -v python >/dev/null 2>&1 && PY_CANDIDATES+=("$(command -v python)")
  for candidate in "${PY_CANDIDATES[@]}"; do
    if python_ok "$candidate"; then PYTHON_BIN="$candidate"; break; fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERRO: nenhum Python disponível importa python-docx e tomllib."
  echo "Em Debian 12, tente: sudo apt install python3-docx"
  exit 4
fi

echo "PASS: Python selecionado: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import docx, sys
print(f"PASS: python-docx {docx.__version__} em {sys.executable}")
PY

for f in "$ORG_FILE" "$MODEL_DOCX" "$PROFILE_FILE" "$APPLY_PROFILE_PY" "$GENERATE_STY_PY"; do
  [[ -f "$f" ]] || { echo "ERRO: arquivo obrigatório ausente: $f"; exit 5; }
done
mkdir -p "$MEDIA_DIR"

PROFILE_SHA256="$(sha256sum "$PROFILE_FILE" | awk '{print $1}')"
PROFILE_NAME="$($PYTHON_BIN - "$PROFILE_FILE" <<'PY'
import sys,tomllib
from pathlib import Path
with Path(sys.argv[1]).open('rb') as f: p=tomllib.load(f)
print(p.get('perfil',{}).get('nome','FGV/ABNT'))
PY
)"

# Hifenização pt-BR no PDF: o DOCX é marcado como pt-BR e autoHyphenation.
# Para o LibreOffice efetivamente hifenizar na conversão, os padrões pt-BR devem estar instalados.
HYPHEN_PTBR_FOUND=0
for h in /usr/share/hyphen/hyph_pt_BR.dic /usr/share/myspell/dicts/hyph_pt_BR.dic /usr/share/libreoffice/share/extensions/*/hyph_pt_BR.dic; do
  [[ -f "$h" ]] && HYPHEN_PTBR_FOUND=1 && break
done
REQUIRE_PTBR_HYPHEN="${REQUIRE_PTBR_HYPHEN:-1}"
if [[ "$HYPHEN_PTBR_FOUND" -eq 0 ]]; then
  echo "ERRO: padrões de hifenização pt-BR do LibreOffice não detectados."
  echo "No Debian 12, instale: sudo apt update && sudo apt install hyphen-pt-br"
  echo "O DOCX é configurado como pt-BR e autoHyphenation, mas a conversão para PDF"
  echo "precisa dos padrões do LibreOffice para materializar a hifenização."
  if [[ "$REQUIRE_PTBR_HYPHEN" == "1" ]]; then
    echo "Para compilar apenas para diagnóstico sem esse gate: REQUIRE_PTBR_HYPHEN=0 ./compilar_org_pdf.sh"
    exit 7
  fi
  echo "AVISO: gate de hifenização ignorado por REQUIRE_PTBR_HYPHEN=0."
else
  echo "PASS: padrões de hifenização pt-BR detectados para o LibreOffice."
fi

echo "============================================================"
echo "EP04 simplificado | Org -> DOCX (mesmo perfil FGV/ABNT do relatório final) -> PDF"
echo "ORG:      $ORG_FILE"
echo "Modelo:   $MODEL_DOCX"
echo "Perfil:   $PROFILE_FILE"
echo "Perfil:   $PROFILE_NAME"
echo "SHA-256:  $PROFILE_SHA256"
echo "Log:      $LOG"
echo "============================================================"

# Materializa também um .sty derivado do MESMO perfil. O backend corrente usa
# DOCX/LibreOffice, mas o .sty mantém o backend LaTeX sincronizado para uso futuro.
"$PYTHON_BIN" "$GENERATE_STY_PY" "$PROFILE_FILE" "$GENERATED_STY"
STY_SHA256="$(sha256sum "$GENERATED_STY" | awk '{print $1}')"

WORK_ORG="$BUILD_DIR/$STEM.org"
WORK_DOCX="$BUILD_DIR/$STEM.docx"
PATCHED_DOCX="$BUILD_DIR/${STEM}_FGV_ABNT.docx"

# Reconcilia caminhos das imagens sem alterar o Org canônico.
"$PYTHON_BIN" - "$ORG_FILE" "$WORK_ORG" "$MEDIA_DIR" <<'PY'
from pathlib import Path
import re, sys
src,dst,media=map(Path,sys.argv[1:4])
text=src.read_text(encoding='utf-8')
pat=re.compile(r'\[\[([^\]\n]+?\.(?:png|jpe?g))\]\]',re.I)
missing=[]
def repl(m):
    base=Path(re.sub(r'^file:','',m.group(1),flags=re.I)).name
    cand=media/base
    if cand.exists(): return f'[[file:media/{base}]]'
    missing.append(m.group(1)); return m.group(0)
text=pat.sub(repl,text)
if missing:
    raise SystemExit('Imagens ausentes em media/: '+', '.join(sorted(set(missing))))
dst.write_text(text,encoding='utf-8')
PY

pandoc -f org -t docx \
  --reference-doc="$MODEL_DOCX" \
  --resource-path="$ROOT:$MEDIA_DIR" \
  "$WORK_ORG" -o "$WORK_DOCX"

# Aplica todas as regras visuais e ABNT a partir do TOML canônico.
"$PYTHON_BIN" "$APPLY_PROFILE_PY" "$WORK_DOCX" "$PATCHED_DOCX" "$PROFILE_FILE"

OUTDIR="$BUILD_DIR/pdf"
mkdir -p "$OUTDIR"
LO_PROFILE="$BUILD_DIR/lo_profile"
mkdir -p "$LO_PROFILE"
libreoffice -env:UserInstallation="file://$LO_PROFILE" --headless \
  --convert-to pdf --outdir "$OUTDIR" "$PATCHED_DOCX" >/dev/null
BUILT_PDF="$OUTDIR/$(basename "${PATCHED_DOCX%.docx}.pdf")"
[[ -s "$BUILT_PDF" ]] || { echo "ERRO: PDF não criado"; exit 8; }

PAGES="$(pdfinfo "$BUILT_PDF" | awk -F: '/^Pages:/ {gsub(/[[:space:]]/,"",$2);print $2}')"
BYTES="$(pdftotext "$BUILT_PDF" - | wc -c)"

# Publicação final: DOCX + PDF.
cp -f "$PATCHED_DOCX" "$OUTPUT_DOCX"
cp -f "$BUILT_PDF" "$OUTPUT_PDF"
cp -f "$LOG" "$CANONICAL_LOG"

# Validação estrutural do DOCX publicado e conferência do SHA do perfil gravado.
"$PYTHON_BIN" - "$OUTPUT_DOCX" "$PROFILE_SHA256" <<'PYDOCX'
from pathlib import Path
from docx import Document
import sys, zipfile
p=Path(sys.argv[1]); expected=sys.argv[2]
if not p.is_file() or p.stat().st_size < 10000:
    raise SystemExit(f'ERRO: DOCX final ausente ou anormalmente pequeno: {p}')
with zipfile.ZipFile(p) as z:
    if 'word/document.xml' not in z.namelist():
        raise SystemExit('ERRO: DOCX final inválido: word/document.xml ausente')
doc=Document(p)
if len(doc.paragraphs) < 20:
    raise SystemExit('ERRO: DOCX final contém poucos parágrafos; publicação não validada')
if expected not in (doc.core_properties.comments or ''):
    raise SystemExit('ERRO: DOCX não registra o SHA-256 do perfil aplicado')
print(f'PASS: DOCX válido: {p} ({p.stat().st_size} bytes; {len(doc.paragraphs)} parágrafos)')
print('PASS: rastreabilidade do perfil presente no DOCX')
PYDOCX

# Gate específico de identidade visual e hifenização do relatório simplificado.
"$PYTHON_BIN" - "$OUTPUT_DOCX" <<'PYLAYOUT'
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
from zipfile import ZipFile
import sys
p=Path(sys.argv[1]); doc=Document(p); sec=doc.sections[0]; normal=doc.styles['Normal']
def near(a,b,tol=0.08): return abs(a-b) <= tol
assert near(sec.page_width.cm,21.0) and near(sec.page_height.cm,29.7), 'página não é A4'
assert near(sec.top_margin.cm,3.0) and near(sec.left_margin.cm,3.0), 'margens superior/esquerda divergentes'
assert near(sec.right_margin.cm,2.0) and near(sec.bottom_margin.cm,2.0), 'margens direita/inferior divergentes'
assert normal.font.name == 'Times New Roman', f'fonte Normal divergente: {normal.font.name}'
assert normal.font.size and near(normal.font.size.pt,9.5,0.05), f'tamanho Normal divergente: {normal.font.size.pt if normal.font.size else None}'
assert normal.paragraph_format.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY, 'corpo não está justificado'
with ZipFile(p) as z:
    settings=z.read('word/settings.xml').decode('utf-8')
    styles=z.read('word/styles.xml').decode('utf-8')
assert 'autoHyphenation' in settings, 'autoHyphenation ausente'
assert 'pt-BR' in styles, 'idioma pt-BR ausente dos estilos'
print('PASS: layout simplificado = perfil FGV/ABNT do relatório final (A4, margens 3/3/2/2, TNR 9,5, justificado, pt-BR, autoHyphenation)')
PYLAYOUT

DOCX_SIZE="$(stat -c '%s' "$OUTPUT_DOCX")"
PDF_SIZE="$(stat -c '%s' "$OUTPUT_PDF")"
DOCX_SHA256="$(sha256sum "$OUTPUT_DOCX" | awk '{print $1}')"
PDF_SHA256="$(sha256sum "$OUTPUT_PDF" | awk '{print $1}')"

echo "PASS: DOCX criado: $OUTPUT_DOCX"
echo "PASS: PDF criado:  $OUTPUT_PDF"
echo "Páginas PDF: $PAGES (faixa oficial do simplificado: 4–6)"
echo "Texto extraído do PDF: $BYTES bytes"
[[ "$PAGES" -ge 4 && "$PAGES" -le 6 ]] || { echo "ERRO: relatório simplificado fora da faixa de 4–6 páginas"; exit 9; }
echo "Perfil SHA-256: $PROFILE_SHA256"
echo "STY gerado: $GENERATED_STY"
echo "STY SHA-256: $STY_SHA256"
echo "DOCX: $DOCX_SIZE bytes | SHA-256: $DOCX_SHA256"
echo "PDF:  $PDF_SIZE bytes | SHA-256: $PDF_SHA256"
echo "Log canônico: $CANONICAL_LOG"

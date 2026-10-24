#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ORG_FILE="${ORG_FILE:-$ROOT/EP04_relatorio_final_licenciamento_ambiental.org}"
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
[[ -d "$MEDIA_DIR" ]] || { echo "ERRO: media/ não encontrado: $MEDIA_DIR"; exit 6; }

PROFILE_SHA256="$(sha256sum "$PROFILE_FILE" | awk '{print $1}')"
PROFILE_NAME="$($PYTHON_BIN - "$PROFILE_FILE" <<'PY'
import sys,tomllib
from pathlib import Path
with Path(sys.argv[1]).open('rb') as f: p=tomllib.load(f)
print(p.get('perfil',{}).get('nome','FGV/ABNT'))
PY
)"

echo "============================================================"
echo "EP04 | Org -> DOCX (perfil FGV/ABNT) -> PDF"
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

DOCX_SIZE="$(stat -c '%s' "$OUTPUT_DOCX")"
PDF_SIZE="$(stat -c '%s' "$OUTPUT_PDF")"
DOCX_SHA256="$(sha256sum "$OUTPUT_DOCX" | awk '{print $1}')"
PDF_SHA256="$(sha256sum "$OUTPUT_PDF" | awk '{print $1}')"

echo "PASS: DOCX criado: $OUTPUT_DOCX"
echo "PASS: PDF criado:  $OUTPUT_PDF"
echo "Páginas PDF: $PAGES (alvo atual: 12)"
echo "Texto extraído do PDF: $BYTES bytes"
[[ "$PAGES" == "12" ]] || echo "AVISO: paginação divergiu do PDF-alvo de 12 páginas."
echo "Perfil SHA-256: $PROFILE_SHA256"
echo "STY gerado: $GENERATED_STY"
echo "STY SHA-256: $STY_SHA256"
echo "DOCX: $DOCX_SIZE bytes | SHA-256: $DOCX_SHA256"
echo "PDF:  $PDF_SIZE bytes | SHA-256: $PDF_SHA256"
echo "Log canônico: $CANONICAL_LOG"

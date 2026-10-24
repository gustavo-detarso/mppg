#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ORG="${ORG_FILE:-$ROOT/EP04_PUIA_licenciamento_ambiental.org}"
REF="${REF_DOCX:-$ROOT/EP04_PUIA_reference.docx}"
STEM="$(basename "${ORG%.org}")"
OUT_DOCX="$ROOT/$STEM.docx"
OUT_PDF="$ROOT/$STEM.pdf"
BUILD="$ROOT/.build"
LOG="$ROOT/compilacao_puia.log"
rm -rf "$BUILD" && mkdir -p "$BUILD/pdf" "$BUILD/lo"
exec > >(tee "$LOG") 2>&1

for c in libreoffice pdfinfo; do
  command -v "$c" >/dev/null 2>&1 || { echo "ERRO: falta $c"; exit 3; }
done

# Seleciona um Python que realmente tenha python-docx. Isso evita o caso comum
# em máquinas com pyenv, nas quais `python3` aponta para um ambiente que não
# enxerga o pacote Debian instalado para /usr/bin/python3.
PYTHON_BIN="${PYTHON_BIN:-}"
python_ok() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  "$candidate" - <<'PY' >/dev/null 2>&1
import docx, sys
PY
}

if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(readlink -f "$PYTHON_BIN" 2>/dev/null || printf '%s' "$PYTHON_BIN")"
  python_ok "$PYTHON_BIN" || {
    echo "ERRO: PYTHON_BIN não consegue importar python-docx: $PYTHON_BIN"
    echo "Em Debian 12, instale o pacote do sistema com: sudo apt install python3-docx"
    exit 4
  }
else
  PY_CANDIDATES=()
  command -v python3 >/dev/null 2>&1 && PY_CANDIDATES+=("$(command -v python3)")
  [[ -x /usr/bin/python3 ]] && PY_CANDIDATES+=("/usr/bin/python3")
  command -v python >/dev/null 2>&1 && PY_CANDIDATES+=("$(command -v python)")

  for candidate in "${PY_CANDIDATES[@]}"; do
    if python_ok "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERRO: nenhum Python disponível consegue importar python-docx."
  echo "Debian 12: sudo apt update && sudo apt install python3-docx"
  echo "Alternativa em venv/pyenv: python -m pip install python-docx"
  exit 4
fi

[[ -f "$ORG" ]] || { echo "ERRO: fonte Org ausente: $ORG"; exit 4; }
[[ -f "$REF" ]] || { echo "ERRO: DOCX de referência ausente: $REF"; exit 4; }
[[ -f "$ROOT/compilar_puia_org.py" ]] || { echo "ERRO: compilador Python ausente"; exit 4; }

echo "============================================================"
echo "EP04 PUIA | Org -> DOCX -> PDF"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import docx, sys
print(f"python-docx {docx.__version__} | {sys.executable}")
PY
echo "Org:    $ORG"
echo "Modelo: $REF"
echo "Log:    $LOG"
echo "============================================================"

"$PYTHON_BIN" "$ROOT/compilar_puia_org.py" "$ORG" "$REF" "$OUT_DOCX"
libreoffice -env:UserInstallation="file://$BUILD/lo" --headless \
  --convert-to pdf --outdir "$BUILD/pdf" "$OUT_DOCX" >/dev/null
cp -f "$BUILD/pdf/$STEM.pdf" "$OUT_PDF"
PAGES="$(pdfinfo "$OUT_PDF" | awk -F: '/^Pages:/ {gsub(/[[:space:]]/,"",$2);print $2}')"
echo "PASS: PUIA DOCX: $OUT_DOCX"
echo "PASS: PUIA PDF:  $OUT_PDF"
echo "Páginas: $PAGES"
[[ "$PAGES" == "1" ]] || { echo "ERRO: PUIA deve permanecer em uma página"; exit 5; }
rm -rf "$BUILD"

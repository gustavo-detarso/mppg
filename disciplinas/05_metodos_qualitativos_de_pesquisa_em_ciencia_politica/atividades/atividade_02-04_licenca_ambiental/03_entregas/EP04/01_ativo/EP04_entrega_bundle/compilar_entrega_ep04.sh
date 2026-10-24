#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="$ROOT/logs/compilacao_entrega_${STAMP}.log"
mkdir -p "$ROOT/logs"
exec > >(tee "$LOG") 2>&1

echo "============================================================"
echo "EP04 — compilação integrada da entrega"
echo "Root: $ROOT"
echo "Log:  $LOG"
echo "============================================================"

echo
echo "[PREFLIGHT] Python compatível com python-docx"
PYTHON_BIN="${PYTHON_BIN:-}"
python_ok() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  "$candidate" - <<'PYDOCX' >/dev/null 2>&1
import docx, tomllib, sys
PYDOCX
}
if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(readlink -f "$PYTHON_BIN" 2>/dev/null || printf '%s' "$PYTHON_BIN")"
  python_ok "$PYTHON_BIN" || {
    echo "ERRO: PYTHON_BIN não importa python-docx/tomllib: $PYTHON_BIN"
    echo "Debian 12: sudo apt update && sudo apt install python3-docx"
    exit 4
  }
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
  echo "ERRO: nenhum Python disponível importa python-docx/tomllib."
  echo "Debian 12: sudo apt update && sudo apt install python3-docx"
  echo "Alternativa no ambiente pyenv/venv ativo: python -m pip install python-docx"
  exit 4
fi
export PYTHON_BIN
echo "PASS: Python selecionado: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PYDOCX'
import docx, sys
print(f"PASS: python-docx {docx.__version__} em {sys.executable}")
PYDOCX

echo
echo "[1/4] Relatório final congelado: PRESERVADO"
echo "INFO: não recompilado por padrão para preservar o ajuste manual homologado de colunas."
if [[ "${REBUILD_REPORTS_HISTORICAL:-0}" == "1" ]]; then
  echo "AVISO: REBUILD_REPORTS_HISTORICAL=1 — recompilando fontes históricas; novo QA será obrigatório."
  ( cd "$ROOT/relatorio" && ./compilar_org_pdf.sh )
fi

echo
echo "[2/4] Relatório simplificado histórico: PRESERVADO"
if [[ "${REBUILD_REPORTS_HISTORICAL:-0}" == "1" ]]; then
  ( cd "$ROOT/relatorio_simplificado" && ./compilar_org_pdf.sh )
fi

echo
echo "[3/4] PUIA: Org -> DOCX + PDF"
( cd "$ROOT/puia" && ./compilar_puia_org.sh )

echo
echo "[4/4] Apresentação: Org + modelo do projeto -> PPTX + PDF Beamer"
( cd "$ROOT/apresentacao" && ./compilar_beamer_org.sh )

echo
echo "[CHECK] Artefatos obrigatórios"
req=(
  "$ROOT/relatorio/EP04_relatorio_final_licenciamento_ambiental.docx"
  "$ROOT/relatorio/EP04_relatorio_final_licenciamento_ambiental.pdf"
  "$ROOT/relatorio_simplificado/EP04_relatorio_final_simplificado_licenciamento_ambiental.docx"
  "$ROOT/relatorio_simplificado/EP04_relatorio_final_simplificado_licenciamento_ambiental.pdf"
  "$ROOT/puia/EP04_PUIA_licenciamento_ambiental.docx"
  "$ROOT/puia/EP04_PUIA_licenciamento_ambiental.pdf"
  "$ROOT/apresentacao/EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL.pptx"
  "$ROOT/apresentacao/EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL.pdf"
  "$ROOT/apresentacao/EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.pdf"
  "$ROOT/planilhas/EP04_Coding_Frequency_01_PreMerge_QualCoder_Consolidado.xlsx"
  "$ROOT/planilhas/EP04_Coding_Frequency_02_Temas_PosMerge_QualCoder_Consolidado.xlsx"
)
for f in "${req[@]}"; do [[ -s "$f" ]] || { echo "ERRO: ausente/vazio: $f"; exit 8; }; echo "PASS: ${f#$ROOT/}"; done

echo
echo "Concluído. Log: $LOG"

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/compilacao_apresentacao_org.log"
exec > >(tee "$LOG") 2>&1

echo "============================================================"
echo "EP04 — Apresentação: Org + modelo do projeto -> PPTX + PDF Beamer"
echo "Root: $ROOT"
echo "Log:  $LOG"
echo "============================================================"

ORG="$ROOT/EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.org"
MODELO="$ROOT/modelo/Modelo_de_Apresentacao_Relatorio_Final_Tema5.5.pptx"
PY_SCRIPT="$ROOT/gerar_apresentacao_do_org.py"

for f in "$ORG" "$MODELO" "$PY_SCRIPT"; do
  [[ -s "$f" ]] || { echo "ERRO: arquivo obrigatório ausente/vazio: $f"; exit 2; }
  echo "PASS: encontrado ${f#$ROOT/}"
done

PYTHON_CANDIDATES=()
[[ -n "${PYTHON_BIN:-}" ]] && PYTHON_CANDIDATES+=("$PYTHON_BIN")
command -v python3 >/dev/null 2>&1 && PYTHON_CANDIDATES+=("$(command -v python3)")
[[ -x /usr/bin/python3 ]] && PYTHON_CANDIDATES+=("/usr/bin/python3")
PYTHON_OK=""
for p in "${PYTHON_CANDIDATES[@]}"; do
  [[ -x "$p" ]] || continue
  if "$p" - <<'PYTEST' >/dev/null 2>&1
import json, zipfile, xml.etree.ElementTree
PYTEST
  then
    PYTHON_OK="$p"; break
  fi
done

if [[ -z "$PYTHON_OK" ]]; then
  echo "ERRO: nenhum Python executável compatível foi encontrado."
  echo "Debian 12: verifique a instalação do python3 (biblioteca padrão é suficiente nesta versão)."
  echo "Esta versão não exige python-pptx, lxml nem Pillow para gerar a apresentação."
  exit 3
fi

echo "PASS: Python apresentação (sem dependências externas): $PYTHON_OK"

for bin in libreoffice pdftoppm pdflatex; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERRO: dependência não encontrada: $bin"; exit 4; }
  echo "PASS: dependência $bin -> $(command -v "$bin")"
done

"$PYTHON_OK" "$PY_SCRIPT"

REQ=(
  "$ROOT/EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL.pptx"
  "$ROOT/EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL.pdf"
  "$ROOT/EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.pdf"
  "$ROOT/beamer_render/EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.tex"
)
for f in "${REQ[@]}"; do
  [[ -s "$f" ]] || { echo "ERRO: saída ausente/vazia: $f"; exit 5; }
  echo "PASS: saída ${f#$ROOT/}"
done

echo "Concluído: PPTX e PDF Beamer gerados a partir do Org usando o modelo do projeto."

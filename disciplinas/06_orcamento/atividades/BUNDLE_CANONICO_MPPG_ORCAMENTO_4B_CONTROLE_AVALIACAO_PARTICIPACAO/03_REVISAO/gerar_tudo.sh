#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_BUNDLE="$(cd "$HERE/.." && pwd)"
OUT_DIR="$ROOT_BUNDLE/05_ENTREGAS"

MAIN_STEM="REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO"
GLOSS_STEM="GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO"

"$HERE/compilar.sh"
"$HERE/glossario/compilar_glossario.sh"
"$HERE/gerar_docx.sh"

for f in \
  "$OUT_DIR/${MAIN_STEM}.pdf" \
  "$OUT_DIR/${MAIN_STEM}.docx" \
  "$OUT_DIR/${GLOSS_STEM}.pdf" \
  "$OUT_DIR/${GLOSS_STEM}.docx"
do
  [[ -s "$f" ]] || { echo "ERRO: entrega esperada ausente ou vazia: $f"; exit 10; }
done

cmp -s "$HERE/${MAIN_STEM}.pdf" "$OUT_DIR/${MAIN_STEM}.pdf" || {
  echo "ERRO: PDF da revisão em 05_ENTREGAS diverge do PDF canônico de 03_REVISAO."
  exit 11
}

echo "GATE_DELIVERY_SET=PASS"
echo "PASS: revisão + glossário gerados em PDF e DOCX e materializados em 05_ENTREGAS/."

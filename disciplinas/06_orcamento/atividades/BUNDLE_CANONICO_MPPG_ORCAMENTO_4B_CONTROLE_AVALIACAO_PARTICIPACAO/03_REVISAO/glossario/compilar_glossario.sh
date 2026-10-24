#!/usr/bin/env bash
set -Eeuo pipefail

STEM="GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REV_DIR="$(cd "$HERE/.." && pwd)"
ROOT_BUNDLE="$(cd "$REV_DIR/.." && pwd)"
ORG="$HERE/${STEM}.org"
BIB="$REV_DIR/REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.bib"
CONTENT="$HERE/GLOSSARIO_CONTEUDO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.org"
OUT_DIR="$ROOT_BUNDLE/05_ENTREGAS"
PDF="$OUT_DIR/${STEM}.pdf"
BUILD="$HERE/.build"
LOG="$HERE/compilacao_glossario.log"
SUCCESS=0

cleanup_on_exit() {
  local rc=$?
  if [[ $rc -eq 0 && $SUCCESS -eq 1 && "${MPPG_KEEP_BUILD:-0}" != "1" ]]; then
    rm -rf "$BUILD" "$LOG" "$HERE/${STEM}.tex" "$HERE/${STEM}.pdf.tmp"
  elif [[ $rc -ne 0 ]]; then
    printf '\nFALHA: intermediários preservados em %s e %s\n' "$BUILD" "$LOG" >&2
  fi
  return "$rc"
}
trap cleanup_on_exit EXIT

rm -rf "$BUILD"
rm -f "$LOG" "$HERE/${STEM}.tex" "$HERE/${STEM}.pdf.tmp"
mkdir -p "$BUILD" "$OUT_DIR"
: > "$LOG"
exec > >(tee "$LOG") 2>&1

[[ -f "$ORG" ]] || { echo "ERRO: ORG do glossário ausente."; exit 2; }
[[ -f "$BIB" ]] || { echo "ERRO: BIB canônico ausente."; exit 2; }
[[ -f "$CONTENT" ]] || { echo "ERRO: conteúdo canônico do glossário ausente."; exit 2; }
[[ "$(grep -c '^\*\* ' "$CONTENT")" -eq 10 ]] || { echo "ERRO: glossário deve conter exatamente 10 verbetes."; exit 3; }
[[ "$(grep -Fc '*Não confundir com:*' "$CONTENT")" -eq 10 ]] || { echo "ERRO: cada verbete deve ter 'Não confundir com'."; exit 3; }

CANDIDATES=(
  "${MPPG_ACADEMIC_PIPELINE_ROOT:-}"
  "$HOME/Documentos/mppg/software/academic_pipeline_mppg"
  "$HOME/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade"
  "$HOME/Documentos/mppg/software/academic_pipeline"
)
PIPE=""
for c in "${CANDIDATES[@]}"; do
  [[ -n "$c" ]] || continue
  if [[ -f "$c/app_bundle/misc/academic-writing.el" ]] && { [[ -f "$c/app_bundle/institutions/fgv/latex/fgv-paper.sty" ]] || [[ -f "$c/app_bundle/misc/fgv/fgv-paper.sty" ]]; }; then
    PIPE="$(cd "$c" && pwd)"; break
  fi
done
[[ -n "$PIPE" ]] || { echo "ERRO: Academic Pipeline MPPG não localizado."; exit 4; }
APP="$PIPE/app_bundle"; AW="$APP/misc/academic-writing.el"
if [[ -f "$APP/institutions/fgv/latex/fgv-paper.sty" ]]; then LATEX_DIR="$APP/institutions/fgv/latex"; else LATEX_DIR="$APP/misc/fgv"; fi
if [[ -d "$APP/institutions/fgv/assets" ]]; then ASSET_DIR="$APP/institutions/fgv/assets"; else ASSET_DIR="$APP/misc"; fi
export TEXINPUTS="$LATEX_DIR//:$ASSET_DIR//:$APP/misc//:${TEXINPUTS:-}"
export BIBINPUTS="$REV_DIR//:${BIBINPUTS:-}"
export BSTINPUTS="$REV_DIR//:${BSTINPUTS:-}"
for cmd in emacs lualatex biber kpsewhich python3; do command -v "$cmd" >/dev/null || { echo "ERRO: $cmd ausente"; exit 5; }; done

cat > "$BUILD/export.el" <<ELISP
(require 'org)
(require 'ox-latex)
(load-file "${AW}")
(setq org-confirm-babel-evaluate nil)
(find-file "${ORG}")
(org-latex-export-to-latex)
ELISP
emacs --batch -Q -l "$BUILD/export.el"
[[ -f "$HERE/${STEM}.tex" ]] || { echo "ERRO: exportação ORG→TEX falhou."; exit 6; }
mv "$HERE/${STEM}.tex" "$BUILD/${STEM}.tex"
TEX="$BUILD/${STEM}.tex"

python3 - "$TEX" "$(basename "$BIB")" <<'PYPATCH'
from pathlib import Path
import re, sys
p=Path(sys.argv[1]); bib=sys.argv[2]; s=p.read_text(encoding='utf-8')
s=re.sub(r'(?m)^\\PassOptionsToPackage\{[^}]*\}\{biblatex\}\s*$','',s)
s=re.sub(r'(?m)^\\usepackage(?:\[[^\]]*\])?\{biblatex\}\s*$','',s)
s=re.sub(r'(?m)^\\addbibresource\{[^}]+\}\s*$','',s)
config=r'\usepackage[backend=biber,style=abnt,language=brazil,sorting=nyt,giveninits=true,uniquename=false,uniquelist=false]{biblatex}'+'\n'+rf'\addbibresource{{{bib}}}'+'\n'
s=s.replace(r'\begin{document}',config+r'\begin{document}',1)
p.write_text(s,encoding='utf-8')
PYPATCH

cd "$HERE"
REL=".build/${STEM}.tex"
lualatex -interaction=nonstopmode -file-line-error -shell-escape -output-directory="$BUILD" "$REL"
biber "$BUILD/$STEM"
lualatex -interaction=nonstopmode -file-line-error -shell-escape -output-directory="$BUILD" "$REL"
lualatex -interaction=nonstopmode -file-line-error -shell-escape -output-directory="$BUILD" "$REL"
[[ -s "$BUILD/${STEM}.pdf" ]] || { echo "ERRO: PDF do glossário não gerado."; exit 7; }
if command -v pdftotext >/dev/null; then
  pdftotext "$BUILD/${STEM}.pdf" "$BUILD/validacao.txt"
  for term in 'Controle interno' 'Controle externo' 'Auditoria' 'Controle social' 'Conselho gestor' 'Orçamento participativo' 'Indicadores de desempenho' 'Avaliação de políticas públicas' 'Ciclo de políticas públicas' 'Risco fiscal'; do
    grep -Fqi "$term" "$BUILD/validacao.txt" || { echo "ERRO: verbete ausente no PDF: $term"; exit 8; }
  done
fi
install -m 0644 "$BUILD/${STEM}.pdf" "$PDF.tmp"
mv -f "$PDF.tmp" "$PDF"
SUCCESS=1
echo "PASS: glossário PDF canônico atualizado em $PDF"

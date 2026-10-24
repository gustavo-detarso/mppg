#!/usr/bin/env bash
set -Eeuo pipefail

STEM="REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORG="$HERE/${STEM}.org"
BIB="$HERE/${STEM}.bib"
PDF="$HERE/${STEM}.pdf"
GLOSS_CONTENT="$HERE/glossario/GLOSSARIO_CONTEUDO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.org"
BUILD="$HERE/.build"
COMPILE_LOG="$HERE/compilacao.log"
SUCCESS=0

cleanup_on_exit() {
  local rc=$?
  if [[ $rc -eq 0 && $SUCCESS -eq 1 ]]; then
    if [[ "${MPPG_KEEP_BUILD:-0}" == "1" ]]; then
      printf '\nMPPG_KEEP_BUILD=1: intermediários preservados em %s e %s\n' "$BUILD" "$COMPILE_LOG"
    else
      rm -rf "$BUILD"
      rm -f "$COMPILE_LOG" "$HERE/${STEM}.tex" "$HERE/${STEM}.pdf.tmp"
    fi
  else
    printf '\n=== FALHA ===\nIntermediários preservados para diagnóstico:\n  %s\n  %s\n' "$BUILD" "$COMPILE_LOG" >&2
  fi
  return "$rc"
}
trap cleanup_on_exit EXIT

# Limpa resíduos de uma execução anterior que tenha terminado com sucesso.
rm -rf "$BUILD"
rm -f "$COMPILE_LOG" "$HERE/${STEM}.tex" "$HERE/${STEM}.pdf.tmp"
mkdir -p "$BUILD"
: > "$COMPILE_LOG"
exec > >(tee "$COMPILE_LOG") 2>&1

printf '=== MPPG / FGV-PAPER — compilação canônica ===\n'
printf 'ORG: %s\nBIB: %s\nPDF canônico: %s\n' "$ORG" "$BIB" "$PDF"

[[ -f "$ORG" ]] || { echo "ERRO: ORG não encontrado: $ORG"; exit 2; }
[[ -f "$BIB" ]] || { echo "ERRO: BIB não encontrado: $BIB"; exit 2; }
[[ -f "$GLOSS_CONTENT" ]] || { echo "ERRO: conteúdo canônico do glossário não encontrado: $GLOSS_CONTENT"; exit 2; }
[[ -f "$HERE/assets/FLUXOGRAMA_CORPUS_PRISMA.png" ]] || { echo "ERRO: asset FLUXOGRAMA ausente."; exit 2; }
[[ -f "$HERE/assets/GRAFICO_MARGEM_LOA2026.png" ]] || { echo "ERRO: asset GRAFICO ausente."; exit 2; }
[[ -f "$HERE/assets/FIG_CADEIA_ANALITICA.png" ]] || { echo "ERRO: asset CADEIA ausente."; exit 2; }

# Gates bibliográficos persistentes
for pattern in \
  'shortauthor  = {{CÂMARA DOS DEPUTADOS}}' \
  'author       = {da Silva, Glauco Peres and Carvalho, Carlos Eduardo}' \
  'author       = {Novaes, Flávio Santos and dos Santos, Maria Elisabete Pereira}' \
  'author       = {de Souza, Fábio Jacinto Barreto and Silva, Suylan Almeida Midlej e}' \
  'author       = {Torrezan, Raphael Guilherme Araujo and de Paiva, Claudio Cesar}'
do
  grep -Fq "$pattern" "$BIB" || { echo "ERRO: gate bibliográfico falhou: $pattern"; exit 21; }
done
echo "GATE_BIB_CANONICO=PASS"


# Gate do glossário canônico: 10 verbetes e inclusão no Apêndice A.
for term in \
  'Controle interno' \
  'Controle externo (TCU)' \
  'Auditoria' \
  'Controle social' \
  'Conselho gestor' \
  'Orçamento participativo' \
  'Indicadores de desempenho' \
  'Avaliação de políticas públicas' \
  'Ciclo de políticas públicas' \
  'Risco fiscal'
do
  grep -Fq "** $term" "$GLOSS_CONTENT" || { echo "ERRO: verbete obrigatório ausente do glossário: $term"; exit 28; }
done
[[ "$(grep -Fc '*Não confundir com:*' "$GLOSS_CONTENT")" -eq 10 ]] || { echo "ERRO: glossário deve ter 10 campos 'Não confundir com'."; exit 29; }
grep -Fq 'APÊNDICE A — GLOSSÁRIO ORÇAMENTÁRIO DO SUBTEMA 4B' "$ORG" || { echo "ERRO: Apêndice A do glossário ausente no ORG principal."; exit 30; }
grep -Fq '#+INCLUDE: "glossario/GLOSSARIO_CONTEUDO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.org"' "$ORG" || { echo "ERRO: ORG principal não inclui a fonte canônica do glossário."; exit 31; }
echo "GATE_GLOSSARIO_CANONICO=PASS"

CANDIDATES=(
  "${MPPG_ACADEMIC_PIPELINE_ROOT:-}"
  "$HOME/Documentos/mppg/software/academic_pipeline_mppg"
  "$HOME/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade"
  "$HOME/Documentos/mppg/software/academic_pipeline"
)
ROOT=""
for c in "${CANDIDATES[@]}"; do
  [[ -n "$c" ]] || continue
  if [[ -f "$c/app_bundle/misc/academic-writing.el" ]] && \
     { [[ -f "$c/app_bundle/institutions/fgv/latex/fgv-paper.sty" ]] || [[ -f "$c/app_bundle/misc/fgv/fgv-paper.sty" ]]; }; then
    ROOT="$(cd "$c" && pwd)"
    break
  fi
done
[[ -n "$ROOT" ]] || { echo "ERRO: não localizei o Academic Pipeline MPPG."; exit 3; }

APP="$ROOT/app_bundle"
AW="$APP/misc/academic-writing.el"
if [[ -f "$APP/institutions/fgv/latex/fgv-paper.sty" ]]; then
  LATEX_DIR="$APP/institutions/fgv/latex"
else
  LATEX_DIR="$APP/misc/fgv"
fi
if [[ -d "$APP/institutions/fgv/assets" ]]; then
  ASSET_DIR="$APP/institutions/fgv/assets"
else
  ASSET_DIR="$APP/misc"
fi

export TEXINPUTS="$LATEX_DIR//:$ASSET_DIR//:$APP/misc//:${TEXINPUTS:-}"
export BIBINPUTS="$HERE//:${BIBINPUTS:-}"
export BSTINPUTS="$HERE//:${BSTINPUTS:-}"

for cmd in emacs lualatex biber kpsewhich python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERRO: comando não encontrado: $cmd"; exit 4; }
done
[[ -n "$(kpsewhich fgv-paper.sty || true)" ]] || { echo "ERRO: fgv-paper.sty não está visível."; exit 5; }
[[ -n "$(kpsewhich abnt.bbx || true)" && -n "$(kpsewhich abnt.cbx || true)" ]] || { echo "ERRO: biblatex-abnt não instalado."; exit 6; }

ELISP="$BUILD/export.el"
cat > "$ELISP" <<ELISP
(require 'org)
(require 'ox-latex)
(load-file "${AW}")
(setq org-confirm-babel-evaluate nil)
(find-file "${ORG}")
(let ((tex (org-latex-export-to-latex)))
  (princ (format "MPPG_TEX=%s\\n" tex)))
ELISP

emacs --batch -Q -l "$ELISP"
TEX_ROOT="$HERE/${STEM}.tex"
[[ -f "$TEX_ROOT" ]] || { echo "ERRO: exportação ORG -> TEX não gerou $TEX_ROOT"; exit 7; }
mv -f "$TEX_ROOT" "$BUILD/${STEM}.tex"
TEX="$BUILD/${STEM}.tex"

python3 - "$TEX" "$(basename "$BIB")" <<'PYPATCH'
from pathlib import Path
import re, sys
tex_path = Path(sys.argv[1])
bib_name = sys.argv[2]
s = tex_path.read_text(encoding='utf-8', errors='strict')
s = re.sub(r'(?m)^\\PassOptionsToPackage\{[^}]*\}\{biblatex\}\s*$', '', s)
s = re.sub(r'(?m)^\\usepackage(?:\[[^\]]*\])?\{biblatex\}\s*$', '', s)
s = re.sub(r'(?m)^\\addbibresource\{[^}]+\}\s*$', '', s)
s = re.sub(r'(?m)^\\usepackage(?:\[[^\]]*\])?\{abntex2cite\}\s*$', '', s)
s = re.sub(r'(?m)^\\bibliographystyle\{[^}]+\}\s*$', '', s)
s = re.sub(r'(?m)^\\bibliography\{[^}]+\}\s*$', '', s)
config = (
    r'\usepackage[backend=biber,style=abnt,language=brazil,sorting=nyt,giveninits=true,uniquename=false,uniquelist=false]{biblatex}'
    + '\n' + rf'\addbibresource{{{bib_name}}}' + '\n'
)
marker = r'\begin{document}'
if marker not in s:
    raise SystemExit('ERRO: \\begin{document} não encontrado.')
s = s.replace(marker, config + marker, 1)
tex_path.write_text(s, encoding='utf-8')
checks = {
    'GATE_BIBLATEX_ABNT_AUTHOR_DATE': r'\usepackage[backend=biber,style=abnt,language=brazil,sorting=nyt,giveninits=true,uniquename=false,uniquelist=false]{biblatex}',
    'GATE_TABLE_LAYOUT_TABULARX': r'\begin{tabularx}',
    'GATE_FIGURES_FIXED_PLACEMENT': r'\begin{figure}[H]',
    'GATE_QUADRO1_KEEP_TOGETHER': r'\Needspace{0.70\textheight}',
    'GATE_HEADHEIGHT_PATCH': r'\setlength{\headheight}{14.5pt}',
    'GATE_COVER_PAGEANCHOR_PATCH': r'\hypersetup{pageanchor=false}',
}
for gate, token in checks.items():
    if token not in s:
        raise SystemExit(f'ERRO: {gate} falhou.')
    print(gate + '=PASS')
if s.count(r'\begin{figure}[H]') < 3:
    raise SystemExit('ERRO: menos de 3 figuras [H] no TEX.')
if 'DECLARAÇÃO DE USO DE INTELIGÊNCIA ARTIFICIAL' not in s or r'\clearpage' not in s:
    raise SystemExit('ERRO: declaração de IA sem proteção de nova página.')
print('GATE_AI_DECLARATION_FRESH_PAGE=PASS')
PYPATCH

cd "$HERE"
TEX_REL=".build/${STEM}.tex"
lualatex -interaction=nonstopmode -file-line-error -shell-escape -output-directory="$BUILD" "$TEX_REL"
biber "$BUILD/$STEM"
lualatex -interaction=nonstopmode -file-line-error -shell-escape -output-directory="$BUILD" "$TEX_REL"
lualatex -interaction=nonstopmode -file-line-error -shell-escape -output-directory="$BUILD" "$TEX_REL"

BUILT_PDF="$BUILD/${STEM}.pdf"
LATEX_LOG="$BUILD/${STEM}.log"
[[ -f "$BUILT_PDF" ]] || { echo "ERRO: PDF não foi criado."; exit 9; }
[[ -f "$LATEX_LOG" ]] || { echo "ERRO: log LaTeX não foi criado."; exit 9; }

# Valida o PDF temporário ANTES de substituir o PDF canônico anterior.
if command -v pdftotext >/dev/null 2>&1; then
  TXT="$BUILD/${STEM}_validacao.txt"
  pdftotext "$BUILT_PDF" "$TXT"
  if grep -Eq '\[[0-9]{1,3}\]' "$TXT"; then
    echo "ERRO: citação numérica residual detectada."
    grep -En '\[[0-9]{1,3}\]' "$TXT" | head -20 || true
    exit 10
  fi
  echo "GATE_PDF_NO_NUMERIC_CITATIONS=PASS"
  grep -Fq "APÊNDICE A" "$TXT" || { echo "ERRO: Apêndice A não apareceu no PDF final."; exit 32; }
  grep -Fq "Risco fiscal" "$TXT" || { echo "ERRO: glossário incompleto no PDF final."; exit 33; }
  echo "GATE_PDF_GLOSSARIO_APPENDIX=PASS"
fi

if grep -Eq 'Overfull \\hbox|Overfull \\vbox' "$LATEX_LOG"; then
  echo "AVISO: caixas overfull encontradas:"
  grep -E 'Overfull \\hbox|Overfull \\vbox' "$LATEX_LOG" | head -30 || true
else
  echo "GATE_NO_OVERFULL_BOXES=PASS"
fi
if grep -Fq 'headheight is too small' "$LATEX_LOG"; then
  echo "ERRO: warning de headheight persiste."; exit 26
else
  echo "GATE_NO_HEADHEIGHT_WARNING=PASS"
fi
if grep -Fq "ignoring duplicate destination with the name 'page.1'" "$LATEX_LOG"; then
  echo "ERRO: destination page.1 duplicada persiste."; exit 27
else
  echo "GATE_NO_DUPLICATE_PAGE1_DESTINATION=PASS"
fi

# Sobrescrita atômica: só ocorre depois de compilação e validação bem-sucedidas.
install -m 0644 "$BUILT_PDF" "$PDF.tmp"
mv -f "$PDF.tmp" "$PDF"
SUCCESS=1

printf '\n=== PASS ===\nPDF canônico sobrescrito: %s\n' "$PDF"
if [[ "${MPPG_KEEP_BUILD:-0}" == "1" ]]; then
  printf 'Intermediários serão preservados por solicitação (MPPG_KEEP_BUILD=1).\n'
else
  printf 'Intermediários e log serão removidos automaticamente ao encerrar.\n'
fi

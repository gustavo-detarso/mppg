#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-/home/gustavodetarso/Documentos/mppg/software/academic_pipeline}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$BASE/app_bundle"
cp -R "$SRC_DIR/app_bundle/"* "$BASE/app_bundle/"
cp "$SRC_DIR/requirements.txt" "$BASE/requirements_rc10.txt"

printf '\nrc10.7-conformidade instalada em: %s\n' "$BASE/app_bundle"
printf '\nVerificando arquivos locais obrigatórios:\n'

if [ ! -f "$BASE/app_bundle/misc/academic-writing.el" ]; then
  echo "AVISO: falta $BASE/app_bundle/misc/academic-writing.el"
else
  echo "OK: $BASE/app_bundle/misc/academic-writing.el"
fi
if [ ! -f "$BASE/app_bundle/misc/fgv.png" ]; then
  echo "AVISO: falta $BASE/app_bundle/misc/fgv.png"
else
  echo "OK: $BASE/app_bundle/misc/fgv.png"
fi
if [ ! -f "$BASE/app_bundle/misc/fgv/fgv-paper.sty" ]; then
  echo "AVISO: falta $BASE/app_bundle/misc/fgv/fgv-paper.sty"
fi
if [ ! -f "$BASE/app_bundle/misc/fgv/fgv-dissertacao.sty" ]; then
  echo "AVISO: falta $BASE/app_bundle/misc/fgv/fgv-dissertacao.sty"
fi

printf '\nComandos recomendados após instalar:\n'
printf 'cd %s\n' "$BASE"
printf 'pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --doctor\n'
printf 'pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --explain-profile fgv\n'
printf 'pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config app_bundle/config/examples/paper_rc10_exemplo.toml --check-config\n'
printf 'pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config app_bundle/config/examples/paper_rc10_exemplo.toml --check-institution-compliance\n'
printf 'pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config app_bundle/config/examples/paper_rc10_exemplo.toml\n'

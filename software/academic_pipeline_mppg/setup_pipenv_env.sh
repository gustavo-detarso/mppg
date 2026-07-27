#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "/home/gustavodetarso/.pyenv/versions/3.11.13/bin/python" ]]; then
    PYTHON_BIN="/home/gustavodetarso/.pyenv/versions/3.11.13/bin/python"
  else
    PYTHON_BIN="3.11"
  fi
fi

echo "==> Criando ambiente Pipenv com Python: $PYTHON_BIN"
pipenv --python "$PYTHON_BIN"

echo "==> Instalando bibliotecas Python principais"
pipenv install openai pydantic python-dotenv pypdf python-docx openpyxl

echo "==> Instalando dependências de desenvolvimento"
pipenv install --dev pytest

echo "==> Gerando Pipfile.lock"
pipenv lock

echo "==> Validando imports principais"
pipenv run python - <<'PY'
import openai, pydantic, dotenv, pypdf, docx, openpyxl
print("Bibliotecas Python principais OK")
PY

echo "==> Validando sintaxe dos módulos do pipeline"
find app_bundle/scripts/pipeline -name "*.py" -print0 | xargs -0 pipenv run python -m py_compile

echo ""
echo "Ambiente Pipenv criado com sucesso."
echo "Próximos passos:"
echo "1. Garanta que .env esteja na raiz: $ROOT_DIR/.env"
echo "2. Copie app_bundle/misc/academic-writing.el"
echo "3. Copie app_bundle/misc/fgv.png"
echo "4. Rode: pipenv run python -m academic_pipeline --doctor"

#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob dotglob

DRY_RUN="${1:-}"

get_base() {
  local name="${1%/}"
  name="${name##*/}"

  case "$name" in
    *.json) name="${name%.json}" ;;
    *.pdf)  name="${name%.pdf}" ;;
    *.org)  name="${name%.org}" ;;
    *.svg)  name="${name%.svg}" ;;
    *.bib)  name="${name%.bib}" ;;
    *.bbl)  name="${name%.bbl}" ;;
    *.tex)  name="${name%.tex}" ;;
  esac

  # remove marcadores técnicos do final do nome
  name="${name%_fulltext_cache}"
  name="${name%_debug}"
  name="${name%_prisma}"
  name="${name%_prisma}"   # cobre casos do tipo *_prisma_prisma

  printf '%s\n' "$name"
}

choose_dest() {
  local base="$1"

  # se já existir uma pasta <base>_prisma, usa ela como raiz da pesquisa
  if [[ -d "${base}_prisma" ]]; then
    printf '%s\n' "${base}_prisma"
  else
    printf '%s\n' "$base"
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "$*"
  else
    "$@"
  fi
}

for item in *; do
  [[ "$item" == "organizar_pesquisas.sh" ]] && continue
  [[ ! -e "$item" ]] && continue

  base="$(get_base "$item")"
  [[ -z "$base" ]] && continue

  dest="$(choose_dest "$base")"

  # não mover a própria pasta destino
  if [[ "$item" == "$dest" ]]; then
    continue
  fi

  run_cmd mkdir -p -- "$dest"

  if [[ -e "$dest/$item" ]]; then
    echo "Pulando '$item' porque já existe em '$dest/'"
    continue
  fi

  run_cmd mv -- "$item" "$dest/"
done

echo "Organização concluída."

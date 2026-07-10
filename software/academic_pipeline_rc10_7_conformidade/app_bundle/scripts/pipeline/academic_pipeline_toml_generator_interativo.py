#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gerador interativo completo de TOML para academic_pipeline rc10.7.42.

O objetivo é reduzir erro operacional: o usuário escolhe um preset explicável,
responde apenas os campos necessários e o script gera um TOML completo com as
camadas atuais do pipeline: instituição, corpus local, pesquisa, documento,
bibliografia, LaTeX, DOCX, prompts, mapa mental, relatório PRISMA,
conformidade e qualidade.

Pode ser usado para:
- atividade local;
- paper local;
- paper com relatório PRISMA;
- dissertação local;
- dissertação com relatório PRISMA;
- relatório PRISMA autônomo;
- somente renderização a partir de document.json;
- modo avançado por componentes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import tomllib
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from prisma_busca_externa import (
    expand_provider_selection,
    provider_selection_choices,
    provider_statuses,
)
from bibliography_manager import (
    expand_metadata_provider_selection,
    metadata_provider_selection_choices,
    metadata_provider_statuses,
)

try:
    import readline  # noqa: F401
except Exception:  # pragma: no cover
    readline = None  # type: ignore


# -----------------------------------------------------------------------------
# Localização do app_bundle
# -----------------------------------------------------------------------------


def find_app_bundle(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if candidate.name == "app_bundle":
            return candidate
        if (candidate / "app_bundle").is_dir():
            return (candidate / "app_bundle").resolve()
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if candidate.name == "app_bundle":
            return candidate
    raise RuntimeError("Não consegui localizar app_bundle. Execute a partir da raiz do academic_pipeline ou de app_bundle.")


APP = find_app_bundle()
ROOT = APP.parent


# -----------------------------------------------------------------------------
# Utilitários básicos
# -----------------------------------------------------------------------------


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    repl = {
        "á": "a", "à": "a", "ã": "a", "â": "a", "ä": "a",
        "é": "e", "ê": "e", "è": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "õ": "o", "ô": "o", "ö": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ç": "c", "ñ": "n",
    }
    for a, b in repl.items():
        value = value.replace(a, b)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "projeto"


def toml_escape(value: Any) -> str:
    s = str(value if value is not None else "")
    return (
        s.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace('"', '\\"')
    )

def tstr(value: Any) -> str:
    return f'"{toml_escape(value)}"'


def tbool(value: bool) -> str:
    return "true" if bool(value) else "false"


def tlist(items: list[str], indent: str = "") -> str:
    if not items:
        return "[]"
    body = ",\n".join(f'{indent}  {tstr(i)}' for i in items)
    return "[\n" + body + f"\n{indent}]"


def rel_for_toml(path: Path | str, config_dir: Path) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    if raw.startswith("profile://"):
        return raw
    p = Path(raw).expanduser()
    if not p.is_absolute():
        # Mantém relativo se o usuário já informou relativo; normaliza separadores.
        return raw.replace("\\", "/")
    try:
        return os.path.relpath(p.resolve(), config_dir.resolve()).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def ensure_project_dir(project_slug: str) -> Path:
    p = APP / "projetos" / project_slug
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def maybe_copy_to_project(src: str, dst: Path) -> str:
    src = (src or "").strip()
    if not src:
        return str(dst)
    p = Path(src).expanduser()
    if p.exists() and p.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.is_file():
            shutil.copy2(p, dst)
            return str(dst)
    return src


def install_path_completion() -> None:
    if readline is None:
        return
    try:
        import glob

        def complete(text: str, state: int) -> str | None:
            expanded = os.path.expanduser(text)
            matches = glob.glob(expanded + "*")
            results: list[str] = []
            for m in matches:
                suffix = "/" if os.path.isdir(m) else ""
                item = m + suffix
                if text.startswith("~"):
                    item = item.replace(os.path.expanduser("~"), "~", 1)
                results.append(item)
            try:
                return results[state]
            except IndexError:
                return None

        readline.set_completer_delims(" \t\n")
        readline.set_completer(complete)
        readline.parse_and_bind("tab: complete")
    except Exception:
        return


# -----------------------------------------------------------------------------
# Entrada interativa
# -----------------------------------------------------------------------------

TUI_THEME = os.getenv("ACADEMIC_PIPELINE_TUI_THEME", "").strip().lower()
TUI_STAGE_TITLE = "Configuração"
# Quando informado pela Central Operacional, este diretório recebe o TOML.
# Sem override, o wizard pergunta explicitamente onde o projeto será salvo.
WIZARD_PROJECT_DIR_OVERRIDE: Path | None = None


def set_wizard_project_dir(value: str | Path | None) -> None:
    global WIZARD_PROJECT_DIR_OVERRIDE
    raw = str(value or "").strip()
    WIZARD_PROJECT_DIR_OVERRIDE = Path(raw).expanduser().resolve() if raw else None


def bundle_rel(data: dict[str, Any], relative_to_bundle: str) -> str:
    """Gera caminho relativo ao TOML, inclusive quando salvo fora de app_bundle."""
    return rel_for_toml(APP / relative_to_bundle, data["config_dir"])


def set_tui_theme(value: str | None) -> None:
    """Ativa o front-end prompt_toolkit para o wizard quando solicitado."""
    global TUI_THEME
    TUI_THEME = str(value or "").strip().lower()


def tui_theme_enabled() -> bool:
    return TUI_THEME == "fgv"


def _fgv_ui() -> Any:
    try:
        import academic_pipeline_tui_widgets as widgets
    except ImportError as exc:  # pragma: no cover - depende da instalação local
        raise RuntimeError(
            "O tema visual FGV requer prompt-toolkit. Execute: pipenv install prompt-toolkit"
        ) from exc
    return widgets


def _dialog_title() -> str:
    return f"Academic Pipeline — {TUI_STAGE_TITLE}"


def _looks_like_path(prompt: str) -> bool:
    value = str(prompt or "").lower()
    tokens = ("arquivo", "pasta", "caminho", "diretório", "diretorio", "zip", "logo", "plantuml", "document.json", ".pdf", ".org")
    return any(token in value for token in tokens)


def _path_picker_suffixes(prompt: str) -> tuple[str, ...]:
    """Restringe o navegador visual quando o rótulo indica um tipo de arquivo.

    A filtragem atua apenas no navegador acionado por F2; o campo continua
    aceitando digitação e colagem manual de qualquer caminho válido.
    """
    value = str(prompt or "").lower()
    if "arquivo estruturado" in value or "dados estruturados" in value:
        return (".toml", ".json", ".yaml", ".yml", ".md", ".org", ".txt")
    if "documentos-base.zip" in value or "orientacoes.zip" in value:
        return (".zip",)
    if "document.json" in value:
        return (".json",)
    if "doi" in value and "manifest" in value:
        return (".csv",)
    if "logo" in value:
        return (".png", ".jpg", ".jpeg", ".svg")
    if "plantuml" in value:
        return (".jar",)
    if ".pdf" in value:
        return (".pdf",)
    if ".org" in value:
        return (".org",)
    return ()


def _path_picker_directories_only(prompt: str) -> bool:
    value = str(prompt or "").lower()
    return (
        ("pasta" in value or "diretório" in value or "diretorio" in value)
        and not any(token in value for token in ("arquivo", "zip", "document.json", ".pdf", ".org"))
    )


def ask(prompt: str, default: str = "") -> str:
    if tui_theme_enabled():
        path_like = _looks_like_path(prompt)
        value = _fgv_ui().input_text(
            _dialog_title(),
            prompt,
            default=default,
            path_completion=path_like,
            only_directories=_path_picker_directories_only(prompt),
            allowed_suffixes=_path_picker_suffixes(prompt),
        )
        if value is None:
            raise SystemExit("Geração de TOML cancelada pelo usuário.")
        value = value.strip()
        return value if value else default
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def ask_required(prompt: str, default: str = "") -> str:
    while True:
        value = ask(prompt, default)
        if value.strip():
            return value.strip()
        if tui_theme_enabled():
            _fgv_ui().message(_dialog_title(), "Este campo é obrigatório.")
        else:
            print("Campo obrigatório.")


def ask_bool(prompt: str, default: bool = True) -> bool:
    if tui_theme_enabled():
        return bool(_fgv_ui().confirm(_dialog_title(), prompt, default=default))
    d = "s" if default else "n"
    while True:
        value = input(f"{prompt} (s/n) [{d}]: ").strip().lower()
        if not value:
            return default
        if value in {"s", "sim", "y", "yes"}:
            return True
        if value in {"n", "nao", "não", "no"}:
            return False
        print("Responda s ou n.")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    if tui_theme_enabled():
        values = [(choice, choice) for choice in choices]
        selected = _fgv_ui().select_one(
            _dialog_title(),
            prompt,
            values,
            default=default if default in choices else choices[0],
        )
        if selected is None:
            raise SystemExit("Geração de TOML cancelada pelo usuário.")
        return str(selected)
    choices_norm = {c.lower(): c for c in choices}
    while True:
        value = ask(prompt + " (" + "/".join(choices) + ")", default).lower()
        if value in choices_norm:
            return choices_norm[value]
        print("Escolha inválida.")


def ask_list(prompt: str, default: list[str] | None = None) -> list[str]:
    default = default or []
    raw_default = "; ".join(default)
    raw = ask(prompt + " (separe por ;)", raw_default)
    return [x.strip() for x in raw.split(";") if x.strip()]


def ask_many_choice(prompt: str, choices: list[tuple[str, str]], default: list[str]) -> list[str]:
    """Seleciona múltiplos itens e expande a opção explícita ``Todas``.

    A opção ``todas`` permanece visível na lista e, ao ser marcada, é
    convertida em todas as fontes reais antes de o TOML ser gravado. O atalho
    ``A`` do componente visual continua disponível como alternativa rápida.
    """
    keys = [key for key, _label in choices]
    if tui_theme_enabled():
        select_all_value = "todas" if "todas" in keys else None
        selected = _fgv_ui().select_many(
            _dialog_title(),
            prompt,
            choices,
            defaults=[item for item in default if item in keys],
            select_all_value=select_all_value,
            select_all_values=[item for item in keys if item != select_all_value],
        )
        if selected is None:
            raise SystemExit("Geração de TOML cancelada pelo usuário.")
        normalised = [str(item) for item in selected if str(item) in keys]
    else:
        shown = "; ".join(f"{key}={label}" for key, label in choices)
        aliases = {key.lower(): key for key in keys}
        while True:
            raw = ask(prompt + " (separe por ; — " + shown + ")", "; ".join(default))
            values = [item.strip().lower().replace("-", "_").replace(" ", "_") for item in raw.split(";") if item.strip()]
            normalised = [aliases[item] for item in values if item in aliases]
            invalid = [item for item in values if item not in aliases]
            if invalid:
                print("Opções inválidas: " + ", ".join(invalid))
                continue
            if not normalised:
                print("Selecione ao menos uma opção.")
                continue
            break
    expanded = expand_provider_selection(normalised)
    if not expanded:
        _notify_structured_input_error("Selecione ao menos uma fonte de descoberta.")
        return ask_many_choice(prompt, choices, default)
    return expanded


def ask_metadata_sources(default: list[str] | None = None) -> list[str]:
    """Seleciona fontes implementadas para metadados de PDFs/DOIs locais.

    Esta seleção é independente das fontes de descoberta PRISMA. A opção
    ``[Todas]`` expande somente Crossref, OpenAlex, Semantic Scholar e Scopus,
    que são os adaptadores efetivamente implementados em bibliography_manager.
    """
    choices = metadata_provider_selection_choices()
    keys = [key for key, _label in choices]
    default = [item for item in (default or ["crossref", "openalex", "semantic_scholar", "scopus"]) if item in keys]
    prompt = (
        "Selecione as fontes para enriquecimento de metadados por DOI. "
        "Os estados abaixo refletem apenas a presença de credenciais no .env; valores não são exibidos. "
        "Marque [Todas] para selecionar todos os adaptadores compatíveis."
    )
    if tui_theme_enabled():
        selected = _fgv_ui().select_many(
            _dialog_title(),
            prompt,
            choices,
            defaults=default,
            select_all_value="todas",
            select_all_values=[item for item in keys if item != "todas"],
        )
        if selected is None:
            raise SystemExit("Geração de TOML cancelada pelo usuário.")
        normalised = [str(item) for item in selected if str(item) in keys]
    else:
        statuses = metadata_provider_statuses()
        print("\nFontes de metadados disponíveis (credenciais sem exibir valores):")
        for source in [key for key in keys if key != "todas"]:
            item = statuses[source]
            print(f"- {item['label']}: {item['status']}")
        shown = "; ".join(f"{key}={label}" for key, label in choices)
        aliases = {key.lower(): key for key in keys}
        while True:
            raw = ask("Fontes de metadados (separe por ; — " + shown + ")", "; ".join(default))
            values = [item.strip().lower().replace("-", "_").replace(" ", "_") for item in raw.split(";") if item.strip()]
            normalised = [aliases[item] for item in values if item in aliases]
            invalid = [item for item in values if item not in aliases]
            if invalid:
                print("Opções inválidas: " + ", ".join(invalid))
                continue
            break
    expanded = expand_metadata_provider_selection(normalised)
    if not expanded:
        _notify_structured_input_error("Selecione ao menos uma fonte de metadados ou desative o enriquecimento.")
        return ask_metadata_sources(default)
    return expanded


def ask_positive_int(prompt: str, default: int, *, minimum: int = 1, maximum: int = 1000) -> int:
    while True:
        raw = ask(prompt, str(default)).strip()
        try:
            value = int(raw)
        except ValueError:
            _notify_structured_input_error("Informe um número inteiro.")
            continue
        if not minimum <= value <= maximum:
            _notify_structured_input_error(f"Informe um valor entre {minimum} e {maximum}.")
            continue
        return value


# -----------------------------------------------------------------------------
# Dados estruturados de pesquisa
# -----------------------------------------------------------------------------

STRUCTURED_RESEARCH_SUFFIXES = {".toml", ".json", ".yaml", ".yml", ".md", ".org", ".txt"}
# Modelo neutro distribuído com o bundle. Ele nunca é alterado diretamente
# pelo assistente: quando escolhido, é copiado para a pasta do projeto com
# um nome estável e a cópia passa a ser a fonte registrada no TOML final.
# Exemplos temáticos (como o de PMF) permanecem apenas como referência e não
# são usados como fallback automático.
# Os modelos estruturados pertencem ao perfil institucional escolhido no
# início do wizard. Eles são sempre buscados em ``misc/<instituicao>/`` e
# copiados para a pasta do projeto antes de qualquer edição.
STRUCTURED_RESEARCH_TEMPLATE_FILENAME = "modelo_relatorio_prisma_busca_orientada.toml"
PROJECT_RESEARCH_MODEL_FILENAME = "dados_pesquisa_prisma.toml"
# O modelo de paper mantém o sufixo institucional, como
# ``modelo_paper_local_fgv.toml`` dentro de ``misc/fgv``.
STRUCTURED_PAPER_TEMPLATE_BASENAME = "modelo_paper_local"
PROJECT_PAPER_MODEL_FILENAME = "dados_paper.toml"
RESEARCH_FIELD_LABELS: dict[str, str] = {
    "tema": "Tema",
    "recorte": "Recorte",
    "objetivo": "Objetivo",
    "pergunta_pesquisa": "Pergunta de pesquisa",
    "hipotese": "Hipótese/tese orientadora",
    "palavras_chave": "Palavras-chave",
    "tipo_estudo": "Tipo de estudo",
}
# Idiomas inicialmente considerados na revisão PRISMA com busca externa.
# O usuário ainda pode editar a lista no wizard, mas o padrão evita uma
# restrição indevida a estudos publicados apenas em português.
PRISMA_BUSCA_DEFAULT_LANGUAGES = ["português", "inglês", "espanhol"]

RESEARCH_FIELD_ALIASES: dict[str, set[str]] = {
    "tema": {"tema", "topic", "assunto", "tema_principal"},
    "recorte": {"recorte", "delimitacao", "delimitacao_da_pesquisa", "scope", "escopo"},
    "objetivo": {"objetivo", "objetivo_geral", "objective", "objetivos"},
    "pergunta_pesquisa": {
        "pergunta_de_pesquisa", "pergunta_pesquisa", "questao_de_pesquisa",
        "questao_pesquisa", "research_question", "pergunta_orientadora",
    },
    "hipotese": {
        "hipotese", "hipotese_tese_orientadora", "tese_orientadora",
        "hypothesis", "thesis", "tese",
    },
    "palavras_chave": {"palavras_chave", "palavra_chave", "keywords", "keyword", "termos_de_busca"},
    "tipo_estudo": {"tipo_de_estudo", "tipo_estudo", "study_type", "tipo_de_pesquisa"},
}


PAPER_FIELD_LABELS: dict[str, str] = {
    "tese_central": "Tese/argumento central do paper",
    "estrutura_desejada": "Estrutura desejada",
    "argumentos_obrigatorios": "Argumentos ou pontos obrigatórios",
    "orientacoes_metodologicas": "Orientações metodológicas",
    "limites_do_escopo": "Limites do escopo",
    "tom_de_redacao": "Tom de redação",
    "instrucoes_adicionais": "Instruções adicionais",
}
PAPER_LIST_FIELDS = {"estrutura_desejada", "argumentos_obrigatorios"}
PAPER_FIELD_ALIASES: dict[str, set[str]] = {
    "tese_central": {"tese_central", "tese", "argumento_central", "central_thesis", "main_argument"},
    "estrutura_desejada": {"estrutura_desejada", "estrutura", "secoes", "secoes_desejadas", "desired_structure"},
    "argumentos_obrigatorios": {"argumentos_obrigatorios", "pontos_obrigatorios", "argumentos_chave", "required_arguments"},
    "orientacoes_metodologicas": {"orientacoes_metodologicas", "diretrizes_metodologicas", "metodologia", "methodological_guidance"},
    "limites_do_escopo": {"limites_do_escopo", "limites", "delimitacoes", "scope_limits"},
    "tom_de_redacao": {"tom_de_redacao", "tom", "estilo_de_redacao", "writing_tone"},
    "instrucoes_adicionais": {"instrucoes_adicionais", "orientacoes_adicionais", "observacoes", "additional_instructions"},
}


def _normalise_research_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _canonical_research_field(value: Any) -> str | None:
    normalised = _normalise_research_key(value)
    for field, aliases in RESEARCH_FIELD_ALIASES.items():
        if normalised in aliases:
            return field
    return None


def _canonical_paper_field(value: Any) -> str | None:
    normalised = _normalise_research_key(value)
    for field, aliases in PAPER_FIELD_ALIASES.items():
        if normalised in aliases:
            return field
    return None


def _coerce_research_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_coerce_research_text(item) for item in value]
        return "; ".join(item for item in parts if item)
    if isinstance(value, dict):
        return ""
    return str(value).strip()


def _coerce_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw_items = re.split(r"[;\n,]+", str(value))
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        clean = re.sub(r"^[\-\*\u2022]\s*", "", str(item)).strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            result.append(clean)
    return result


def _coerce_paper_list(value: Any) -> list[str]:
    """Normaliza listas de seções e argumentos do modelo do paper."""
    return _coerce_keywords(value)


def _iter_mapping_layers(value: Any) -> list[dict[str, Any]]:
    """Percorre dicionários priorizando blocos usuais de metadados de pesquisa."""
    if not isinstance(value, dict):
        return []
    layers: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(mapping: dict[str, Any]) -> None:
        marker = id(mapping)
        if marker not in seen:
            layers.append(mapping)
            seen.add(marker)

    priority = {"pesquisa", "research", "dados_pesquisa", "metadados", "metadata", "projeto"}
    for key, child in value.items():
        if _normalise_research_key(key) in priority and isinstance(child, dict):
            add(child)
    add(value)

    def walk(mapping: dict[str, Any]) -> None:
        for child in mapping.values():
            if isinstance(child, dict) and id(child) not in seen:
                add(child)
                walk(child)

    for layer in list(layers):
        walk(layer)
    return layers


def _parse_labelled_research_text(text: str) -> dict[str, Any]:
    """Lê campos simples de Markdown, Org, TXT ou YAML sem dependência extra."""
    result: dict[str, Any] = {}
    current: str | None = None
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:[#>*\-\u2022]\s*)*([^:]{2,100}?)\s*:\s*(.*)$", line)
        if match:
            field = _canonical_research_field(match.group(1))
            if field:
                current = field
                value = match.group(2).strip()
                if field == "palavras_chave":
                    result[field] = _coerce_keywords(value)
                else:
                    result[field] = value
                continue
        if current == "palavras_chave" and re.match(r"^[\-\*\u2022]\s+", line):
            existing = _coerce_keywords(result.get(current, []))
            existing.extend(_coerce_keywords(re.sub(r"^[\-\*\u2022]\s+", "", line)))
            result[current] = _coerce_keywords(existing)
    return result


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Fallback deliberadamente simples para YAML de metadados sem PyYAML."""
    return _parse_labelled_research_text(text)


def _parse_labelled_paper_text(text: str) -> dict[str, Any]:
    """Lê rótulos de pesquisa e de redação de paper em MD/Org/TXT/YAML."""
    result: dict[str, Any] = {}
    current: str | None = None
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:[#>*\-\u2022]\s*)*([^:]{2,100}?)\s*:\s*(.*)$", line)
        if match:
            field = _canonical_research_field(match.group(1)) or _canonical_paper_field(match.group(1))
            if field:
                current = field
                value = match.group(2).strip()
                if field == "palavras_chave":
                    result[field] = _coerce_keywords(value)
                elif field in PAPER_LIST_FIELDS:
                    result[field] = _coerce_paper_list(value)
                else:
                    result[field] = value
                continue
        if current and (current == "palavras_chave" or current in PAPER_LIST_FIELDS) and re.match(r"^[\-\*\u2022]\s+", line):
            cleaner = re.sub(r"^[\-\*\u2022]\s+", "", line)
            existing = _coerce_keywords(result.get(current, []))
            existing.extend(_coerce_keywords(cleaner))
            result[current] = _coerce_keywords(existing)
    return result


def _read_structured_paper_file(
    raw_path: str,
    *,
    allow_declared_empty_fields: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Lê um arquivo de dados e diretrizes para ``paper_local_fgv``.

    Aceita os mesmos formatos do modelo de pesquisa e separa os blocos
    ``[pesquisa]`` e ``[paper]`` quando o arquivo é TOML/JSON/YAML.
    """
    path = Path(str(raw_path or "")).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Arquivo inexistente ou não é arquivo regular: {path}")
    suffix = path.suffix.lower()
    if suffix not in STRUCTURED_RESEARCH_SUFFIXES:
        allowed = ", ".join(sorted(STRUCTURED_RESEARCH_SUFFIXES))
        raise ValueError(f"Formato não suportado. Use um destes: {allowed}")
    source_text = path.read_text(encoding="utf-8", errors="replace")
    try:
        if suffix == ".toml":
            payload: Any = tomllib.loads(source_text)
        elif suffix == ".json":
            payload = json.loads(source_text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
                payload = yaml.safe_load(source_text)
            except ModuleNotFoundError:
                payload = _parse_labelled_paper_text(source_text)
        else:
            payload = _parse_labelled_paper_text(source_text)
    except (ValueError, TypeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Não foi possível ler o arquivo estruturado: {exc}") from exc

    values: dict[str, Any] = {
        **{field: ([] if field == "palavras_chave" else "") for field in RESEARCH_FIELD_LABELS},
        **{field: ([] if field in PAPER_LIST_FIELDS else "") for field in PAPER_FIELD_LABELS},
    }
    declared_fields: set[str] = set()
    if isinstance(payload, dict):
        for layer in _iter_mapping_layers(payload):
            normalised = {_normalise_research_key(key): value for key, value in layer.items()}
            for field, aliases in RESEARCH_FIELD_ALIASES.items():
                for alias in aliases:
                    if alias not in normalised:
                        continue
                    declared_fields.add(field)
                    if not values[field]:
                        raw_value = normalised[alias]
                        values[field] = _coerce_keywords(raw_value) if field == "palavras_chave" else _coerce_research_text(raw_value)
                    break
            for field, aliases in PAPER_FIELD_ALIASES.items():
                for alias in aliases:
                    if alias not in normalised:
                        continue
                    declared_fields.add(field)
                    if not values[field]:
                        raw_value = normalised[alias]
                        values[field] = _coerce_paper_list(raw_value) if field in PAPER_LIST_FIELDS else _coerce_research_text(raw_value)
                    break
    else:
        raise ValueError("O arquivo estruturado deve conter um objeto/dicionário ou campos identificados por rótulos.")

    all_labels = {**RESEARCH_FIELD_LABELS, **PAPER_FIELD_LABELS}
    has_values = any(values[field] for field in all_labels)
    if not has_values and not (allow_declared_empty_fields and declared_fields):
        readable = ", ".join(all_labels.values())
        raise ValueError(f"Não encontrei campos reconhecidos. Use rótulos como: {readable}.")
    return path, values


def _read_structured_research_file(
    raw_path: str,
    *,
    allow_declared_empty_fields: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Lê metadados estruturados de pesquisa.

    ``allow_declared_empty_fields`` é usado exclusivamente pelo modelo neutro
    copiado para a pasta do projeto: os campos já existem, mas ainda precisam
    ser preenchidos no assistente. Arquivos externos continuam exigindo ao
    menos um valor reconhecido para evitar aceitar um arquivo irrelevante.
    """
    path = Path(str(raw_path or "")).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Arquivo inexistente ou não é arquivo regular: {path}")
    suffix = path.suffix.lower()
    if suffix not in STRUCTURED_RESEARCH_SUFFIXES:
        allowed = ", ".join(sorted(STRUCTURED_RESEARCH_SUFFIXES))
        raise ValueError(f"Formato não suportado. Use um destes: {allowed}")
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        if suffix == ".toml":
            payload: Any = tomllib.loads(text)
        elif suffix == ".json":
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
                payload = yaml.safe_load(text)
            except ModuleNotFoundError:
                payload = _parse_simple_yaml(text)
        else:
            payload = _parse_labelled_research_text(text)
    except (ValueError, TypeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Não foi possível ler o arquivo estruturado: {exc}") from exc

    values: dict[str, Any] = {field: ([] if field == "palavras_chave" else "") for field in RESEARCH_FIELD_LABELS}
    declared_fields: set[str] = set()
    if isinstance(payload, dict):
        for layer in _iter_mapping_layers(payload):
            normalised = {_normalise_research_key(key): value for key, value in layer.items()}
            for field, aliases in RESEARCH_FIELD_ALIASES.items():
                for alias in aliases:
                    if alias not in normalised:
                        continue
                    declared_fields.add(field)
                    if not values[field]:
                        raw_value = normalised[alias]
                        values[field] = _coerce_keywords(raw_value) if field == "palavras_chave" else _coerce_research_text(raw_value)
                    break
    else:
        raise ValueError("O arquivo estruturado deve conter um objeto/dicionário ou campos identificados por rótulos.")

    has_values = any(values[field] for field in RESEARCH_FIELD_LABELS)
    if not has_values and not (allow_declared_empty_fields and declared_fields):
        readable = ", ".join(RESEARCH_FIELD_LABELS.values())
        raise ValueError(f"Não encontrei campos reconhecidos. Use rótulos como: {readable}.")
    return path, values


def _notify_structured_input_error(message: str) -> None:
    if tui_theme_enabled():
        _fgv_ui().message(_dialog_title(), message)
    else:
        print(f"\n{message}\n")


def ask_structured_research_file(prompt: str, *, required_fields: tuple[str, ...] = ()) -> tuple[str, dict[str, Any]]:
    """Solicita um arquivo e repete a seleção se não houver os campos necessários."""
    while True:
        raw = ask_required(prompt, "")
        try:
            _path, values = _read_structured_research_file(raw)
        except ValueError as exc:
            _notify_structured_input_error(str(exc))
            continue
        missing = [RESEARCH_FIELD_LABELS[field] for field in required_fields if not values.get(field)]
        if missing:
            _notify_structured_input_error(
                "O arquivo não contém os campos exigidos nesta etapa: " + ", ".join(missing) + "."
            )
            continue
        return raw, values


def _show_imported_research_values(path: str, values: dict[str, Any], fields: tuple[str, ...]) -> None:
    lines = [f"Arquivo: {Path(path).expanduser()}", ""]
    for field in fields:
        value = values.get(field, [])
        rendered = "; ".join(value) if isinstance(value, list) else str(value or "—")
        lines.append(f"{RESEARCH_FIELD_LABELS[field]}: {rendered}")
    text = "\n".join(lines)
    if tui_theme_enabled():
        _fgv_ui().message(_dialog_title(), text)
    else:
        print("\n" + text + "\n")


def review_imported_research_values(
    path: str,
    values: dict[str, Any],
    fields: tuple[str, ...],
    *,
    default_edit: bool = False,
) -> dict[str, Any]:
    """Permite revisar valores extraídos antes de salvá-los no TOML."""
    _show_imported_research_values(path, values, fields)
    if not ask_bool("Deseja editar os valores extraídos antes de continuar?", default_edit):
        return values
    reviewed = dict(values)
    for field in fields:
        if field == "palavras_chave":
            reviewed[field] = ask_list(RESEARCH_FIELD_LABELS[field], _coerce_keywords(reviewed.get(field)))
        else:
            reviewed[field] = ask(RESEARCH_FIELD_LABELS[field], _coerce_research_text(reviewed.get(field)))
    return reviewed


def _institution_template_path(data: dict[str, Any], candidates: list[str], *, description: str) -> Path:
    """Localiza um modelo no diretório ``misc/<instituicao>/``.

    O perfil institucional é escolhido antes das etapas de pesquisa e de
    diretrizes. Não há fallback silencioso para FGV: em uma instalação com
    outra instituição, a ausência do modelo deve ser explícita para evitar a
    cópia de regras de uma instituição errada.
    """
    institution = slugify(str(data.get("institution") or "")).strip()
    if not institution:
        raise ValueError("Nenhum perfil institucional foi definido para localizar o modelo estruturado.")
    directory = (APP / "misc" / institution).resolve()
    expected = [directory / name for name in candidates]
    for candidate in expected:
        if candidate.is_file():
            return candidate
    readable = "\n- ".join(str(item) for item in expected)
    raise ValueError(
        f"O modelo institucional de {description} não foi encontrado para o perfil '{institution}'. "
        f"Esperado em:\n- {readable}"
    )


def _structured_research_template_path(data: dict[str, Any]) -> Path:
    return _institution_template_path(
        data,
        [STRUCTURED_RESEARCH_TEMPLATE_FILENAME],
        description="dados de pesquisa PRISMA",
    )


def _structured_paper_template_path(data: dict[str, Any]) -> Path:
    institution = slugify(str(data.get("institution") or ""))
    return _institution_template_path(
        data,
        [f"{STRUCTURED_PAPER_TEMPLATE_BASENAME}_{institution}.toml", f"{STRUCTURED_PAPER_TEMPLATE_BASENAME}.toml"],
        description="diretrizes de paper",
    )


def _open_project_model_in_editor(path: Path) -> bool:
    """Abre uma cópia de modelo no editor configurado, com fallback guiado.

    Retorna ``True`` somente quando um editor externo foi iniciado e fechado;
    nesse caso o arquivo é relido para validar o TOML editado. ``EDITOR`` ou
    ``VISUAL`` é respeitado, sem impor editor, terminal ou interface gráfica.
    """
    choice = ask_choice(
        "Como deseja editar a cópia do modelo institucional",
        [
            "abrir no editor configurado (VISUAL/EDITOR)",
            "preencher os campos pelo assistente",
        ],
        "abrir no editor configurado (VISUAL/EDITOR)",
    )
    if choice != "abrir no editor configurado (VISUAL/EDITOR)":
        return False
    configured = str(os.getenv("VISUAL") or os.getenv("EDITOR") or "").strip()
    if not configured:
        _notify_structured_input_error(
            "Nenhum editor foi configurado em VISUAL ou EDITOR. "
            "Use a opção de preenchimento pelo assistente ou defina, por exemplo: export EDITOR='emacs -nw'."
        )
        return False
    try:
        command = shlex.split(configured)
    except ValueError as exc:
        _notify_structured_input_error(f"Não foi possível interpretar o editor configurado: {exc}")
        return False
    if not command:
        _notify_structured_input_error("EDITOR/VISUAL não contém um comando utilizável.")
        return False
    try:
        result = subprocess.run([*command, str(path)], check=False)
    except OSError as exc:
        _notify_structured_input_error(f"Não foi possível abrir o editor configurado: {exc}")
        return False
    if result.returncode != 0:
        _notify_structured_input_error(
            f"O editor foi encerrado com código {result.returncode}. O arquivo será relido; corrija-o se a validação falhar."
        )
    return True


def _toml_literal_for_research_field(field: str, value: Any) -> str:
    if field == "palavras_chave":
        items = _coerce_keywords(value)
        return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in items) + "]"
    return json.dumps(_coerce_research_text(value), ensure_ascii=False)


def _persist_research_values_in_project_model(path: Path, values: dict[str, Any], fields: tuple[str, ...]) -> None:
    """Atualiza apenas os campos de [pesquisa] da cópia TOML do projeto.

    O método preserva os comentários e o restante do modelo. O recurso é
    restrito a .toml porque o modelo padrão distribuído pelo bundle é TOML.
    """
    if path.suffix.lower() != ".toml":
        return
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = text.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        if re.match(r"^\s*\[\s*pesquisa\s*\]\s*(?:#.*)?$", line, flags=re.IGNORECASE):
            section_start = index
            continue
        if section_start is not None and index > section_start and re.match(r"^\s*\[.*?\]\s*(?:#.*)?$", line):
            section_end = index
            break

    if section_start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.append("[pesquisa]\n")
        section_start = len(lines) - 1
        section_end = len(lines)

    field_set = set(fields)
    updated: set[str] = set()
    assignment = re.compile(r"^(\s*)([^#=]+?)\s*=.*$")
    for index in range(section_start + 1, section_end):
        match = assignment.match(lines[index])
        if not match:
            continue
        field = _canonical_research_field(match.group(2))
        if field not in field_set:
            continue
        indent = match.group(1)
        lines[index] = f"{indent}{field} = {_toml_literal_for_research_field(field, values.get(field))}\n"
        updated.add(field)

    missing = [field for field in fields if field not in updated]
    if missing:
        additions = [f"{field} = {_toml_literal_for_research_field(field, values.get(field))}\n" for field in missing]
        lines[section_end:section_end] = additions
    path.write_text("".join(lines).rstrip() + "\n", encoding="utf-8")


def _project_model_destination(data: dict[str, Any], template: Path) -> Path:
    project_dir = Path(data["project_dir"]).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / PROJECT_RESEARCH_MODEL_FILENAME


def _choose_project_model_destination(data: dict[str, Any], template: Path) -> Path:
    target = _project_model_destination(data, template)
    while target.exists():
        action = ask_choice(
            "Já existe uma cópia do modelo estruturado na pasta do projeto. Como deseja prosseguir",
            [
                "usar a cópia existente",
                "restaurar a cópia a partir do modelo do bundle",
                "informar outro nome na pasta do projeto",
            ],
            "usar a cópia existente",
        )
        if action == "usar a cópia existente":
            return target
        if action == "restaurar a cópia a partir do modelo do bundle":
            shutil.copy2(template, target)
            return target
        raw_name = ask_required("Nome do arquivo de dados estruturados na pasta do projeto", PROJECT_RESEARCH_MODEL_FILENAME)
        candidate = (Path(data["project_dir"]).resolve() / raw_name).resolve()
        if candidate.parent != Path(data["project_dir"]).resolve():
            _notify_structured_input_error("Informe apenas o nome do arquivo; o modelo deve permanecer na pasta do projeto.")
            continue
        if candidate.suffix.lower() != ".toml":
            _notify_structured_input_error("O modelo estruturado deve usar a extensão .toml.")
            continue
        target = candidate
    shutil.copy2(template, target)
    return target


def ask_structured_research_source(
    data: dict[str, Any],
    prompt: str,
    *,
    required_fields: tuple[str, ...] = (),
    editable_fields: tuple[str, ...] = (),
    source_override: str | None = None,
) -> tuple[str, dict[str, Any], bool]:
    """Escolhe arquivo existente ou uma cópia editável do modelo institucional."""
    while True:
        choice = source_override or ask_choice(
            "Como deseja fornecer o arquivo estruturado",
            [
                "usar arquivo estruturado já existente",
                "copiar e editar o modelo institucional na pasta do projeto",
            ],
            "usar arquivo estruturado já existente",
        )
        source_override = None
        if choice == "usar arquivo estruturado já existente":
            raw_path, values = ask_structured_research_file(prompt, required_fields=required_fields)
            return raw_path, values, False

        try:
            template = _structured_research_template_path(data)
            destination = _choose_project_model_destination(data, template)
            edited_in_external_editor = _open_project_model_in_editor(destination)
            _path, values = _read_structured_research_file(
                str(destination),
                allow_declared_empty_fields=True,
            )
        except ValueError as exc:
            _notify_structured_input_error(str(exc))
            continue

        fields_to_edit = editable_fields or tuple(RESEARCH_FIELD_LABELS)
        _notify_structured_input_error(
            "Modelo copiado para a pasta do projeto:\n"
            f"{destination}\n\n"
            "Os ajustes feitos a seguir serão gravados nessa cópia; o arquivo do bundle não será alterado."
        )
        reviewed = values if edited_in_external_editor else review_imported_research_values(
            str(destination),
            values,
            fields_to_edit,
            default_edit=True,
        )
        _persist_research_values_in_project_model(destination, reviewed, fields_to_edit)

        missing = [RESEARCH_FIELD_LABELS[field] for field in required_fields if not reviewed.get(field)]
        if missing:
            _notify_structured_input_error(
                "A cópia do modelo ainda não contém os campos exigidos nesta etapa: " + ", ".join(missing) + "."
            )
            continue
        return str(destination), reviewed, True


# -----------------------------------------------------------------------------
# Arquivo estruturado para paper local
# -----------------------------------------------------------------------------

PAPER_STRUCTURED_FIELDS = (
    "tema",
    "recorte",
    "objetivo",
    "pergunta_pesquisa",
    "hipotese",
    "palavras_chave",
    "tipo_estudo",
    "tese_central",
    "estrutura_desejada",
    "argumentos_obrigatorios",
    "orientacoes_metodologicas",
    "limites_do_escopo",
    "tom_de_redacao",
    "instrucoes_adicionais",
)
PAPER_CORE_RESEARCH_FIELDS = ("tema", "recorte", "objetivo", "pergunta_pesquisa", "tipo_estudo")


def _paper_field_label(field: str) -> str:
    return PAPER_FIELD_LABELS.get(field) or RESEARCH_FIELD_LABELS.get(field) or field


def _show_imported_paper_values(path: str, values: dict[str, Any], fields: tuple[str, ...]) -> None:
    lines = [f"Arquivo: {Path(path).expanduser()}", ""]
    for field in fields:
        value = values.get(field, [])
        rendered = "; ".join(value) if isinstance(value, list) else str(value or "—")
        lines.append(f"{_paper_field_label(field)}: {rendered}")
    message = "\n".join(lines)
    if tui_theme_enabled():
        _fgv_ui().message(_dialog_title(), message)
    else:
        print("\n" + message + "\n")


def review_imported_paper_values(
    path: str,
    values: dict[str, Any],
    fields: tuple[str, ...],
    *,
    default_edit: bool = False,
) -> dict[str, Any]:
    """Mostra e permite ajustar os dados estruturados antes de salvar o TOML."""
    _show_imported_paper_values(path, values, fields)
    if not ask_bool("Deseja editar os valores extraídos antes de continuar?", default_edit):
        return values
    reviewed = dict(values)
    for field in fields:
        if field == "palavras_chave" or field in PAPER_LIST_FIELDS:
            existing = _coerce_keywords(reviewed.get(field))
            reviewed[field] = ask_list(_paper_field_label(field), existing)
        else:
            reviewed[field] = ask(_paper_field_label(field), _coerce_research_text(reviewed.get(field)))
    return reviewed


def _toml_literal_for_paper_field(field: str, value: Any) -> str:
    if field == "palavras_chave" or field in PAPER_LIST_FIELDS:
        items = _coerce_keywords(value)
        return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in items) + "]"
    return json.dumps(_coerce_research_text(value), ensure_ascii=False)


def _persist_toml_section_fields(
    path: Path,
    section_name: str,
    values: dict[str, Any],
    fields: tuple[str, ...],
    canonical_field: Any,
    literal: Any,
) -> None:
    """Atualiza campos de uma seção TOML preservando comentários não relacionados.

    As listas do modelo podem ocupar múltiplas linhas. Ao substituir uma lista,
    o bloco anterior é consumido por inteiro para não deixar sobras inválidas de
    TOML na cópia editável do projeto.
    """
    if path.suffix.lower() != ".toml":
        return
    source = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = source.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    header = re.compile(rf"^\s*\[\s*{re.escape(section_name)}\s*\]\s*(?:#.*)?$", flags=re.IGNORECASE)
    for index, line in enumerate(lines):
        if header.match(line):
            section_start = index
            continue
        if section_start is not None and index > section_start and re.match(r"^\s*\[.*?\]\s*(?:#.*)?$", line):
            section_end = index
            break
    if section_start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.append(f"[{section_name}]\n")
        section_start = len(lines) - 1
        section_end = len(lines)

    before = lines[: section_start + 1]
    body = lines[section_start + 1 : section_end]
    after = lines[section_end:]
    field_set = set(fields)
    updated: set[str] = set()
    assignment = re.compile(r"^(\s*)([^#=]+?)\s*=.*$")
    new_body: list[str] = []
    index = 0
    while index < len(body):
        line = body[index]
        match = assignment.match(line)
        field = canonical_field(match.group(2)) if match else None
        if field not in field_set:
            new_body.append(line)
            index += 1
            continue

        new_body.append(f"{match.group(1)}{field} = {literal(field, values.get(field))}\n")
        updated.add(field)
        rhs = line.split("=", 1)[1]
        bracket_depth = rhs.count("[") - rhs.count("]")
        index += 1
        while bracket_depth > 0 and index < len(body):
            bracket_depth += body[index].count("[") - body[index].count("]")
            index += 1

    for field in fields:
        if field not in updated:
            new_body.append(f"{field} = {literal(field, values.get(field))}\n")
    path.write_text("".join(before + new_body + after).rstrip() + "\n", encoding="utf-8")


def _project_paper_model_destination(data: dict[str, Any], template: Path) -> Path:
    project_dir = Path(data["project_dir"]).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / PROJECT_PAPER_MODEL_FILENAME


def _choose_project_paper_model_destination(data: dict[str, Any], template: Path) -> Path:
    target = _project_paper_model_destination(data, template)
    while target.exists():
        action = ask_choice(
            "Já existe uma cópia do modelo de paper na pasta do projeto. Como deseja prosseguir",
            [
                "usar a cópia existente",
                "restaurar a cópia a partir do modelo do bundle",
                "informar outro nome na pasta do projeto",
            ],
            "usar a cópia existente",
        )
        if action == "usar a cópia existente":
            return target
        if action == "restaurar a cópia a partir do modelo do bundle":
            shutil.copy2(template, target)
            return target
        raw_name = ask_required("Nome do arquivo de diretrizes estruturadas do paper", PROJECT_PAPER_MODEL_FILENAME)
        candidate = (Path(data["project_dir"]).resolve() / raw_name).resolve()
        if candidate.parent != Path(data["project_dir"]).resolve():
            _notify_structured_input_error("Informe apenas o nome do arquivo; o modelo deve permanecer na pasta do projeto.")
            continue
        if candidate.suffix.lower() != ".toml":
            _notify_structured_input_error("O modelo estruturado do paper deve usar a extensão .toml.")
            continue
        target = candidate
    shutil.copy2(template, target)
    return target


def ask_structured_paper_source(
    data: dict[str, Any],
    prompt: str,
    *,
    required_fields: tuple[str, ...] = (),
    editable_fields: tuple[str, ...] = PAPER_STRUCTURED_FIELDS,
) -> tuple[str, dict[str, Any], bool]:
    """Seleciona ou cria a fonte estruturada de dados e diretrizes do paper."""
    while True:
        choice = ask_choice(
            "Como deseja fornecer o arquivo estruturado do paper",
            [
                "usar arquivo estruturado já existente",
                "copiar e editar o modelo institucional na pasta do projeto",
            ],
            "usar arquivo estruturado já existente",
        )
        if choice == "usar arquivo estruturado já existente":
            raw_path = ask_required(prompt, "")
            try:
                _path, values = _read_structured_paper_file(raw_path)
            except ValueError as exc:
                _notify_structured_input_error(str(exc))
                continue
            missing = [_paper_field_label(field) for field in required_fields if not values.get(field)]
            if missing:
                _notify_structured_input_error("O arquivo não contém os campos exigidos nesta etapa: " + ", ".join(missing) + ".")
                continue
            return raw_path, values, False

        try:
            template = _structured_paper_template_path(data)
            destination = _choose_project_paper_model_destination(data, template)
            edited_in_external_editor = _open_project_model_in_editor(destination)
            _path, values = _read_structured_paper_file(
                str(destination),
                allow_declared_empty_fields=True,
            )
        except ValueError as exc:
            _notify_structured_input_error(str(exc))
            continue

        _notify_structured_input_error(
            "Modelo copiado para a pasta do projeto:\n"
            f"{destination}\n\n"
            "Os ajustes feitos a seguir serão gravados nessa cópia; o arquivo do bundle não será alterado."
        )
        reviewed = values if edited_in_external_editor else review_imported_paper_values(
            str(destination),
            values,
            editable_fields,
            default_edit=True,
        )
        research_fields = tuple(field for field in editable_fields if field in RESEARCH_FIELD_LABELS)
        paper_fields = tuple(field for field in editable_fields if field in PAPER_FIELD_LABELS)
        if research_fields:
            _persist_toml_section_fields(
                destination,
                "pesquisa",
                reviewed,
                research_fields,
                _canonical_research_field,
                _toml_literal_for_research_field,
            )
        if paper_fields:
            _persist_toml_section_fields(
                destination,
                "paper",
                reviewed,
                paper_fields,
                _canonical_paper_field,
                _toml_literal_for_paper_field,
            )
        missing = [_paper_field_label(field) for field in required_fields if not reviewed.get(field)]
        if missing:
            _notify_structured_input_error(
                "A cópia do modelo ainda não contém os campos exigidos nesta etapa: " + ", ".join(missing) + "."
            )
            continue
        return str(destination), reviewed, True


# -----------------------------------------------------------------------------
# Wizard UI
# -----------------------------------------------------------------------------

WIZARD_NO_CLEAR = os.getenv("ACADEMIC_PIPELINE_TOML_NO_CLEAR", "").strip().lower() in {"1", "true", "s", "sim", "yes", "y"}


def set_wizard_no_clear(value: bool) -> None:
    global WIZARD_NO_CLEAR
    WIZARD_NO_CLEAR = bool(value)


def clear_screen() -> None:
    if tui_theme_enabled():
        return
    if WIZARD_NO_CLEAR:
        print("\n" + "=" * 92 + "\n")
        return
    cmd = "cls" if os.name == "nt" else "clear"
    os.system(cmd)


def wizard_header(title: str, index: int | None = None, total: int | None = None, preset: "Preset | None" = None) -> None:
    global TUI_STAGE_TITLE
    stage = f"Etapa {index}/{total} — {title}" if index is not None and total is not None else title
    if preset is not None:
        stage = f"{stage} | {preset.label}"
    TUI_STAGE_TITLE = stage
    if tui_theme_enabled():
        return
    clear_screen()
    print("academic_pipeline TOML Wizard")
    if preset is not None:
        print(f"Perfil: {preset.key} — {preset.label}")
    if index is not None and total is not None:
        print(f"Etapa {index}/{total} — {title}")
    else:
        print(title)
    print("-" * 92)
    print("Comandos de navegação ao final da etapa: próxima | voltar | refazer | resumo | cancelar")
    print("Use Ctrl+C para cancelar imediatamente.\n")


def short_value(value: Any, max_len: int = 110) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        s = "; ".join(str(x) for x in value)
    else:
        s = str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def wizard_summary(data: dict[str, Any]) -> str:
    preset = data.get("preset")
    preset_key = getattr(preset, "key", "") if preset else ""
    rows: list[tuple[str, list[tuple[str, Any]]]] = [
        ("Projeto e perfil", [
            ("nome", data.get("project_name")),
            ("slug", data.get("project_slug")),
            ("perfil", preset_key),
            ("instituição", data.get("institution")),
            ("layout", data.get("layout")),
            ("TOML", data.get("config_path")),
        ]),
        ("Metadados", [
            ("título", data.get("titulo")),
            ("autor", data.get("autor")),
            ("curso", data.get("curso")),
            ("turma", data.get("turma")),
            ("disciplina", data.get("disciplina")),
            ("professor", data.get("professor")),
            ("data", data.get("data")),
        ]),
        ("Dados acadêmicos da atividade/pesquisa", [
            ("fonte dos dados", data.get("atividade_dados_modo")),
            ("dados da atividade por IA", data.get("atividade_gerar_dados_ia")),
            ("arquivo com dados da atividade", data.get("atividade_dados_paths")),
            ("tema", data.get("tema")),
            ("recorte", data.get("recorte")),
            ("objetivo", data.get("objetivo")),
            ("pergunta orientadora", data.get("pergunta_pesquisa")),
            ("hipótese/tese", data.get("hipotese")),
            ("enunciado/orientação", data.get("orientacao_professor")),
            ("palavras-chave", "automáticas pela IA" if data.get("gerar_palavras_chave_ia") else data.get("palavras_chave")),
            ("idiomas do corpus", data.get("idiomas")),
            ("resumo no idioma principal", data.get("gerar_resumo_principal")),
            ("resumos adicionais", data.get("idiomas_resumo_adicionais")),
            ("palavras-chave nos resumos adicionais", data.get("gerar_palavras_chave_resumo_adicionais")),
            ("versões adicionais por IA", data.get("idiomas_adicionais_saida")),
        ]),
        ("Proveniência dos dados da revisão", [
            ("fonte tema/recorte/objetivo", data.get("fonte_dados_pesquisa")),
            ("arquivo de dados da pesquisa", data.get("dados_pesquisa_path")),
            ("fonte das palavras-chave", data.get("fonte_palavras_chave")),
            ("arquivo de palavras-chave", data.get("palavras_chave_path")),
            ("fonte da hipótese/tese", data.get("fonte_hipotese")),
            ("arquivo da hipótese/tese", data.get("hipotese_path")),
        ]),
        ("Busca bibliográfica externa", [
            ("bases", data.get("bases_busca")),
            ("consulta geral", data.get("consulta_geral")),
            ("limite por base", data.get("limite_por_base")),
            ("meta de estudos incluídos", data.get("meta_estudos_incluidos")),
            ("corpus local complementar", data.get("corpus_local_opcional")),
        ]),
        ("Orientações estruturadas do paper", [
            ("fonte", data.get("fonte_orientacoes_paper")),
            ("arquivo estruturado", data.get("orientacoes_paper_path")),
            ("tese central", data.get("tese_central")),
            ("estrutura desejada", data.get("estrutura_desejada")),
            ("argumentos obrigatórios", data.get("argumentos_obrigatorios")),
            ("tom de redação", data.get("tom_de_redacao")),
        ]),
        ("Corpus, orientações e prompts", [
            ("input_zip", data.get("documentos_input_zip")),
            ("input_dir", data.get("documentos_input_dir")),
            ("orientações", data.get("orientacoes_paths")),
            ("modo das orientações gerais", data.get("orientacao_geral_modo")),
            ("orientação geral digitada", data.get("orientacao_geral_inline")),
            ("prompt específico", data.get("prompt_extra_paths")),
            ("orientação inline", data.get("orientacao_professor")),
            ("doi_manifest", data.get("doi_manifest_path")),
        ]),
        ("Saídas", [
            ("document_output_dir", data.get("document_output_dir") or "../../output/documento"),
            ("subdiretório", data.get("create_document_subdir")),
            ("ORG", data.get("exportar_org")),
            ("PDF", data.get("exportar_pdf")),
            ("DOCX", data.get("exportar_docx")),
            ("mapa mental", data.get("gerar_mapa_mental")),
            ("estilo", data.get("estilo")),
        ]),
    ]
    if not any(
        data.get(key)
        for key in (
            "fonte_dados_pesquisa", "dados_pesquisa_path", "fonte_palavras_chave",
            "palavras_chave_path", "fonte_hipotese", "hipotese_path",
        )
    ):
        rows = [item for item in rows if item[0] != "Proveniência dos dados da revisão"]

    if not any(
        data.get(key)
        for key in (
            "fonte_orientacoes_paper", "orientacoes_paper_path", "tese_central",
            "estrutura_desejada", "argumentos_obrigatorios", "tom_de_redacao",
        )
    ):
        rows = [item for item in rows if item[0] != "Orientações estruturadas do paper"]

    out: list[str] = []
    for section, items in rows:
        out.append(section + ":")
        for k, v in items:
            if v not in (None, "", [], {}):
                out.append(f"  - {k}: {short_value(v)}")
        out.append("")
    return "\n".join(out).rstrip()


def show_summary(data: dict[str, Any]) -> None:
    if tui_theme_enabled():
        _fgv_ui().message(_dialog_title(), wizard_summary(data) or "Ainda não há dados suficientes.")
        return
    print("\nResumo atual da configuração")
    print("=" * 92)
    print(wizard_summary(data) or "Ainda não há dados suficientes.")
    print("=" * 92 + "\n")


def ask_stage_action(allow_back: bool = True) -> str:
    if tui_theme_enabled():
        values = [("proxima", "[P] Próxima etapa"), ("refazer", "[R] Refazer esta etapa"), ("resumo", "[S] Ver resumo atual")]
        if allow_back:
            values.insert(1, ("voltar", "[V] Voltar à etapa anterior"))
        values.append(("cancelar", "[C] Cancelar sem salvar"))
        selected = _fgv_ui().select_one(
            _dialog_title(),
            "Revise o que foi informado e escolha o próximo movimento.",
            values,
            default="proxima",
        )
        return str(selected or "cancelar")
    choices = {"p", "proxima", "próxima", "voltar", "v", "refazer", "r", "resumo", "cancelar", "c"}
    while True:
        raw = input("Ação [próxima/voltar/refazer/resumo/cancelar] (Enter=próxima): ").strip().lower()
        if not raw:
            return "proxima"
        if raw in choices:
            if raw in {"p", "proxima", "próxima"}:
                return "proxima"
            if raw in {"v", "voltar"}:
                return "voltar" if allow_back else "refazer"
            if raw in {"r", "refazer"}:
                return "refazer"
            if raw == "resumo":
                return "resumo"
            if raw in {"c", "cancelar"}:
                return "cancelar"
        print("Escolha uma ação válida: próxima, voltar, refazer, resumo ou cancelar.")


def ask_final_action() -> str:
    if tui_theme_enabled():
        selected = _fgv_ui().select_one(
            _dialog_title(),
            "Revise a configuração antes de gravar o TOML.",
            [
                ("salvar", "[S] Salvar TOML"),
                ("editar", "[E] Editar uma etapa"),
                ("cancelar", "[C] Cancelar sem salvar"),
            ],
            default="salvar",
        )
        return str(selected or "cancelar")
    while True:
        raw = input("Salvar TOML? [s=salvar / e=editar etapa / n=cancelar] [s]: ").strip().lower()
        if not raw or raw in {"s", "sim", "salvar"}:
            return "salvar"
        if raw in {"e", "editar"}:
            return "editar"
        if raw in {"n", "nao", "não", "cancelar"}:
            return "cancelar"
        print("Responda s, e ou n.")


def ask_stage_number(total: int) -> int:
    if tui_theme_enabled():
        selected = _fgv_ui().select_one(
            _dialog_title(),
            "Selecione a etapa que deseja editar.",
            [(index, f"Etapa {index + 1}") for index in range(total)],
            default=0,
        )
        if selected is None:
            raise SystemExit("Geração de TOML cancelada pelo usuário.")
        return int(selected)
    while True:
        raw = input(f"Qual etapa deseja editar? [1-{total}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= total:
            return int(raw) - 1
        print("Número inválido.")

def list_institutions() -> list[str]:
    base = APP / "institutions"
    if not base.exists():
        return ["fgv"]
    items = sorted(p.name for p in base.iterdir() if p.is_dir())
    return items or ["fgv"]


def load_institution_profile_toml(institution: str) -> dict[str, Any]:
    path = APP / "institutions" / institution / "institution_profile.toml"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def available_layouts_for_institution(institution: str) -> dict[str, dict[str, Any]]:
    profile = load_institution_profile_toml(institution)
    layouts = profile.get("layouts", {}) if isinstance(profile.get("layouts"), dict) else {}
    return {str(k): v for k, v in layouts.items() if isinstance(v, dict)}


def default_layout_for_preset(institution: str, preset: "Preset") -> str:
    profile = load_institution_profile_toml(institution)
    layouts = profile.get("layouts", {}) if isinstance(profile.get("layouts"), dict) else {}
    content_types = profile.get("document_content_types", {}) if isinstance(profile.get("document_content_types"), dict) else {}
    if preset.key == "resumo_artigos_local_fgv" and isinstance(content_types.get("resumo_artigos"), dict):
        val = str(content_types["resumo_artigos"].get("default_layout") or "").strip()
        if val:
            return val
    genero = preset.document_type
    if preset.key == "resumo_artigos_local_fgv":
        genero = "atividade"
    for layout_id, spec in layouts.items():
        if str(spec.get("genero_academico") or spec.get("genero") or "").strip().lower() == genero:
            return layout_id
    if institution == "fgv":
        return {"atividade": "atividade_fgv", "paper": "paper_fgv", "dissertacao": "dissertacao_fgv"}.get(genero, "paper_fgv")
    return genero


def latex_class_for_layout(institution: str, layout_id: str, genero: str) -> str:
    spec = available_layouts_for_institution(institution).get(layout_id, {})
    val = str(spec.get("classe_latex") or spec.get("latex_class") or "").strip()
    if val:
        return val
    if genero == "dissertacao":
        return "fgv-dissertacao" if institution == "fgv" else "article"
    return "fgv-paper" if institution == "fgv" else "article"


# -----------------------------------------------------------------------------
# Perfis
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    document_type: str
    local_corpus: bool
    prisma_report: bool
    executar_documento: bool
    executar_pesquisa: bool
    render_only: bool = False
    default_toml: str = "config.toml"


PRESETS: list[Preset] = [
    Preset(
        key="atividade_local_fgv",
        label="Atividade local FGV",
        description="Gera atividade acadêmica com Ficha Técnica FGV a partir de corpus local. Não faz busca externa nem relatório PRISMA.",
        document_type="atividade",
        local_corpus=True,
        prisma_report=False,
        executar_documento=True,
        executar_pesquisa=False,
        default_toml="atividade_config.toml",
    ),
    Preset(
        key="resumo_artigos_local_fgv",
        label="Resumo analítico de artigos locais FGV",
        description="Gera documento FGV com Ficha Técnica, introdução, resumos individuais, comparação, síntese analítica, considerações finais, referências e mapa mental opcional a partir de corpus local. Não faz busca externa nem relatório PRISMA.",
        # Mantém tipo_documento=atividade para preservar a Ficha Técnica FGV no ORG/PDF/DOCX.
        # A especialização semântica fica no preset, no papertype, no prompt e na seção [resumo_artigos].
        document_type="atividade",
        local_corpus=True,
        prisma_report=False,
        executar_documento=True,
        executar_pesquisa=False,
        default_toml="resumo_artigos_config.toml",
    ),
    Preset(
        key="paper_local_fgv",
        label="Paper local FGV",
        description="Gera paper acadêmico FGV a partir de PDFs/DOCX/TXT locais, com ORG/PDF/DOCX, conformidade e qualidade.",
        document_type="paper",
        local_corpus=True,
        prisma_report=False,
        executar_documento=True,
        executar_pesquisa=False,
        default_toml="paper_config.toml",
    ),
    Preset(
        key="paper_prisma_fgv",
        label="Paper + relatório PRISMA FGV",
        description="Gera paper e saída própria de relatório PRISMA. O relatório usa corpus local, prisma_report.json ou diretório de pesquisa/triagem existente.",
        document_type="paper",
        local_corpus=True,
        prisma_report=True,
        executar_documento=True,
        executar_pesquisa=True,
        default_toml="paper_prisma_config.toml",
    ),
    Preset(
        key="dissertacao_local_fgv",
        label="Dissertação local FGV",
        description="Gera dissertação FGV a partir de corpus local, com campos de orientador, área, linha e pré-textuais básicos.",
        document_type="dissertacao",
        local_corpus=True,
        prisma_report=False,
        executar_documento=True,
        executar_pesquisa=False,
        default_toml="dissertacao_config.toml",
    ),
    Preset(
        key="dissertacao_prisma_fgv",
        label="Dissertação + relatório PRISMA FGV",
        description="Gera dissertação e relatório PRISMA auditável a partir de corpus local ou dados de pesquisa existentes.",
        document_type="dissertacao",
        local_corpus=True,
        prisma_report=True,
        executar_documento=True,
        executar_pesquisa=True,
        default_toml="dissertacao_prisma_config.toml",
    ),
    Preset(
        key="relatorio_prisma_fgv",
        label="Relatório PRISMA autônomo FGV",
        description="Gera apenas o relatório de pesquisa/PRISMA. Observação: a versão atual ainda exige corpus local, prisma_report.json ou diretório de pesquisa existente como insumo.",
        document_type="relatorio_prisma",
        local_corpus=True,
        prisma_report=True,
        executar_documento=False,
        executar_pesquisa=True,
        default_toml="relatorio_prisma_config.toml",
    ),
    Preset(
        key="relatorio_prisma_busca_orientada_fgv",
        label="Relatório PRISMA com busca orientada FGV",
        description="Executa busca bibliográfica externa rastreável nas fontes configuráveis, deduplica registros, gera triagem humana e produz relatório PRISMA preliminar/final em ORG e PDF no layout institucional escolhido. O fluxo começa pela descoberta externa e não exige corpus local.",
        document_type="relatorio_prisma",
        local_corpus=False,
        prisma_report=True,
        executar_documento=False,
        executar_pesquisa=True,
        default_toml="relatorio_prisma_busca_orientada_config.toml",
    ),
    Preset(
        key="somente_renderizar_fgv",
        label="Somente renderizar document.json FGV",
        description="Cria TOML para renderizar um document.json existente em ORG/PDF/DOCX, sem recriar conteúdo pela IA.",
        document_type="paper",
        local_corpus=False,
        prisma_report=False,
        executar_documento=True,
        executar_pesquisa=False,
        render_only=True,
        default_toml="render_config.toml",
    ),
]


def choose_preset() -> Preset:
    print("\nGerador interativo de TOML — academic_pipeline rc10.7.42")
    print("\nPerfis disponíveis:\n")
    for i, p in enumerate(PRESETS, start=1):
        print(f"{i}. {p.label}")
        print(textwrap.fill("   " + p.description, width=92, subsequent_indent="   "))
    print(f"{len(PRESETS) + 1}. Modo avançado por componentes")
    print("   Monta um perfil customizado escolhendo documento, corpus, relatório PRISMA e saídas.")
    while True:
        choice = ask("\nEscolha o perfil", "1")
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(PRESETS):
                return PRESETS[n - 1]
            if n == len(PRESETS) + 1:
                return choose_advanced_preset()
        by_key = {p.key: p for p in PRESETS}
        if choice in by_key:
            return by_key[choice]
        print("Escolha inválida.")


def choose_advanced_preset() -> Preset:
    print("\nModo avançado por componentes")
    document_type = ask_choice("Tipo de documento", ["atividade", "paper", "dissertacao", "relatorio_prisma"], "paper")
    local_corpus = ask_bool("Usar corpus local?", True)
    prisma_report = ask_bool("Gerar relatório PRISMA como saída própria?", document_type == "relatorio_prisma")
    executar_documento = document_type != "relatorio_prisma" and ask_bool("Gerar documento acadêmico final?", True)
    executar_pesquisa = prisma_report or ask_bool("Marcar executar_pesquisa=true?", False)
    default_toml = f"{document_type}_config.toml" if document_type != "relatorio_prisma" else "relatorio_prisma_config.toml"
    return Preset(
        key="custom_avancado",
        label="Custom avançado",
        description="Perfil customizado montado por componentes.",
        document_type=document_type,
        local_corpus=local_corpus,
        prisma_report=prisma_report,
        executar_documento=executar_documento,
        executar_pesquisa=executar_pesquisa,
        default_toml=default_toml,
    )


# -----------------------------------------------------------------------------
# Coleta de dados
# -----------------------------------------------------------------------------


def collect_common(preset: Preset) -> dict[str, Any]:
    institutions = list_institutions()
    default_institution = "fgv" if "fgv" in institutions else institutions[0]
    print(f"\nPerfil selecionado: {preset.label}")
    print(textwrap.fill(preset.description, width=92))

    if preset.key == "resumo_artigos_local_fgv":
        default_project_name = "resumo_artigos_encontro_X"
    elif preset.document_type == "atividade":
        default_project_name = "atividade_aula_2"
    else:
        default_project_name = f"{preset.document_type}_meu_tema"
    project_name = ask_required("Nome do projeto", default_project_name)
    project_slug = slugify(project_name)
    default_project_dir = APP / "projetos" / project_slug
    if WIZARD_PROJECT_DIR_OVERRIDE is not None:
        project_dir = WIZARD_PROJECT_DIR_OVERRIDE
    else:
        raw_project_dir = ask(
            "Diretório do projeto para salvar o TOML e os arquivos auxiliares",
            str(default_project_dir),
        )
        project_dir = Path(raw_project_dir).expanduser() if raw_project_dir.strip() else default_project_dir
        project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    institution = ask_choice("Perfil institucional", institutions, default_institution)

    layouts = available_layouts_for_institution(institution)
    default_layout = default_layout_for_preset(institution, preset)
    if layouts:
        print("\nLayouts disponíveis para " + institution + ":")
        for layout_id, spec in layouts.items():
            desc = str(spec.get("description") or spec.get("descricao") or "").strip()
            genero = str(spec.get("genero_academico") or "").strip()
            suffix = ""
            if genero or desc:
                suffix = " — " + "; ".join(x for x in [genero, desc] if x)
            print(f"- {layout_id}{suffix}")
        layout = ask_choice("Layout institucional", list(layouts.keys()), default_layout if default_layout in layouts else next(iter(layouts.keys())))
    else:
        layout = ask("Layout institucional", default_layout)

    while True:
        toml_name = ask("Nome do arquivo TOML (vazio para usar o padrão interno)", "").strip() or preset.default_toml
        candidate = Path(toml_name).expanduser()
        if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != toml_name:
            _notify_structured_input_error(
                "Informe somente o nome do TOML, sem diretórios nem caminho absoluto. "
                "O arquivo final será salvo dentro do diretório do projeto escolhido."
            )
            continue
        if candidate.suffix.casefold() != ".toml":
            _notify_structured_input_error("O arquivo final deve ter a extensão .toml.")
            continue
        config_path = (project_dir / candidate.name).resolve()
        break

    genero_academico = "atividade" if preset.key == "resumo_artigos_local_fgv" else preset.document_type
    tipo_conteudo = "resumo_artigos" if preset.key == "resumo_artigos_local_fgv" else preset.document_type
    classe_latex = latex_class_for_layout(institution, layout, genero_academico)

    data: dict[str, Any] = {
        "preset": preset,
        "project_name": project_name,
        "project_slug": project_slug,
        "project_dir": project_dir,
        "institution": institution,
        "layout": layout,
        "genero_academico": genero_academico,
        "tipo_conteudo": tipo_conteudo,
        "classe_latex": classe_latex,
        "config_path": config_path,
        "config_dir": project_dir,
    }
    return data


def collect_metadata(data: dict[str, Any]) -> None:
    preset: Preset = data["preset"]
    print("\nMetadados acadêmicos")
    year = str(date.today().year)
    if preset.key == "resumo_artigos_local_fgv":
        default_title = "Resumo analítico dos textos"
    else:
        default_title = {
            "atividade": "Atividade acadêmica",
            "paper": "Título do paper",
            "dissertacao": "Título da dissertação: subtítulo se houver",
            "relatorio_prisma": "Relatório de Pesquisa PRISMA",
        }.get(preset.document_type, "Documento acadêmico")
    data["titulo"] = ask("Título do trabalho", default_title)
    data["autor"] = ask("Autor/aluno", "Gustavo M. Mendes de Tarso")
    data["curso"] = ask("Curso", "Mestrado Acadêmico em Políticas Públicas e Governo")
    data["turma"] = ask("Turma", "2026.1")
    data["polo"] = ask("Pólo/cidade", "Brasília")
    data["disciplina"] = ask("Disciplina", "")
    data["professor"] = ask("Professor", "")
    data["data"] = ask("Data/ano", year)

    if preset.document_type == "dissertacao":
        print("\nCampos específicos de dissertação")
        data["area_de_concentracao"] = ask("Área de concentração", "Políticas Públicas e Governo")
        data["linha_pesquisa"] = ask("Linha de pesquisa", "")
        data["orientador"] = ask("Orientador", data.get("professor", ""))
        data["coorientador"] = ask("Coorientador", "")
        data["data_aprovacao"] = ask("Data de aprovação", "")
        data["natureza_trabalho"] = ask(
            "Natureza do trabalho",
            f"Dissertação apresentada à Fundação Getulio Vargas, como requisito para obtenção do título de Mestre em {data['area_de_concentracao']}.",
        )
    else:
        data["area_de_concentracao"] = ""
        data["linha_pesquisa"] = ""
        data["orientador"] = ""
        data["coorientador"] = ""
        data["data_aprovacao"] = ""
        data["natureza_trabalho"] = ""


DEFAULT_RESUMO_ARTIGOS_TEMA = (
    "Corpus local de textos acadêmicos. Produzir leitura analítica aprofundada de cada texto, "
    "reconstruindo problema, objetivo, argumento central, estrutura conceitual, método/evidências, "
    "contribuições, limites e diálogo com o restante do corpus, sem forçar unidade temática artificial."
)

DEFAULT_RESUMO_ARTIGOS_RECORTE = (
    "Comparar os textos apenas em aspectos transversais realmente presentes no corpus: problema abordado, "
    "argumento central, conceitos utilizados, tipo de abordagem, método/evidências, contribuição para a disciplina "
    "e limites analíticos. Não forçar convergências ou unidade temática artificial."
)

DEFAULT_RESUMO_ARTIGOS_OBJETIVO = (
    "Elaborar análise acadêmica aprofundada de cada texto do corpus local, comparando conexões reais entre "
    "eles e produzindo uma síntese interpretativa útil para a disciplina, com fidelidade ao material fornecido "
    "e sem transformar o resultado em resumo meramente sinóptico."
)

DEFAULT_RESUMO_ARTIGOS_PALAVRAS = [
    "resumo analítico",
    "textos acadêmicos",
    "análise comparativa",
    "políticas públicas",
    "administração pública",
]


def _print_default_block(title: str, value: str) -> None:
    print(f"\n{title} padrão:")
    print(textwrap.fill(value, width=92, initial_indent="  ", subsequent_indent="  "))


def _parse_ai_keyword_response(content: Any) -> list[str]:
    text = str(content or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = text
    if isinstance(payload, dict):
        for key in ("palavras_chave", "keywords", "termos"):
            if key in payload:
                return _coerce_keywords(payload[key])
    return _coerce_keywords(payload)


def suggest_prisma_keywords_with_ai(context: dict[str, Any]) -> list[str]:
    """Gera termos candidatos no wizard e devolve-os para revisão explícita."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("A sugestão por IA requer as dependências openai e python-dotenv do ambiente Pipenv.") from exc

    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não foi encontrada no ambiente ou no arquivo .env.")

    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    prompt = (
        "Você apoia a formulação de estratégias de busca para uma revisão PRISMA. "
        "Com base somente no contexto informado, proponha entre 6 e 12 palavras-chave ou expressões de busca, "
        "em português e inglês quando isso ampliar a recuperação. Não escreva uma string booleana, não crie "
        "critérios de inclusão/exclusão e não trate a hipótese como fato. Retorne exclusivamente JSON válido no formato "
        '{"palavras_chave":["termo 1","termo 2"]}.\\n\\n'
        f"Tema: {context.get('tema', '')}\\n"
        f"Recorte: {context.get('recorte', '')}\\n"
        f"Objetivo: {context.get('objetivo', '')}\\n"
        f"Pergunta de pesquisa: {context.get('pergunta_pesquisa', '')}\\n"
        f"Tipo de estudo: {context.get('tipo_estudo', '')}"
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Responda com precisão metodológica e apenas o JSON solicitado."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content if response.choices else ""
    except Exception as exc:
        raise RuntimeError(f"Não foi possível obter sugestões de palavras-chave pela IA: {exc}") from exc
    keywords = _parse_ai_keyword_response(content)
    if not keywords:
        raise RuntimeError("A IA não retornou palavras-chave utilizáveis.")
    return keywords


def _collect_optional_hypothesis(
    data: dict[str, Any],
    *,
    primary_path: str,
    primary_values: dict[str, Any],
) -> None:
    data["hipotese"] = ""
    data["fonte_hipotese"] = "nao_informada"
    data["hipotese_path"] = ""
    data["hipotese_paths"] = []
    if not ask_bool(
        "Deseja informar hipótese/tese orientadora? Ela será registrada como contexto e não entrará automaticamente nos critérios de inclusão/exclusão.",
        False,
    ):
        return

    choices = ["outro arquivo estruturado", "digitar manualmente"]
    if primary_path:
        choices.insert(0, "mesmo arquivo estruturado")
    while True:
        source = ask_choice("Como deseja informar a hipótese/tese orientadora", choices, choices[0])
        if source == "mesmo arquivo estruturado":
            value = _coerce_research_text(primary_values.get("hipotese"))
            raw_path = primary_path
        elif source == "outro arquivo estruturado":
            raw_path, imported, _model_edited = ask_structured_research_source(
                data,
                "Arquivo estruturado com hipótese/tese orientadora",
                required_fields=("hipotese",),
                editable_fields=("hipotese",),
            )
            value = _coerce_research_text(imported.get("hipotese"))
        else:
            raw_path = ""
            value = ask_required("Hipótese/tese orientadora", "")
        if not value:
            _notify_structured_input_error("Não encontrei uma hipótese/tese orientadora nessa fonte. Escolha outra fonte ou digite o conteúdo.")
            continue
        data["hipotese"] = value
        data["fonte_hipotese"] = {
            "mesmo arquivo estruturado": "mesmo_arquivo_estruturado",
            "outro arquivo estruturado": "arquivo_estruturado",
            "digitar manualmente": "manual",
        }[source]
        if raw_path:
            stored = rel_for_toml(raw_path, data["config_dir"])
            data["hipotese_path"] = stored
            data["hipotese_paths"] = [stored]
        return


def _collect_keywords(
    data: dict[str, Any],
    *,
    primary_path: str,
    primary_values: dict[str, Any],
) -> None:
    data["palavras_chave"] = []
    data["gerar_palavras_chave_ia"] = False
    data["fonte_palavras_chave"] = ""
    data["palavras_chave_path"] = ""
    data["palavras_chave_paths"] = []

    choices = ["outro arquivo estruturado", "digitar manualmente", "solicitar sugestões à IA"]
    if primary_path:
        choices.insert(0, "mesmo arquivo estruturado")
    while True:
        source = ask_choice(
            "Como deseja informar as palavras-chave da revisão",
            choices,
            "mesmo arquivo estruturado" if primary_path else "solicitar sugestões à IA",
        )
        if source == "mesmo arquivo estruturado":
            values = _coerce_keywords(primary_values.get("palavras_chave"))
            raw_path = primary_path
        elif source == "outro arquivo estruturado":
            raw_path, imported, _model_edited = ask_structured_research_source(
                data,
                "Arquivo estruturado com palavras-chave",
                required_fields=("palavras_chave",),
                editable_fields=("palavras_chave",),
            )
            values = _coerce_keywords(imported.get("palavras_chave"))
        elif source == "digitar manualmente":
            raw_path = ""
            values = ask_list("Palavras-chave", [])
        else:
            raw_path = ""
            try:
                values = suggest_prisma_keywords_with_ai(data)
            except RuntimeError as exc:
                _notify_structured_input_error(str(exc))
                continue
            _show_imported_research_values("Sugestões geradas pela IA", {"palavras_chave": values}, ("palavras_chave",))
            if ask_bool("Deseja editar as palavras-chave sugeridas pela IA?", False):
                values = ask_list("Palavras-chave", values)

        if not values:
            _notify_structured_input_error("Não encontrei palavras-chave nessa fonte. Escolha outro arquivo, digite os termos ou solicite sugestões à IA.")
            continue
        data["palavras_chave"] = values
        data["fonte_palavras_chave"] = {
            "mesmo arquivo estruturado": "mesmo_arquivo_estruturado",
            "outro arquivo estruturado": "arquivo_estruturado",
            "digitar manualmente": "manual",
            "solicitar sugestões à IA": "ia",
        }[source]
        if raw_path:
            stored = rel_for_toml(raw_path, data["config_dir"])
            data["palavras_chave_path"] = stored
            data["palavras_chave_paths"] = [stored]
        return


def _balanced_prisma_keyword_blocks(keywords: list[str], *, groups: int = 3) -> list[list[str]]:
    """Distribui termos em blocos curtos para consultas independentes.

    A estratégia não converte os termos em uma única frase longa. Cada bloco
    será consultado isoladamente em cada fonte e permanecerá rastreável no
    protocolo, nos arquivos JSON e na planilha de triagem.
    """
    values = [str(item).strip() for item in keywords if str(item).strip()]
    if not values:
        return []
    count = max(1, min(int(groups), len(values)))
    base, remainder = divmod(len(values), count)
    chunks: list[list[str]] = []
    offset = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        chunks.append(values[offset:offset + size])
        offset += size
    return [chunk for chunk in chunks if chunk]


def _default_prisma_search_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Propõe blocos complementares com consultas curtas e editáveis.

    Cada consulta será executada separadamente em cada fonte. Isso evita enviar
    uma cadeia extensa de palavras-chave, em idiomas distintos, como se fosse
    uma única expressão para todas as APIs.
    """
    keywords = [str(item).strip() for item in data.get("palavras_chave", []) if str(item).strip()]
    context = " ".join([str(data.get("tema") or ""), *keywords]).casefold()

    if any(token in context for token in ("incapacidade", "disability", "perícia", "pericia")):
        return [
            {
                "id": "incapacidade_avaliacao",
                "rotulo": "Benefícios por incapacidade e avaliação médica",
                "consultas": [
                    "disability benefits",
                    "work disability",
                    "medical disability assessment",
                ],
            },
            {
                "id": "analise_documental_telepericia",
                "rotulo": "Análise documental e avaliação remota",
                "consultas": [
                    "documentary review",
                    "telemedicine",
                    "telehealth",
                    "remote medical assessment",
                ],
            },
            {
                "id": "filas_capacidade_alocacao",
                "rotulo": "Filas, capacidade e alocação de serviços",
                "consultas": [
                    "waiting time",
                    "backlog",
                    "service capacity",
                    "case allocation",
                ],
            },
        ]

    chunks = _balanced_prisma_keyword_blocks(keywords, groups=3)
    if not chunks:
        fallback = str(data.get("tema") or "").strip() or "termo principal da revisão"
        chunks = [[fallback]]
    return [
        {
            "id": f"bloco_{index}",
            "rotulo": f"Bloco temático {index}",
            "consultas": chunk,
        }
        for index, chunk in enumerate(chunks, 1)
    ]



PRISMA_PROTOCOL_DEFAULT_INCLUSION = [
    "Aderência substantiva ao tema, recorte, objetivo e pergunta de pesquisa.",
    "Estudo com contribuição direta para fluxos de análise, triagem, teleperícia, perícia presencial, capacidade ou governança correlata.",
    "Metadados suficientes para identificação bibliográfica e triagem humana.",
]
PRISMA_PROTOCOL_DEFAULT_EXCLUSION = [
    "Fora do tema ou do recorte.",
    "Duplicado.",
    "Sem relação substantiva com a pergunta de pesquisa.",
]


def _coerce_text_list(value: Any) -> list[str]:
    """Normaliza listas textuais sem quebrar critérios que contenham vírgulas."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw = [item.strip() for item in re.split(r"[;\n]+", str(value)) if item.strip()]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _normalise_prisma_block(value: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalised = {_normalise_research_key(key): item for key, item in value.items()}
    label = str(
        normalised.get("rotulo")
        or normalised.get("label")
        or normalised.get("nome")
        or normalised.get("titulo")
        or ""
    ).strip()
    raw_id = str(normalised.get("id") or normalised.get("identificador") or "").strip()
    identifier = slugify(raw_id or label) if (raw_id or label) else ""
    raw_queries = (
        normalised.get("consultas")
        or normalised.get("queries")
        or normalised.get("consulta")
        or normalised.get("query")
        or []
    )
    queries = _coerce_text_list(raw_queries)
    if not label or not queries:
        return None
    placeholders = {"substitua", "substituir", "termo principal da revisão", "consulta curta"}
    if any(any(token in query.casefold() for token in placeholders) for query in queries):
        return None
    return {
        "id": identifier or f"bloco_{index}",
        "rotulo": label,
        "consultas": queries,
    }


def _protocol_layers(payload: Any) -> list[dict[str, Any]]:
    """Prioriza seções de protocolo, sem deixar de aceitar arquivos simples."""
    if not isinstance(payload, dict):
        return []
    result: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(item: Any) -> None:
        if isinstance(item, dict) and id(item) not in seen:
            result.append(item)
            seen.add(id(item))

    for key, value in payload.items():
        if _normalise_research_key(key) in {
            "busca_prisma", "protocolo_prisma", "estrategia_busca", "estrategia_de_busca", "criterios",
        }:
            add(value)
    for layer in _iter_mapping_layers(payload):
        add(layer)
    return result


def _read_structured_prisma_protocol_file(
    raw_path: str,
    *,
    allow_declared_empty_fields: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Lê critérios e estratégia de busca de um arquivo estruturado.

    O formato preferencial é TOML com ``[busca_prisma]`` e
    ``[[busca_prisma.estrategias]]``. Também aceita ``[criterios]`` e aliases
    simples para facilitar arquivos previamente preparados pelo usuário.
    """
    path = Path(str(raw_path or "")).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Arquivo inexistente ou não é arquivo regular: {path}")
    suffix = path.suffix.lower()
    if suffix not in STRUCTURED_RESEARCH_SUFFIXES:
        allowed = ", ".join(sorted(STRUCTURED_RESEARCH_SUFFIXES))
        raise ValueError(f"Formato não suportado. Use um destes: {allowed}")
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        if suffix == ".toml":
            payload: Any = tomllib.loads(source)
        elif suffix == ".json":
            payload = json.loads(source)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
                payload = yaml.safe_load(source)
            except ModuleNotFoundError:
                payload = _parse_simple_yaml(source)
        else:
            payload = _parse_simple_yaml(source)
    except (ValueError, TypeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Não foi possível ler o arquivo estruturado: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("O protocolo estruturado deve conter um objeto/dicionário.")

    values: dict[str, Any] = {
        "criterios_inclusao": [],
        "criterios_exclusao": [],
        "estrategia_busca": "",
        "consulta_geral": "",
        "blocos_busca": [],
        "limite_por_base": None,
        "limite_scopus_por_consulta": None,
    }
    declared: set[str] = set()
    aliases = {
        "criterios_inclusao": {"criterios_inclusao", "inclusao", "inclusion_criteria", "criterios_de_inclusao"},
        "criterios_exclusao": {"criterios_exclusao", "exclusao", "exclusion_criteria", "criterios_de_exclusao"},
        "estrategia_busca": {"estrategia", "estrategia_busca", "strategy", "search_strategy"},
        "consulta_geral": {"consulta_geral", "consulta", "query", "search_query"},
        "blocos_busca": {"estrategias", "blocos", "blocos_busca", "blocos_tematicos", "search_blocks"},
        "limite_por_base": {"limite_por_base", "max_registros_por_base", "records_per_source"},
        "limite_scopus_por_consulta": {"limite_scopus_por_consulta", "max_registros_scopus", "scopus_records_per_query"},
    }
    for layer in _protocol_layers(payload):
        normalised = {_normalise_research_key(key): item for key, item in layer.items()}
        for field, names in aliases.items():
            value = next((normalised[name] for name in names if name in normalised), None)
            if value is None:
                continue
            declared.add(field)
            if field in {"criterios_inclusao", "criterios_exclusao"}:
                if not values[field]:
                    values[field] = _coerce_text_list(value)
            elif field == "blocos_busca":
                if not values[field] and isinstance(value, list):
                    blocks = [_normalise_prisma_block(item, index) for index, item in enumerate(value, 1)]
                    values[field] = [item for item in blocks if item]
            elif field in {"limite_por_base", "limite_scopus_por_consulta"}:
                if values[field] is None:
                    values[field] = value
            elif not values[field]:
                values[field] = str(value or "").strip()

    strategy = _normalise_research_key(values.get("estrategia_busca"))
    if strategy in {"blocos", "blocos_tematicos", "blocos_independentes", "thematic_blocks"}:
        values["estrategia_busca"] = "blocos_tematicos"
    elif strategy in {"consulta_unica", "consulta", "single_query", "query"}:
        values["estrategia_busca"] = "consulta_unica"
    elif values.get("blocos_busca"):
        values["estrategia_busca"] = "blocos_tematicos"
    elif values.get("consulta_geral"):
        values["estrategia_busca"] = "consulta_unica"

    has_content = bool(
        values["criterios_inclusao"]
        or values["criterios_exclusao"]
        or values["blocos_busca"]
        or values["consulta_geral"]
    )
    if not has_content and not (allow_declared_empty_fields and declared):
        raise ValueError(
            "Não encontrei critérios ou estratégia de busca. Use [busca_prisma] com "
            "criterios_inclusao, criterios_exclusao e [[busca_prisma.estrategias]]."
        )
    return path, values


def _coerce_protocol_limit(value: Any, *, label: str, default: int, minimum: int, maximum: int) -> int:
    if value in {None, ""}:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} deve ser um número inteiro entre {minimum} e {maximum}.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{label} deve estar entre {minimum} e {maximum}.")
    return number


def _show_imported_prisma_protocol(path: str, values: dict[str, Any]) -> None:
    lines = [f"Arquivo: {Path(path).expanduser()}", ""]
    strategy = str(values.get("estrategia_busca") or "não identificada")
    lines.append(f"Estratégia: {strategy}")
    if strategy == "consulta_unica":
        lines.append(f"Consulta geral: {values.get('consulta_geral') or '—'}")
    blocks = values.get("blocos_busca") or []
    if blocks:
        lines.append("Blocos:")
        for block in blocks:
            queries = "; ".join(block.get("consultas") or [])
            lines.append(f"- {block.get('rotulo') or 'Sem rótulo'}: {queries}")
    lines.append("Critérios de inclusão:")
    lines.extend(f"- {item}" for item in (values.get("criterios_inclusao") or ["—"]))
    lines.append("Critérios de exclusão:")
    lines.extend(f"- {item}" for item in (values.get("criterios_exclusao") or ["—"]))
    message = "\n".join(lines)
    if tui_theme_enabled():
        _fgv_ui().message(_dialog_title(), message)
    else:
        print("\n" + message + "\n")


def _apply_structured_prisma_protocol(data: dict[str, Any], path: str, values: dict[str, Any], *, source: str) -> None:
    strategy = str(values.get("estrategia_busca") or "").strip()
    blocks = list(values.get("blocos_busca") or [])
    query = str(values.get("consulta_geral") or "").strip()
    inclusion = _coerce_text_list(values.get("criterios_inclusao"))
    exclusion = _coerce_text_list(values.get("criterios_exclusao"))
    if strategy == "blocos_tematicos" and not blocks:
        raise ValueError("A estratégia por blocos exige ao menos um bloco com rótulo e uma consulta.")
    if strategy == "consulta_unica" and not query:
        raise ValueError("A estratégia de consulta única exige consulta_geral preenchida.")
    if strategy not in {"blocos_tematicos", "consulta_unica"}:
        raise ValueError("Defina estrategia = 'blocos_tematicos' ou 'consulta_unica' no arquivo estruturado.")
    if not inclusion or not exclusion:
        raise ValueError("O arquivo estruturado deve conter ao menos um critério de inclusão e um de exclusão.")
    data["estrategia_busca"] = strategy
    data["blocos_busca"] = blocks if strategy == "blocos_tematicos" else []
    data["consulta_geral"] = query if strategy == "consulta_unica" else ""
    data["limite_por_base"] = _coerce_protocol_limit(
        values.get("limite_por_base"),
        label="limite_por_base",
        default=25 if strategy == "blocos_tematicos" else 100,
        minimum=5 if strategy == "blocos_tematicos" else 10,
        maximum=100 if strategy == "blocos_tematicos" else 200,
    )
    data["limite_scopus_por_consulta"] = _coerce_protocol_limit(
        values.get("limite_scopus_por_consulta"),
        label="limite_scopus_por_consulta",
        default=min(int(data["limite_por_base"]), 10),
        minimum=1,
        maximum=25,
    )
    data["criterios_inclusao"] = inclusion
    data["criterios_exclusao"] = exclusion
    stored = rel_for_toml(path, data["config_dir"])
    data["fonte_estrategia_busca"] = source
    data["estrategia_busca_path"] = stored
    data["fonte_criterios_prisma"] = source
    data["criterios_prisma_path"] = stored


def _collect_manual_prisma_criteria(data: dict[str, Any]) -> None:
    data["criterios_inclusao"] = ask_list("Critérios de inclusão", PRISMA_PROTOCOL_DEFAULT_INCLUSION)
    data["criterios_exclusao"] = ask_list("Critérios de exclusão", PRISMA_PROTOCOL_DEFAULT_EXCLUSION)
    data["fonte_estrategia_busca"] = "manual"
    data["estrategia_busca_path"] = ""
    data["fonte_criterios_prisma"] = "manual"
    data["criterios_prisma_path"] = ""



def _toml_array_literal(values: list[str], indent: str = "") -> str:
    if not values:
        return "[]"
    return "[\n" + ",\n".join(f"{indent}  {json.dumps(item, ensure_ascii=False)}" for item in values) + f"\n{indent}]"


def _render_prisma_protocol_model_section(values: dict[str, Any]) -> list[str]:
    """Renderiza a seção editável do modelo, sem incluir credenciais."""
    strategy = str(values.get("estrategia_busca") or "blocos_tematicos")
    blocks = list(values.get("blocos_busca") or [])
    lines = [
        "[busca_prisma]\n",
        f"estrategia = {json.dumps(strategy, ensure_ascii=False)}\n",
        f"consulta_geral = {json.dumps(str(values.get('consulta_geral') or ''), ensure_ascii=False)}\n",
        f"limite_por_base = {int(values.get('limite_por_base') or (25 if strategy == 'blocos_tematicos' else 100))}\n",
        f"limite_scopus_por_consulta = {int(values.get('limite_scopus_por_consulta') or 10)}\n",
        "criterios_inclusao = " + _toml_array_literal(_coerce_text_list(values.get("criterios_inclusao"))) + "\n",
        "criterios_exclusao = " + _toml_array_literal(_coerce_text_list(values.get("criterios_exclusao"))) + "\n",
    ]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        lines.extend([
            "\n",
            "[[busca_prisma.estrategias]]\n",
            f"id = {json.dumps(str(block.get('id') or ''), ensure_ascii=False)}\n",
            f"rotulo = {json.dumps(str(block.get('rotulo') or ''), ensure_ascii=False)}\n",
            "consultas = " + _toml_array_literal(_coerce_text_list(block.get("consultas"))) + "\n",
        ])
    return lines


def _persist_prisma_protocol_in_project_model(path: Path, values: dict[str, Any]) -> None:
    """Substitui somente o bloco [busca_prisma] da cópia do projeto.

    As subtabelas ``[[busca_prisma.estrategias]]`` fazem parte do mesmo bloco.
    Qualquer outra seção do modelo é preservada literalmente.
    """
    if path.suffix.lower() != ".toml":
        return
    source = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = source.splitlines(keepends=True)
    start: int | None = None
    end = len(lines)
    section_header = re.compile(r"^\s*\[\s*busca_prisma\s*\]\s*(?:#.*)?$", flags=re.IGNORECASE)
    child_header = re.compile(r"^\s*\[\[\s*busca_prisma\.[^\]]+\]\]\s*(?:#.*)?$", flags=re.IGNORECASE)
    any_header = re.compile(r"^\s*\[\[?\s*([^\]]+)\s*\]?\]\s*(?:#.*)?$")
    for index, line in enumerate(lines):
        if start is None:
            if section_header.match(line):
                start = index
            continue
        if child_header.match(line):
            continue
        match = any_header.match(line)
        if match:
            header_name = match.group(1).strip().casefold()
            if not header_name.startswith("busca_prisma.") and header_name != "busca_prisma":
                end = index
                break
    rendered = _render_prisma_protocol_model_section(values)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("\n")
        path.write_text("".join(lines + rendered).rstrip() + "\n", encoding="utf-8")
        return
    path.write_text("".join(lines[:start] + rendered + lines[end:]).rstrip() + "\n", encoding="utf-8")


def _collect_guided_prisma_protocol_values(data: dict[str, Any]) -> dict[str, Any]:
    """Mantém uma via assistida quando EDITOR/VISUAL não estiver configurado."""
    temporary = dict(data)
    _collect_prisma_search_strategy(temporary)
    _collect_manual_prisma_criteria(temporary)
    return {
        "estrategia_busca": temporary.get("estrategia_busca"),
        "consulta_geral": temporary.get("consulta_geral", ""),
        "blocos_busca": temporary.get("blocos_busca", []),
        "limite_por_base": temporary.get("limite_por_base"),
        "limite_scopus_por_consulta": temporary.get("limite_scopus_por_consulta"),
        "criterios_inclusao": temporary.get("criterios_inclusao", []),
        "criterios_exclusao": temporary.get("criterios_exclusao", []),
    }


def _collect_prisma_strategy_and_criteria(data: dict[str, Any], *, primary_path: str = "") -> None:
    choices = [
        "digitar estratégia e critérios no assistente",
        "usar arquivo estruturado já existente",
        "copiar e editar o modelo institucional na pasta do projeto",
    ]
    if primary_path:
        choices.insert(0, "usar o mesmo arquivo estruturado da pesquisa")
    choice = ask_choice("Como deseja fornecer blocos, consultas e critérios PRISMA", choices, choices[0])
    if choice == "digitar estratégia e critérios no assistente":
        _collect_prisma_search_strategy(data)
        _collect_manual_prisma_criteria(data)
        return

    source = "arquivo_estruturado"
    if choice == "usar o mesmo arquivo estruturado da pesquisa":
        path = primary_path
        source = "mesmo_arquivo_estruturado"
    elif choice == "usar arquivo estruturado já existente":
        path = ask_required("Arquivo estruturado com blocos, consultas e critérios PRISMA", "")
    else:
        try:
            template = _structured_research_template_path(data)
            destination = _choose_project_model_destination(data, template)
            edited_in_external_editor = _open_project_model_in_editor(destination)
            if not edited_in_external_editor:
                guided_values = _collect_guided_prisma_protocol_values(data)
                _persist_prisma_protocol_in_project_model(destination, guided_values)
            path = str(destination)
            source = "modelo_institucional_editado"
        except ValueError as exc:
            _notify_structured_input_error(str(exc))
            return _collect_prisma_strategy_and_criteria(data, primary_path=primary_path)

    try:
        _path, values = _read_structured_prisma_protocol_file(path)
        _show_imported_prisma_protocol(path, values)
        _apply_structured_prisma_protocol(data, path, values, source=source)
    except ValueError as exc:
        _notify_structured_input_error(str(exc))
        return _collect_prisma_strategy_and_criteria(data, primary_path=primary_path)


def _collect_prisma_search_strategy(data: dict[str, Any]) -> None:
    """Coleta estratégia reprodutível sem juntar todas as palavras-chave."""
    choice = ask_choice(
        "Estratégia de busca bibliográfica",
        ["blocos temáticos independentes (recomendado)", "consulta única (compatibilidade)"],
        "blocos temáticos independentes (recomendado)",
    )
    if choice == "consulta única (compatibilidade)":
        consulta_padrao = " ".join(data.get("palavras_chave", [])).strip() or str(data.get("tema") or "").strip()
        data["estrategia_busca"] = "consulta_unica"
        data["blocos_busca"] = []
        if ask_bool("Deseja editar a consulta livre enviada às bases? Ela será registrada exatamente como for executada.", False):
            data["consulta_geral"] = ask_required("Consulta geral de busca", consulta_padrao)
        else:
            data["consulta_geral"] = consulta_padrao
        data["limite_por_base"] = ask_positive_int("Máximo de registros por base", 100, minimum=10, maximum=200)
        data["limite_scopus_por_consulta"] = min(int(data["limite_por_base"]), 10)
        return

    suggested = _default_prisma_search_blocks(data)
    print(
        "A busca será executada separadamente para cada bloco em cada fonte. "
        "Use consultas curtas; o ponto e vírgula indica alternativas, especialmente no Scopus."
    )
    blocks: list[dict[str, Any]] = []
    for index, item in enumerate(suggested, 1):
        label = ask(f"Rótulo do bloco {index}", str(item["rotulo"]))
        default_queries = "; ".join(str(value) for value in item.get("consultas", []) if str(value).strip())
        raw_queries = ask_required(
            f"Consultas curtas do bloco {index} (separe alternativas por ;)",
            default_queries,
        )
        queries = [value.strip() for value in raw_queries.split(";") if value.strip()]
        if not queries:
            _notify_structured_input_error("Informe ao menos uma consulta curta para cada bloco.")
            return _collect_prisma_search_strategy(data)
        ident = re.sub(r"[^a-z0-9_]+", "_", _normalise_research_key(label)).strip("_") or f"bloco_{index}"
        blocks.append({"id": ident, "rotulo": label.strip() or f"Bloco temático {index}", "consultas": queries})

    data["estrategia_busca"] = "blocos_tematicos"
    data["blocos_busca"] = blocks
    data["consulta_geral"] = ""
    data["limite_por_base"] = ask_positive_int(
        "Máximo de registros por bloco em cada base",
        25,
        minimum=5,
        maximum=100,
    )
    data["limite_scopus_por_consulta"] = ask_positive_int(
        "Máximo de registros por bloco no Scopus (limite seguro para a credencial atual)",
        10,
        minimum=1,
        maximum=25,
    )


def collect_prisma_busca_orientada(data: dict[str, Any]) -> None:
    """Coleta exclusiva do perfil novo; não altera o fluxo legado de PRISMA."""
    print("\nDados da revisão PRISMA com busca orientada")
    print(
        "Escolha entre importar um arquivo estruturado ou preencher manualmente. "
        "A hipótese/tese é opcional e não será convertida em critério automático de seleção. "
        "A busca externa será executada antes da triagem. Quando ativada, a IA apenas prioriza e justifica a ordem de revisão; ela não decide inclusão ou exclusão final."
    )
    data["dados_pesquisa_paths"] = []
    data["fonte_dados_pesquisa"] = ""
    data["dados_pesquisa_path"] = ""
    data["orientacao_professor"] = ""
    # Este perfil inicia na busca externa. Corpus local, manifestos DOI,
    # diretórios de triagem prévia e prompts de documento não são insumos do
    # primeiro ciclo e, por isso, não são perguntados no wizard.
    data["documentos_input_zip"] = ""
    data["documentos_input_dir"] = ""
    data["corpus_local_opcional"] = ""
    data["orientacoes_paths"] = []
    data["orientacao_geral_inline"] = ""
    data["orientacao_geral_modo"] = "nenhuma"
    data["doi_manifest_path"] = ""
    data["prompt_extra_paths"] = []

    mode = ask_choice(
        "Como deseja informar tema, recorte, objetivo, pergunta de pesquisa e tipo de estudo",
        [
            "usar arquivo estruturado já existente",
            "copiar e editar o modelo institucional na pasta do projeto",
            "digitar manualmente",
        ],
        "usar arquivo estruturado já existente",
    )
    primary_path = ""
    primary_values: dict[str, Any] = {}
    core_fields = ("tema", "recorte", "objetivo", "pergunta_pesquisa", "tipo_estudo")

    if mode != "digitar manualmente":
        source_override = (
            "usar arquivo estruturado já existente"
            if mode == "usar arquivo estruturado já existente"
            else "copiar e editar o modelo institucional na pasta do projeto"
        )
        primary_path, primary_values, model_was_edited = ask_structured_research_source(
            data,
            "Arquivo estruturado com tema, recorte, objetivo, pergunta de pesquisa e tipo de estudo",
            required_fields=core_fields,
            editable_fields=core_fields,
            source_override=source_override,
        )
        if not model_was_edited:
            primary_values = review_imported_research_values(primary_path, primary_values, core_fields)
        for field in core_fields:
            data[field] = primary_values.get(field, [] if field == "palavras_chave" else "")
        data["fonte_dados_pesquisa"] = "modelo_institucional_editado" if model_was_edited else "arquivo_estruturado"
        stored = rel_for_toml(primary_path, data["config_dir"])
        data["dados_pesquisa_path"] = stored
        data["dados_pesquisa_paths"] = [stored]
    else:
        data["tema"] = ask("Tema", "")
        data["recorte"] = ask("Recorte", "")
        data["objetivo"] = ask("Objetivo", "")
        data["pergunta_pesquisa"] = ask("Pergunta de pesquisa", "")
        data["tipo_estudo"] = ask("Tipo de estudo", "relatório PRISMA")
        data["fonte_dados_pesquisa"] = "manual"

    if not str(data.get("tipo_estudo") or "").strip():
        data["tipo_estudo"] = "relatório PRISMA"

    _collect_optional_hypothesis(data, primary_path=primary_path, primary_values=primary_values)
    _collect_keywords(data, primary_path=primary_path, primary_values=primary_values)
    data["idiomas"] = ask_list("Idiomas", PRISMA_BUSCA_DEFAULT_LANGUAGES)

    print("\nProtocolo de busca bibliográfica externa")
    statuses = provider_statuses()
    print("Credenciais detectadas no .env (sem exibir valores):")
    for provider in [key for key, _label in provider_selection_choices() if key != "todas"]:
        item = statuses[provider]
        print(f"- {item['label']}: {item['status']}")
    print("A opção [Todas] seleciona todos os buscadores. Fontes sem credencial obrigatória registram o motivo e não interrompem as demais consultas.")
    data["bases_busca"] = ask_many_choice(
        "Selecione as fontes de descoberta bibliográfica. Marque [Todas] para consultar todas as fontes configuráveis.",
        provider_selection_choices(),
        ["crossref", "openalex", "semantic_scholar"],
    )
    _collect_prisma_strategy_and_criteria(data, primary_path=primary_path)
    data["limite_triagem_inicial"] = ask_positive_int("Máximo de registros enviados para a planilha de triagem inicial", 250, minimum=10, maximum=1000)
    data["pre_triagem_ia"] = ask_bool(
        "Executar pré-triagem assistida por IA antes do corte da planilha? A IA só ordena e justifica prioridades; a decisão final permanece humana.",
        True,
    )
    if data["pre_triagem_ia"]:
        data["pre_triagem_ia_modelo"] = ask("Modelo da pré-triagem IA (vazio para usar [openai].model/OPENAI_MODEL)", "")
        data["pre_triagem_ia_lote"] = ask_positive_int("Registros por lote da pré-triagem IA", 20, minimum=5, maximum=30)
        data["pre_triagem_ia_max_registros"] = ask_positive_int("Máximo de registros deduplicados a avaliar com IA", 1500, minimum=10, maximum=5000)
        data["pre_triagem_ia_reserva_incertos"] = ask_positive_int("Reserva de itens incertos/falhos para revisão humana na planilha", 40, minimum=0, maximum=1000)
        data["pre_triagem_ia_min_confianca"] = ask_positive_int("Confiança mínima da IA antes de dispensar a reserva de incerteza (0–100)", 55, minimum=0, maximum=100)
        data["pre_triagem_ia_max_chars_resumo"] = ask_positive_int("Máximo de caracteres do resumo enviados por registro à IA", 700, minimum=200, maximum=1500)
    else:
        data["pre_triagem_ia_modelo"] = ""
        data["pre_triagem_ia_lote"] = 20
        data["pre_triagem_ia_max_registros"] = 1500
        data["pre_triagem_ia_reserva_incertos"] = 40
        data["pre_triagem_ia_min_confianca"] = 55
        data["pre_triagem_ia_max_chars_resumo"] = 700
    data["meta_estudos_incluidos"] = ask_positive_int("Meta de estudos incluídos após a triagem", 15, minimum=1, maximum=500)
    data["ano_inicio"] = ask("Ano inicial de publicação (vazio para não limitar)", "")
    data["ano_fim"] = ask("Ano final de publicação (vazio para não limitar)", "")
    # Credenciais e e-mails específicos devem permanecer no .env; o TOML não
    # recebe nem exibe esses valores. Este campo é somente fallback opcional.
    data["email_contato_busca"] = ask("E-mail de contato alternativo para APIs sem e-mail próprio no .env (vazio para usar apenas o .env)", "")
    unpaywall = statuses["unpaywall"]
    data["enriquecer_unpaywall"] = ask_bool(
        f"Complementar links de acesso aberto com Unpaywall? Status: {unpaywall['status']}.",
        bool(unpaywall["available"]),
    )
    data["limite_unpaywall"] = ask_positive_int(
        "Máximo de registros com DOI a consultar no Unpaywall",
        100,
        minimum=0,
        maximum=250,
    ) if data["enriquecer_unpaywall"] else 0
    data["busca_externa_ativa"] = True


def _set_default_paper_directives(data: dict[str, Any]) -> None:
    data.setdefault("tese_central", "")
    data.setdefault("estrutura_desejada", [])
    data.setdefault("argumentos_obrigatorios", [])
    data.setdefault("orientacoes_metodologicas", "")
    data.setdefault("limites_do_escopo", "")
    data.setdefault("tom_de_redacao", "acadêmico analítico")
    data.setdefault("instrucoes_adicionais", "")




def _paper_structured_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if text in {"1", "true", "s", "sim", "yes", "y"}:
        return True
    if text in {"0", "false", "n", "nao", "não", "no"}:
        return False
    return default


def _paper_structured_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _paper_structured_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = default
    return max(low, min(candidate, high))


def _read_paper_structured_render_config(raw_path: str) -> dict[str, Any]:
    '''Lê somente as seções de saída/resumo do modelo estruturado do paper.'''
    path = Path(str(raw_path or "")).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}
    suffix = path.suffix.lower()
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".toml":
            payload: Any = tomllib.loads(source)
        elif suffix == ".json":
            payload = json.loads(source)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
                payload = yaml.safe_load(source)
            except ModuleNotFoundError:
                return {}
        else:
            return {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _apply_paper_structured_render_config(data: dict[str, Any], raw_path: str) -> None:
    '''Importa título e escolhas de saída do mesmo TOML estruturado do paper.'''
    payload = _read_paper_structured_render_config(raw_path)
    if not payload:
        return
    metadata = payload.get("metadados", {}) if isinstance(payload.get("metadados"), dict) else {}
    output = payload.get("idiomas_saida", {}) if isinstance(payload.get("idiomas_saida"), dict) else {}
    abstracts = payload.get("resumos_paper", {}) if isinstance(payload.get("resumos_paper"), dict) else {}

    title = str(metadata.get("titulo_trabalho") or metadata.get("titulo") or "").strip()
    subtitle = str(metadata.get("subtitulo") or "").strip()
    if title:
        data["titulo"] = title if not subtitle or ":" in title else title + ": " + subtitle

    if output:
        data["idioma_principal_saida"] = str(output.get("principal") or "pt-BR").strip() or "pt-BR"
        data["gerar_traducao_ia"] = _paper_structured_bool(output.get("gerar_traducao_ia"), False)
        data["idiomas_adicionais_saida"] = _paper_structured_list(output.get("idiomas_adicionais"))
        data["preservar_referencias_originais"] = _paper_structured_bool(output.get("preservar_referencias_originais"), True)
        data["translation_max_chars_batch"] = _paper_structured_int(output.get("max_chars_por_lote"), 12000, 2500, 24000)
        data["_paper_structured_translation_settings"] = True

    if abstracts:
        data["gerar_resumo_principal"] = _paper_structured_bool(abstracts.get("gerar_resumo_principal"), True)
        data["idioma_resumo_principal"] = str(abstracts.get("principal") or data.get("idioma_principal_saida") or "pt-BR").strip() or "pt-BR"
        data["gerar_resumo_adicional"] = _paper_structured_bool(abstracts.get("gerar_resumo_adicional"), False)
        data["idiomas_resumo_adicionais"] = _paper_structured_list(abstracts.get("idiomas_adicionais"))
        data["gerar_palavras_chave_resumo_adicionais"] = _paper_structured_bool(abstracts.get("gerar_palavras_chave_adicionais"), True)
        data["resumo_max_palavras"] = _paper_structured_int(abstracts.get("max_palavras"), 250, 150, 350)
        data["_paper_structured_abstract_settings"] = True

def collect_paper_local_research(data: dict[str, Any]) -> None:
    """Coleta dados e diretrizes do ``paper_local_fgv`` sem alterar outros perfis."""
    print("\nDados e diretrizes do paper a partir de corpus local")
    print(
        "O ZIP ou a pasta de PDFs será solicitado na próxima etapa. Aqui você pode "
        "usar um arquivo estruturado para orientar a redação do paper ou preencher "
        "os campos manualmente."
    )
    data["orientacoes_paper_paths"] = []
    data["orientacoes_paper_path"] = ""
    data["fonte_orientacoes_paper"] = ""
    data["orientacao_professor"] = ""
    _set_default_paper_directives(data)

    mode = ask_choice(
        "Como deseja informar os dados da pesquisa e as diretrizes de redação do paper",
        ["arquivo estruturado", "digitar manualmente"],
        "arquivo estruturado",
    )
    if mode == "arquivo estruturado":
        raw_path, values, model_was_edited = ask_structured_paper_source(
            data,
            "Arquivo estruturado com dados da pesquisa e diretrizes do paper",
            required_fields=PAPER_CORE_RESEARCH_FIELDS,
            editable_fields=PAPER_STRUCTURED_FIELDS,
        )
        if not model_was_edited:
            values = review_imported_paper_values(raw_path, values, PAPER_STRUCTURED_FIELDS)
        for field in RESEARCH_FIELD_LABELS:
            data[field] = values.get(field, [] if field == "palavras_chave" else "")
        for field in PAPER_FIELD_LABELS:
            data[field] = values.get(field, [] if field in PAPER_LIST_FIELDS else "")
        data["fonte_orientacoes_paper"] = "arquivo_estruturado"
        stored = rel_for_toml(raw_path, data["config_dir"])
        data["orientacoes_paper_path"] = stored
        data["orientacoes_paper_paths"] = [stored]
        _apply_paper_structured_render_config(data, raw_path)
    else:
        data["tema"] = ask("Tema", "")
        data["recorte"] = ask("Recorte", "")
        data["objetivo"] = ask("Objetivo", "")
        data["pergunta_pesquisa"] = ask("Pergunta de pesquisa", "")
        data["hipotese"] = ask("Hipótese/tese orientadora (opcional)", "")
        data["palavras_chave"] = ask_list("Palavras-chave", [])
        data["tipo_estudo"] = ask("Tipo de estudo", "paper acadêmico")
        data["fonte_orientacoes_paper"] = "manual"
        if ask_bool("Deseja informar diretrizes adicionais para a redação do paper?", False):
            data["tese_central"] = ask("Tese/argumento central do paper", "")
            data["estrutura_desejada"] = ask_list(
                "Estrutura desejada (separe por ;)",
                ["Introdução", "Referencial analítico", "Análise e discussão", "Conclusão"],
            )
            data["argumentos_obrigatorios"] = ask_list("Argumentos ou pontos obrigatórios (separe por ;)", [])
            data["orientacoes_metodologicas"] = ask("Orientações metodológicas", "")
            data["limites_do_escopo"] = ask("Limites do escopo", "")
            data["tom_de_redacao"] = ask("Tom de redação", "acadêmico analítico")
            data["instrucoes_adicionais"] = ask("Instruções adicionais", "")

    if not str(data.get("tipo_estudo") or "").strip():
        data["tipo_estudo"] = "paper acadêmico"
    if not str(data.get("tom_de_redacao") or "").strip():
        data["tom_de_redacao"] = "acadêmico analítico"
    data["gerar_palavras_chave_ia"] = False
    data["idiomas"] = ask_list("Idiomas do corpus", ["português", "inglês", "espanhol"])


def collect_research(data: dict[str, Any]) -> None:
    preset: Preset = data["preset"]

    if preset.key == "relatorio_prisma_busca_orientada_fgv":
        collect_prisma_busca_orientada(data)
        return

    if preset.key == "paper_local_fgv":
        collect_paper_local_research(data)
        return

    if preset.key == "resumo_artigos_local_fgv":
        print("\nDados do resumo analítico de artigos")
        _print_default_block("Tema/foco de leitura", DEFAULT_RESUMO_ARTIGOS_TEMA)
        _print_default_block("Recorte/foco comparativo", DEFAULT_RESUMO_ARTIGOS_RECORTE)
        _print_default_block("Objetivo do resumo analítico", DEFAULT_RESUMO_ARTIGOS_OBJETIVO)
        usar_padrao = ask_bool("Usar essas orientações padrão de tema, recorte e objetivo?", True)
        data["tema_recorte_objetivo_paths"] = []
        if usar_padrao:
            data["tema"] = DEFAULT_RESUMO_ARTIGOS_TEMA
            data["recorte"] = DEFAULT_RESUMO_ARTIGOS_RECORTE
            data["objetivo"] = DEFAULT_RESUMO_ARTIGOS_OBJETIVO
        else:
            usar_arquivo_tro = ask_bool("Deseja apontar um arquivo/pasta/ZIP com diretrizes de tema, recorte e objetivo?", True)
            if usar_arquivo_tro:
                tro_path = ask_required("Arquivo/pasta/ZIP com diretrizes de tema, recorte e objetivo", "")
                data["tema_recorte_objetivo_paths"] = [rel_for_toml(tro_path, data["config_dir"])]
                # Mantém valores mínimos no TOML para estabilidade do pipeline; o arquivo informado entra em [orientacoes].paths.
                data["tema"] = DEFAULT_RESUMO_ARTIGOS_TEMA
                data["recorte"] = DEFAULT_RESUMO_ARTIGOS_RECORTE
                data["objetivo"] = DEFAULT_RESUMO_ARTIGOS_OBJETIVO
            else:
                print("\nEdite manualmente tema, recorte e objetivo.")
                data["tema"] = ask("Tema/foco de leitura", DEFAULT_RESUMO_ARTIGOS_TEMA)
                data["recorte"] = ask("Recorte/foco comparativo", DEFAULT_RESUMO_ARTIGOS_RECORTE)
                data["objetivo"] = ask("Objetivo do resumo analítico", DEFAULT_RESUMO_ARTIGOS_OBJETIVO)
        data["pergunta_pesquisa"] = ""
        data["hipotese"] = ""
        data["orientacao_professor"] = ""
        data["orientacao_complementar_paths"] = []
        if ask_bool("Deseja informar orientação textual complementar?", False):
            modo_orientacao = ask_choice("Como deseja informar a orientação textual complementar", ["digitar", "arquivo"], "digitar")
            if modo_orientacao == "digitar":
                data["orientacao_professor"] = ask("Digite a orientação textual complementar", "")
            else:
                orient_comp = ask_required("Arquivo/pasta/ZIP com orientação textual complementar", "")
                data["orientacao_complementar_paths"] = [rel_for_toml(orient_comp, data["config_dir"])]
        if ask_bool("Informar palavras-chave manualmente?", False):
            data["palavras_chave"] = ask_list("Palavras-chave", DEFAULT_RESUMO_ARTIGOS_PALAVRAS)
        else:
            data["palavras_chave"] = []
        data["idiomas"] = ask_list("Idiomas", ["português"])
        data["tipo_estudo"] = "resumo analítico de artigos"
        data["resumo_nivel_detalhamento"] = ask_choice("Profundidade analítica", ["medio", "médio", "alto", "profundo", "exaustivo"], "profundo")
        prof = str(data["resumo_nivel_detalhamento"]).strip().lower()
        if prof == "exaustivo":
            defaults = {"palavras": "1500", "paragrafos": "11", "comparacao": "1300", "sintese": "1100", "chars_doc": "18000", "chars_total": "120000"}
        elif prof in {"profundo", "alto"}:
            defaults = {"palavras": "1200", "paragrafos": "9", "comparacao": "1100", "sintese": "900", "chars_doc": "14000", "chars_total": "95000"}
        else:
            defaults = {"palavras": "850", "paragrafos": "6", "comparacao": "800", "sintese": "700", "chars_doc": "10000", "chars_total": "70000"}
        usar_parametros = ask_bool("Usar parâmetros padrão de profundidade para esse modo?", True)
        if usar_parametros:
            data["resumo_min_palavras_por_artigo"] = defaults["palavras"]
            data["resumo_min_paragrafos_por_artigo"] = defaults["paragrafos"]
            data["resumo_min_palavras_comparacao"] = defaults["comparacao"]
            data["resumo_min_palavras_sintese"] = defaults["sintese"]
            data["resumo_max_chars_por_documento"] = defaults["chars_doc"]
            data["resumo_max_chars_total_corpus"] = defaults["chars_total"]
        else:
            data["resumo_min_palavras_por_artigo"] = ask("Mínimo aproximado de palavras por artigo", defaults["palavras"])
            data["resumo_min_paragrafos_por_artigo"] = ask("Mínimo de parágrafos por artigo", defaults["paragrafos"])
            data["resumo_min_palavras_comparacao"] = ask("Mínimo aproximado de palavras na comparação", defaults["comparacao"])
            data["resumo_min_palavras_sintese"] = ask("Mínimo aproximado de palavras na síntese analítica", defaults["sintese"])
            data["resumo_max_chars_por_documento"] = ask("Máximo de caracteres do texto-base enviados por documento", defaults["chars_doc"])
            data["resumo_max_chars_total_corpus"] = ask("Máximo total de caracteres do corpus enviados à IA", defaults["chars_total"])
        data["resumo_comparar_textos"] = ask_bool("Comparar convergências e divergências entre os textos?", True)
        data["resumo_sintese_analitica"] = ask_bool("Incluir síntese analítica final?", True)
        data["resumo_apontar_limites"] = ask_bool("Apontar limites/tensões dos textos quando pertinente?", True)
        data["resumo_matriz_analitica"] = ask_bool("Exigir matriz/eixos analíticos em cada texto?", True)
        data["resumo_tabela_comparativa"] = ask_bool("Incluir tabela comparativa sintética antes da comparação?", True)
        data["resumo_dialogo_entre_textos"] = ask_bool("Exigir diálogo explícito entre os textos?", True)
        return

    if preset.document_type == "atividade":
        print("\nDados da atividade")
        print("Você pode fornecer os dados centrais em um único arquivo ou preenchê-los manualmente.")
        data["atividade_dados_paths"] = []
        modo_dados = ask_choice(
            "Como deseja fornecer tema, pergunta orientadora e enunciado da atividade",
            ["ia", "arquivo", "manual"],
            "ia",
        )
        data["atividade_dados_modo"] = modo_dados
        data["atividade_gerar_dados_ia"] = (modo_dados == "ia")
        if modo_dados == "ia":
            print("\nDados da atividade: serão inferidos automaticamente pela IA a partir do corpus, das orientações, do prompt específico e dos metadados informados.")
            print("A IA deverá inferir: tema da atividade, pergunta orientadora, objetivo/recorte, enunciado operacional e palavras-chave.")
            data["tema"] = ""
            data["recorte"] = ""
            data["objetivo"] = ""
            data["pergunta_pesquisa"] = ""
            data["hipotese"] = ""
            data["orientacao_professor"] = ""
        elif modo_dados == "arquivo":
            dados_path = ask_required(
                "Arquivo/pasta/ZIP com tema, pergunta orientadora e enunciado/orientação da atividade",
                "",
            )
            data["atividade_dados_paths"] = [rel_for_toml(dados_path, data["config_dir"])]
            data["tema"] = ""
            data["recorte"] = ""
            data["objetivo"] = ""
            data["pergunta_pesquisa"] = ""
            data["hipotese"] = ""
            data["orientacao_professor"] = ""
        else:
            data["tema"] = ask("Tema da atividade", "")
            data["pergunta_pesquisa"] = ask("Pergunta orientadora da atividade", "")
            data["orientacao_professor"] = ask("Enunciado/orientação do professor", "")
            data["recorte"] = ""
            data["objetivo"] = ""
            data["hipotese"] = ""

        data["palavras_chave"] = []
        data["gerar_palavras_chave_ia"] = True
        print("Palavras-chave: serão inferidas automaticamente pela IA a partir dos dados da atividade, das orientações e do corpus.")
        data["idiomas"] = ask_list("Idiomas", ["português"])
        data["tipo_estudo"] = ask("Tipo de entrega", "atividade acadêmica")
        return

    print("\nTema, recorte e objetivo da pesquisa")
    data["tema"] = ask("Tema", "")
    data["recorte"] = ask("Recorte", "")
    data["objetivo"] = ask("Objetivo", "")
    data["pergunta_pesquisa"] = ask("Pergunta de pesquisa", "")
    data["hipotese"] = ask("Hipótese/tese orientadora", "")
    data["palavras_chave"] = ask_list("Palavras-chave", [])
    data["idiomas"] = ask_list("Idiomas", ["português"])
    if preset.document_type == "dissertacao":
        default_tipo = "dissertação"
    elif preset.document_type == "relatorio_prisma":
        default_tipo = "relatório PRISMA"
    else:
        default_tipo = "paper acadêmico"
    data["tipo_estudo"] = ask("Tipo de estudo", default_tipo)

def collect_sources(data: dict[str, Any]) -> None:
    preset: Preset = data["preset"]
    project_dir: Path = data["project_dir"]
    config_dir: Path = data["config_dir"]
    if preset.key == "relatorio_prisma_busca_orientada_fgv":
        # O fluxo especializado não pede corpus local, DOI manifest ou
        # orientações de documento antes da busca. PDFs selecionados podem ser
        # associados posteriormente na planilha de triagem, após a descoberta.
        data.setdefault("documentos_input_zip", "")
        data.setdefault("documentos_input_dir", "")
        data.setdefault("corpus_local_opcional", "")
        data.setdefault("orientacoes_paths", [])
        data.setdefault("doi_manifest_path", "")
        data.setdefault("prompt_extra_paths", [])
        return
    elif preset.render_only:
        print("\nSomente renderização")
        data["document_json"] = ask_required("Caminho do document.json existente", "")
        data["documentos_input_zip"] = ""
        data["documentos_input_dir"] = ""
        data["orientacoes_paths"] = []
        data["doi_manifest_path"] = ""
        data["prompt_extra_paths"] = []
        return

    print("\nEntradas do corpus e orientações")
    if preset.local_corpus:
        # A TUI usa uma lista visual; o modo textual preserva `zip`/`dir`
        # para compatibilidade com scripts e preenchimentos automatizados.
        if tui_theme_enabled():
            mode_label = ask_choice(
                "Formato do corpus local",
                ["arquivo ZIP (documentos-base.zip)", "pasta com documentos"],
                "arquivo ZIP (documentos-base.zip)",
            )
            input_mode = "zip" if mode_label.startswith("arquivo ZIP") else "dir"
        else:
            while True:
                raw_mode = ask("Formato do corpus local (zip/dir)", "zip").strip().lower()
                if raw_mode in {"zip", "arquivo", "arquivo zip", "arquivo_zip"}:
                    input_mode = "zip"
                    break
                if raw_mode in {"dir", "pasta", "diretório", "diretorio"}:
                    input_mode = "dir"
                    break
                print("Escolha zip ou dir.")
        initial_path = ""

        if input_mode == "zip":
            default_zip = project_dir / "documentos-base.zip"
            src = initial_path or ask_required("Caminho do documentos-base.zip", "")
            if src and ask_bool("Copiar esse arquivo para a pasta do projeto como documentos-base.zip, se existir?", False):
                src = maybe_copy_to_project(src, default_zip)
            data["documentos_input_zip"] = rel_for_toml(src, config_dir)
            data["documentos_input_dir"] = ""
        else:
            src_dir = initial_path or ask_required("Caminho da pasta de documentos", "")
            data["documentos_input_zip"] = ""
            data["documentos_input_dir"] = rel_for_toml(src_dir, config_dir)
    else:
        data["documentos_input_zip"] = ""
        data["documentos_input_dir"] = ""

    # Orientações como ZIP/path são lidas por collect_orientation_docs.
    default_orient = project_dir / "orientacoes.zip"
    orient_paths: list[str] = []
    orient_paths.extend(data.get("tema_recorte_objetivo_paths", []))
    orient_paths.extend(data.get("orientacao_complementar_paths", []))
    orient_paths.extend(data.get("atividade_dados_paths", []))
    data["orientacao_geral_inline"] = ""

    if preset.document_type == "atividade":
        modo_orientacao_geral = ask_choice(
            "Como deseja fornecer orientações gerais da aula/roteiro/rubrica",
            ["nenhuma", "arquivo", "manual"],
            "nenhuma",
        )
        data["orientacao_geral_modo"] = modo_orientacao_geral
        if modo_orientacao_geral == "arquivo":
            orient = ask_required("Arquivo/pasta/ZIP com orientações gerais da aula/roteiro/rubrica", "")
            if ask_bool("Copiar orientações gerais para a pasta do projeto como orientacoes.zip, se for arquivo existente?", False):
                orient = maybe_copy_to_project(orient, default_orient)
            orient_paths.append(rel_for_toml(orient, config_dir))
        elif modo_orientacao_geral == "manual":
            data["orientacao_geral_inline"] = ask("Digite as orientações gerais da aula/roteiro/rubrica", "")
    else:
        orient = ask("Arquivo/pasta/ZIP com orientações gerais da aula/roteiro/rubrica, se houver (vazio se não houver)", "")
        if orient.strip():
            if ask_bool("Copiar orientações gerais para a pasta do projeto como orientacoes.zip, se for arquivo existente?", False):
                orient = maybe_copy_to_project(orient, default_orient)
            orient_paths.append(rel_for_toml(orient, config_dir))

    # Remove duplicatas preservando ordem.
    seen_orient: set[str] = set()
    data["orientacoes_paths"] = [p for p in orient_paths if p and not (p in seen_orient or seen_orient.add(p))]

    # DOI manifest: opcional. A partir da rc10.7.22, nenhum caminho de arquivo é sugerido por padrão.
    default_doi = project_dir / "doi_manifest.csv"
    doi = ask("Caminho do doi_manifest.csv (vazio se não houver)", "")
    if doi.strip():
        doi_path = Path(doi).expanduser()
        if not doi_path.exists() and ask_bool("Criar doi_manifest.csv vazio nesse caminho?", False):
            target = doi_path if doi_path.is_absolute() else (config_dir / doi_path)
            write_text(target, "arquivo,doi")
            doi = str(target)
        data["doi_manifest_path"] = rel_for_toml(doi, config_dir)
    else:
        data["doi_manifest_path"] = ""

    # Prompt específico adicional: opcional. Se o usuário deixar vazio, nada é gravado no TOML.
    extra_prompts: list[str] = []
    default_add_prompt = False
    if ask_bool("Adicionar prompt específico do projeto/documento?", default_add_prompt):
        if preset.key == "resumo_artigos_local_fgv":
            default_prompt = project_dir / "prompt_base_resumo_artigos.txt"
        else:
            default_prompt = project_dir / ("prompt_base_atividade.txt" if preset.document_type == "atividade" else "prompt_extra.txt")
        prompt_path = ask("Caminho do prompt específico (vazio para não adicionar)", "")
        if prompt_path.strip():
            if Path(prompt_path).expanduser().exists() and ask_bool("Copiar prompt para a pasta do projeto?", False):
                prompt_path = maybe_copy_to_project(prompt_path, default_prompt)
            extra_prompts.append(rel_for_toml(prompt_path, config_dir))
    data["prompt_extra_paths"] = extra_prompts



PAPER_TRANSLATION_LANGUAGE_OPTIONS: dict[str, str] = {
    "inglês": "en",
    "espanhol": "es",
}


def _collect_paper_abstracts(data: dict[str, Any]) -> None:
    '''Configura resumo obrigatório no idioma principal e versões adicionais opcionais.'''
    data.setdefault("gerar_resumo_principal", True)
    data.setdefault("idioma_resumo_principal", data.get("idioma_principal_saida") or "pt-BR")
    data.setdefault("gerar_resumo_adicional", False)
    data.setdefault("idiomas_resumo_adicionais", [])
    data.setdefault("gerar_palavras_chave_resumo_adicionais", True)
    data.setdefault("resumo_max_palavras", 250)
    if bool(data.get("_paper_structured_abstract_settings")):
        print("Configuração de resumos carregada do arquivo estruturado do paper.")
        return
    if not ask_bool(
        "Além do resumo no idioma principal do paper, deseja gerar resumo em outro idioma?",
        bool(data.get("gerar_resumo_adicional", False)),
    ):
        data["gerar_resumo_adicional"] = False
        data["idiomas_resumo_adicionais"] = []
        return
    selected: list[str] = list(data.get("idiomas_resumo_adicionais", []))
    while True:
        choice = ask_choice(
            "Idioma do resumo adicional",
            ["inglês", "espanhol", "outro idioma"],
            "inglês",
        )
        if choice == "outro idioma":
            value = ask_required("Informe o idioma do resumo adicional", "")
        else:
            value = PAPER_TRANSLATION_LANGUAGE_OPTIONS[choice]
        if value not in selected:
            selected.append(value)
        if not ask_bool("Gerar resumo também em outro idioma?", False):
            break
    data["gerar_resumo_adicional"] = bool(selected)
    data["idiomas_resumo_adicionais"] = selected
    data["gerar_palavras_chave_resumo_adicionais"] = ask_bool(
        "Gerar também palavras-chave nos idiomas adicionais?",
        bool(data.get("gerar_palavras_chave_resumo_adicionais", True)),
    )
    print(
        "Os resumos serão sintetizados a partir do document.json canônico já validado. "
        "A escolha de resumo adicional não cria uma versão integral traduzida do paper."
    )


def _collect_paper_language_versions(data: dict[str, Any]) -> None:
    '''Configura versões adicionais traduzidas sem duplicar a geração do paper.'''
    data.setdefault("idioma_principal_saida", "pt-BR")
    data.setdefault("gerar_traducao_ia", False)
    data.setdefault("idiomas_adicionais_saida", [])
    data.setdefault("preservar_referencias_originais", True)
    data.setdefault("translation_max_chars_batch", 12000)
    if bool(data.get("_paper_structured_translation_settings")):
        print("Configuração de versões adicionais carregada do arquivo estruturado do paper.")
        return
    if not ask_bool(
        "Gerar versão adicional do paper em outro idioma com IA?",
        bool(data.get("gerar_traducao_ia", False)),
    ):
        data["gerar_traducao_ia"] = False
        data["idiomas_adicionais_saida"] = []
        return
    selected: list[str] = list(data.get("idiomas_adicionais_saida", []))
    while True:
        choice = ask_choice(
            "Idioma da versão adicional",
            ["inglês", "espanhol", "outro idioma"],
            "inglês",
        )
        if choice == "outro idioma":
            value = ask_required("Informe o idioma de destino", "")
        else:
            value = PAPER_TRANSLATION_LANGUAGE_OPTIONS[choice]
        if value not in selected:
            selected.append(value)
        if not ask_bool("Gerar também outra versão adicional?", False):
            break
    data["gerar_traducao_ia"] = bool(selected)
    data["idiomas_adicionais_saida"] = selected
    print(
        "As versões adicionais serão traduzidas a partir do document.json canônico já gerado. "
        "As referências, citações, DOI, URLs, fórmulas, números, siglas e nomes próprios serão preservados."
    )


def collect_outputs_and_options(data: dict[str, Any]) -> None:
    preset: Preset = data["preset"]
    project_dir: Path = data["project_dir"]
    config_dir: Path = data["config_dir"]
    if preset.key == "relatorio_prisma_busca_orientada_fgv":
        print("\nSaídas do fluxo PRISMA com busca orientada")
        print(
            "A busca inicial gera protocolo, registros brutos e deduplicados, planilha de triagem "
            "e um relatório PRISMA preliminar. Após a importação da triagem humana, o mesmo TOML "
            "gera a versão final do relatório e a matriz de estudos incluídos. ORG e PDF seguem o "
            "layout institucional selecionado no início do assistente."
        )
        default_document_output = project_dir / "output"
        raw_document_output = ask(
            "Diretório de saída do protocolo, registros, triagem e relatório PRISMA",
            str(default_document_output),
        )
        document_output = Path(raw_document_output).expanduser() if raw_document_output.strip() else default_document_output
        document_output = document_output.resolve()
        data["document_output_dir"] = rel_for_toml(document_output, config_dir)
        data["research_output_dir"] = rel_for_toml(project_dir / "output_pesquisa", config_dir)
        data["work_dir"] = rel_for_toml(project_dir / ".academic_pipeline" / "work", config_dir)
        data["cache_dir"] = rel_for_toml(project_dir / ".academic_pipeline" / "cache", config_dir)
        data["create_document_subdir"] = True
        data["exportar_org"] = ask_bool("Gerar ORG do relatório PRISMA?", True)
        data["exportar_pdf"] = ask_bool("Gerar PDF do relatório PRISMA no layout institucional?", True)
        # A escolha da engine repete o comportamento dos perfis que já
        # compilam PDF: lualatex, xelatex ou pdflatex. O valor ficará no TOML
        # e será aplicado tanto ao relatório preliminar como ao consolidado.
        data["pdf_engine"] = ask_choice(
            "Engine PDF",
            ["lualatex", "xelatex", "pdflatex"],
            "lualatex",
        ) if data["exportar_pdf"] else "lualatex"
        # A busca não gera um paper/document.json, portanto DOCX, Pandoc/CSL,
        # DOI manifest e enriquecimento de PDFs continuam fora deste perfil.
        data["exportar_docx"] = False
        data["gerar_mapa_mental"] = False
        data["plantuml_jar_path"] = ""
        data["estilo"] = "abnt"
        data["latex_style"] = "abnt"
        data["latex_options"] = "backend=biber,style=abnt,sorting=nty,giveninits=true"
        data["docx_csl"] = bundle_rel(data, "templates/csl/associacao-brasileira-de-normas-tecnicas.csl")
        data["enriquecer_metadados"] = False
        data["fontes_metadados"] = []
        data["extrair_doi_dos_pdfs"] = False
        data["buscar_metadados_por_doi"] = False
        data["usar_pandoc_docx"] = False
        # O título já foi informado nos metadados; não há motivo para pedir
        # uma segunda formulação no fim do fluxo.
        data["relatorio_prisma_titulo"] = str(data.get("titulo") or "Relatório de Pesquisa PRISMA")
        data["prisma_json_path"] = ""
        data["pesquisa_dir_existente"] = ""
        data["rel_exportar_org"] = bool(data["exportar_org"] or data["exportar_pdf"])
        data["rel_exportar_pdf"] = bool(data["exportar_pdf"])
        data["rel_exportar_docx"] = False
        data["rel_exportar_xlsx"] = True
        data["rel_exportar_fluxograma"] = False
        data["conformidade"] = False
        data["qualidade"] = False
        return

    print("\nSaídas e opções")
    default_document_output = project_dir / "output"
    raw_document_output = ask(
        "Diretório de saída dos documentos (PDF, ORG, DOCX e relatórios)",
        str(default_document_output),
    )
    document_output = Path(raw_document_output).expanduser() if raw_document_output.strip() else default_document_output
    document_output = document_output.resolve()
    data["document_output_dir"] = rel_for_toml(document_output, config_dir)
    data["research_output_dir"] = rel_for_toml(project_dir / "output_pesquisa", config_dir)
    data["work_dir"] = rel_for_toml(project_dir / ".academic_pipeline" / "work", config_dir)
    data["cache_dir"] = rel_for_toml(project_dir / ".academic_pipeline" / "cache", config_dir)
    data["create_document_subdir"] = True
    data["exportar_org"] = ask_bool("Gerar ORG?", True)
    data["exportar_pdf"] = ask_bool("Gerar PDF?", True)
    data["exportar_docx"] = ask_bool("Gerar DOCX?", True)
    data["gerar_mapa_mental"] = ask_bool("Gerar mapa mental após referências?", preset.key == "resumo_artigos_local_fgv")
    data["plantuml_jar_path"] = ""
    if data["gerar_mapa_mental"]:
        data["plantuml_jar_path"] = ask("Caminho do plantuml.jar (vazio para PLANTUML_JAR/env)", "")

    style = ask_choice("Estilo bibliográfico", ["abnt", "apa", "outro"], "abnt")
    if style == "outro":
        style = ask("Nome do estilo biblatex instalado", "authoryear-chicago")
    data["estilo"] = style
    if style == "abnt":
        data["latex_style"] = "abnt"
        data["latex_options"] = "backend=biber,style=abnt,sorting=nty,giveninits=true"
        data["docx_csl"] = bundle_rel(data, "templates/csl/associacao-brasileira-de-normas-tecnicas.csl")
    elif style == "apa":
        data["latex_style"] = "apa"
        data["latex_options"] = "backend=biber,style=apa,sorting=nyt"
        data["docx_csl"] = bundle_rel(data, "templates/csl/apa.csl")
    else:
        data["latex_style"] = style
        data["latex_options"] = f"backend=biber,style={style}"
        data["docx_csl"] = ""

    data["enriquecer_metadados"] = ask_bool("Buscar/enriquecer metadados por DOI/buscadores?", preset.document_type not in {"atividade"})
    if data["enriquecer_metadados"]:
        data["fontes_metadados"] = ask_metadata_sources(
            ["crossref", "openalex", "semantic_scholar", "scopus"]
        )
    else:
        data["fontes_metadados"] = ["crossref", "openalex"]
    data["extrair_doi_dos_pdfs"] = ask_bool("Tentar extrair DOI dos PDFs?", True)
    data["buscar_metadados_por_doi"] = data["enriquecer_metadados"] and ask_bool("Buscar metadados quando DOI estiver disponível?", True)

    data["usar_pandoc_docx"] = False
    if data["exportar_docx"]:
        data["usar_pandoc_docx"] = ask_bool("Usar Pandoc/CSL para DOCX, se disponível?", False)
    data["pdf_engine"] = ask_choice("Engine PDF", ["lualatex", "xelatex", "pdflatex"], "lualatex")

    if preset.key == "paper_local_fgv":
        _collect_paper_abstracts(data)
        _collect_paper_language_versions(data)
    else:
        data["gerar_resumo_principal"] = False
        data["idioma_resumo_principal"] = "pt-BR"
        data["gerar_resumo_adicional"] = False
        data["idiomas_resumo_adicionais"] = []
        data["gerar_palavras_chave_resumo_adicionais"] = True
        data["resumo_max_palavras"] = 250
        data["idioma_principal_saida"] = "pt-BR"
        data["gerar_traducao_ia"] = False
        data["idiomas_adicionais_saida"] = []
        data["preservar_referencias_originais"] = True
        data["translation_max_chars_batch"] = 12000

    if preset.prisma_report:
        print("\nRelatório PRISMA")
        data["relatorio_prisma_titulo"] = ask("Título do relatório PRISMA", "Relatório de Pesquisa PRISMA")
        data["prisma_json_path"] = ask("prisma_report.json existente (vazio se não houver)", "")
        data["pesquisa_dir_existente"] = ask("Diretório de pesquisa/triagem existente (vazio se não houver)", "")
        data["rel_exportar_pdf"] = ask_bool("Exportar PDF do relatório PRISMA?", True)
        data["rel_exportar_docx"] = ask_bool("Exportar DOCX do relatório PRISMA?", True)
        data["rel_exportar_xlsx"] = ask_bool("Exportar XLSX do relatório PRISMA?", True)
        data["rel_exportar_fluxograma"] = ask_bool("Exportar fluxograma PRISMA?", True)
    else:
        data["relatorio_prisma_titulo"] = ""
        data["prisma_json_path"] = ""
        data["pesquisa_dir_existente"] = ""
        data["rel_exportar_pdf"] = False
        data["rel_exportar_docx"] = False
        data["rel_exportar_xlsx"] = False
        data["rel_exportar_fluxograma"] = False

    data["conformidade"] = ask_bool("Gerar relatório de conformidade institucional?", True)
    data["qualidade"] = ask_bool("Gerar relatório de qualidade?", True)


# -----------------------------------------------------------------------------
# Renderização do TOML
# -----------------------------------------------------------------------------


def document_type_name(tipo: str) -> str:
    if tipo == "resumo_artigos":
        return "Resumo analítico de artigos"
    if tipo == "atividade":
        return "Atividade acadêmica"
    if tipo == "dissertacao":
        return "Dissertação"
    if tipo == "relatorio_prisma":
        return "Relatório de Pesquisa PRISMA"
    return "Paper acadêmico"


def prompt_paths_for_type(data: dict[str, Any]) -> dict[str, list[str]]:
    """Resolve os prompts do bundle relativamente ao TOML ativo.

    Isso permite salvar o TOML na pasta externa da disciplina sem quebrar os
    caminhos internos de prompts, templates e perfis institucionais.
    """
    preset: Preset = data["preset"]
    tipo = preset.document_type
    extras = data.get("prompt_extra_paths", [])
    paper: list[str] = []
    atividade: list[str] = []
    dissertacao: list[str] = []
    prisma: list[str] = []
    resumo_artigos: list[str] = []
    p_atividade = bundle_rel(data, "prompts/document/atividade.txt")
    p_resumo = bundle_rel(data, "prompts/document/resumo_artigos.txt")
    p_paper = bundle_rel(data, "prompts/document/paper.txt")
    p_dissertacao = bundle_rel(data, "prompts/document/dissertacao.txt")
    p_prisma = bundle_rel(data, "prompts/prisma/relatorio_prisma.txt")

    if preset.key == "resumo_artigos_local_fgv":
        atividade = [p_atividade, p_resumo, *extras]
        resumo_artigos = [p_resumo, *extras]
    elif tipo == "paper":
        paper = [p_paper, *extras]
    elif tipo == "atividade":
        atividade = [p_atividade, *extras]
    elif tipo == "dissertacao":
        dissertacao = [p_dissertacao, *extras]
    else:
        prisma = [p_prisma, *extras]
    if preset.prisma_report and p_prisma not in prisma:
        prisma.append(p_prisma)
    return {
        "paper_paths": paper,
        "atividade_paths": atividade,
        "resumo_artigos_paths": resumo_artigos,
        "dissertacao_paths": dissertacao,
        "prisma_paths": prisma,
    }

def build_orientacao_inline(data: dict[str, Any]) -> str:
    preset: Preset = data["preset"]
    professor = str(data.get("orientacao_professor") or "").strip()
    if preset.key == "resumo_artigos_local_fgv":
        parts = [
            "O documento deve ser produzido exclusivamente a partir do corpus local.",
            "Todos os textos/artigos/capítulos fornecidos devem ser mencionados e resumidos individualmente.",
            "Não use revisão bibliográfica externa como base principal e não acrescente autores, títulos ou conceitos não presentes no corpus, salvo se a orientação do usuário exigir explicitamente.",
            "Estrutura preferencial: 1 INTRODUÇÃO; 2 RESUMO INDIVIDUAL DOS TEXTOS; 3 COMPARAÇÃO ENTRE OS TEXTOS; 4 SÍNTESE ANALÍTICA; 5 CONSIDERAÇÕES FINAIS. Não crie seção textual de REFERÊNCIAS/BIBLIOGRAFIA; o renderizador insere a bibliografia automaticamente a partir do .bib. Se o mapa mental estiver ativado, ele será inserido após a bibliografia renderizada.",
            "Cada resumo individual deve explicitar objetivo do texto, problema central, argumento principal, conceitos relevantes, contribuição para o tema/foco da leitura e limites ou tensões quando pertinentes.",
        ]
        if bool(data.get("resumo_comparar_textos", True)):
            parts.append("A comparação deve destacar convergências, divergências, complementaridades e diferenças de objeto, método ou escala entre os textos.")
        if bool(data.get("resumo_sintese_analitica", True)):
            parts.append("A síntese analítica deve articular o conjunto do corpus em uma interpretação própria, evitando mera justaposição de resumos.")
        if bool(data.get("resumo_apontar_limites", True)):
            parts.append("Quando pertinente, identifique limites, tensões analíticas ou lacunas dos textos, sem extrapolar além do corpus.")
        if professor:
            parts.append("Orientação específica do professor/usuário: " + professor)
        return "\n".join(parts)
    if preset.key == "paper_local_fgv":
        parts = [
            "Produza um paper acadêmico analítico exclusivamente a partir do corpus local fornecido. "
            "Não atribua aos estudos resultados, conceitos ou conclusões que não estejam sustentados pelos documentos."
        ]
        tese = str(data.get("tese_central") or "").strip()
        if tese:
            parts.append("Tese/argumento central a desenvolver: " + tese)
        estrutura = _coerce_paper_list(data.get("estrutura_desejada"))
        if estrutura:
            parts.append("Estrutura preferencial do paper: " + " → ".join(estrutura) + ".")
        argumentos = _coerce_paper_list(data.get("argumentos_obrigatorios"))
        if argumentos:
            parts.append("Pontos que devem ser tratados explicitamente: " + "; ".join(argumentos) + ".")
        metodologia = str(data.get("orientacoes_metodologicas") or "").strip()
        if metodologia:
            parts.append("Orientações metodológicas: " + metodologia)
        limites = str(data.get("limites_do_escopo") or "").strip()
        if limites:
            parts.append("Limites de escopo: " + limites)
        tom = str(data.get("tom_de_redacao") or "").strip()
        if tom:
            parts.append("Tom de redação: " + tom + ".")
        adicionais = str(data.get("instrucoes_adicionais") or "").strip()
        if adicionais:
            parts.append("Instruções adicionais: " + adicionais)
        if professor:
            parts.append("Orientação específica do professor/usuário: " + professor)
        return "\n".join(parts)
    if preset.document_type == "atividade":
        parts: list[str] = []
        if bool(data.get("atividade_gerar_dados_ia", False)):
            parts.append(
                "Dados da atividade devem ser inferidos pela IA a partir do corpus local, das orientações gerais, "
                "do prompt específico e dos metadados acadêmicos disponíveis. Inferir tema, recorte, objetivo, "
                "pergunta orientadora, enunciado operacional e palavras-chave sem inventar conteúdo fora do material fornecido."
            )
        if professor:
            parts.append(professor)
        geral = str(data.get("orientacao_geral_inline") or "").strip()
        if geral:
            parts.append(geral)
        if bool(data.get("gerar_palavras_chave_ia", False)):
            parts.append(
                "Inferir automaticamente palavras-chave acadêmicas a partir do tema, da pergunta orientadora, "
                "do enunciado, das orientações e do corpus local; não solicitar palavras-chave ao usuário."
            )
        return "\n".join(parts)
    return ""

def render_toml(data: dict[str, Any]) -> str:
    preset: Preset = data["preset"]
    project_slug = data["project_slug"]
    tipo = preset.document_type if preset.document_type != "relatorio_prisma" else "paper"
    logical_tipo = "resumo_artigos" if preset.key == "resumo_artigos_local_fgv" else tipo
    doc_prefix = project_slug
    rel_prefix = f"relatorio_prisma_{project_slug}" if preset.prisma_report else ""
    prompt_groups = prompt_paths_for_type(data)

    # Os caminhos podem estar no próprio diretório da disciplina ou em
    # app_bundle/projetos. Sempre são relativos ao TOML final.
    output_documento = str(data.get("document_output_dir") or bundle_rel(data, "output/documento"))
    output_pesquisa = str(data.get("research_output_dir") or bundle_rel(data, "output/pesquisa"))
    output_work = str(data.get("work_dir") or bundle_rel(data, "output/work"))
    output_cache = str(data.get("cache_dir") or bundle_rel(data, "output/cache"))

    if preset.render_only:
        # Render-only ainda precisa de seções suficientes para PDF/DOCX/conformidade.
        executar_pesquisa = False
        executar_documento = True
    else:
        executar_pesquisa = preset.executar_pesquisa
        executar_documento = preset.executar_documento

    lines: list[str] = []
    lines.append(f"# Gerado por academic_pipeline_toml_generator_interativo.py")
    lines.append(f"# Preset: {preset.key} — {preset.label}")
    lines.append(f"# Descrição: {preset.description}")
    if preset.render_only:
        lines.append(f"# Uso: academic_pipeline_rc10.py --config {data['config_path']} --somente-renderizar --document-json {data.get('document_json', '')}")
    lines.append("")

    lines.append("[projeto]")
    lines.append(f"nome = {tstr(project_slug)}")
    lines.append(f"descricao = {tstr(preset.description)}")
    lines.append(f"preset = {tstr(preset.key)}")
    lines.append("")

    lines.append("[instituicao]")
    lines.append(f"perfil = {tstr(data['institution'])}")
    lines.append("")

    lines.append("[openai]")
    lines.append(f"model = {tstr(os.getenv('OPENAI_MODEL', 'gpt-5.4'))}")
    lines.append("")

    lines.append("[pipeline]")
    pipeline_mode = "somente_renderizar" if preset.render_only else ("busca_externa" if preset.key == "relatorio_prisma_busca_orientada_fgv" else "documentos_locais")
    lines.append(f"modo_entrada = {tstr(pipeline_mode)}")
    lines.append(f"executar_pesquisa = {tbool(executar_pesquisa)}")
    lines.append(f"executar_documento = {tbool(executar_documento)}")
    lines.append("executar_bundle = false")
    if data.get("pesquisa_dir_existente"):
        lines.append(f"pesquisa_dir_existente = {tstr(rel_for_toml(data['pesquisa_dir_existente'], data['config_dir']))}")
    else:
        lines.append("pesquisa_dir_existente = \"\"")
    lines.append("")

    lines.append("[paths]")
    lines.append("# A partir da rc10.7.20, todos os diretórios de saída/cache/trabalho ficam aqui.")
    lines.append(f"document_output_dir = {tstr(output_documento)}")
    lines.append(f"research_output_dir = {tstr(output_pesquisa)}")
    lines.append(f"work_dir = {tstr(output_work)}")
    lines.append(f"cache_dir = {tstr(output_cache)}")
    lines.append(f"document_prefix = {tstr(doc_prefix)}")
    lines.append(f"research_prefix = {tstr(rel_prefix or 'relatorio_prisma')}")
    lines.append(f"create_document_subdir = {tbool(bool(data.get('create_document_subdir', True)))}")
    lines.append("create_research_subdir = true")
    lines.append("create_work_subdir = true")
    lines.append("create_cache_subdir = true")
    lines.append("")

    lines.append("[orientacoes]")
    lines.append(f"paths = {tlist(data.get('orientacoes_paths', []))}")
    lines.append(f"inline = {tstr(build_orientacao_inline(data))}")
    lines.append("")

    lines.append("[documentos_locais]")
    lines.append(f"ativos = {tbool(preset.local_corpus and not preset.render_only)}")
    lines.append("modo_entrada = \"documentos_locais\"")
    lines.append(f"input_zip = {tstr(data.get('documentos_input_zip', ''))}")
    lines.append(f"input_dir = {tstr(data.get('documentos_input_dir', ''))}")
    lines.append("tipos = [\"pdf\", \"docx\", \"txt\", \"md\", \"org\"]")
    lines.append("recursive = true")
    lines.append("limpar_extracao_anterior = true")
    lines.append("copiar_para_fulltext_cache = true")
    lines.append("limpar_cache_anterior = true")
    lines.append("auto_detect_bib = true")
    lines.append("gerar_bib_revisado_ia = true")
    lines.append(f"enriquecer_metadados_buscadores = {tbool(data.get('enriquecer_metadados', False))}")
    lines.append(f"fontes_metadados = {tlist(data.get('fontes_metadados', []))}")
    lines.append("min_score_match_metadados = 0.82")
    lines.append(f"extrair_doi_dos_pdfs = {tbool(data.get('extrair_doi_dos_pdfs', True))}")
    lines.append(f"doi_manifest_path = {tstr(data.get('doi_manifest_path', ''))}")
    lines.append("preferir_doi_manual = true")
    lines.append(f"buscar_metadados_por_doi = {tbool(data.get('buscar_metadados_por_doi', False))}")
    lines.append("incluir_notas_metadados_inferidos = false")
    lines.append("deduplicar_bib = true")
    lines.append("deduplicar_referencias = true")
    lines.append("autor_padrao = \"\"")
    lines.append("ano_padrao = \"s.d.\"")
    lines.append("")

    lines.append("[pesquisa]")
    lines.append(f"tema = {tstr(data.get('tema', ''))}")
    lines.append(f"recorte = {tstr(data.get('recorte', ''))}")
    lines.append(f"objetivo = {tstr(data.get('objetivo', ''))}")
    lines.append(f"pergunta_pesquisa = {tstr(data.get('pergunta_pesquisa', ''))}")
    lines.append(f"hipotese = {tstr(data.get('hipotese', ''))}")
    lines.append(f"palavras_chave = {tlist(data.get('palavras_chave', []))}")
    lines.append(f"gerar_palavras_chave_ia = {tbool(bool(data.get('gerar_palavras_chave_ia', False)))}")
    default_idiomas = PRISMA_BUSCA_DEFAULT_LANGUAGES if preset.key == "relatorio_prisma_busca_orientada_fgv" else ["português"]
    lines.append(f"idiomas = {tlist(data.get('idiomas', default_idiomas))}")
    lines.append(f"tipo_estudo = {tstr(data.get('tipo_estudo', document_type_name(logical_tipo)))}")
    if preset.key == "relatorio_prisma_busca_orientada_fgv":
        lines.append(f"fonte_dados_pesquisa = {tstr(data.get('fonte_dados_pesquisa', ''))}")
        lines.append(f"dados_pesquisa_path = {tstr(data.get('dados_pesquisa_path', ''))}")
        lines.append(f"fonte_palavras_chave = {tstr(data.get('fonte_palavras_chave', ''))}")
        lines.append(f"palavras_chave_path = {tstr(data.get('palavras_chave_path', ''))}")
        lines.append(f"fonte_hipotese = {tstr(data.get('fonte_hipotese', ''))}")
        lines.append(f"hipotese_path = {tstr(data.get('hipotese_path', ''))}")
    if preset.key == "paper_local_fgv":
        lines.append(f"fonte_orientacoes_paper = {tstr(data.get('fonte_orientacoes_paper', ''))}")
        lines.append(f"orientacoes_paper_path = {tstr(data.get('orientacoes_paper_path', ''))}")
    # Campos textuais opcionais precisam usar tstr: TOML não aceita ``chave =``.
    lines.append(f"periodo = {tstr(data.get('periodo', ''))}")
    bases_default = data.get("bases_busca", ["crossref", "openalex", "semantic_scholar"]) if preset.key == "relatorio_prisma_busca_orientada_fgv" else ["Crossref", "OpenAlex", "Semantic Scholar", "Scopus"]
    lines.append(f"bases = {tlist([str(item) for item in bases_default])}")
    lines.append("")

    if preset.key == "relatorio_prisma_busca_orientada_fgv":
        lines.append("[busca_prisma]")
        lines.append("ativo = true")
        lines.append("modo = \"busca_externa\"")
        lines.append(f"bases = {tlist([str(item) for item in data.get('bases_busca', ['crossref', 'openalex', 'semantic_scholar'])])}")
        lines.append(f"estrategia = {tstr(data.get('estrategia_busca', 'consulta_unica'))}")
        lines.append(f"consulta_geral = {tstr(data.get('consulta_geral', ''))}")
        lines.append(f"limite_por_base = {int(data.get('limite_por_base', 100))}")
        lines.append(f"limite_scopus_por_consulta = {int(data.get('limite_scopus_por_consulta', min(int(data.get('limite_por_base', 100)), 10)))}")
        lines.append(f"limite_triagem_inicial = {int(data.get('limite_triagem_inicial', 250))}")
        lines.append(f"pre_triagem_ia = {tbool(bool(data.get('pre_triagem_ia', False)))}")
        lines.append(f"pre_triagem_ia_modelo = {tstr(data.get('pre_triagem_ia_modelo', ''))}")
        lines.append(f"pre_triagem_ia_lote = {int(data.get('pre_triagem_ia_lote', 20))}")
        lines.append(f"pre_triagem_ia_max_registros = {int(data.get('pre_triagem_ia_max_registros', 1500))}")
        lines.append(f"pre_triagem_ia_reserva_incertos = {int(data.get('pre_triagem_ia_reserva_incertos', 40))}")
        lines.append(f"pre_triagem_ia_min_confianca = {int(data.get('pre_triagem_ia_min_confianca', 55))}")
        lines.append(f"pre_triagem_ia_max_chars_resumo = {int(data.get('pre_triagem_ia_max_chars_resumo', 700))}")
        lines.append("# A IA só prioriza a leitura; inclusão/exclusão final continua obrigatoriamente humana.")
        lines.append(f"meta_estudos_incluidos = {int(data.get('meta_estudos_incluidos', 15))}")
        lines.append(f"ano_inicio = {tstr(data.get('ano_inicio', ''))}")
        lines.append(f"ano_fim = {tstr(data.get('ano_fim', ''))}")
        lines.append(f"email_contato = {tstr(data.get('email_contato_busca', ''))}")
        lines.append("# Chaves e e-mails específicos ficam no .env da raiz do software; nunca são gravados neste TOML.")
        lines.append(f"enriquecer_unpaywall = {tbool(bool(data.get('enriquecer_unpaywall', False)))}")
        lines.append(f"limite_unpaywall = {int(data.get('limite_unpaywall', 0) or 0)}")
        lines.append(f"criterios_inclusao = {tlist(data.get('criterios_inclusao', []))}")
        lines.append(f"criterios_exclusao = {tlist(data.get('criterios_exclusao', []))}")
        lines.append(f"fonte_estrategia_busca = {tstr(data.get('fonte_estrategia_busca', 'manual'))}")
        lines.append(f"estrategia_busca_path = {tstr(data.get('estrategia_busca_path', ''))}")
        lines.append(f"fonte_criterios = {tstr(data.get('fonte_criterios_prisma', 'manual'))}")
        lines.append(f"criterios_path = {tstr(data.get('criterios_prisma_path', ''))}")
        lines.append(f"corpus_local_opcional = {tstr(data.get('corpus_local_opcional', ''))}")
        lines.append("decisao_automatica_inclusao = false")
        lines.append("baixar_texto_completo_automaticamente = false")
        for block in data.get('blocos_busca', []):
            if not isinstance(block, dict):
                continue
            queries = [str(value).strip() for value in block.get('consultas', []) if str(value).strip()]
            if not queries and str(block.get('consulta') or '').strip():
                queries = [str(block.get('consulta')).strip()]
            if not queries:
                continue
            lines.append("")
            lines.append("[[busca_prisma.estrategias]]")
            lines.append(f"id = {tstr(block.get('id', ''))}")
            lines.append(f"rotulo = {tstr(block.get('rotulo', ''))}")
            lines.append(f"consultas = {tlist(queries)}")
        lines.append("")

    if preset.key == "paper_local_fgv":
        lines.append("[paper]")
        lines.append("ativo = true")
        lines.append(f"fonte_orientacoes = {tstr(data.get('fonte_orientacoes_paper', ''))}")
        lines.append(f"arquivo_estruturado = {tstr(data.get('orientacoes_paper_path', ''))}")
        lines.append(f"tese_central = {tstr(data.get('tese_central', ''))}")
        lines.append(f"estrutura_desejada = {tlist(data.get('estrutura_desejada', []))}")
        lines.append(f"argumentos_obrigatorios = {tlist(data.get('argumentos_obrigatorios', []))}")
        lines.append(f"orientacoes_metodologicas = {tstr(data.get('orientacoes_metodologicas', ''))}")
        lines.append(f"limites_do_escopo = {tstr(data.get('limites_do_escopo', ''))}")
        lines.append(f"tom_de_redacao = {tstr(data.get('tom_de_redacao', 'acadêmico analítico'))}")
        lines.append(f"instrucoes_adicionais = {tstr(data.get('instrucoes_adicionais', ''))}")
        lines.append("")

    lines.append("[resumo_artigos]")
    resumo_ativo = preset.key == "resumo_artigos_local_fgv"
    lines.append(f"ativo = {tbool(resumo_ativo)}")
    lines.append(f"geracao_em_etapas = {tbool(resumo_ativo)}")
    lines.append("# Quando true, o pipeline gera o document.json em chamadas separadas:")
    lines.append("# análise do artigo 1/N, análise do artigo 2/N, comparação e síntese.")
    lines.append("# Isso exibe progresso no terminal e salva checkpoints .checkpoint_*.json no output.")
    lines.append("modo = \"analitico_comparativo\"")
    lines.append(f"usar_apenas_corpus_local = {tbool(resumo_ativo)}")
    lines.append(f"incluir_introducao = {tbool(resumo_ativo)}")
    lines.append(f"incluir_resumos_individuais = {tbool(resumo_ativo)}")
    lines.append(f"incluir_comparacao = {tbool(bool(data.get('resumo_comparar_textos', resumo_ativo)))}")
    lines.append(f"incluir_sintese_analitica = {tbool(bool(data.get('resumo_sintese_analitica', resumo_ativo)))}")
    lines.append(f"incluir_consideracoes_finais = {tbool(resumo_ativo)}")
    lines.append("incluir_referencias = true")
    lines.append(f"incluir_mapa_mental = {tbool(data.get('gerar_mapa_mental', False))}")
    lines.append(f"uma_secao_por_artigo = {tbool(resumo_ativo)}")
    lines.append(f"comparar_convergencias = {tbool(bool(data.get('resumo_comparar_textos', resumo_ativo)))}")
    lines.append(f"comparar_divergencias = {tbool(bool(data.get('resumo_comparar_textos', resumo_ativo)))}")
    lines.append(f"apontar_limites = {tbool(bool(data.get('resumo_apontar_limites', resumo_ativo)))}")
    lines.append(f"nivel_detalhamento = {tstr(data.get('resumo_nivel_detalhamento', 'profundo' if resumo_ativo else 'medio'))}")
    lines.append(f"profundidade_analitica = {tstr(data.get('resumo_nivel_detalhamento', 'profundo' if resumo_ativo else 'medio'))}")
    if resumo_ativo:
        lines.append(f"min_palavras_por_artigo = {int(str(data.get('resumo_min_palavras_por_artigo') or 1200).strip() or 1200)}")
        lines.append(f"min_paragrafos_por_artigo = {int(str(data.get('resumo_min_paragrafos_por_artigo') or 9).strip() or 9)}")
        lines.append(f"min_palavras_comparacao = {int(str(data.get('resumo_min_palavras_comparacao') or 1100).strip() or 1100)}")
        lines.append(f"min_palavras_sintese = {int(str(data.get('resumo_min_palavras_sintese') or 900).strip() or 900)}")
        lines.append(f"max_chars_por_documento = {int(str(data.get('resumo_max_chars_por_documento') or 14000).strip() or 14000)}")
        lines.append(f"max_chars_total_corpus = {int(str(data.get('resumo_max_chars_total_corpus') or 95000).strip() or 95000)}")
        lines.append(f"exigir_matriz_analitica_por_texto = {tbool(bool(data.get('resumo_matriz_analitica', True)))}")
        lines.append(f"incluir_tabela_comparativa = {tbool(bool(data.get('resumo_tabela_comparativa', True)))}")
        lines.append(f"exigir_dialogo_entre_textos = {tbool(bool(data.get('resumo_dialogo_entre_textos', True)))}")
        lines.append("evitar_resumo_sinoptico = true")
        lines.append('eixos_analise = ["problema/questão central", "objetivo e escopo", "argumento/tese principal", "conceitos/categorias", "método/evidências", "achados/contribuições", "limites/tensões", "diálogo com o corpus"]')
    lines.append("")

    lines.append("[atividade]")
    lines.append(f"curso = {tstr(data.get('curso', ''))}")
    lines.append(f"turma = {tstr(data.get('turma', ''))}")
    lines.append(f"polo = {tstr(data.get('polo', ''))}")
    lines.append(f"disciplina = {tstr(data.get('disciplina', ''))}")
    lines.append(f"professor = {tstr(data.get('professor', ''))}")
    lines.append(f"aluno = {tstr(data.get('autor', ''))}")
    lines.append(f"data = {tstr(data.get('data', ''))}")
    lines.append(f"titulo_trabalho = {tstr(data.get('titulo', ''))}")
    lines.append(f"gerar_dados_atividade_ia = {tbool(bool(data.get('atividade_gerar_dados_ia', False)))}")
    lines.append(f"fonte_dados_atividade = {tstr(data.get('atividade_dados_modo', ''))}")
    lines.append("")

    lines.append("[documento]")
    lines.append(f"tipo_documento = {tstr(tipo)}")
    lines.append(f"tipo_conteudo = {tstr(data.get('tipo_conteudo', logical_tipo))}")
    lines.append(f"genero_academico = {tstr(data.get('genero_academico', tipo))}")
    lines.append(f"layout = {tstr(data.get('layout', ''))}")
    lines.append(f"classe_latex = {tstr(data.get('classe_latex', ''))}")
    # Caminhos/prefixo ficam em [paths].
    lines.append(f"titulo_trabalho = {tstr(data.get('titulo', ''))}")
    lines.append(f"autor = {tstr(data.get('autor', ''))}")
    lines.append("inferir_campos_vazios_ia = true")
    lines.append(f"institution_name = {tstr(data.get('institution', 'fgv'))}")
    lines.append("program_name = \"\"")
    lines.append(f"course_name = {tstr(data.get('curso', ''))}")
    lines.append(f"discipline_name = {tstr(data.get('disciplina', ''))}")
    lines.append(f"professor_name = {tstr(data.get('professor', ''))}")
    lines.append(f"city_name = {tstr(data.get('polo', 'Brasília'))}")
    lines.append(f"ano = {tstr(data.get('data', ''))}")
    lines.append(f"data = {tstr(data.get('data', ''))}")
    papertype = "Resumo analítico de artigos" if preset.key == "resumo_artigos_local_fgv" else document_type_name(tipo)
    perfil_redacao = "academico_analitico_comparativo" if preset.key == "resumo_artigos_local_fgv" else "academico_analitico"
    lines.append(f"papertype = {tstr(papertype)}")
    lines.append(f"perfil_redacao = {tstr(perfil_redacao)}")
    lines.append("covernote = \"Trabalho acadêmico elaborado para a disciplina.\"")
    if tipo == "dissertacao":
        lines.append(f"area_de_concentracao = {tstr(data.get('area_de_concentracao', ''))}")
        lines.append(f"linha_pesquisa = {tstr(data.get('linha_pesquisa', ''))}")
        lines.append(f"orientador = {tstr(data.get('orientador', ''))}")
        lines.append(f"coorientador = {tstr(data.get('coorientador', ''))}")
        lines.append(f"natureza_trabalho = {tstr(data.get('natureza_trabalho', ''))}")
        lines.append(f"data_aprovacao = {tstr(data.get('data_aprovacao', ''))}")
    lines.append(f"estilo_citacao = {tstr(data.get('estilo', 'abnt'))}")
    lines.append(f"exportar_org = {tbool(data.get('exportar_org', True))}")
    lines.append(f"exportar_pdf = {tbool(data.get('exportar_pdf', True))}")
    lines.append(f"exportar_docx = {tbool(data.get('exportar_docx', True))}")
    lines.append("gerar_documento_json = true")
    lines.append("modo_renderizacao = \"document_model\"")
    lines.append("usar_citacoes_latex_diretas = true")
    lines.append("validar_org_final = true")
    lines.append("falhar_se_org_tiver_chave_crua = true")
    lines.append("falhar_se_org_tiver_empty_citation = true")
    lines.append("falhar_se_org_tiver_mencao_tecnica = true")
    lines.append("")

    lines.append("[idiomas_saida]")
    lines.append(f"principal = {tstr(data.get('idioma_principal_saida', 'pt-BR'))}")
    lines.append(f"gerar_traducao_ia = {tbool(bool(data.get('gerar_traducao_ia', False)))}")
    lines.append(f"idiomas_adicionais = {tlist([str(item) for item in data.get('idiomas_adicionais_saida', [])])}")
    lines.append(f"preservar_referencias_originais = {tbool(bool(data.get('preservar_referencias_originais', True)))}")
    lines.append(f"max_chars_por_lote = {int(data.get('translation_max_chars_batch', 12000) or 12000)}")
    lines.append("# A tradução atua no document.json canônico; não recria análise, corpus ou bibliografia.")
    lines.append("")

    lines.append("[resumos_paper]")
    lines.append(f"ativo = {tbool(preset.key == 'paper_local_fgv')}")
    lines.append(f"principal = {tstr(data.get('idioma_resumo_principal', 'pt-BR'))}")
    lines.append(f"gerar_resumo_principal = {tbool(bool(data.get('gerar_resumo_principal', False)))}")
    lines.append(f"gerar_resumo_adicional = {tbool(bool(data.get('gerar_resumo_adicional', False)))}")
    lines.append(f"idiomas_adicionais = {tlist([str(item) for item in data.get('idiomas_resumo_adicionais', [])])}")
    lines.append(f"gerar_palavras_chave_adicionais = {tbool(bool(data.get('gerar_palavras_chave_resumo_adicionais', True)))}")
    lines.append(f"max_palavras = {int(data.get('resumo_max_palavras', 250) or 250)}")
    lines.append("# O resumo é sintetizado do document.json validado e inserido no ORG, PDF e DOCX.")
    lines.append("")

    lines.append("[bibliografia]")
    lines.append("# O .bib é neutro. O estilo visual é aplicado na renderização.")
    lines.append(f"estilo_citacao = {tstr(data.get('estilo', 'abnt'))}")
    lines.append(f"latex_style = {tstr(data.get('latex_style', 'abnt'))}")
    lines.append(f"latex_options = {tstr(data.get('latex_options', 'backend=biber,style=abnt,sorting=nty,giveninits=true'))}")
    lines.append(f"docx_csl = {tstr(data.get('docx_csl', ''))}")
    lines.append(f"buscar_metadados_por_doi = {tbool(data.get('buscar_metadados_por_doi', False))}")
    lines.append(f"enriquecer_metadados_buscadores = {tbool(data.get('enriquecer_metadados', False))}")
    lines.append("")

    lines.append("[relatorio_pesquisa]")
    lines.append(f"ativo = {tbool(preset.prisma_report)}")
    lines.append("tipo = \"prisma\"")
    lines.append(f"titulo = {tstr(data.get('relatorio_prisma_titulo', 'Relatório de Pesquisa PRISMA'))}")
    # Caminhos/prefixo ficam em [paths].
    lines.append("exportar_json = true")
    lines.append(f"exportar_org = {tbool(data.get('rel_exportar_org', True))}")
    lines.append(f"exportar_pdf = {tbool(data.get('rel_exportar_pdf', False))}")
    lines.append(f"exportar_docx = {tbool(data.get('rel_exportar_docx', False))}")
    lines.append(f"exportar_xlsx = {tbool(data.get('rel_exportar_xlsx', False))}")
    lines.append(f"exportar_fluxograma = {tbool(data.get('rel_exportar_fluxograma', False))}")
    lines.append("validar = true")
    lines.append("falhar_se_invalido = false")
    lines.append(f"prisma_json_path = {tstr(rel_for_toml(data.get('prisma_json_path', ''), data['config_dir']))}")
    lines.append(f"pesquisa_dir_existente = {tstr(rel_for_toml(data.get('pesquisa_dir_existente', ''), data['config_dir']))}")
    criterios_inclusao = data.get("criterios_inclusao", []) if preset.key == "relatorio_prisma_busca_orientada_fgv" else [
        "Aderência substantiva ao tema, recorte e objetivo.",
        "Relação direta com os textos-base ou com o problema de pesquisa.",
        "Disponibilidade de metadados mínimos ou DOI para identificação bibliográfica.",
    ]
    criterios_exclusao = data.get("criterios_exclusao", []) if preset.key == "relatorio_prisma_busca_orientada_fgv" else [
        "Fora do tema ou do recorte.",
        "Duplicado.",
        "Ausência de relação substantiva com a pergunta de pesquisa.",
    ]
    lines.append(f"criterios_inclusao = {tlist([str(item) for item in criterios_inclusao])}")
    lines.append(f"criterios_exclusao = {tlist([str(item) for item in criterios_exclusao])}")
    lines.append("")

    lines.append("[docx]")
    lines.append(f"ativo = {tbool(data.get('exportar_docx', True))}")
    # Preferência por referência institucional quando existir.
    lines.append(f"reference_docx = {tstr(bundle_rel(data, 'institutions/' + data['institution'] + '/docx/reference_fgv.docx'))}")
    lines.append(f"usar_pandoc = {tbool(data.get('usar_pandoc_docx', False))}")
    lines.append(f"csl_path = {tstr(data.get('docx_csl', ''))}")
    lines.append("falhar_se_pandoc_falhar = false")
    lines.append("incluir_capa = true")
    lines.append("incluir_referencias = true")
    lines.append(f"incluir_mapa_mental = {tbool(data.get('gerar_mapa_mental', False))}")
    lines.append("")

    lines.append("[latex]")
    lines.append(f"pdf_engine = {tstr(data.get('pdf_engine', 'lualatex'))}")
    lines.append(f"org_latex_class_init = {tstr(bundle_rel(data, 'misc/academic-writing.el'))}")
    lines.append(f"latex_extra_path = {tstr(bundle_rel(data, 'misc/fgv'))}")
    lines.append(f"fgv_logo_path = {tstr(bundle_rel(data, 'misc/fgv.png'))}")
    lines.append("")

    lines.append("[prompts]")
    lines.append("ativos = true")
    lines.append(f"global_paths = {tlist([bundle_rel(data, 'prompts/global/orientacao_geral_execucao.txt')])}")
    lines.append("institution_paths = [\"profile://prompts/fgv_geral.txt\"]")
    lines.append(f"research_paths = {tlist([bundle_rel(data, 'prompts/research/triagem_prompt.txt'), bundle_rel(data, 'prompts/research/diretivas_extras.txt')])}")
    for key in ("paper_paths", "atividade_paths", "resumo_artigos_paths", "dissertacao_paths", "prisma_paths"):
        lines.append(f"{key} = {tlist(prompt_groups[key])}")
    lines.append("document_paths = []")
    lines.append("")

    lines.append("[mapa_mental]")
    lines.append(f"gerar = {tbool(data.get('gerar_mapa_mental', False))}")
    lines.append(f"ativo = {tbool(data.get('gerar_mapa_mental', False))}")
    lines.append("posicao = \"apos_referencias\"")
    lines.append("titulo = \"Mapa mental dos textos analisados\"")
    lines.append("arquivo = \"mapa_mental\"")
    lines.append("formato = \"png\"")
    lines.append(f"renderizar = {tbool(data.get('gerar_mapa_mental', False))}")
    lines.append(f"inserir_no_org = {tbool(data.get('gerar_mapa_mental', False))}")
    lines.append(f"plantuml_jar_path = {tstr(data.get('plantuml_jar_path', ''))}")
    lines.append("plantuml_limit_size = 8192")
    lines.append("colorido = true")
    lines.append("sobrescrever_cores_existentes = true")
    lines.append('cores_niveis = ["#D9EAF7", "#DFF2E1", "#FFF2CC", "#F8D7DA", "#F8D7DA", "#F8D7DA", "#F8D7DA"]')
    lines.append("falhar_se_nao_renderizar = false")
    lines.append("")

    lines.append("[conformidade]")
    lines.append(f"ativo = {tbool(data.get('conformidade', True))}")
    lines.append("gerar_relatorio = true")
    lines.append("")

    lines.append("[qualidade]")
    lines.append(f"ativo = {tbool(data.get('qualidade', True))}")
    lines.append("gerar_relatorio = true")
    lines.append("")

    lines.append("[controle]")
    lines.append("nao_interativo = true")
    lines.append("dry_run = false")
    lines.append("mock_run = false")
    rendered_toml = "\n".join(lines).rstrip() + "\n"
    try:
        tomllib.loads(rendered_toml)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(
            "O assistente gerou um TOML inválido antes de salvar o arquivo. "
            "Nenhum TOML foi escrito; revise a configuração do wizard."
        ) from exc
    return rendered_toml


# -----------------------------------------------------------------------------
# Execução
# -----------------------------------------------------------------------------


def generate_interactive(non_interactive_profile: str | None = None, project_name: str | None = None, no_clear: bool = False, project_dir: str | Path | None = None) -> Path:
    """Executa o gerador em modo assistente por etapas.

    A navegação é por etapa, não por pergunta individual, para preservar a
    compatibilidade com os coletores existentes. Ao final de cada etapa, o
    usuário pode avançar, voltar, refazer, ver resumo ou cancelar. Antes de
    gravar o TOML, há uma revisão final com opção de editar qualquer etapa.
    """
    set_wizard_no_clear(no_clear or WIZARD_NO_CLEAR)
    set_wizard_project_dir(project_dir)
    install_path_completion()
    if non_interactive_profile:
        preset_map = {p.key: p for p in PRESETS}
        if non_interactive_profile not in preset_map:
            raise RuntimeError(f"Preset não encontrado: {non_interactive_profile}. Opções: {', '.join(preset_map)}")
        preset = preset_map[non_interactive_profile]
    else:
        wizard_header("Seleção do perfil", None, None, None)
        preset = choose_preset()

    data: dict[str, Any] = {}

    def stage_common() -> None:
        nonlocal data
        new_data = collect_common(preset)
        data.clear()
        data.update(new_data)

    if preset.key == "atividade_local_fgv":
        research_stage_title = "Dados da atividade"
    elif preset.key == "relatorio_prisma_busca_orientada_fgv":
        research_stage_title = "Dados e protocolo da busca PRISMA"
    else:
        research_stage_title = "Tema, recorte e objetivo"
    if preset.key == "relatorio_prisma_busca_orientada_fgv":
        stages: list[tuple[str, Any]] = [
            ("Projeto e perfil", stage_common),
            ("Metadados acadêmicos", lambda: collect_metadata(data)),
            (research_stage_title, lambda: collect_research(data)),
            ("Saídas da busca e triagem PRISMA", lambda: collect_outputs_and_options(data)),
        ]
    else:
        stages = [
            ("Projeto e perfil", stage_common),
            ("Metadados acadêmicos", lambda: collect_metadata(data)),
            (research_stage_title, lambda: collect_research(data)),
            ("Corpus local, orientações e prompts", lambda: collect_sources(data)),
            ("Saídas, bibliografia e mapa mental", lambda: collect_outputs_and_options(data)),
        ]

    idx = 0
    while idx < len(stages):
        title, fn = stages[idx]
        wizard_header(title, idx + 1, len(stages), preset)
        if idx > 0 and not data:
            idx = 0
            continue
        fn()
        while True:
            action = ask_stage_action(allow_back=idx > 0)
            if action == "resumo":
                show_summary(data)
                continue
            if action == "cancelar":
                raise SystemExit("Geração de TOML cancelada pelo usuário.")
            if action == "refazer":
                break
            if action == "voltar":
                idx = max(0, idx - 1)
                break
            idx += 1
            break

    # Revisão final: permite editar qualquer etapa antes de salvar.
    while True:
        wizard_header("Revisão final", None, None, preset)
        if tui_theme_enabled():
            show_summary(data)
        else:
            print(wizard_summary(data))
            print("\nNada foi salvo ainda. Revise os caminhos e metadados antes de confirmar.\n")
        action = ask_final_action()
        if action == "salvar":
            break
        if action == "cancelar":
            raise SystemExit("Geração de TOML cancelada pelo usuário.")
        if action == "editar":
            if not tui_theme_enabled():
                for i, (title, _) in enumerate(stages, start=1):
                    print(f"{i}. {title}")
            edit_idx = ask_stage_number(len(stages))
            title, fn = stages[edit_idx]
            wizard_header(title, edit_idx + 1, len(stages), preset)
            fn()

    toml = render_toml(data)
    project_dir = Path(data["project_dir"]).expanduser().resolve()
    config_path = Path(data["config_path"]).expanduser().resolve()
    try:
        config_path.relative_to(project_dir)
    except ValueError as exc:
        raise RuntimeError(
            "O TOML final deve ficar dentro do diretório do projeto. "
            f"Diretório do projeto: {project_dir}; destino informado: {config_path}."
        ) from exc

    structured_sources: list[Path] = []
    for key in ("dados_pesquisa_path", "estrategia_busca_path", "criterios_prisma_path"):
        raw = str(data.get(key) or "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        structured_sources.append(candidate.resolve())
    for raw in data.get("dados_pesquisa_paths") or []:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        structured_sources.append(candidate.resolve())
    if config_path in set(structured_sources):
        raise RuntimeError(
            "O TOML final não pode sobrescrever o arquivo estruturado usado como fonte de pesquisa. "
            "Escolha um nome distinto, por exemplo 'prisma_fluxo_pmf.toml', dentro da pasta do projeto."
        )

    if config_path.exists():
        if not ask_bool(f"O TOML já existe ({config_path}). Sobrescrever?", False):
            while True:
                alt = ask("Informe outro nome de arquivo (vazio para config_novo.toml)", "").strip() or "config_novo.toml"
                candidate = Path(alt).expanduser()
                if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != alt:
                    _notify_structured_input_error("Informe somente um nome de arquivo, sem diretórios nem caminho absoluto.")
                    continue
                if candidate.suffix.casefold() != ".toml":
                    _notify_structured_input_error("O arquivo final deve ter a extensão .toml.")
                    continue
                config_path = (project_dir / candidate.name).resolve()
                if config_path in set(structured_sources):
                    _notify_structured_input_error("Esse nome coincide com um arquivo estruturado de entrada. Escolha outro nome.")
                    continue
                break
    write_text(config_path, toml)

    try:
        config_for_cli = str(config_path.relative_to(ROOT))
    except ValueError:
        config_for_cli = str(config_path)
    command_lines = [
        "TOML gerado com sucesso:",
        f"- {config_path}",
        "",
        "Próximos comandos sugeridos:",
        f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_for_cli} --show-prompts",
        f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_for_cli} --check-config",
    ]
    if preset.render_only:
        doc_json = data.get("document_json") or "CAMINHO/document.document.json"
        command_lines.append(f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_for_cli} --somente-renderizar --document-json {doc_json}")
    else:
        command_lines.append(f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_for_cli}")
        if preset.key == "relatorio_prisma_busca_orientada_fgv":
            command_lines.append("# A busca, triagem e consolidação PRISMA serão gravadas em [paths].research_output_dir.")
        else:
            command_lines.append("# A saída principal ficou em [paths].document_output_dir. Você ainda pode sobrescrever por CLI com --output-dir /caminho/de/saida")
    if tui_theme_enabled():
        _fgv_ui().message("Academic Pipeline — TOML salvo", "\n".join(command_lines))
    else:
        print("\n".join(command_lines))
    return config_path

def print_profiles() -> None:
    for p in PRESETS:
        print(f"{p.key}: {p.label}")
        print(textwrap.fill("  " + p.description, width=92, subsequent_indent="  "))
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gerador interativo completo de TOML para academic_pipeline rc10.7.42")
    parser.add_argument("--list-profiles", action="store_true", help="Lista presets disponíveis e encerra")
    parser.add_argument("--profile", default="", help="Inicia diretamente em um preset, ex.: atividade_local_fgv")
    parser.add_argument("--project-name", default="", help="Reservado para automações futuras")
    parser.add_argument("--project-dir", default="", help="Diretório que receberá o TOML do projeto; se omitido, o wizard pergunta")
    parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela entre as etapas do wizard")
    parser.add_argument("--tui-theme", choices=["", "fgv"], default="", help="Usa diálogos visuais prompt_toolkit com paleta FGV")
    args = parser.parse_args(argv)
    set_tui_theme(args.tui_theme)
    if args.list_profiles:
        print_profiles()
        return 0
    generate_interactive(
        non_interactive_profile=args.profile or None,
        project_name=args.project_name or None,
        no_clear=bool(args.no_clear),
        project_dir=args.project_dir or None,
    )
    return 0













# >>> PATCH_WIZARD_DOCUMENTOS_LOCAIS_V4 >>>
# Camada de compatibilidade para o wizard de TOML. Atua somente nos perfis
# locais e não modifica o formato esperado pelo gerador original.


def _wiz_norm(value: object) -> str:
    import unicodedata as _unicodedata

    text = str(value or "").strip().lower()
    return "".join(
        char for char in _unicodedata.normalize("NFD", text)
        if _unicodedata.category(char) != "Mn"
    )


def _wiz_default_from_prompt(prompt: str) -> str:
    import re as _re

    values = _re.findall(r"\[([^\]]*)\]", prompt)
    return values[-1].strip() if values else ""


def _wiz_toml_section_set(text: str, section: str, key: str, literal: str) -> str:
    """Define uma chave TOML sem alterar as demais chaves da seção."""
    import re as _re

    section_re = _re.compile(
        rf"(?ms)(^\[{_re.escape(section)}\]\n)(.*?)(?=^\[|\Z)"
    )

    def _replace(match: object) -> str:
        header = match.group(1)
        body = match.group(2)
        key_re = _re.compile(rf"(?m)^([ \t]*{_re.escape(key)}[ \t]*=[ \t]*).*$" )
        if key_re.search(body):
            body = key_re.sub(rf"\g<1>{literal}", body, count=1)
        else:
            if body and not body.endswith("\n"):
                body += "\n"
            body += f"{key} = {literal}\n"
        return header + body

    if section_re.search(text):
        return section_re.sub(_replace, text, count=1)
    suffix = "" if text.endswith("\n") else "\n"
    return text + suffix + f"\n[{section}]\n{key} = {literal}\n"


def _wiz_toml_set_bool(text: str, section: str, key: str, value: bool) -> str:
    return _wiz_toml_section_set(text, section, key, "true" if value else "false")


def _wiz_toml_set_str(text: str, section: str, key: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return _wiz_toml_section_set(text, section, key, f'"{escaped}"')


def _wiz_append_orientation(text: str, instruction: str) -> str:
    """Acrescenta uma instrução ao [orientacoes].inline sem duplicá-la."""
    import re as _re

    if instruction in text:
        return text

    section_re = _re.compile(r"(?ms)(^\[orientacoes\]\n)(.*?)(?=^\[|\Z)")

    def _replace(match: object) -> str:
        header = match.group(1)
        body = match.group(2)
        triple_re = _re.compile(r'(?ms)^(\s*inline\s*=\s*""")(.*?)(""")')
        triple = triple_re.search(body)
        if triple:
            rendered = triple.group(1) + triple.group(2).rstrip() + "\n\n" + instruction + triple.group(3)
            return header + body[:triple.start()] + rendered + body[triple.end():]

        single_re = _re.compile(r'(?m)^(\s*inline\s*=\s*")((?:[^"\\]|\\.)*)("\s*)$')
        single = single_re.search(body)
        if single:
            escaped = instruction.replace("\\", "\\\\").replace('"', '\\"')
            rendered = single.group(1) + single.group(2).rstrip() + "\\n\\n" + escaped + single.group(3)
            return header + body[:single.start()] + rendered + body[single.end():]

        escaped = instruction.replace("\\", "\\\\").replace('"', '\\"')
        if body and not body.endswith("\n"):
            body += "\n"
        return header + body + f'inline = "{escaped}"\n'

    if section_re.search(text):
        return section_re.sub(_replace, text, count=1)
    escaped = instruction.replace("\\", "\\\\").replace('"', '\\"')
    suffix = "" if text.endswith("\n") else "\n"
    return text + suffix + f'\n[orientacoes]\ninline = "{escaped}"\n'


def _wiz_is_local_toml(text: str) -> bool:
    import re as _re

    return bool(
        _re.search(r'(?m)^\s*modo_entrada\s*=\s*"documentos_locais"\s*$', text)
        or "[documentos_locais]" in text
    )


def _wiz_disable_references(text: str) -> str:
    settings = (
        ("documentos_locais", "auto_detect_bib", False),
        ("documentos_locais", "gerar_bib_revisado_ia", False),
        ("documentos_locais", "enriquecer_metadados_buscadores", False),
        ("documentos_locais", "extrair_doi_dos_pdfs", False),
        ("documentos_locais", "buscar_metadados_por_doi", False),
        ("documento", "usar_citacoes_latex_diretas", False),
        ("documento", "referencias_formais", False),
        ("resumo_artigos", "incluir_referencias", False),
        ("docx", "incluir_referencias", False),
        ("idiomas_saida", "preservar_referencias_originais", False),
        ("bibliografia", "ativo", False),
        ("bibliografia", "gerar_arquivo_bib", False),
        ("bibliografia", "buscar_metadados_por_doi", False),
        ("bibliografia", "enriquecer_metadados_buscadores", False),
        ("mapa_mental", "gerar", False),
        ("mapa_mental", "ativo", False),
        ("mapa_mental", "inserir_no_org", False),
    )
    for section, key, value in settings:
        text = _wiz_toml_set_bool(text, section, key, value)

    return _wiz_append_orientation(
        text,
        "Não inclua citações no corpo do texto, notas bibliográficas, seção "
        "Referências, lista bibliográfica ou arquivo .bib. Use exclusivamente "
        "as informações fornecidas no corpus local e não invente fontes, "
        "autores, obras, links ou dados.",
    )


def _wiz_apply_text_ready_policy(text: str) -> str:
    text = _wiz_toml_set_bool(text, "atividade", "gerar_dados_atividade_ia", False)
    text = _wiz_toml_set_str(text, "atividade", "fonte_dados_atividade", "arquivo")
    text = _wiz_toml_set_bool(text, "documento", "inferir_campos_vazios_ia", False)
    return _wiz_append_orientation(
        text,
        "O corpus local foi fornecido como texto pronto. Preserve a estrutura, "
        "os argumentos, as distinções e as ressalvas do texto-base. Limite-se à "
        "revisão, organização e adequação ao formato institucional; não acrescente "
        "dados, exemplos, fontes ou interpretações externas.",
    )


class _WizLocalFlowState:
    def __init__(self) -> None:
        import sys as _sys

        args = " ".join(_wiz_norm(arg) for arg in _sys.argv)
        self.local_candidate = "local" in args or "documentos_locais" in args
        self.local_confirmed = False
        self.project_name = ""
        self.project_dir = None
        self.content_mode = ""
        self.corpus_mode = ""
        self.corpus_original = None
        self.corpus_staged = False
        self.corpus_toml_kind = ""
        self.corpus_runtime_path = None
        self.references_formal = None
        self.pdf_enabled = None
        self.docx_enabled = None
        self.skip_notice_printed = False
        self.zip_copy_choice = None
        self.text_ready_file = None
        self.reuse_text_ready_as_corpus = False
        self.reuse_notice_printed = False

    @property
    def local(self) -> bool:
        return self.local_candidate or self.local_confirmed


class _WizInputController:
    SUPPORTED_FILE_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".org"}

    def __init__(self, original_input: object, state: _WizLocalFlowState) -> None:
        self.original_input = original_input
        self.state = state
        self.awaiting_corpus_path = False

    def ask(self, prompt: str = "") -> str:
        normalized = _wiz_norm(prompt)

        if "nome do projeto" in normalized:
            answer = self.original_input(prompt)
            self.state.project_name = answer.strip() or _wiz_default_from_prompt(prompt)
            return answer

        if "diretorio do projeto" in normalized and "toml" in normalized:
            answer = self.original_input(prompt)
            rendered = answer.strip() or _wiz_default_from_prompt(prompt)
            if rendered:
                from pathlib import Path as _Path
                self.state.project_dir = _Path(rendered).expanduser()
            return answer

        if "perfil institucional" in normalized and self.state.local_candidate:
            print("- Perfil institucional: fgv (definido pelo perfil local).")
            return "fgv"

        if "layout institucional" in normalized and self.state.local_candidate:
            answer = self.original_input(
                "Layout institucional [atividade_fgv] (Enter mantém; informe outro código para alterar): "
            )
            return answer.strip() or "atividade_fgv"

        if "nome do arquivo toml" in normalized:
            default_name = self.state.project_name or _wiz_default_from_prompt(prompt) or "projeto"
            if not default_name.endswith(".toml"):
                default_name += ".toml"
            answer = self.original_input(f"Nome do arquivo TOML [{default_name}] (Enter para usar): ")
            return answer.strip() or default_name

        if normalized.startswith("acao ["):
            answer = self.original_input(
                "[Enter] Continuar | [v] Voltar | [e] Editar etapa | [r] Resumo | [c] Cancelar: "
            ).strip()
            aliases = {
                "v": "voltar", "e": "refazer", "editar": "refazer", "r": "resumo",
                "c": "cancelar", "p": "proxima", "proxima": "proxima", "proxima etapa": "proxima",
            }
            return aliases.get(_wiz_norm(answer), answer)

        if self.state.local_candidate and "como deseja fornecer tema" in normalized and "pergunta orientadora" in normalized:
            answer = self.original_input(
                "Como o conteúdo principal será fornecido? "
                "(texto_pronto/corpus_para_analise/arquivo_estruturado/manual) [texto_pronto]: "
            )
            choice = _wiz_norm(answer) or "texto_pronto"
            aliases = {
                "texto_pronto": "texto_pronto", "texto": "texto_pronto", "pronto": "texto_pronto",
                "corpus_para_analise": "corpus_para_analise", "corpus": "corpus_para_analise", "analise": "corpus_para_analise",
                "arquivo_estruturado": "arquivo_estruturado", "arquivo": "arquivo_estruturado",
                "manual": "manual",
            }
            if choice not in aliases:
                print("Opção inválida. Selecione texto_pronto, corpus_para_analise, arquivo_estruturado ou manual.")
                return self.ask(prompt)
            self.state.content_mode = aliases[choice]
            if self.state.content_mode == "texto_pronto":
                self._ask_text_ready_file_and_optional_reuse()
            mapping = {
                "texto_pronto": "ia",
                "corpus_para_analise": "ia",
                "arquivo_estruturado": "arquivo",
                "manual": "manual",
            }
            return mapping[self.state.content_mode]

        if "formato do corpus local" in normalized:
            self.state.local_confirmed = True
            if self.state.reuse_text_ready_as_corpus and self.state.corpus_runtime_path:
                if not self.state.reuse_notice_printed:
                    print("- Corpus local: reutilizando o mesmo arquivo de texto pronto já informado.")
                    self.state.reuse_notice_printed = True
                self.state.corpus_mode = "arquivo"
                self.awaiting_corpus_path = True
                return "dir"
            answer = self.original_input("Formato do conteúdo local (arquivo/zip/dir) [arquivo]: ")
            choice = _wiz_norm(answer) or "arquivo"
            aliases = {
                "arquivo": "arquivo", "file": "arquivo", "zip": "zip",
                "dir": "dir", "diretorio": "dir", "pasta": "dir",
            }
            if choice not in aliases:
                print("Opção inválida. Informe arquivo, zip ou dir.")
                return self.ask(prompt)
            self.state.corpus_mode = aliases[choice]
            self.awaiting_corpus_path = True
            return "dir" if self.state.corpus_mode == "arquivo" else self.state.corpus_mode

        if self.awaiting_corpus_path and ("caminho" in normalized or "diretorio" in normalized):
            self.awaiting_corpus_path = False
            if self.state.reuse_text_ready_as_corpus and self.state.corpus_runtime_path:
                return str(self.state.corpus_runtime_path)
            return self._ask_and_prepare_corpus()

        if self.state.local and "doi_manifest" in normalized:
            self._ensure_reference_policy()
            if self.state.references_formal is False:
                return ""
            return self.original_input(prompt)

        if self.state.local and "gerar mapa mental" in normalized and "referencia" in normalized:
            self._ensure_reference_policy()
            if self.state.references_formal is False:
                self._print_skip_notice()
                return "n"
            return self.original_input(prompt)

        if self.state.local and self.state.references_formal is False:
            if "estilo bibliografico" in normalized:
                self._print_skip_notice()
                return "abnt"
            if "buscar/enriquecer metadados" in normalized:
                return "n"
            if "tentar extrair doi" in normalized:
                return "n"

        if "gerar pdf" in normalized:
            answer = self.original_input(prompt)
            choice = _wiz_norm(answer) or _wiz_norm(_wiz_default_from_prompt(prompt))
            self.state.pdf_enabled = choice in {"s", "sim", "y", "yes", "true", "1"}
            return answer

        if "gerar docx" in normalized:
            answer = self.original_input(prompt)
            choice = _wiz_norm(answer) or _wiz_norm(_wiz_default_from_prompt(prompt))
            self.state.docx_enabled = choice in {"s", "sim", "y", "yes", "true", "1"}
            return answer

        if "usar pandoc" in normalized and self.state.docx_enabled is False:
            return "n"

        if "engine pdf" in normalized and self.state.pdf_enabled is False:
            return "lualatex"

        if "usar pandoc" in normalized and self.state.docx_enabled is True:
            return self.original_input("Usar Pandoc para gerar DOCX, se disponível? [n]: ")

        if "copiar esse arquivo" in normalized or "armazenar uma copia" in normalized:
            if self.state.zip_copy_choice is not None:
                return "s" if self.state.zip_copy_choice else "n"
            return self.original_input("Armazenar uma cópia do conteúdo-base dentro do projeto? [s]: ") or "s"

        return self.original_input(prompt)

    def _ask_supported_file(self, label: str) -> object:
        from pathlib import Path as _Path

        while True:
            raw = self.original_input(label).strip()
            candidate = _Path(raw).expanduser()
            if not raw:
                print("Campo obrigatório.")
                continue
            if not candidate.exists() or not candidate.is_file():
                print("Erro: arquivo inexistente. Informe um arquivo existente e legível.")
                continue
            if candidate.suffix.lower() not in self.SUPPORTED_FILE_SUFFIXES:
                formats = ", ".join(sorted(self.SUPPORTED_FILE_SUFFIXES))
                print(f"Formato não suportado. Informe um destes formatos: {formats}.")
                continue
            return candidate.resolve()

    def _ask_text_ready_file_and_optional_reuse(self) -> None:
        candidate = self._ask_supported_file(
            "Caminho do texto pronto (.txt/.md/.org/.docx/.pdf): "
        )
        self.state.text_ready_file = candidate
        answer = self.original_input(
            "Usar este mesmo arquivo de texto pronto como corpus local? [S]: "
        ).strip()
        reuse = _wiz_norm(answer or "s") in {"s", "sim", "y", "yes", "true", "1"}
        self.state.reuse_text_ready_as_corpus = reuse
        if not reuse:
            print("- Será solicitado um corpus local separado na etapa seguinte.")
            return
        self.state.corpus_mode = "arquivo"
        self.state.corpus_original = candidate
        self._prepare_known_corpus(candidate, "arquivo")

    def _ensure_reference_policy(self) -> None:
        if self.state.references_formal is not None:
            return
        answer = self.original_input(
            "\nEste documento local depende de fontes acadêmicas formais, "
            "com citações e referências bibliográficas? [s/N]: "
        )
        choice = _wiz_norm(answer)
        self.state.references_formal = choice in {"s", "sim", "y", "yes", "true", "1"}
        if self.state.references_formal is False:
            print("Configuração sem referências selecionada. Perguntas de DOI, estilo, .bib e mapa mental serão ignoradas.")

    def _print_skip_notice(self) -> None:
        if not self.state.skip_notice_printed:
            print("- Opções bibliográficas desativadas para este documento local.")
            self.state.skip_notice_printed = True

    def _ask_and_prepare_corpus(self) -> str:
        from pathlib import Path as _Path

        while True:
            raw = self.original_input("Caminho do conteúdo local (arquivo/zip/dir): ").strip()
            candidate = _Path(raw).expanduser()
            if not raw:
                print("Campo obrigatório.")
                continue
            if not candidate.exists():
                print("Erro: arquivo ou diretório inexistente. Informe um caminho existente e legível.")
                continue
            if self.state.corpus_mode == "arquivo":
                if not candidate.is_file() or candidate.suffix.lower() not in self.SUPPORTED_FILE_SUFFIXES:
                    print("Para 'arquivo', informe um .pdf, .docx, .txt, .md ou .org existente.")
                    continue
            elif self.state.corpus_mode == "zip":
                if not candidate.is_file() or candidate.suffix.lower() != ".zip":
                    print("Para 'zip', informe um arquivo .zip existente.")
                    continue
            elif self.state.corpus_mode == "dir" and not candidate.is_dir():
                print("Para 'dir', informe um diretório existente.")
                continue
            break

        self.state.corpus_original = candidate.resolve()
        return self._prepare_known_corpus(candidate.resolve(), self.state.corpus_mode)

    def _prepare_known_corpus(self, candidate: object, corpus_mode: str) -> str:
        from pathlib import Path as _Path
        import shutil as _shutil

        candidate = _Path(candidate).resolve()
        copy_answer = self.original_input(
            "Armazenar uma cópia do conteúdo-base dentro do projeto? [s]: "
        ).strip()
        copy_inside = _wiz_norm(copy_answer or "s") in {"s", "sim", "y", "yes", "true", "1"}

        if corpus_mode == "zip":
            self.state.zip_copy_choice = copy_inside
            self.state.corpus_toml_kind = "zip"
            self.state.corpus_runtime_path = candidate
            return str(candidate)

        project_dir = self.state.project_dir
        if project_dir is None:
            project_dir = _Path.cwd() / (self.state.project_name or "projeto")
        project_dir = _Path(project_dir).expanduser()

        if copy_inside:
            destination = project_dir / "fontes"
            destination.mkdir(parents=True, exist_ok=True)
            if corpus_mode == "arquivo":
                target = destination / candidate.name
                if not target.exists() or candidate != target.resolve():
                    _shutil.copy2(candidate, target)
            else:
                try:
                    destination.resolve().relative_to(candidate)
                except ValueError:
                    _shutil.copytree(candidate, destination, dirs_exist_ok=True)
                else:
                    raise RuntimeError(
                        "O diretório do projeto está dentro do diretório do corpus. "
                        "Escolha outro diretório de projeto ou não copie o corpus para dentro dele."
                    )
            self.state.corpus_staged = True
            self.state.corpus_toml_kind = "dir"
            self.state.corpus_runtime_path = destination
            print(f"- Conteúdo-base armazenado em: {destination}")
            return str(destination)

        if corpus_mode == "arquivo":
            link_dir = project_dir / ".academic_pipeline" / "corpus_externo"
            link_dir.mkdir(parents=True, exist_ok=True)
            link_path = link_dir / candidate.name
            try:
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(candidate)
            except OSError:
                print("Aviso: não foi possível criar link simbólico; será usado o diretório original do arquivo.")
                self.state.corpus_toml_kind = "dir"
                self.state.corpus_runtime_path = candidate.parent
                return str(candidate.parent)
            self.state.corpus_toml_kind = "external_file_link"
            self.state.corpus_runtime_path = link_dir
            return str(link_dir)

        self.state.corpus_toml_kind = "dir"
        self.state.corpus_runtime_path = candidate
        return str(candidate)


def _wiz_apply_state_to_toml(text: str, state: _WizLocalFlowState) -> str:
    if not _wiz_is_local_toml(text):
        return text

    if state.references_formal is False:
        text = _wiz_disable_references(text)

    if state.content_mode == "texto_pronto":
        text = _wiz_apply_text_ready_policy(text)

    if state.corpus_staged and state.corpus_toml_kind == "dir":
        text = _wiz_toml_set_str(text, "documentos_locais", "input_zip", "")
        text = _wiz_toml_set_str(text, "documentos_locais", "input_dir", "fontes")

    return text


def _wiz_run_generate_interactive_with_local_flow(
    target: object,
    *args: object,
    **kwargs: object,
) -> object:
    """Aplica o fluxo local ao ponto de entrada real do wizard.

    ``academic_pipeline_rc10.py --init-toml`` importa e chama
    ``generate_interactive()`` diretamente; por isso a interceptação precisa
    estar aqui, e não apenas em ``main()`` deste módulo.
    """
    import builtins as _builtins
    import pathlib as _pathlib

    state = _WizLocalFlowState()

    # O comando principal fornece o preset como argumento da função, não
    # necessariamente em sys.argv do módulo do wizard.
    requested_profile = str(
        kwargs.get("non_interactive_profile")
        or (args[0] if args else "")
        or ""
    )
    if "local" in _wiz_norm(requested_profile) or "documentos_locais" in _wiz_norm(requested_profile):
        state.local_candidate = True

    original_input = _builtins.input
    original_write_text = _pathlib.Path.write_text
    processed_paths: set[str] = set()
    controller = _WizInputController(original_input, state)

    def _input_wrapper(prompt: str = "") -> str:
        return controller.ask(prompt)

    def _write_text_wrapper(
        path: object,
        data: object,
        *write_args: object,
        **write_kwargs: object,
    ) -> int:
        rendered = data
        if isinstance(data, str) and str(path).lower().endswith(".toml"):
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key not in processed_paths and _wiz_is_local_toml(data):
                processed_paths.add(key)
                if state.references_formal is None and state.local:
                    controller._ensure_reference_policy()
                rendered = _wiz_apply_state_to_toml(data, state)
        return original_write_text(path, rendered, *write_args, **write_kwargs)

    _builtins.input = _input_wrapper
    _pathlib.Path.write_text = _write_text_wrapper
    try:
        return target(*args, **kwargs)
    finally:
        _builtins.input = original_input
        _pathlib.Path.write_text = original_write_text


_generate_interactive_before_wizard_documentos_locais_v4 = generate_interactive


def _generate_interactive_with_wizard_documentos_locais_v4(
    *args: object,
    **kwargs: object,
) -> object:
    return _wiz_run_generate_interactive_with_local_flow(
        _generate_interactive_before_wizard_documentos_locais_v4,
        *args,
        **kwargs,
    )


# Importação pelo academic_pipeline_rc10.py passa a receber esta função.
# Execução direta deste arquivo também funciona, pois main() chama o nome
# global generate_interactive em tempo de execução.
generate_interactive = _generate_interactive_with_wizard_documentos_locais_v4
# <<< PATCH_WIZARD_DOCUMENTOS_LOCAIS_V4 <<<


# >>> PATCH_POLITICA_REFERENCIAS_FORMAIS_V5 >>>
# Política nativa de referências para documentos locais. Esta camada atua em
# collect_outputs_and_options() e render_toml(), não após o TOML ser salvo.

_WIZ_V5_REFERENCE_POLICY: bool | None = None


def _v5_is_local_document(data: dict[str, Any]) -> bool:
    preset = data.get("preset")
    return bool(
        getattr(preset, "local_corpus", False)
        and not getattr(preset, "render_only", False)
        and str(getattr(preset, "key", "")) != "relatorio_prisma_busca_orientada_fgv"
    )


def _v5_reference_default(data: dict[str, Any]) -> bool:
    """Atividade local começa sem referências; paper/dissertação preservam o padrão acadêmico."""
    preset = data.get("preset")
    key = str(getattr(preset, "key", ""))
    return key in {"paper_local_fgv", "paper_prisma_fgv", "dissertacao_local_fgv", "dissertacao_prisma_fgv", "resumo_artigos_local_fgv"}


def _v5_normalise_prompt(value: object) -> str:
    import unicodedata as _unicodedata

    raw = str(value or "").strip().casefold()
    return "".join(
        char for char in _unicodedata.normalize("NFD", raw)
        if _unicodedata.category(char) != "Mn"
    )


def _v5_configure_reference_policy(data: dict[str, Any]) -> bool:
    """Pergunta uma única vez e armazena a decisão no próprio estado do wizard."""
    global _WIZ_V5_REFERENCE_POLICY
    if "incluir_referencias_formais" in data:
        value = bool(data["incluir_referencias_formais"])
        _WIZ_V5_REFERENCE_POLICY = value
        return value

    default = _v5_reference_default(data)
    value = ask_bool(
        "Este documento local deve conter citações e referências bibliográficas formais?",
        default,
    )
    data["incluir_referencias_formais"] = bool(value)
    _WIZ_V5_REFERENCE_POLICY = bool(value)
    if not value:
        print(
            "- Política sem referências selecionada: citações, .bib, DOI, estilo bibliográfico "
            "e mapa mental após referências serão desativados no TOML."
        )
    return bool(value)


_v5_collect_outputs_and_options_original = collect_outputs_and_options


def collect_outputs_and_options(data: dict[str, Any]) -> None:
    """Coleta saídas e suprime perguntas bibliográficas quando a política for sem referências."""
    if not _v5_is_local_document(data):
        return _v5_collect_outputs_and_options_original(data)

    include_references = _v5_configure_reference_policy(data)
    if include_references:
        return _v5_collect_outputs_and_options_original(data)

    # A função original continua coletando ORG/PDF/DOCX/conformidade/qualidade,
    # mas estes retornos impedem perguntas incompatíveis e gravam valores seguros.
    original_bool = globals()["ask_bool"]
    original_choice = globals()["ask_choice"]

    def policy_bool(prompt: str, default: bool = True) -> bool:
        normalized = _v5_normalise_prompt(prompt)
        suppressed = (
            "gerar mapa mental apos referencias",
            "buscar/enriquecer metadados por doi/buscadores",
            "tentar extrair doi dos pdfs",
            "buscar metadados quando doi estiver disponivel",
        )
        if any(token in normalized for token in suppressed):
            return False
        return original_bool(prompt, default)

    def policy_choice(prompt: str, choices: list[str], default: str) -> str:
        if "estilo bibliografico" in _v5_normalise_prompt(prompt):
            return "abnt"
        return original_choice(prompt, choices, default)

    globals()["ask_bool"] = policy_bool
    globals()["ask_choice"] = policy_choice
    try:
        _v5_collect_outputs_and_options_original(data)
    finally:
        globals()["ask_bool"] = original_bool
        globals()["ask_choice"] = original_choice

    data["incluir_referencias_formais"] = False
    data["gerar_mapa_mental"] = False
    data["enriquecer_metadados"] = False
    data["fontes_metadados"] = []
    data["extrair_doi_dos_pdfs"] = False
    data["buscar_metadados_por_doi"] = False
    data["preservar_referencias_originais"] = False


_v5_render_toml_original = render_toml


def render_toml(data: dict[str, Any]) -> str:
    """Renderiza valores coerentes, sem depender de mutação posterior do arquivo."""
    text = _v5_render_toml_original(data)
    if _v5_is_local_document(data) and not bool(data.get("incluir_referencias_formais", True)):
        text = _wiz_disable_references(text)
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError("A política de referências gerou TOML inválido.") from exc
    return text


# Compatibilidade: v3/v4 interceptam Path.write_text e poderiam perguntar outra
# vez ao salvar. A decisão já foi tomada na etapa de saídas, então a sincronizamos.
if "_WizInputController" in globals():
    _v5_original_ensure_reference_policy = _WizInputController._ensure_reference_policy

    def _v5_ensure_reference_policy(self: object) -> None:
        policy = globals().get("_WIZ_V5_REFERENCE_POLICY")
        if policy is not None:
            self.state.references_formal = bool(policy)
            return
        return _v5_original_ensure_reference_policy(self)

    _WizInputController._ensure_reference_policy = _v5_ensure_reference_policy
# <<< PATCH_POLITICA_REFERENCIAS_FORMAIS_V5 <<<

# >>> PATCH_CORRECAO_REFERENCIAS_V5_2 >>>
# Compatibilização da política de referências com os helpers locais v3/v4.
# A função abaixo é chamada pela renderização final e também pelo interceptor
# de salvamento do wizard, portanto a decisão se mantém em todos os fluxos.

_wiz_disable_references_pre_v5_2 = _wiz_disable_references


def _wiz_disable_references(text: str) -> str:
    text = _wiz_disable_references_pre_v5_2(text)
    for _section, _key in (
        ("documento", "referencias_formais"),
        ("bibliografia", "ativo"),
        ("bibliografia", "gerar_bib"),
        ("mapa_mental", "renderizar"),
    ):
        text = _wiz_toml_set_bool(text, _section, _key, False)
    text = _wiz_toml_set_str(text, "controle", "politica_referencias", "sem_referencias")
    return text

# <<< PATCH_CORRECAO_REFERENCIAS_V5_2 <<<


# >>> PATCH_REFERENCIAS_FORMAIS_EFETIVAS_V6_GENERATOR >>>
# A escolha "sem referências" já era aplicada por _wiz_disable_references.
# Este marcador registra que a política explícita abaixo deve constar do TOML:
# [documento] referencias_formais = false
# [bibliografia] ativo = false
# O runtime usa essas duas chaves como fonte de verdade.
# <<< PATCH_REFERENCIAS_FORMAIS_EFETIVAS_V6_GENERATOR <<<

if __name__ == "__main__":
    raise SystemExit(main())

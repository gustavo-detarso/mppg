#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gerador interativo completo de TOML para academic_pipeline rc10.7.41.

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
import os
import re
import shutil
import sys
import textwrap
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

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


def ask(prompt: str, default: str = "") -> str:
    if tui_theme_enabled():
        value = _fgv_ui().input_text(
            _dialog_title(),
            prompt,
            default=default,
            path_completion=_looks_like_path(prompt),
            only_directories=(
                ("pasta" in str(prompt or "").lower() or "diretório" in str(prompt or "").lower())
                and not any(token in str(prompt or "").lower() for token in ("arquivo", "zip", "document.json", ".pdf", ".org"))
            ),
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
            ("enunciado/orientação", data.get("orientacao_professor")),
            ("palavras-chave", "automáticas pela IA" if data.get("gerar_palavras_chave_ia") else data.get("palavras_chave")),
            ("idiomas", data.get("idiomas")),
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
    print("\nGerador interativo de TOML — academic_pipeline rc10.7.41")
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
    project_dir = ensure_project_dir(project_slug)
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

    toml_name = ask("Nome do arquivo TOML (vazio para usar o padrão interno)", "").strip() or preset.default_toml
    config_path = project_dir / toml_name

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


def collect_research(data: dict[str, Any]) -> None:
    preset: Preset = data["preset"]

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
    if preset.render_only:
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
        raw_mode = ask("Entrada do corpus: digite zip, dir ou cole diretamente o caminho", "zip").strip()
        # UX rc10.7.13: se o usuário colar um caminho .zip aqui, interpretar automaticamente como ZIP.
        candidate = Path(raw_mode).expanduser() if raw_mode else Path("")
        if raw_mode.lower() in {"zip", "dir"}:
            input_mode = raw_mode.lower()
            initial_path = ""
        elif raw_mode.lower().endswith(".zip"):
            input_mode = "zip"
            initial_path = raw_mode
        elif raw_mode and candidate.exists() and candidate.is_dir():
            input_mode = "dir"
            initial_path = raw_mode
        else:
            print("Entrada não reconhecida como caminho; assumindo modo ZIP.")
            input_mode = "zip"
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


def collect_outputs_and_options(data: dict[str, Any]) -> None:
    preset: Preset = data["preset"]
    print("\nSaídas e opções")
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
        data["docx_csl"] = "../../templates/csl/associacao-brasileira-de-normas-tecnicas.csl"
    elif style == "apa":
        data["latex_style"] = "apa"
        data["latex_options"] = "backend=biber,style=apa,sorting=nyt"
        data["docx_csl"] = "../../templates/csl/apa.csl"
    else:
        data["latex_style"] = style
        data["latex_options"] = f"backend=biber,style={style}"
        data["docx_csl"] = ""

    data["enriquecer_metadados"] = ask_bool("Buscar/enriquecer metadados por DOI/buscadores?", preset.document_type not in {"atividade"})
    if data["enriquecer_metadados"]:
        data["fontes_metadados"] = ask_list("Fontes de metadados", ["crossref", "openalex", "semantic_scholar", "scopus"])
    else:
        data["fontes_metadados"] = ["crossref", "openalex"]
    data["extrair_doi_dos_pdfs"] = ask_bool("Tentar extrair DOI dos PDFs?", True)
    data["buscar_metadados_por_doi"] = data["enriquecer_metadados"] and ask_bool("Buscar metadados quando DOI estiver disponível?", True)

    data["usar_pandoc_docx"] = False
    if data["exportar_docx"]:
        data["usar_pandoc_docx"] = ask_bool("Usar Pandoc/CSL para DOCX, se disponível?", False)
    data["pdf_engine"] = ask_choice("Engine PDF", ["lualatex", "xelatex", "pdflatex"], "lualatex")

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
    preset: Preset = data["preset"]
    tipo = preset.document_type
    extras = data.get("prompt_extra_paths", [])
    paper: list[str] = []
    atividade: list[str] = []
    dissertacao: list[str] = []
    prisma: list[str] = []
    resumo_artigos: list[str] = []

    if preset.key == "resumo_artigos_local_fgv":
        # O tipo canônico continua sendo atividade para preservar a Ficha Técnica.
        # Por isso o prompt especializado entra em atividade_paths.
        atividade = ["../../prompts/document/atividade.txt", "../../prompts/document/resumo_artigos.txt", *extras]
        resumo_artigos = ["../../prompts/document/resumo_artigos.txt", *extras]
    elif tipo == "paper":
        paper = ["../../prompts/document/paper.txt", *extras]
    elif tipo == "atividade":
        atividade = ["../../prompts/document/atividade.txt", *extras]
    elif tipo == "dissertacao":
        dissertacao = ["../../prompts/document/dissertacao.txt", *extras]
    else:
        prisma = ["../../prompts/prisma/relatorio_prisma.txt", *extras]
    if preset.prisma_report and "../../prompts/prisma/relatorio_prisma.txt" not in prisma:
        prisma.append("../../prompts/prisma/relatorio_prisma.txt")
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

    # Como o TOML fica em app_bundle/projetos/<slug>, estes caminhos são relativos ao arquivo.
    output_documento = "../../output/documento"
    output_pesquisa = "../../output/pesquisa"
    output_work = "../../output/work"
    output_cache = "../../output/cache"

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
    lines.append(f"modo_entrada = {tstr('somente_renderizar' if preset.render_only else 'documentos_locais')}")
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
    lines.append("create_document_subdir = true")
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
    lines.append(f"idiomas = {tlist(data.get('idiomas', ['português']))}")
    lines.append(f"tipo_estudo = {tstr(data.get('tipo_estudo', document_type_name(logical_tipo)))}")
    lines.append("periodo = \"\"")
    lines.append("bases = [\"Crossref\", \"OpenAlex\", \"Semantic Scholar\", \"Scopus\"]")
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
    lines.append("exportar_org = true")
    lines.append(f"exportar_pdf = {tbool(data.get('rel_exportar_pdf', False))}")
    lines.append(f"exportar_docx = {tbool(data.get('rel_exportar_docx', False))}")
    lines.append(f"exportar_xlsx = {tbool(data.get('rel_exportar_xlsx', False))}")
    lines.append(f"exportar_fluxograma = {tbool(data.get('rel_exportar_fluxograma', False))}")
    lines.append("validar = true")
    lines.append("falhar_se_invalido = false")
    lines.append(f"prisma_json_path = {tstr(rel_for_toml(data.get('prisma_json_path', ''), data['config_dir']))}")
    lines.append(f"pesquisa_dir_existente = {tstr(rel_for_toml(data.get('pesquisa_dir_existente', ''), data['config_dir']))}")
    lines.append("criterios_inclusao = [")
    lines.append("  \"Aderência substantiva ao tema, recorte e objetivo.\",")
    lines.append("  \"Relação direta com os textos-base ou com o problema de pesquisa.\",")
    lines.append("  \"Disponibilidade de metadados mínimos ou DOI para identificação bibliográfica.\"")
    lines.append("]")
    lines.append("criterios_exclusao = [")
    lines.append("  \"Fora do tema ou do recorte.\",")
    lines.append("  \"Duplicado.\",")
    lines.append("  \"Ausência de relação substantiva com a pergunta de pesquisa.\"")
    lines.append("]")
    lines.append("")

    lines.append("[docx]")
    lines.append(f"ativo = {tbool(data.get('exportar_docx', True))}")
    # Preferência por referência institucional quando existir.
    lines.append(f"reference_docx = {tstr('../../institutions/' + data['institution'] + '/docx/reference_fgv.docx')}")
    lines.append(f"usar_pandoc = {tbool(data.get('usar_pandoc_docx', False))}")
    lines.append(f"csl_path = {tstr(data.get('docx_csl', ''))}")
    lines.append("falhar_se_pandoc_falhar = false")
    lines.append("incluir_capa = true")
    lines.append("incluir_referencias = true")
    lines.append(f"incluir_mapa_mental = {tbool(data.get('gerar_mapa_mental', False))}")
    lines.append("")

    lines.append("[latex]")
    lines.append(f"pdf_engine = {tstr(data.get('pdf_engine', 'lualatex'))}")
    lines.append("org_latex_class_init = \"../../misc/academic-writing.el\"")
    lines.append("latex_extra_path = \"../../misc/fgv\"")
    lines.append("fgv_logo_path = \"../../misc/fgv.png\"")
    lines.append("")

    lines.append("[prompts]")
    lines.append("ativos = true")
    lines.append("global_paths = [\"../../prompts/global/orientacao_geral_execucao.txt\"]")
    lines.append("institution_paths = [\"profile://prompts/fgv_geral.txt\"]")
    lines.append("research_paths = [")
    lines.append("  \"../../prompts/research/triagem_prompt.txt\",")
    lines.append("  \"../../prompts/research/diretivas_extras.txt\"")
    lines.append("]")
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
    return "\n".join(lines).rstrip() + "\n"


# -----------------------------------------------------------------------------
# Execução
# -----------------------------------------------------------------------------


def generate_interactive(non_interactive_profile: str | None = None, project_name: str | None = None, no_clear: bool = False) -> Path:
    """Executa o gerador em modo assistente por etapas.

    A navegação é por etapa, não por pergunta individual, para preservar a
    compatibilidade com os coletores existentes. Ao final de cada etapa, o
    usuário pode avançar, voltar, refazer, ver resumo ou cancelar. Antes de
    gravar o TOML, há uma revisão final com opção de editar qualquer etapa.
    """
    set_wizard_no_clear(no_clear or WIZARD_NO_CLEAR)
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

    research_stage_title = "Dados da atividade" if preset.key == "atividade_local_fgv" else "Tema, recorte e objetivo"
    stages: list[tuple[str, Any]] = [
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
    config_path: Path = data["config_path"]
    if config_path.exists():
        if not ask_bool(f"O TOML já existe ({config_path}). Sobrescrever?", False):
            alt = ask("Informe outro nome de arquivo (vazio para config_novo.toml)", "").strip() or "config_novo.toml"
            config_path = data["project_dir"] / alt
    write_text(config_path, toml)

    command_lines = [
        "TOML gerado com sucesso:",
        f"- {config_path}",
        "",
        "Próximos comandos sugeridos:",
        f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_path.relative_to(ROOT)} --show-prompts",
        f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_path.relative_to(ROOT)} --check-config",
    ]
    if preset.render_only:
        doc_json = data.get("document_json") or "CAMINHO/document.document.json"
        command_lines.append(f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_path.relative_to(ROOT)} --somente-renderizar --document-json {doc_json}")
    else:
        command_lines.append(f"pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config {config_path.relative_to(ROOT)}")
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
    parser = argparse.ArgumentParser(description="Gerador interativo completo de TOML para academic_pipeline rc10.7.41")
    parser.add_argument("--list-profiles", action="store_true", help="Lista presets disponíveis e encerra")
    parser.add_argument("--profile", default="", help="Inicia diretamente em um preset, ex.: atividade_local_fgv")
    parser.add_argument("--project-name", default="", help="Reservado para automações futuras")
    parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela entre as etapas do wizard")
    parser.add_argument("--tui-theme", choices=["", "fgv"], default="", help="Usa diálogos visuais prompt_toolkit com paleta FGV")
    args = parser.parse_args(argv)
    set_tui_theme(args.tui_theme)
    if args.list_profiles:
        print_profiles()
        return 0
    generate_interactive(non_interactive_profile=args.profile or None, project_name=args.project_name or None, no_clear=bool(args.no_clear))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

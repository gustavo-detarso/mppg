#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface gráfica FGV para o Academic Pipeline.

Esta camada substitui o fluxo exclusivamente terminal quando o usuário executa
``academic_pipeline_rc10.py --gui``. A interface é uma GUI nativa em Tkinter:
janela gráfica, navegação por etapas, seleção de arquivos/pastas, autocompletar
caminhos, revisão visual, validação e geração monitorada do trabalho.

O pipeline canônico continua sendo a única fonte de geração. Esta interface
apenas monta o TOML pelo mesmo renderizador oficial e chama os comandos
``--check-config`` e de geração completa em subprocesso.
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:  # Carregamento tardio para manter o CLI funcional em instalações sem Tk.
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except Exception as exc:  # pragma: no cover - depende do sistema do usuário
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None

from academic_pipeline_toml_generator_interativo import (
    PRESETS,
    latex_class_for_layout,
    rel_for_toml,
    render_toml,
    slugify,
    write_text,
)


HERE = Path(__file__).resolve()
APP = HERE.parents[2]
ROOT = APP.parent
PIPELINE = HERE.with_name("academic_pipeline_rc10.py")
GUI_STATE_PATH = APP / ".academic_pipeline_gui_state.toml"

# Paleta inspirada no sistema visual FGV usado nos ativos já presentes no bundle.
FGV_NAVY = "#003A70"
FGV_NAVY_DARK = "#00264D"
FGV_BLUE = "#0067B1"
FGV_CYAN = "#00A3E0"
FGV_BG = "#F4F7FA"
FGV_CARD = "#FFFFFF"
FGV_BORDER = "#D7E0E8"
FGV_TEXT = "#13283D"
FGV_MUTED = "#5E7184"
FGV_SUCCESS = "#138A5B"
FGV_WARNING = "#B76E00"
FGV_DANGER = "#B42318"


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    hint: str


STEPS: tuple[Step, ...] = (
    Step("project", "1. Projeto", "Identificação acadêmica e arquivo TOML"),
    Step("activity", "2. Dados da atividade", "IA, arquivo único ou preenchimento manual"),
    Step("inputs", "3. Corpus e orientações", "Textos-base, orientações e prompt específico"),
    Step("outputs", "4. Saídas", "PDF, ORG, DOCX e auditorias"),
    Step("review", "5. Revisar e gerar", "Salvar, validar, gerar e abrir resultados"),
)


def _safe_resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _existing_path(raw: str) -> Path | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        path = Path(value).expanduser()
        return path.resolve() if path.exists() else None
    except OSError:
        return None


def _is_zip_path(raw: str) -> bool:
    return (raw or "").strip().lower().endswith(".zip")


def _path_display(path: Path) -> str:
    try:
        home = Path.home().resolve()
        return "~" + str(path.resolve()).removeprefix(str(home)) if str(path.resolve()).startswith(str(home)) else str(path)
    except OSError:
        return str(path)


def path_suggestions(raw: str, *, limit: int = 12) -> list[str]:
    """Retorna sugestões de caminho sem realizar varredura custosa no disco."""
    value = (raw or "").strip()
    if not value:
        candidates = [Path.home(), APP / "projetos", APP / "output", ROOT]
        return [str(p) + (os.sep if p.is_dir() else "") for p in candidates if p.exists()]

    expanded = os.path.expanduser(value)
    # Quando o texto termina em separador, o próprio texto é diretório-base.
    if expanded.endswith(os.sep):
        parent = Path(expanded)
        prefix = ""
    else:
        parent = Path(os.path.dirname(expanded) or ".")
        prefix = os.path.basename(expanded)
    try:
        if not parent.exists() or not parent.is_dir():
            return []
        items = sorted(parent.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return []

    results: list[str] = []
    for item in items:
        if prefix and not item.name.lower().startswith(prefix.lower()):
            continue
        candidate = str(item)
        if value.startswith("~"):
            try:
                candidate = "~" + candidate.removeprefix(str(Path.home()))
            except Exception:
                pass
        if item.is_dir():
            candidate += os.sep
        results.append(candidate)
        if len(results) >= limit:
            break
    return results


class AutocompletePathEntry(ttk.Entry):
    """Entrada de caminho com autocomplete, Tab e lista suspensa."""

    def __init__(self, master: tk.Misc, variable: tk.StringVar, **kwargs: Any) -> None:
        super().__init__(master, textvariable=variable, **kwargs)
        self.variable = variable
        self.popup: tk.Toplevel | None = None
        self.listbox: tk.Listbox | None = None
        self._suggestions: list[str] = []
        self._after_id: str | None = None
        self.bind("<KeyRelease>", self._on_key_release, add=True)
        self.bind("<FocusOut>", self._on_focus_out, add=True)
        self.bind("<Tab>", self._on_tab, add=True)
        self.bind("<Down>", self._on_down, add=True)
        self.bind("<Escape>", self._on_escape, add=True)

    def _on_key_release(self, event: tk.Event) -> None:
        if event.keysym in {"Up", "Down", "Return", "Escape", "Tab", "Shift_L", "Shift_R", "Control_L", "Control_R"}:
            return
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(100, self.show_suggestions)

    def _on_focus_out(self, _event: tk.Event) -> None:
        self.after(180, self.hide_suggestions)

    def _on_tab(self, _event: tk.Event) -> str:
        if not self._suggestions:
            self.show_suggestions()
        if self._suggestions:
            self._choose(0)
        return "break"

    def _on_down(self, _event: tk.Event) -> str | None:
        if self.listbox and self._suggestions:
            self.listbox.focus_set()
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.listbox.activate(0)
            return "break"
        return None

    def _on_escape(self, _event: tk.Event) -> str:
        self.hide_suggestions()
        return "break"

    def show_suggestions(self) -> None:
        self._after_id = None
        suggestions = path_suggestions(self.variable.get())
        self._suggestions = suggestions
        if not suggestions:
            self.hide_suggestions()
            return

        if self.popup is None:
            self.popup = tk.Toplevel(self)
            self.popup.overrideredirect(True)
            self.popup.attributes("-topmost", True)
            self.listbox = tk.Listbox(
                self.popup,
                height=min(len(suggestions), 8),
                activestyle="none",
                selectmode=tk.SINGLE,
                bg=FGV_CARD,
                fg=FGV_TEXT,
                highlightthickness=1,
                highlightbackground=FGV_BORDER,
                selectbackground=FGV_CYAN,
                selectforeground="#FFFFFF",
                font=("DejaVu Sans", 10),
            )
            self.listbox.pack(fill=tk.BOTH, expand=True)
            self.listbox.bind("<ButtonRelease-1>", self._choose_selected)
            self.listbox.bind("<Return>", self._choose_selected)
            self.listbox.bind("<Escape>", lambda _e: self.hide_suggestions())
            self.listbox.bind("<FocusOut>", lambda _e: self.after(180, self.hide_suggestions()))

        assert self.listbox is not None
        self.listbox.delete(0, tk.END)
        for item in suggestions:
            self.listbox.insert(tk.END, item)
        self.listbox.configure(height=min(len(suggestions), 8))
        self.popup.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        width = max(self.winfo_width(), 460)
        self.popup.geometry(f"{width}x{min(len(suggestions), 8) * 24 + 4}+{x}+{y}")
        self.popup.deiconify()

    def hide_suggestions(self) -> None:
        if self.popup is not None:
            self.popup.withdraw()

    def _choose_selected(self, _event: tk.Event | None = None) -> str:
        if self.listbox is None:
            return "break"
        selection = self.listbox.curselection()
        if selection:
            self._choose(int(selection[0]))
        return "break"

    def _choose(self, index: int) -> None:
        if not self._suggestions:
            return
        index = max(0, min(index, len(self._suggestions) - 1))
        self.variable.set(self._suggestions[index])
        self.icursor(tk.END)
        self.focus_set()
        self.hide_suggestions()


class ScrollableFrame(ttk.Frame):
    """Frame rolável, usado para formulários longos sem poluir a janela."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=FGV_BG)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="App.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.content.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_width)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add=True)
        self.canvas.bind_all("<Button-4>", lambda _e: self.canvas.yview_scroll(-3, "units"), add=True)
        self.canvas.bind_all("<Button-5>", lambda _e: self.canvas.yview_scroll(3, "units"), add=True)

    def _sync_scroll_region(self, _event: tk.Event | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * int(event.delta / 120) if event.delta else 0
        if delta:
            try:
                self.canvas.yview_scroll(delta, "units")
            except tk.TclError:
                return

    def to_top(self) -> None:
        self.canvas.yview_moveto(0)


class AcademicPipelineGUI:
    """Wizard gráfico focado no perfil de atividade local FGV."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Academic Pipeline — Atividades FGV")
        self.root.geometry("1320x850")
        self.root.minsize(1080, 720)
        self.root.configure(bg=FGV_BG)
        self._busy = False
        self._worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._active_process: subprocess.Popen[str] | None = None
        self.current_step = 0
        self.step_frames: dict[str, ScrollableFrame] = {}
        self.step_buttons: dict[str, ttk.Button] = {}
        self._logo: tk.PhotoImage | None = None
        self._build_variables()
        self._configure_style()
        self._build_layout()
        self._render_step("project")
        self.root.after(120, self._poll_worker_queue)

    # ------------------------------------------------------------------
    # Estado, dados e TOML
    # ------------------------------------------------------------------

    def _build_variables(self) -> None:
        today_year = str(time.localtime().tm_year)
        self.project_name = tk.StringVar(value="atividade_aula")
        self.toml_name = tk.StringVar(value="atividade_config.toml")
        self.title = tk.StringVar(value="Atividade acadêmica")
        self.author = tk.StringVar(value="Gustavo M. Mendes de Tarso")
        self.course = tk.StringVar(value="Mestrado Acadêmico em Políticas Públicas e Governo")
        self.class_group = tk.StringVar(value="2026.1")
        self.city = tk.StringVar(value="Brasília")
        self.discipline = tk.StringVar(value="")
        self.professor = tk.StringVar(value="")
        self.academic_year = tk.StringVar(value=today_year)

        self.data_mode = tk.StringVar(value="ia")
        self.activity_data_path = tk.StringVar(value="")
        self.manual_topic = tk.StringVar(value="")
        self.manual_question = tk.StringVar(value="")
        self.manual_instruction = tk.StringVar(value="")

        self.corpus_mode = tk.StringVar(value="zip")
        self.corpus_path = tk.StringVar(value="")
        self.orientation_mode = tk.StringVar(value="nenhuma")
        self.orientation_path = tk.StringVar(value="")
        self.prompt_path = tk.StringVar(value="")
        self.doi_manifest_path = tk.StringVar(value="")
        self.orientation_inline = ""

        self.export_org = tk.BooleanVar(value=True)
        self.export_pdf = tk.BooleanVar(value=True)
        self.export_docx = tk.BooleanVar(value=True)
        self.extract_doi = tk.BooleanVar(value=True)
        self.enrich_metadata = tk.BooleanVar(value=False)
        self.conformity = tk.BooleanVar(value=True)
        self.quality = tk.BooleanVar(value=True)
        self.output_override = tk.StringVar(value="")
        self.pdf_engine = tk.StringVar(value="lualatex")
        self.status = tk.StringVar(value="Pronto para configurar a atividade.")
        self.config_path_var = tk.StringVar(value="")
        self.output_path_var = tk.StringVar(value="")

    def _preset(self) -> Any:
        for preset in PRESETS:
            if preset.key == "atividade_local_fgv":
                return preset
        raise RuntimeError("Preset atividade_local_fgv não encontrado no gerador.")

    def project_slug(self) -> str:
        return slugify(self.project_name.get())

    def project_dir(self) -> Path:
        return APP / "projetos" / self.project_slug()

    def config_path(self) -> Path:
        name = self.toml_name.get().strip() or "atividade_config.toml"
        if not name.lower().endswith(".toml"):
            name += ".toml"
        return self.project_dir() / name

    def output_dir(self) -> Path:
        base = _existing_path(self.output_override.get())
        if base is not None:
            return base / self.project_slug()
        return APP / "output" / "documento" / self.project_slug()

    def _activity_data(self) -> dict[str, Any]:
        preset = self._preset()
        project_dir = self.project_dir()
        config_path = self.config_path()
        project_dir.mkdir(parents=True, exist_ok=True)
        config_dir = project_dir

        mode = self.data_mode.get().strip() or "ia"
        activity_data_rel = rel_for_toml(self.activity_data_path.get(), config_dir) if mode == "arquivo" and self.activity_data_path.get().strip() else ""
        corpus_rel = rel_for_toml(self.corpus_path.get(), config_dir) if self.corpus_path.get().strip() else ""
        orient_rel = rel_for_toml(self.orientation_path.get(), config_dir) if self.orientation_mode.get() == "arquivo" and self.orientation_path.get().strip() else ""
        prompt_rel = rel_for_toml(self.prompt_path.get(), config_dir) if self.prompt_path.get().strip() else ""
        doi_rel = rel_for_toml(self.doi_manifest_path.get(), config_dir) if self.doi_manifest_path.get().strip() else ""

        manual = mode == "manual"
        orient_paths: list[str] = []
        if activity_data_rel:
            orient_paths.append(activity_data_rel)
        if orient_rel:
            orient_paths.append(orient_rel)

        # Remove duplicadas preservando ordem.
        seen: set[str] = set()
        orient_paths = [item for item in orient_paths if item and not (item in seen or seen.add(item))]

        estilo = "abnt"
        latex_style = "abnt"
        latex_options = "backend=biber,style=abnt,sorting=nty,giveninits=true"
        docx_csl = "../../templates/csl/associacao-brasileira-de-normas-tecnicas.csl"

        data: dict[str, Any] = {
            "preset": preset,
            "project_name": self.project_name.get().strip(),
            "project_slug": self.project_slug(),
            "project_dir": project_dir,
            "institution": "fgv",
            "layout": "atividade_fgv",
            "genero_academico": "atividade",
            "tipo_conteudo": "atividade",
            "classe_latex": latex_class_for_layout("fgv", "atividade_fgv", "atividade"),
            "config_path": config_path,
            "config_dir": config_dir,
            "titulo": self.title.get().strip() or "Atividade acadêmica",
            "autor": self.author.get().strip(),
            "curso": self.course.get().strip(),
            "turma": self.class_group.get().strip(),
            "polo": self.city.get().strip() or "Brasília",
            "disciplina": self.discipline.get().strip(),
            "professor": self.professor.get().strip(),
            "data": self.academic_year.get().strip(),
            "area_de_concentracao": "",
            "linha_pesquisa": "",
            "orientador": "",
            "coorientador": "",
            "data_aprovacao": "",
            "natureza_trabalho": "",
            "atividade_dados_modo": mode,
            "atividade_gerar_dados_ia": mode == "ia",
            "atividade_dados_paths": [activity_data_rel] if activity_data_rel else [],
            "tema": self.manual_topic.get().strip() if manual else "",
            "recorte": "",
            "objetivo": "",
            "pergunta_pesquisa": self.manual_question.get().strip() if manual else "",
            "hipotese": "",
            "orientacao_professor": self.manual_instruction.get().strip() if manual else "",
            "palavras_chave": [],
            "gerar_palavras_chave_ia": True,
            "idiomas": ["português"],
            "tipo_estudo": "atividade acadêmica",
            "documentos_input_zip": corpus_rel if self.corpus_mode.get() == "zip" else "",
            "documentos_input_dir": corpus_rel if self.corpus_mode.get() == "dir" else "",
            "orientacoes_paths": orient_paths,
            "orientacao_geral_modo": self.orientation_mode.get(),
            "orientacao_geral_inline": self.orientation_inline.strip() if self.orientation_mode.get() == "manual" else "",
            "doi_manifest_path": doi_rel,
            "prompt_extra_paths": [prompt_rel] if prompt_rel else [],
            "exportar_org": bool(self.export_org.get()),
            "exportar_pdf": bool(self.export_pdf.get()),
            "exportar_docx": bool(self.export_docx.get()),
            "gerar_mapa_mental": False,
            "plantuml_jar_path": "",
            "estilo": estilo,
            "latex_style": latex_style,
            "latex_options": latex_options,
            "docx_csl": docx_csl,
            "enriquecer_metadados": bool(self.enrich_metadata.get()),
            "fontes_metadados": ["crossref", "openalex"],
            "extrair_doi_dos_pdfs": bool(self.extract_doi.get()),
            "buscar_metadados_por_doi": bool(self.enrich_metadata.get()),
            "usar_pandoc_docx": False,
            "pdf_engine": self.pdf_engine.get().strip() or "lualatex",
            "relatorio_prisma_titulo": "",
            "prisma_json_path": "",
            "pesquisa_dir_existente": "",
            "rel_exportar_pdf": False,
            "rel_exportar_docx": False,
            "rel_exportar_xlsx": False,
            "rel_exportar_fluxograma": False,
            "conformidade": bool(self.conformity.get()),
            "qualidade": bool(self.quality.get()),
        }
        return data

    def _write_toml(self) -> Path:
        data = self._activity_data()
        config_path: Path = data["config_path"]
        write_text(config_path, render_toml(data))
        self.config_path_var.set(str(config_path))
        self.output_path_var.set(str(self.output_dir()))
        return config_path

    # ------------------------------------------------------------------
    # Layout e estilo visual
    # ------------------------------------------------------------------

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        default_font = ("DejaVu Sans", 10)
        heading_font = ("DejaVu Sans", 15, "bold")
        small_font = ("DejaVu Sans", 9)
        style.configure("App.TFrame", background=FGV_BG)
        style.configure("Card.TFrame", background=FGV_CARD)
        style.configure("Sidebar.TFrame", background=FGV_NAVY)
        style.configure("Header.TFrame", background=FGV_CARD)
        style.configure("TLabel", background=FGV_BG, foreground=FGV_TEXT, font=default_font)
        style.configure("Card.TLabel", background=FGV_CARD, foreground=FGV_TEXT, font=default_font)
        style.configure("CardMuted.TLabel", background=FGV_CARD, foreground=FGV_MUTED, font=small_font)
        style.configure("Heading.TLabel", background=FGV_BG, foreground=FGV_NAVY, font=heading_font)
        style.configure("Subheading.TLabel", background=FGV_BG, foreground=FGV_MUTED, font=("DejaVu Sans", 10))
        style.configure("SidebarTitle.TLabel", background=FGV_NAVY, foreground="#FFFFFF", font=("DejaVu Sans", 14, "bold"))
        style.configure("SidebarText.TLabel", background=FGV_NAVY, foreground="#DCEBFA", font=small_font)
        style.configure("Step.TButton", background=FGV_NAVY, foreground="#FFFFFF", borderwidth=0, padding=(16, 10), anchor="w", font=("DejaVu Sans", 10, "bold"))
        style.map("Step.TButton", background=[("active", FGV_BLUE), ("pressed", FGV_NAVY_DARK)])
        style.configure("StepActive.TButton", background=FGV_CYAN, foreground="#FFFFFF", borderwidth=0, padding=(16, 10), anchor="w", font=("DejaVu Sans", 10, "bold"))
        style.map("StepActive.TButton", background=[("active", FGV_CYAN), ("pressed", FGV_BLUE)])
        style.configure("Primary.TButton", background=FGV_BLUE, foreground="#FFFFFF", borderwidth=0, padding=(14, 9), font=("DejaVu Sans", 10, "bold"))
        style.map("Primary.TButton", background=[("active", FGV_CYAN), ("disabled", "#9CB7CE")])
        style.configure("Secondary.TButton", background="#E7EEF5", foreground=FGV_NAVY, borderwidth=0, padding=(13, 9), font=("DejaVu Sans", 10, "bold"))
        style.map("Secondary.TButton", background=[("active", "#D6E5F3")])
        style.configure("Danger.TButton", background="#FCE8E6", foreground=FGV_DANGER, borderwidth=0, padding=(12, 8), font=("DejaVu Sans", 10, "bold"))
        style.map("Danger.TButton", background=[("active", "#F8D1CD")])
        style.configure("TEntry", padding=7, fieldbackground="#FFFFFF", bordercolor=FGV_BORDER, lightcolor=FGV_BORDER, darkcolor=FGV_BORDER)
        style.configure("TCombobox", padding=7, fieldbackground="#FFFFFF", bordercolor=FGV_BORDER, lightcolor=FGV_BORDER, darkcolor=FGV_BORDER)
        style.configure("TCheckbutton", background=FGV_CARD, foreground=FGV_TEXT, font=default_font)
        style.map("TCheckbutton", background=[("active", FGV_CARD)])
        style.configure("TRadiobutton", background=FGV_CARD, foreground=FGV_TEXT, font=default_font)
        style.map("TRadiobutton", background=[("active", FGV_CARD)])
        style.configure("TSeparator", background=FGV_BORDER)
        style.configure("Status.TLabel", background=FGV_CARD, foreground=FGV_MUTED, font=("DejaVu Sans", 9))

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        shell = ttk.Frame(self.root, style="App.TFrame")
        shell.grid(row=0, column=0, sticky="nsew")
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(1, weight=1)

        sidebar = ttk.Frame(shell, style="Sidebar.TFrame", width=285)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

        main = ttk.Frame(shell, style="App.TFrame")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(2, weight=1)
        main.grid_columnconfigure(0, weight=1)
        self._build_header(main)
        self._build_progress(main)

        self.content_stack = ttk.Frame(main, style="App.TFrame")
        self.content_stack.grid(row=2, column=0, sticky="nsew", padx=(26, 26), pady=(8, 0))
        self.content_stack.grid_rowconfigure(0, weight=1)
        self.content_stack.grid_columnconfigure(0, weight=1)
        self._build_step_pages()
        self._build_footer(main)

    def _build_sidebar(self, sidebar: ttk.Frame) -> None:
        top = ttk.Frame(sidebar, style="Sidebar.TFrame")
        top.pack(fill=tk.X, padx=18, pady=(24, 20))
        ttk.Label(top, text="Academic Pipeline", style="SidebarTitle.TLabel").pack(anchor="w")
        ttk.Label(top, text="Atividades acadêmicas FGV", style="SidebarText.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=18, pady=(0, 14))
        for step in STEPS:
            button = ttk.Button(sidebar, text=step.label, style="Step.TButton", command=lambda key=step.key: self._render_step(key))
            button.pack(fill=tk.X, padx=12, pady=3)
            self.step_buttons[step.key] = button
        spacer = ttk.Frame(sidebar, style="Sidebar.TFrame")
        spacer.pack(fill=tk.BOTH, expand=True)
        lower = ttk.Frame(sidebar, style="Sidebar.TFrame")
        lower.pack(fill=tk.X, padx=18, pady=(10, 20))
        ttk.Label(lower, text="Atalhos", style="SidebarTitle.TLabel", font=("DejaVu Sans", 10, "bold")).pack(anchor="w")
        ttk.Label(lower, text="Tab completa caminhos\nCtrl+S salva o TOML\nCtrl+Enter gera", style="SidebarText.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Button(lower, text="Abrir pasta do bundle", style="Secondary.TButton", command=lambda: self._open_path(ROOT)).pack(fill=tk.X, pady=(14, 0))

    def _build_header(self, main: ttk.Frame) -> None:
        header = ttk.Frame(main, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        logo_frame = ttk.Frame(header, style="Header.TFrame")
        logo_frame.grid(row=0, column=0, padx=(26, 14), pady=14, sticky="w")
        logo_path = APP / "institutions" / "fgv" / "assets" / "fgv.png"
        if logo_path.exists():
            try:
                image = tk.PhotoImage(file=str(logo_path))
                # Mantém boa proporção sem introduzir Pillow como dependência.
                self._logo = image.subsample(8, 8)
                tk.Label(logo_frame, image=self._logo, bg=FGV_CARD, borderwidth=0).pack(anchor="w")
            except tk.TclError:
                ttk.Label(logo_frame, text="FGV", style="Heading.TLabel").pack(anchor="w")
        else:
            ttk.Label(logo_frame, text="FGV", style="Heading.TLabel").pack(anchor="w")
        texts = ttk.Frame(header, style="Header.TFrame")
        texts.grid(row=0, column=1, sticky="w", pady=12)
        tk.Label(texts, text="Central de atividades acadêmicas", bg=FGV_CARD, fg=FGV_NAVY, font=("DejaVu Sans", 17, "bold")).pack(anchor="w")
        tk.Label(texts, text="Crie, valide e gere sua atividade em um fluxo guiado, sem depender de perguntas no terminal.", bg=FGV_CARD, fg=FGV_MUTED, font=("DejaVu Sans", 10)).pack(anchor="w", pady=(4, 0))
        right = ttk.Frame(header, style="Header.TFrame")
        right.grid(row=0, column=2, padx=(14, 26), pady=12, sticky="e")
        ttk.Button(right, text="Salvar TOML", style="Secondary.TButton", command=self.save_config).pack(anchor="e")
        ttk.Label(right, textvariable=self.status, style="Status.TLabel", wraplength=280, justify="right").pack(anchor="e", pady=(7, 0))

    def _build_progress(self, main: ttk.Frame) -> None:
        bar = ttk.Frame(main, style="App.TFrame")
        bar.grid(row=1, column=0, sticky="ew", padx=26, pady=(18, 2))
        bar.grid_columnconfigure(0, weight=1)
        self.progress_title = ttk.Label(bar, text="", style="Heading.TLabel")
        self.progress_title.grid(row=0, column=0, sticky="w")
        self.progress_hint = ttk.Label(bar, text="", style="Subheading.TLabel")
        self.progress_hint.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.progress_indicator = ttk.Label(bar, text="", style="Subheading.TLabel")
        self.progress_indicator.grid(row=0, column=1, rowspan=2, sticky="e")

    def _build_footer(self, main: ttk.Frame) -> None:
        footer = ttk.Frame(main, style="App.TFrame")
        footer.grid(row=3, column=0, sticky="ew", padx=26, pady=16)
        footer.grid_columnconfigure(1, weight=1)
        self.back_button = ttk.Button(footer, text="← Voltar", style="Secondary.TButton", command=self.previous_step)
        self.back_button.grid(row=0, column=0, sticky="w")
        self.footer_status = ttk.Label(footer, text="", style="Subheading.TLabel")
        self.footer_status.grid(row=0, column=1, sticky="w", padx=16)
        self.next_button = ttk.Button(footer, text="Avançar →", style="Primary.TButton", command=self.next_step)
        self.next_button.grid(row=0, column=2, sticky="e")
        self.root.bind_all("<Control-s>", lambda _e: self.save_config())
        self.root.bind_all("<Control-Return>", lambda _e: self.generate_activity())

    # ------------------------------------------------------------------
    # Componentes de formulário
    # ------------------------------------------------------------------

    def _build_step_pages(self) -> None:
        builders: dict[str, Callable[[ttk.Frame], None]] = {
            "project": self._page_project,
            "activity": self._page_activity,
            "inputs": self._page_inputs,
            "outputs": self._page_outputs,
            "review": self._page_review,
        }
        for step in STEPS:
            frame = ScrollableFrame(self.content_stack, style="App.TFrame")
            frame.grid(row=0, column=0, sticky="nsew")
            self.step_frames[step.key] = frame
            builders[step.key](frame.content)

    def _card(self, parent: tk.Misc, *, title: str, subtitle: str = "") -> ttk.Frame:
        outer = ttk.Frame(parent, style="Card.TFrame", padding=(22, 18))
        outer.pack(fill=tk.X, pady=(0, 14), padx=2)
        ttk.Label(outer, text=title, style="Card.TLabel", font=("DejaVu Sans", 12, "bold")).pack(anchor="w")
        if subtitle:
            ttk.Label(outer, text=subtitle, style="CardMuted.TLabel", wraplength=920, justify="left").pack(anchor="w", pady=(4, 14))
        return outer

    def _field(self, parent: tk.Misc, label: str, variable: tk.StringVar, *, help_text: str = "", path_kind: str | None = None, width: int = 68) -> tk.Widget:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, pady=6)
        ttk.Label(row, text=label, style="Card.TLabel", width=29, anchor="w").pack(side=tk.LEFT, anchor="n", pady=(4, 0))
        if path_kind:
            entry = AutocompletePathEntry(row, variable, width=width)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if path_kind in {"file", "zip"}:
                ttk.Button(row, text="Procurar arquivo", style="Secondary.TButton", command=lambda: self._choose_file(variable, filetypes=self._filetypes_for(path_kind))).pack(side=tk.LEFT, padx=(8, 0))
            elif path_kind == "dir":
                ttk.Button(row, text="Procurar pasta", style="Secondary.TButton", command=lambda: self._choose_dir(variable)).pack(side=tk.LEFT, padx=(8, 0))
            elif path_kind == "any":
                ttk.Button(row, text="Arquivo", style="Secondary.TButton", command=lambda: self._choose_file(variable)).pack(side=tk.LEFT, padx=(8, 0))
                ttk.Button(row, text="Pasta", style="Secondary.TButton", command=lambda: self._choose_dir(variable)).pack(side=tk.LEFT, padx=(5, 0))
            widget: tk.Widget = entry
        else:
            entry = ttk.Entry(row, textvariable=variable, width=width)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            widget = entry
        if help_text:
            ttk.Label(parent, text=help_text, style="CardMuted.TLabel", wraplength=840, justify="left").pack(anchor="w", padx=(233, 0), pady=(0, 5))
        return widget

    def _text_field(self, parent: tk.Misc, label: str, *, height: int = 5, help_text: str = "") -> tk.Text:
        ttk.Label(parent, text=label, style="Card.TLabel").pack(anchor="w", pady=(8, 4))
        text = tk.Text(parent, height=height, wrap=tk.WORD, bg="#FFFFFF", fg=FGV_TEXT, insertbackground=FGV_TEXT, relief=tk.FLAT, highlightthickness=1, highlightbackground=FGV_BORDER, highlightcolor=FGV_CYAN, font=("DejaVu Sans", 10), padx=9, pady=8)
        text.pack(fill=tk.X)
        if help_text:
            ttk.Label(parent, text=help_text, style="CardMuted.TLabel", wraplength=900, justify="left").pack(anchor="w", pady=(5, 0))
        return text

    def _choose_file(self, variable: tk.StringVar, *, filetypes: list[tuple[str, str]] | None = None) -> None:
        current = _existing_path(variable.get())
        initialdir = current if current and current.is_dir() else (current.parent if current else Path.home())
        selected = filedialog.askopenfilename(parent=self.root, initialdir=str(initialdir), filetypes=filetypes or [("Todos os arquivos", "*.*")])
        if selected:
            variable.set(selected)

    def _choose_dir(self, variable: tk.StringVar) -> None:
        current = _existing_path(variable.get())
        initialdir = current if current and current.is_dir() else (current.parent if current else Path.home())
        selected = filedialog.askdirectory(parent=self.root, initialdir=str(initialdir))
        if selected:
            variable.set(selected)

    @staticmethod
    def _filetypes_for(kind: str) -> list[tuple[str, str]]:
        if kind == "zip":
            return [("Arquivos ZIP", "*.zip"), ("Todos os arquivos", "*.*")]
        return [("Documentos e arquivos", "*.txt *.md *.org *.pdf *.docx *.zip"), ("Todos os arquivos", "*.*")]

    # ------------------------------------------------------------------
    # Páginas
    # ------------------------------------------------------------------

    def _page_project(self, parent: ttk.Frame) -> None:
        card = self._card(parent, title="Projeto e ficha acadêmica", subtitle="Defina o identificador do projeto e os dados que irão compor a Ficha Técnica FGV. O TOML será salvo dentro de app_bundle/projetos/<nome-do-projeto>/.")
        self._field(card, "Nome do projeto", self.project_name, help_text="Use um nome curto, por exemplo: atividade_3. O sistema cria uma versão segura para diretórios e arquivos.")
        self._field(card, "Nome do arquivo TOML", self.toml_name, help_text="O padrão é atividade_config.toml. Um TOML organiza os insumos, as saídas e as regras da atividade.")
        ttk.Separator(card).pack(fill=tk.X, pady=14)
        self._field(card, "Título da atividade", self.title)
        self._field(card, "Aluno", self.author)
        self._field(card, "Curso", self.course)
        self._field(card, "Turma", self.class_group)
        self._field(card, "Pólo/cidade", self.city)
        self._field(card, "Disciplina", self.discipline)
        self._field(card, "Professor", self.professor)
        self._field(card, "Data/ano", self.academic_year, help_text="Pode manter apenas o ano ou utilizar uma data mais específica, conforme a orientação da disciplina.")

    def _page_activity(self, parent: ttk.Frame) -> None:
        card = self._card(parent, title="Como deseja fornecer os dados da atividade?", subtitle="Escolha a fonte para tema, pergunta orientadora e enunciado. Em todos os modos, as palavras-chave são inferidas automaticamente pela IA durante a produção do documento.")
        radios = ttk.Frame(card, style="Card.TFrame")
        radios.pack(fill=tk.X, pady=(4, 12))
        options = (
            ("ia", "IA infere os dados", "A IA utiliza corpus, orientações e prompt específico para inferir tema, recorte, pergunta orientadora, enunciado operacional e palavras-chave."),
            ("arquivo", "Usar arquivo único", "Você informa um arquivo, pasta ou ZIP com tema, pergunta orientadora e enunciado/orientação."),
            ("manual", "Preencher manualmente", "Você digita tema, pergunta orientadora e enunciado diretamente nesta tela."),
        )
        for value, label, hint in options:
            line = ttk.Frame(radios, style="Card.TFrame")
            line.pack(fill=tk.X, pady=5)
            ttk.Radiobutton(line, text=label, variable=self.data_mode, value=value, command=self._refresh_data_mode).pack(side=tk.LEFT)
            ttk.Label(line, text=hint, style="CardMuted.TLabel", wraplength=720, justify="left").pack(side=tk.LEFT, padx=(16, 0))

        self.data_dynamic = ttk.Frame(card, style="Card.TFrame")
        self.data_dynamic.pack(fill=tk.X, pady=(6, 0))
        self._refresh_data_mode()

        keywords = self._card(parent, title="Palavras-chave automáticas", subtitle="Não há campo para preenchimento manual: a IA infere entre quatro e seis palavras-chave acadêmicas a partir do tema, pergunta orientadora, orientações e corpus local. Isso reduz inconsistências entre instruções e metadados.")
        ttk.Label(keywords, text="✓ geração automática ativada", style="Card.TLabel", foreground=FGV_SUCCESS, font=("DejaVu Sans", 10, "bold")).pack(anchor="w")

    def _refresh_data_mode(self) -> None:
        self._capture_manual_instruction()
        for child in self.data_dynamic.winfo_children():
            child.destroy()
        mode = self.data_mode.get()
        if mode == "ia":
            ttk.Label(self.data_dynamic, text="A IA inferirá os dados da atividade no momento da geração. Para obter um resultado fiel, informe um corpus local e, quando houver, o enunciado ou roteiro da aula na próxima etapa.", style="CardMuted.TLabel", wraplength=860, justify="left").pack(anchor="w", pady=8)
        elif mode == "arquivo":
            self._field(self.data_dynamic, "Arquivo/pasta/ZIP", self.activity_data_path, path_kind="any", help_text="O conteúdo pode conter campos como Tema da atividade, Pergunta orientadora e Enunciado/orientação do professor. O arquivo é incorporado às orientações do pipeline.")
        else:
            self._field(self.data_dynamic, "Tema da atividade", self.manual_topic)
            self._field(self.data_dynamic, "Pergunta orientadora", self.manual_question)
            self.manual_instruction_text = self._text_field(
                self.data_dynamic,
                "Enunciado/orientação do professor",
                height=5,
                help_text="Campo opcional. Use-o para a orientação principal do professor. Para um conjunto de arquivos ou uma rubrica extensa, prefira a opção Arquivo na etapa Corpus e orientações.",
            )
            if self.manual_instruction.get():
                self.manual_instruction_text.insert("1.0", self.manual_instruction.get())
            self.manual_instruction_text.bind("<FocusOut>", lambda _e: self._capture_manual_instruction())

    def _page_inputs(self, parent: ttk.Frame) -> None:
        corpus = self._card(parent, title="Corpus local", subtitle="A atividade será construída apenas a partir dos textos indicados. Você pode usar um ZIP ou uma pasta com PDFs, DOCX, TXT, MD e ORG.")
        mode_line = ttk.Frame(corpus, style="Card.TFrame")
        mode_line.pack(fill=tk.X, pady=(2, 8))
        ttk.Label(mode_line, text="Tipo de entrada", style="Card.TLabel", width=29, anchor="w").pack(side=tk.LEFT)
        combo = ttk.Combobox(mode_line, textvariable=self.corpus_mode, values=["zip", "dir"], width=12, state="readonly")
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_corpus_mode())
        self.corpus_dynamic = ttk.Frame(corpus, style="Card.TFrame")
        self.corpus_dynamic.pack(fill=tk.X)
        self._refresh_corpus_mode()

        orientations = self._card(parent, title="Orientações gerais da aula", subtitle="Inclua o enunciado completo, roteiro, rubrica ou critérios de avaliação. É uma camada separada dos dados da atividade e ajuda a IA a respeitar a entrega solicitada pelo professor.")
        orientation_mode_line = ttk.Frame(orientations, style="Card.TFrame")
        orientation_mode_line.pack(fill=tk.X, pady=(2, 8))
        ttk.Label(orientation_mode_line, text="Forma de fornecimento", style="Card.TLabel", width=29, anchor="w").pack(side=tk.LEFT)
        combo_or = ttk.Combobox(orientation_mode_line, textvariable=self.orientation_mode, values=["nenhuma", "arquivo", "manual"], width=16, state="readonly")
        combo_or.pack(side=tk.LEFT)
        combo_or.bind("<<ComboboxSelected>>", lambda _e: self._refresh_orientation_mode())
        self.orientation_dynamic = ttk.Frame(orientations, style="Card.TFrame")
        self.orientation_dynamic.pack(fill=tk.X)
        self._refresh_orientation_mode()

        prompt = self._card(parent, title="Prompt específico do projeto", subtitle="Opcional. Use um arquivo TXT/MD/ORG com regras de estrutura, extensão, tom e restrições da atividade. O pipeline já possui um prompt padrão para atividades; este campo adiciona diretrizes específicas.")
        self._field(prompt, "Arquivo de prompt", self.prompt_path, path_kind="file", help_text="Exemplo: prompt_base_atividade_aula_4.txt. Deixe em branco se não houver instrução adicional.")
        self._field(prompt, "Manifesto DOI", self.doi_manifest_path, path_kind="file", help_text="Opcional. Informe apenas se houver um doi_manifest.csv revisado para o corpus.")

    def _refresh_corpus_mode(self) -> None:
        for child in self.corpus_dynamic.winfo_children():
            child.destroy()
        if self.corpus_mode.get() == "zip":
            self._field(self.corpus_dynamic, "Arquivo ZIP do corpus", self.corpus_path, path_kind="zip", help_text="Selecione o ZIP com os textos-base. Tab oferece autocompletar de caminhos; o botão abre o seletor de arquivos.")
        else:
            self._field(self.corpus_dynamic, "Pasta do corpus", self.corpus_path, path_kind="dir", help_text="A pasta será analisada recursivamente. Somente os tipos PDF, DOCX, TXT, MD e ORG são usados no corpus.")

    def _refresh_orientation_mode(self) -> None:
        self._capture_orientation_inline()
        for child in self.orientation_dynamic.winfo_children():
            child.destroy()
        mode = self.orientation_mode.get()
        if mode == "nenhuma":
            ttk.Label(self.orientation_dynamic, text="Nenhuma orientação adicional será anexada. Isso é adequado quando o próprio arquivo de dados da atividade ou o prompt específico já contém todas as instruções relevantes.", style="CardMuted.TLabel", wraplength=860, justify="left").pack(anchor="w", pady=8)
        elif mode == "arquivo":
            self._field(self.orientation_dynamic, "Arquivo/pasta/ZIP", self.orientation_path, path_kind="any", help_text="Indique o enunciado, roteiro, rubrica ou um pacote de orientações da aula.")
        else:
            self.orientation_text = self._text_field(self.orientation_dynamic, "Orientações gerais digitadas", height=7, help_text="Cole aqui o enunciado ou a rubrica. Este texto será gravado no TOML como orientação inline.")
            if self.orientation_inline:
                self.orientation_text.insert("1.0", self.orientation_inline)
            self.orientation_text.bind("<FocusOut>", lambda _e: self._capture_orientation_inline())

    def _capture_orientation_inline(self) -> None:
        widget = getattr(self, "orientation_text", None)
        if widget is not None and widget.winfo_exists():
            self.orientation_inline = widget.get("1.0", "end-1c")

    def _capture_manual_instruction(self) -> None:
        widget = getattr(self, "manual_instruction_text", None)
        if widget is not None and widget.winfo_exists():
            self.manual_instruction.set(widget.get("1.0", "end-1c"))

    def _capture_multiline_fields(self) -> None:
        self._capture_orientation_inline()
        self._capture_manual_instruction()

    def _page_outputs(self, parent: ttk.Frame) -> None:
        exports = self._card(parent, title="Formatos de saída", subtitle="A geração cria um document.json rastreável e, conforme a seleção abaixo, os formatos finais para entrega e edição.")
        checks = ttk.Frame(exports, style="Card.TFrame")
        checks.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(checks, text="Gerar ORG editável", variable=self.export_org).grid(row=0, column=0, sticky="w", padx=(0, 26), pady=5)
        ttk.Checkbutton(checks, text="Gerar PDF", variable=self.export_pdf).grid(row=0, column=1, sticky="w", padx=(0, 26), pady=5)
        ttk.Checkbutton(checks, text="Gerar DOCX", variable=self.export_docx).grid(row=0, column=2, sticky="w", pady=5)
        ttk.Separator(exports).pack(fill=tk.X, pady=14)
        ttk.Checkbutton(exports, text="Tentar extrair DOI dos PDFs", variable=self.extract_doi).pack(anchor="w", pady=4)
        ttk.Checkbutton(exports, text="Buscar/enriquecer metadados por DOI quando disponível", variable=self.enrich_metadata).pack(anchor="w", pady=4)
        ttk.Checkbutton(exports, text="Gerar relatório de conformidade institucional", variable=self.conformity).pack(anchor="w", pady=4)
        ttk.Checkbutton(exports, text="Gerar relatório de qualidade", variable=self.quality).pack(anchor="w", pady=4)

        technical = self._card(parent, title="Pasta de saída e renderização", subtitle="Sem sobrescrita, os resultados ficam em app_bundle/output/documento/<projeto>/. Você pode escolher uma pasta externa apenas para esta execução.")
        self._field(technical, "Pasta de saída opcional", self.output_override, path_kind="dir", help_text="Quando preenchida, o pipeline usa essa pasta como base. O projeto ainda recebe uma subpasta própria para evitar misturar saídas.")
        engine_line = ttk.Frame(technical, style="Card.TFrame")
        engine_line.pack(fill=tk.X, pady=6)
        ttk.Label(engine_line, text="Engine do PDF", style="Card.TLabel", width=29, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(engine_line, textvariable=self.pdf_engine, values=["lualatex", "xelatex", "pdflatex"], state="readonly", width=16).pack(side=tk.LEFT)

    def _page_review(self, parent: ttk.Frame) -> None:
        card = self._card(parent, title="Revisar, validar e gerar", subtitle="Nesta etapa, o sistema monta o TOML, valida a configuração com o pipeline e então executa a geração completa. Nada precisa ser digitado no terminal.")
        self.review_text = tk.Text(card, height=18, wrap=tk.WORD, bg="#F9FBFD", fg=FGV_TEXT, relief=tk.FLAT, highlightthickness=1, highlightbackground=FGV_BORDER, font=("DejaVu Sans Mono", 9), padx=10, pady=10)
        self.review_text.pack(fill=tk.BOTH, expand=True)
        self.review_text.configure(state=tk.DISABLED)

        action_bar = ttk.Frame(card, style="Card.TFrame")
        action_bar.pack(fill=tk.X, pady=(15, 0))
        ttk.Button(action_bar, text="Salvar TOML", style="Secondary.TButton", command=self.save_config).pack(side=tk.LEFT)
        ttk.Button(action_bar, text="Validar", style="Secondary.TButton", command=self.validate_config).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(action_bar, text="Gerar atividade", style="Primary.TButton", command=self.generate_activity).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(action_bar, text="Abrir saída", style="Secondary.TButton", command=lambda: self._open_path(self.output_dir())).pack(side=tk.RIGHT)

        log_card = self._card(parent, title="Log de execução", subtitle="Acompanhamento em tempo real da validação e geração. O texto também pode ser copiado para diagnóstico técnico quando necessário.")
        self.log_text = scrolledtext.ScrolledText(log_card, height=13, wrap=tk.WORD, bg="#0B1E33", fg="#E8F2FA", insertbackground="#FFFFFF", relief=tk.FLAT, font=("DejaVu Sans Mono", 9), padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Navegação e validação de formulários
    # ------------------------------------------------------------------

    def _step_index(self, key: str) -> int:
        return next(index for index, step in enumerate(STEPS) if step.key == key)

    def _render_step(self, key: str) -> None:
        self._capture_multiline_fields()
        index = self._step_index(key)
        if index > self.current_step and not self._validate_through(index - 1, show_message=True):
            return
        self.current_step = index
        frame = self.step_frames[key]
        frame.tkraise()
        frame.to_top()
        step = STEPS[index]
        self.progress_title.configure(text=step.label)
        self.progress_hint.configure(text=step.hint)
        self.progress_indicator.configure(text=f"Etapa {index + 1} de {len(STEPS)}")
        self.footer_status.configure(text="Use Tab para completar caminhos e os botões Procurar para selecionar arquivos e pastas.")
        for item in STEPS:
            self.step_buttons[item.key].configure(style="StepActive.TButton" if item.key == key else "Step.TButton")
        self.back_button.configure(state=tk.NORMAL if index > 0 and not self._busy else tk.DISABLED)
        if index == len(STEPS) - 1:
            self.next_button.configure(text="Gerar atividade", command=self.generate_activity, state=tk.NORMAL if not self._busy else tk.DISABLED)
            self._refresh_review()
        else:
            self.next_button.configure(text="Avançar →", command=self.next_step, state=tk.NORMAL if not self._busy else tk.DISABLED)

    def previous_step(self) -> None:
        if self.current_step > 0:
            self._render_step(STEPS[self.current_step - 1].key)

    def next_step(self) -> None:
        if not self._validate_step(self.current_step, show_message=True):
            return
        if self.current_step < len(STEPS) - 1:
            self._render_step(STEPS[self.current_step + 1].key)

    def _validate_through(self, final_index: int, *, show_message: bool) -> bool:
        for index in range(final_index + 1):
            if not self._validate_step(index, show_message=show_message):
                return False
        return True

    def _validate_step(self, index: int, *, show_message: bool) -> bool:
        self._capture_multiline_fields()
        step = STEPS[index].key
        errors: list[str] = []
        if step == "project":
            if not self.project_name.get().strip():
                errors.append("Informe o nome do projeto.")
            if not self.title.get().strip():
                errors.append("Informe o título da atividade.")
            if not self.author.get().strip():
                errors.append("Informe o nome do aluno.")
        elif step == "activity":
            if self.data_mode.get() == "arquivo":
                path = _existing_path(self.activity_data_path.get())
                if path is None:
                    errors.append("Selecione um arquivo, pasta ou ZIP com os dados da atividade.")
            elif self.data_mode.get() == "manual":
                if not self.manual_topic.get().strip():
                    errors.append("Informe o tema da atividade.")
                if not self.manual_question.get().strip():
                    errors.append("Informe a pergunta orientadora da atividade.")
        elif step == "inputs":
            corpus = _existing_path(self.corpus_path.get())
            if corpus is None:
                errors.append("Informe o corpus local: um arquivo ZIP ou uma pasta com os textos-base.")
            elif self.corpus_mode.get() == "zip" and not corpus.is_file():
                errors.append("O tipo de corpus está como ZIP, mas o caminho informado não é um arquivo.")
            elif self.corpus_mode.get() == "dir" and not corpus.is_dir():
                errors.append("O tipo de corpus está como pasta, mas o caminho informado não é um diretório.")
            if self.orientation_mode.get() == "arquivo" and _existing_path(self.orientation_path.get()) is None:
                errors.append("Selecione o arquivo/pasta/ZIP das orientações gerais ou escolha outro modo.")
            if self.prompt_path.get().strip() and _existing_path(self.prompt_path.get()) is None:
                errors.append("O arquivo de prompt específico informado não foi localizado.")
            if self.doi_manifest_path.get().strip() and _existing_path(self.doi_manifest_path.get()) is None:
                errors.append("O manifesto DOI informado não foi localizado.")
        elif step == "outputs":
            if not any([self.export_org.get(), self.export_pdf.get(), self.export_docx.get()]):
                errors.append("Selecione ao menos um formato de saída: ORG, PDF ou DOCX.")
            override = self.output_override.get().strip()
            if override:
                try:
                    Path(override).expanduser().mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    errors.append(f"Não foi possível criar/acessar a pasta de saída: {exc}")
        if errors:
            self.status.set("Há informações pendentes.")
            if show_message:
                messagebox.showwarning("Revise esta etapa", "\n".join(f"• {item}" for item in errors), parent=self.root)
            return False
        return True

    # ------------------------------------------------------------------
    # Revisão, execução e logs
    # ------------------------------------------------------------------

    def _summary_text(self) -> str:
        data_mode_label = {"ia": "IA inferirá os dados", "arquivo": "arquivo com dados", "manual": "preenchimento manual"}.get(self.data_mode.get(), self.data_mode.get())
        orientation_label = {"nenhuma": "nenhuma", "arquivo": "arquivo/pasta/ZIP", "manual": "texto digitado"}.get(self.orientation_mode.get(), self.orientation_mode.get())
        outputs = ", ".join(name for enabled, name in [(self.export_org.get(), "ORG"), (self.export_pdf.get(), "PDF"), (self.export_docx.get(), "DOCX")] if enabled)
        return "\n".join([
            "REVISÃO DA ATIVIDADE",
            "=" * 72,
            f"Projeto: {self.project_name.get().strip() or '—'}",
            f"TOML: {self.config_path()}",
            f"Título: {self.title.get().strip() or '—'}",
            f"Disciplina: {self.discipline.get().strip() or '—'}",
            f"Professor: {self.professor.get().strip() or '—'}",
            "",
            "DADOS DA ATIVIDADE",
            f"Fonte: {data_mode_label}",
            f"Dados por IA: {'sim' if self.data_mode.get() == 'ia' else 'não'}",
            f"Palavras-chave: inferência automática pela IA",
            "",
            "INSUMOS",
            f"Corpus ({self.corpus_mode.get()}): {self.corpus_path.get().strip() or '—'}",
            f"Orientações gerais: {orientation_label}",
            f"Prompt específico: {self.prompt_path.get().strip() or 'não informado'}",
            f"Manifesto DOI: {self.doi_manifest_path.get().strip() or 'não informado'}",
            "",
            "SAÍDAS",
            f"Formatos: {outputs or '—'}",
            f"Pasta de saída: {self.output_dir()}",
            f"Conformidade: {'ativada' if self.conformity.get() else 'desativada'}",
            f"Qualidade: {'ativada' if self.quality.get() else 'desativada'}",
            "",
            "AÇÃO RECOMENDADA",
            "1. Salve o TOML.  2. Valide.  3. Gere a atividade.",
        ])

    def _refresh_review(self) -> None:
        if not hasattr(self, "review_text"):
            return
        self.review_text.configure(state=tk.NORMAL)
        self.review_text.delete("1.0", tk.END)
        self.review_text.insert("1.0", self._summary_text())
        self.review_text.configure(state=tk.DISABLED)

    def save_config(self) -> bool:
        self._capture_multiline_fields()
        if not self._validate_through(3, show_message=True):
            return False
        try:
            config_path = self._write_toml()
        except Exception as exc:
            self.status.set("Falha ao salvar o TOML.")
            messagebox.showerror("Não foi possível salvar", str(exc), parent=self.root)
            return False
        self.status.set(f"TOML salvo: {config_path.name}")
        self._append_log(f"[CONFIG] TOML salvo em: {config_path}\n")
        self._refresh_review()
        return True

    def validate_config(self) -> None:
        if self._busy:
            return
        if not self.save_config():
            return
        config_path = self.config_path()
        self._run_command([sys.executable, str(PIPELINE), "--config", str(config_path), "--check-config"], label="Validação da configuração")

    def generate_activity(self) -> None:
        if self._busy:
            return
        if not self.save_config():
            return
        if not messagebox.askyesno("Confirmar geração", "A geração poderá chamar a API configurada e produzir os arquivos finais. Deseja continuar?", parent=self.root):
            return
        config_path = self.config_path()
        command = [sys.executable, str(PIPELINE), "--config", str(config_path)]
        if self.output_override.get().strip():
            command.extend(["--output-dir", str(Path(self.output_override.get()).expanduser())])
        self._run_command(command, label="Geração completa da atividade")

    def _run_command(self, command: list[str], *, label: str) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self.status.set(f"Em execução: {label}")
        self._append_log("\n" + "=" * 78 + "\n")
        self._append_log(f"[INÍCIO] {label}\n")
        self._append_log("[COMANDO] " + " ".join(command) + "\n\n")

        def worker() -> None:
            try:
                process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                self._active_process = process
                assert process.stdout is not None
                for line in process.stdout:
                    self._worker_queue.put(("line", line))
                code = process.wait()
                self._worker_queue.put(("done", (label, code)))
            except Exception as exc:  # pragma: no cover - depende de subprocesso externo
                self._worker_queue.put(("error", (label, str(exc))))

        threading.Thread(target=worker, name="academic-pipeline-gui-worker", daemon=True).start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self._worker_queue.get_nowait()
                if kind == "line":
                    self._append_log(str(payload))
                elif kind == "done":
                    label, code = payload
                    self._set_busy(False)
                    if code == 0:
                        self.status.set(f"Concluído: {label}")
                        self._append_log(f"\n[CONCLUÍDO] {label}.\n")
                        self._refresh_review()
                        messagebox.showinfo("Concluído", f"{label} finalizada com sucesso.\n\nSaída: {self.output_dir()}", parent=self.root)
                    else:
                        self.status.set(f"Falha: {label}")
                        self._append_log(f"\n[FALHA] {label}. Código de saída: {code}.\n")
                        messagebox.showerror("Falha na execução", f"{label} terminou com código {code}. Consulte o log nesta tela.", parent=self.root)
                elif kind == "error":
                    label, message = payload
                    self._set_busy(False)
                    self.status.set(f"Falha: {label}")
                    self._append_log(f"\n[ERRO] {message}\n")
                    messagebox.showerror("Erro de execução", message, parent=self.root)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_worker_queue)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self.step_buttons.values():
            button.configure(state=state)
        self.back_button.configure(state=state if self.current_step > 0 else tk.DISABLED)
        self.next_button.configure(state=state)
        if not busy:
            self._render_step(STEPS[self.current_step].key)

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Utilidades de sistema
    # ------------------------------------------------------------------

    def _open_path(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True) if not path.suffix else None
        except OSError:
            pass
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":  # pragma: no cover - plataforma específica
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                raise RuntimeError("Sistema operacional não suportado para abrir arquivos automaticamente.")
        except Exception as exc:
            messagebox.showwarning("Não foi possível abrir", f"Abra manualmente este caminho:\n{path}\n\nDetalhe: {exc}", parent=self.root)


def run_gui() -> int:
    """Inicializa a interface gráfica, com diagnóstico claro em falta de Tk."""
    if tk is None or ttk is None:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "A interface gráfica precisa do módulo tkinter. No Debian/Ubuntu, instale os pacotes do Tk e garanta "
            "que o Python usado pelo Pipenv tenha sido compilado com suporte a Tk.\n\n"
            "Teste: pipenv run python -c \"import tkinter; print(tkinter.TkVersion)\"\n"
            "Detalhe técnico: " + repr(_TK_IMPORT_ERROR)
        )
    root = tk.Tk()
    AcademicPipelineGUI(root)
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interface gráfica FGV do Academic Pipeline")
    parser.parse_args(argv)
    return run_gui()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

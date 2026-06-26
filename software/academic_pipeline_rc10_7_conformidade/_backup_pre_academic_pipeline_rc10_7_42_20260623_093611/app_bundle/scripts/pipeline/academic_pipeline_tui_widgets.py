#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Componentes visuais reutilizáveis da TUI do Academic Pipeline.

A interface é intencionalmente baseada em :mod:`prompt_toolkit`, seguindo o
mesmo padrão operacional da Administração UAC: tela inteira, listas
selecionáveis, atalhos explícitos e campos de texto com suporte a conclusão de
caminhos por ``Tab``.  A paleta foi derivada da marca instalada em
``institutions/fgv/assets/fgv.png``:

- azul FGV escuro: ``#003E7E``;
- azul FGV claro: ``#0595D5``;
- azul claro de foco: ``#BDEBFA``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence


class TUIUnavailable(RuntimeError):
    """Indica que a dependência opcional da TUI ainda não foi instalada."""


def _imports() -> dict[str, Any]:
    try:
        from prompt_toolkit import Application
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import PathCompleter
        from prompt_toolkit.filters import has_focus
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Dimension, Layout
        from prompt_toolkit.layout.containers import HSplit
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Box, Button, CheckboxList, Dialog, Label, RadioList, TextArea
    except ImportError as exc:  # pragma: no cover - depende do ambiente do usuário
        raise TUIUnavailable(
            "A TUI visual requer prompt-toolkit. Instale uma única vez com: "
            "pipenv install prompt-toolkit"
        ) from exc

    # Cores do logotipo FGV distribuído no bundle: #003E7E e #0595D5.
    style = Style.from_dict(
        {
            "dialog": "bg:#003E7E #ffffff",
            "dialog.body": "bg:#002B55 #ffffff",
            "dialog frame.label": "bg:#0595D5 #ffffff bold",
            "dialog frame-label": "bg:#0595D5 #ffffff bold",
            "dialog shadow": "bg:#001A33",
            "button": "bg:#005FA9 #ffffff",
            "button.focused": "bg:#BDEBFA #003E7E bold",
            "radio": "#80D5F1",
            "radio-selected": "fg:#BDEBFA bold",
            "checkbox": "#80D5F1",
            "checkbox-selected": "fg:#BDEBFA bold",
            "bottom-toolbar": "bg:#0595D5 fg:#ffffff",
            "text-area": "bg:#001F3F fg:#ffffff",
            "text-area.focused": "bg:#002E5D fg:#ffffff",
            "label.emphasis": "fg:#BDEBFA bold",
            "status.ok": "fg:#80D5F1 bold",
            "status.warn": "fg:#FFE082 bold",
            "status.error": "fg:#ffffff bg:#B00020 bold",
        }
    )
    return locals()


def _history_path() -> Path:
    root = Path.home() / ".cache" / "academic_pipeline"
    root.mkdir(parents=True, exist_ok=True)
    return root / "tui_input_history.txt"


def _title(title: str) -> str:
    clean = str(title or "Academic Pipeline").strip()
    return clean if clean.startswith("FGV") else f"FGV | {clean}"


def message(title: str, text: str, *, width: int = 112) -> None:
    """Mostra texto rolável em tela inteira."""
    ui = _imports()
    app: Any

    def finish() -> None:
        app.exit(result=None)

    button = ui["Button"](text="[Enter/Esc] Fechar", handler=finish)
    kb = ui["KeyBindings"]()

    @kb.add("enter")
    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _close(event: Any) -> None:
        finish()

    dialog = ui["Dialog"](
        title=_title(title),
        body=ui["Box"](
            body=ui["TextArea"](
                text=str(text or ""),
                read_only=True,
                focus_on_click=True,
                scrollbar=True,
                wrap_lines=True,
                style="class:text-area",
            ),
            padding=1,
        ),
        buttons=[button],
        width=ui["Dimension"](preferred=width),
        modal=True,
        with_background=True,
    )
    app = ui["Application"](
        layout=ui["Layout"](dialog, focused_element=button),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
        style=ui["style"],
    )
    app.run()


def confirm(
    title: str,
    text: str,
    *,
    yes_text: str = "[S] Sim",
    no_text: str = "[N] Não",
    default: bool = False,
) -> bool:
    ui = _imports()
    app: Any

    def finish(value: bool) -> None:
        app.exit(result=value)

    yes = ui["Button"](text=yes_text, handler=lambda: finish(True))
    no = ui["Button"](text=no_text, handler=lambda: finish(False))
    kb = ui["KeyBindings"]()

    @kb.add("s")
    @kb.add("y")
    def _yes(event: Any) -> None:
        finish(True)

    @kb.add("n")
    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _no(event: Any) -> None:
        finish(False)

    dialog = ui["Dialog"](
        title=_title(title),
        body=ui["Box"](body=ui["Label"](text=str(text or "")), padding=1),
        buttons=[yes, no],
        width=ui["Dimension"](preferred=104),
        modal=True,
        with_background=True,
    )
    focus = yes if default else no
    app = ui["Application"](
        layout=ui["Layout"](dialog, focused_element=focus),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
        style=ui["style"],
    )
    try:
        return bool(app.run())
    except (KeyboardInterrupt, EOFError):
        return False


def input_text(
    title: str,
    text: str,
    *,
    default: str = "",
    multiline: bool = False,
    path_completion: bool = False,
    only_directories: bool = False,
    width: int = 112,
) -> str | None:
    """Campo editável com histórico e conclusão de caminhos opcional.

    Em campos de caminho, ``Tab`` abre as sugestões do sistema de arquivos.
    """
    ui = _imports()
    app: Any
    completer = (
        ui["PathCompleter"](expanduser=True, only_directories=only_directories)
        if path_completion
        else None
    )
    field = ui["TextArea"](
        text=str(default or ""),
        multiline=multiline,
        scrollbar=bool(multiline),
        focus_on_click=True,
        style="class:text-area",
        height=None if multiline else 1,
        completer=completer,
        complete_while_typing=False,
        auto_suggest=ui["AutoSuggestFromHistory"](),
        history=ui["FileHistory"](str(_history_path())),
        wrap_lines=True,
    )

    def finish(value: str | None) -> None:
        app.exit(result=value)

    buttons = [
        ui["Button"](text="[Ctrl+S] Confirmar", handler=lambda: finish(field.text)),
        ui["Button"](text="[Esc] Cancelar", handler=lambda: finish(None)),
    ]
    hints = "Ctrl+S confirma • Esc cancela"
    if path_completion:
        hints = "Tab completa caminho • " + hints
    elif not multiline:
        hints = "Enter confirma • " + hints
    body_items: list[Any] = [ui["Label"](text=str(text or "")), ui["Box"](body=field, padding=1)]
    body_items.append(ui["Label"](text=hints, style="class:bottom-toolbar"))
    body = ui["HSplit"](body_items, padding=1)
    dialog = ui["Dialog"](
        title=_title(title),
        body=body,
        buttons=buttons,
        width=ui["Dimension"](preferred=width),
        modal=True,
        with_background=True,
    )
    kb = ui["KeyBindings"]()

    @kb.add("c-s")
    def _save(event: Any) -> None:
        finish(field.text)

    if not multiline:
        @kb.add("enter", filter=ui["has_focus"](field), eager=True)
        def _save_enter(event: Any) -> None:
            finish(field.text)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event: Any) -> None:
        finish(None)

    app = ui["Application"](
        layout=ui["Layout"](dialog, focused_element=field),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
        style=ui["style"],
    )
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


def _bind_shortcut(kb: Any, key: str, handler: Any) -> bool:
    try:
        kb.add(key)(handler)
    except (TypeError, ValueError):
        return False
    return True


def menu(
    title: str,
    text: str,
    options: Sequence[tuple[str, str, Sequence[str]]],
    *,
    width: int = 118,
) -> str | None:
    """Menu vertical selecionável; preserva atalhos declarados pelo chamador."""
    ui = _imports()
    if not options:
        return None
    app: Any
    values = [(value, label) for value, label, _keys in options]
    radio = ui["RadioList"](values=values, default=values[0][0], show_scrollbar=True)

    def finish(value: str | None) -> None:
        app.exit(result=value)

    buttons = [
        ui["Button"](text="[Enter/S] Selecionar", handler=lambda: finish(radio.current_value)),
        ui["Button"](text="[Esc/Q] Sair", handler=lambda: finish(None)),
    ]
    kb = ui["KeyBindings"]()

    @kb.add("enter")
    @kb.add("s")
    def _select(event: Any) -> None:
        finish(radio.current_value)

    used: set[str] = {"enter", "s"}
    for value, _label, keys in options:
        for key in keys:
            key_s = str(key).strip().lower()
            if not key_s or key_s in used:
                continue

            def handler(event: Any, v: str = value) -> None:
                finish(v)

            if _bind_shortcut(kb, key_s, handler):
                used.add(key_s)

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event: Any) -> None:
        finish(None)

    body = ui["HSplit"](
        [
            ui["Label"](text=str(text or "")),
            ui["Box"](body=radio, padding=1),
            ui["Label"](
                text="↑/↓ navegar • Enter/S selecionar • atalhos entre colchetes funcionam • Esc/Q sair",
                style="class:bottom-toolbar",
            ),
        ],
        padding=1,
    )
    dialog = ui["Dialog"](
        title=_title(title),
        body=body,
        buttons=buttons,
        width=ui["Dimension"](preferred=width),
        modal=True,
        with_background=True,
    )
    app = ui["Application"](
        layout=ui["Layout"](dialog, focused_element=radio),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
        style=ui["style"],
    )
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


def select_one(
    title: str,
    text: str,
    values: Sequence[tuple[Any, str]],
    *,
    default: Any | None = None,
    width: int = 112,
) -> Any | None:
    ui = _imports()
    if not values:
        return None
    app: Any
    default = values[0][0] if default is None else default
    radio = ui["RadioList"](values=list(values), default=default, show_scrollbar=True)

    def finish(value: Any | None) -> None:
        app.exit(result=value)

    buttons = [
        ui["Button"](text="[S/Enter] Selecionar", handler=lambda: finish(radio.current_value)),
        ui["Button"](text="[Esc] Cancelar", handler=lambda: finish(None)),
    ]
    kb = ui["KeyBindings"]()

    @kb.add("s")
    @kb.add("enter")
    def _select(event: Any) -> None:
        finish(radio.current_value)

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event: Any) -> None:
        finish(None)

    body = ui["HSplit"](
        [
            ui["Label"](text=str(text or "")),
            ui["Box"](body=radio, padding=1),
            ui["Label"](text="↑/↓ navegar • S/Enter selecionar • Esc cancelar", style="class:bottom-toolbar"),
        ],
        padding=1,
    )
    dialog = ui["Dialog"](
        title=_title(title),
        body=body,
        buttons=buttons,
        width=ui["Dimension"](preferred=width),
        modal=True,
        with_background=True,
    )
    app = ui["Application"](
        layout=ui["Layout"](dialog, focused_element=radio),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
        style=ui["style"],
    )
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


def select_many(
    title: str,
    text: str,
    values: Sequence[tuple[Any, str]],
    *,
    defaults: Iterable[Any] = (),
    width: int = 112,
) -> list[Any] | None:
    ui = _imports()
    app: Any
    checklist = ui["CheckboxList"](values=list(values), default_values=list(defaults))

    def finish(value: list[Any] | None) -> None:
        app.exit(result=value)

    def all_items() -> None:
        checklist.current_values[:] = [v for v, _ in values]
        app.invalidate()

    def clear_items() -> None:
        checklist.current_values.clear()
        app.invalidate()

    buttons = [
        ui["Button"](text="[Ctrl+S] Continuar", handler=lambda: finish(list(checklist.current_values))),
        ui["Button"](text="[A] Todos", handler=all_items),
        ui["Button"](text="[L] Limpar", handler=clear_items),
        ui["Button"](text="[Esc] Cancelar", handler=lambda: finish(None)),
    ]
    body = ui["HSplit"](
        [
            ui["Label"](text=str(text or "")),
            ui["Box"](body=checklist, padding=1),
            ui["Label"](
                text="Espaço marcar • A todos • L limpar • Ctrl+S continuar • Esc cancelar",
                style="class:bottom-toolbar",
            ),
        ],
        padding=1,
    )
    dialog = ui["Dialog"](
        title=_title(title),
        body=body,
        buttons=buttons,
        width=ui["Dimension"](preferred=width),
        modal=True,
        with_background=True,
    )
    kb = ui["KeyBindings"]()

    @kb.add("a")
    def _all(event: Any) -> None:
        all_items()

    @kb.add("l")
    def _clear(event: Any) -> None:
        clear_items()

    @kb.add("c-s")
    def _continue(event: Any) -> None:
        finish(list(checklist.current_values))

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event: Any) -> None:
        finish(None)

    app = ui["Application"](
        layout=ui["Layout"](dialog, focused_element=checklist),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
        style=ui["style"],
    )
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None

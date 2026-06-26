#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Componentes visuais reutilizáveis da TUI do Academic Pipeline.

A TUI usa :mod:`prompt_toolkit` em tela inteira, no padrão operacional da
UAC: listas verticais, atalhos diretos, diálogos modais, histórico de texto e
conclusão de caminhos com ``Tab``.

Nota de implementação importante
--------------------------------
``RadioList`` possui atalhos internos (inclusive ``Enter`` e ``Keys.Any``).
Quando esses atalhos têm precedência sobre os atalhos da aplicação, uma tela
parece selecionável, mas ``Enter`` e teclas como ``1`` apenas movem ou marcam a
lista e não encerram o diálogo. Todos os atalhos de decisão deste módulo são,
por isso, registrados como ``eager=True``. Isso mantém a navegação na lista e
garante que Enter, números e letras fechem a tela com a escolha esperada.
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

    # Paleta FGV com foreground explícito para funcionar em terminais com
    # renderização limitada de cores e sem herança confiável do Dialog.
    style = Style.from_dict(
        {
            "dialog": "bg:#003E7E fg:#FFFFFF",
            "dialog.body": "bg:#EAF6FC fg:#003E7E",
            "dialog frame.label": "bg:#0595D5 fg:#FFFFFF bold",
            "dialog frame-label": "bg:#0595D5 fg:#FFFFFF bold",
            "dialog shadow": "bg:#001A33",
            "label": "bg:#EAF6FC fg:#003E7E",
            "radio-list": "bg:#EAF6FC fg:#003E7E",
            "radio": "bg:#EAF6FC fg:#005FA9",
            "radio-selected": "bg:#BDEBFA fg:#003E7E bold",
            "radio-checked": "bg:#BDEBFA fg:#003E7E bold",
            "checkbox-list": "bg:#EAF6FC fg:#003E7E",
            "checkbox": "bg:#EAF6FC fg:#005FA9",
            "checkbox-selected": "bg:#BDEBFA fg:#003E7E bold",
            "checkbox-checked": "bg:#BDEBFA fg:#003E7E bold",
            "button": "bg:#005FA9 fg:#FFFFFF bold",
            "button.focused": "bg:#BDEBFA fg:#003E7E bold",
            "bottom-toolbar": "bg:#0595D5 fg:#FFFFFF bold",
            "text-area": "bg:#FFFFFF fg:#003E7E",
            "text-area.focused": "bg:#F7FCFF fg:#003E7E",
            "label.emphasis": "bg:#EAF6FC fg:#003E7E bold",
            "status.ok": "bg:#EAF6FC fg:#006D3F bold",
            "status.warn": "bg:#EAF6FC fg:#7A4A00 bold",
            "status.error": "bg:#B00020 fg:#FFFFFF bold",
            "window.too-small": "bg:#B00020 fg:#FFFFFF bold",
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


def _menu_height(count: int) -> int:
    """Reserva linhas suficientes para a lista não desaparecer no diálogo."""
    return max(8, min(24, count + 3))


def _run_application(
    ui: dict[str, Any],
    *,
    layout: Any,
    key_bindings: Any,
    style: Any,
    input_obj: Any | None = None,
    output_obj: Any | None = None,
) -> Any:
    """Executa uma aplicação e permite injeção de I/O nos testes.

    ``input_obj`` e ``output_obj`` são internos; não fazem parte do fluxo do
    usuário. Eles permitem testes reais de teclado com ``PipeInput`` e
    ``DummyOutput`` sem abrir um terminal físico.
    """
    kwargs: dict[str, Any] = {
        "layout": layout,
        "key_bindings": key_bindings,
        "full_screen": True,
        "mouse_support": True,
        "style": style,
    }
    if input_obj is not None:
        kwargs["input"] = input_obj
    if output_obj is not None:
        kwargs["output"] = output_obj
    return ui["Application"](**kwargs)


def message(
    title: str,
    text: str,
    *,
    width: int = 116,
    _input: Any | None = None,
    _output: Any | None = None,
) -> None:
    """Mostra texto rolável em tela inteira."""
    ui = _imports()
    app: Any

    def finish() -> None:
        app.exit(result=None)

    button = ui["Button"](text="[Enter/Esc] Fechar", handler=finish)
    kb = ui["KeyBindings"]()

    @kb.add("enter", eager=True)
    @kb.add("escape", eager=True)
    @kb.add("q", eager=True)
    @kb.add("c-c", eager=True)
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
    app = _run_application(
        ui,
        layout=ui["Layout"](dialog, focused_element=button),
        key_bindings=kb,
        style=ui["style"],
        input_obj=_input,
        output_obj=_output,
    )
    app.run()


def confirm(
    title: str,
    text: str,
    *,
    yes_text: str = "[S] Sim",
    no_text: str = "[N] Não",
    default: bool = False,
    _input: Any | None = None,
    _output: Any | None = None,
) -> bool:
    ui = _imports()
    app: Any

    def finish(value: bool) -> None:
        app.exit(result=value)

    yes = ui["Button"](text=yes_text, handler=lambda: finish(True))
    no = ui["Button"](text=no_text, handler=lambda: finish(False))
    kb = ui["KeyBindings"]()

    @kb.add("s", eager=True)
    @kb.add("y", eager=True)
    def _yes(event: Any) -> None:
        finish(True)

    @kb.add("n", eager=True)
    @kb.add("escape", eager=True)
    @kb.add("q", eager=True)
    @kb.add("c-c", eager=True)
    def _no(event: Any) -> None:
        finish(False)

    @kb.add("enter", eager=True)
    def _default(event: Any) -> None:
        finish(bool(default))

    dialog = ui["Dialog"](
        title=_title(title),
        body=ui["Box"](body=ui["Label"](text=str(text or "")), padding=1),
        buttons=[yes, no],
        width=ui["Dimension"](preferred=104),
        modal=True,
        with_background=True,
    )
    focus = yes if default else no
    app = _run_application(
        ui,
        layout=ui["Layout"](dialog, focused_element=focus),
        key_bindings=kb,
        style=ui["style"],
        input_obj=_input,
        output_obj=_output,
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
    width: int = 116,
    _input: Any | None = None,
    _output: Any | None = None,
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
    body_items: list[Any] = [
        ui["Label"](text=str(text or ""), style="class:label.emphasis"),
        ui["Box"](body=field, padding=1),
        ui["Label"](text=hints, style="class:bottom-toolbar"),
    ]
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

    @kb.add("c-s", eager=True)
    def _save(event: Any) -> None:
        finish(field.text)

    if not multiline:

        @kb.add("enter", filter=ui["has_focus"](field), eager=True)
        def _save_enter(event: Any) -> None:
            finish(field.text)

    @kb.add("escape", eager=True)
    @kb.add("c-c", eager=True)
    def _cancel(event: Any) -> None:
        finish(None)

    app = _run_application(
        ui,
        layout=ui["Layout"](dialog, focused_element=field),
        key_bindings=kb,
        style=ui["style"],
        input_obj=_input,
        output_obj=_output,
    )
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


def _bind_shortcut(kb: Any, key: str, handler: Any) -> bool:
    """Registra atalho de forma segura e com precedência sobre RadioList."""
    try:
        kb.add(key, eager=True)(handler)
    except (TypeError, ValueError):
        return False
    return True


def menu(
    title: str,
    text: str,
    options: Sequence[tuple[str, str, Sequence[str]]],
    *,
    width: int = 122,
    _input: Any | None = None,
    _output: Any | None = None,
) -> str | None:
    """Menu vertical selecionável com atalhos diretos.

    ``select_on_focus=True`` faz com que a opção marcada acompanhe ↑/↓. Assim,
    quando Enter (registrado como eager) confirma a tela, ele retorna a linha
    atualmente destacada — e não a primeira linha do menu.
    """
    ui = _imports()
    if not options:
        return None
    app: Any
    values = [(value, label) for value, label, _keys in options]
    radio = ui["RadioList"](
        values=values,
        default=values[0][0],
        show_scrollbar=True,
        select_on_focus=True,
    )

    def finish(value: str | None) -> None:
        app.exit(result=value)

    buttons = [
        ui["Button"](text="[Enter] Selecionar", handler=lambda: finish(radio.current_value)),
        ui["Button"](text="[Esc] Voltar", handler=lambda: finish(None)),
    ]
    kb = ui["KeyBindings"]()

    @kb.add("enter", eager=True)
    @kb.add("s", eager=True)
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

    @kb.add("escape", eager=True)
    @kb.add("q", eager=True)
    @kb.add("c-c", eager=True)
    def _cancel(event: Any) -> None:
        finish(None)

    body = ui["HSplit"](
        [
            ui["Label"](text=str(text or ""), style="class:label.emphasis"),
            ui["Box"](
                body=radio,
                padding=1,
                height=ui["Dimension"](
                    preferred=_menu_height(len(values)),
                    min=min(8, _menu_height(len(values))),
                ),
            ),
            ui["Label"](
                text="↑/↓ navegar • Enter selecionar • número/letra abre diretamente • Esc/Q voltar",
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
    app = _run_application(
        ui,
        layout=ui["Layout"](dialog, focused_element=radio),
        key_bindings=kb,
        style=ui["style"],
        input_obj=_input,
        output_obj=_output,
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
    width: int = 116,
    _input: Any | None = None,
    _output: Any | None = None,
) -> Any | None:
    """Lista de escolha única com Enter funcionando em todas as telas."""
    ui = _imports()
    if not values:
        return None
    app: Any
    default = values[0][0] if default is None else default
    radio = ui["RadioList"](
        values=list(values),
        default=default,
        show_scrollbar=True,
        select_on_focus=True,
    )

    def finish(value: Any | None) -> None:
        app.exit(result=value)

    buttons = [
        ui["Button"](text="[Enter] Selecionar", handler=lambda: finish(radio.current_value)),
        ui["Button"](text="[Esc] Voltar", handler=lambda: finish(None)),
    ]
    kb = ui["KeyBindings"]()

    @kb.add("s", eager=True)
    @kb.add("enter", eager=True)
    def _select(event: Any) -> None:
        finish(radio.current_value)

    @kb.add("escape", eager=True)
    @kb.add("q", eager=True)
    @kb.add("c-c", eager=True)
    def _cancel(event: Any) -> None:
        finish(None)

    body = ui["HSplit"](
        [
            ui["Label"](text=str(text or ""), style="class:label.emphasis"),
            ui["Box"](
                body=radio,
                padding=1,
                height=ui["Dimension"](
                    preferred=_menu_height(len(values)),
                    min=min(8, _menu_height(len(values))),
                ),
            ),
            ui["Label"](
                text="↑/↓ navegar • Enter selecionar • Esc/Q voltar",
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
    app = _run_application(
        ui,
        layout=ui["Layout"](dialog, focused_element=radio),
        key_bindings=kb,
        style=ui["style"],
        input_obj=_input,
        output_obj=_output,
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
    width: int = 116,
    _input: Any | None = None,
    _output: Any | None = None,
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
        ui["Button"](text="[Esc] Voltar", handler=lambda: finish(None)),
    ]
    body = ui["HSplit"](
        [
            ui["Label"](text=str(text or ""), style="class:label.emphasis"),
            ui["Box"](
                body=checklist,
                padding=1,
                height=ui["Dimension"](
                    preferred=_menu_height(len(values)),
                    min=min(8, _menu_height(len(values))),
                ),
            ),
            ui["Label"](
                text="Espaço marca • A todos • L limpar • Ctrl+S continuar • Esc voltar",
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

    @kb.add("a", eager=True)
    def _all(event: Any) -> None:
        all_items()

    @kb.add("l", eager=True)
    def _clear(event: Any) -> None:
        clear_items()

    @kb.add("c-s", eager=True)
    def _continue(event: Any) -> None:
        finish(list(checklist.current_values))

    @kb.add("escape", eager=True)
    @kb.add("c-c", eager=True)
    def _cancel(event: Any) -> None:
        finish(None)

    app = _run_application(
        ui,
        layout=ui["Layout"](dialog, focused_element=checklist),
        key_bindings=kb,
        style=ui["style"],
        input_obj=_input,
        output_obj=_output,
    )
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None

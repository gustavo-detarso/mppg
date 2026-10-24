"""Entrypoint oficial do Academic Pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from .runtime import run


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o runtime oficial por rotas canônicas nativas."""

    # MPPG_MINUTA_CLI_INTEGRATION_V1
    from sys import argv as _minuta_sys_argv
    _minuta_argv = _minuta_sys_argv[1:] if argv is None else list(argv)
    if _minuta_argv and _minuta_argv[0] == 'minuta':
        from .minuta_runtime import main as _minuta_main
        return _minuta_main(_minuta_argv[1:])
    # MPPG_MINUTA_TUI_V25
    if '--tui' in _minuta_argv:
        from .minuta_tui import run_with_minuta_menu as _minuta_tui_wrap
        return _minuta_tui_wrap(lambda: run(argv))
    return run(argv)

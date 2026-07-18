"""Interface pública estável do Academic Pipeline.

A implementação produtiva ainda reside no runtime legado durante a AP-002.
Este pacote fornece o novo nome de entrada sem quebrar comandos existentes.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o entrypoint oficial, delegando ao runtime compatível."""
    from .cli import main as cli_main

    return cli_main(argv)

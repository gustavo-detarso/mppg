"""Entrypoint oficial do Academic Pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from .runtime import run


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o runtime oficial por rotas canônicas nativas."""

    return run(argv)

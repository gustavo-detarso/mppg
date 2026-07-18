"""Entrypoint oficial do Academic Pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from .legacy import run_legacy


def main(argv: Sequence[str] | None = None) -> int:
    """Encaminha argumentos para o runtime legado durante a transição."""
    return run_legacy(argv)

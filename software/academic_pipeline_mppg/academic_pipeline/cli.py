"""Entrypoint oficial do Academic Pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from .legacy import run_legacy
from .runtime import run


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o runtime oficial com fallback legado enumerado."""

    return run(argv, legacy_runner=run_legacy)

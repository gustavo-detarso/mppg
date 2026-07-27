#!/usr/bin/env python3
"""Launcher fino para a CLI canônica do Academic Pipeline.

A AP-008F.2C desconecta este alias do script físico RC10. O arquivo permanece
como superfície de compatibilidade de nome, mas toda execução é delegada a
``academic_pipeline.cli:main``.
"""

from __future__ import annotations

import sys as _ap008f2c_sys
from pathlib import Path as _AP008F2CPath

_ap008f2c_source_root = _AP008F2CPath(__file__).resolve().parents[3]
if str(_ap008f2c_source_root) not in _ap008f2c_sys.path:
    _ap008f2c_sys.path.insert(0, str(_ap008f2c_source_root))

from academic_pipeline.cli import main as _ap008f2c_canonical_main


def main(argv: list[str] | None = None) -> int:
    """Executa a superfície pública canônica preservando argumentos da CLI."""
    return int(_ap008f2c_canonical_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

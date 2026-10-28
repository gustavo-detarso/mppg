"""Permite executar o Academic Pipeline com ``python -m academic_pipeline``."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    if "--prisma-qualidade-ia" in __import__("sys").argv[1:]:
        from .prisma_qualidade_ia import intercept
        raise SystemExit(intercept(__import__("sys").argv[1:], main))
    raise SystemExit(main())

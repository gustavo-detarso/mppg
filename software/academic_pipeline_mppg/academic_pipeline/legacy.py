"""Ponte controlada para o runtime legado do Academic Pipeline.

Esta camada existe para preservar o comando histórico enquanto os módulos
internos são migrados gradualmente para imports de pacote.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
SOFTWARE_ROOT = PACKAGE_DIR.parent
LEGACY_PIPELINE_DIR = SOFTWARE_ROOT / "app_bundle" / "scripts" / "pipeline"
LEGACY_MODULE_NAME = "academic_pipeline_rc10"
OFFICIAL_PROGRAM_NAME = "academic-pipeline"


class LegacyRuntimeError(RuntimeError):
    """Indica que a ponte não conseguiu carregar ou executar o runtime legado."""


def _same_path(entry: object, target: Path) -> bool:
    """Compara entradas reais de ``sys.path`` sem remover o marcador ``""``."""
    if not isinstance(entry, (str, bytes, os.PathLike)):
        return False

    raw = os.fspath(entry)
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)

    if raw == "":
        return False

    try:
        return Path(raw).expanduser().resolve(strict=False) == target
    except (OSError, RuntimeError, ValueError):
        return raw == str(target)


def ensure_legacy_path() -> Path:
    """Mantém exatamente uma entrada explícita para o runtime legado.

    O pytest e outros launchers podem inserir o mesmo diretório mais de uma
    vez. A ponte remove duplicatas equivalentes e recoloca o caminho uma única
    vez no início, sem alterar o marcador vazio que representa o diretório
    corrente.
    """
    target = LEGACY_PIPELINE_DIR.resolve(strict=False)
    if not target.is_dir():
        raise LegacyRuntimeError(
            "Diretório do runtime legado não encontrado: "
            f"{target}"
        )

    sys.path[:] = [
        entry
        for entry in sys.path
        if not _same_path(entry, target)
    ]
    sys.path.insert(0, str(target))

    return target


def load_legacy_module() -> ModuleType:
    """Carrega o módulo legado e valida a presença de um ``main`` executável."""
    ensure_legacy_path()
    module = importlib.import_module(LEGACY_MODULE_NAME)

    if not callable(getattr(module, "main", None)):
        raise LegacyRuntimeError(
            f"O módulo {LEGACY_MODULE_NAME!r} não fornece main() executável."
        )

    return module


def _normalize_return_code(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise LegacyRuntimeError(
        "O runtime legado retornou um código de saída inválido: "
        f"{value!r}"
    )


def run_legacy(
    argv: Sequence[str] | None = None,
    *,
    program_name: str = OFFICIAL_PROGRAM_NAME,
) -> int:
    """Executa o ``main`` legado preservando e restaurando ``sys.argv``.

    Quando ``argv`` é omitido, são encaminhados os argumentos recebidos pelo
    processo atual. O nome público exibido pelo argparse passa a ser
    ``academic-pipeline``.
    """
    original_argv = sys.argv[:]
    forwarded = original_argv[1:] if argv is None else [str(item) for item in argv]

    module = load_legacy_module()
    sys.argv = [program_name, *forwarded]

    try:
        result = module.main()
    finally:
        sys.argv = original_argv

    return _normalize_return_code(result)

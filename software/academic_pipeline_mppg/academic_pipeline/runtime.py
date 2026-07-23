"""Runtime oficial do Academic Pipeline com primeira onda nativa.

Os comandos ainda não migrados seguem para o adaptador legado por uma decisão
de rota explícita. Os comandos da primeira onda não importam nem executam o
monólito histórico e não modificam o caminho de importação do processo.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app_bundle.scripts.pipeline.diagnostics import PIPELINE_VERSION

from . import cli_parser
from .command_dispatch import (
    dispatch_stage_003,
    dispatch_stage_005,
    dispatch_stage_006,
    dispatch_stage_007,
)

OFFICIAL_PROGRAM_NAME = "academic-pipeline"
FIRST_WAVE_OPTIONS = frozenset(
    {
        "--help",
        "--list-toml-profiles",
        "--list-institutions",
        "--list-layouts",
        "--explain-profile",
    }
)
NATIVE_TRIGGER_OPTIONS = FIRST_WAVE_OPTIONS | {"-h"}
PARSER_BUILDER_NAME = 'build_parser'

LegacyRunner = Callable[[Sequence[str] | None], int]


class RuntimeRoute(str, Enum):
    """Rotas deliberadas do entrypoint durante a transição."""

    NATIVE_FIRST_WAVE = "native_first_wave"
    LEGACY_FALLBACK = "legacy_fallback"


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Dependências explícitas necessárias à primeira onda nativa."""

    path_type: type[Path]
    load_config: Callable[[Path], dict[str, Any]]
    describe_institution_profiles: Callable[..., Any]
    available_layouts: Callable[..., Any]
    resolve_layout_spec: Callable[..., Any]
    explain_profile: Callable[..., Any]

    def as_dispatch_mapping(self) -> Mapping[str, Any]:
        """Adapta campos tipados ao contrato histórico dos dispatchers."""

        return MappingProxyType(
            {
            'Path': self.path_type,
            'available_layouts': self.available_layouts,
            'describe_institution_profiles': self.describe_institution_profiles,
            'explain_profile': self.explain_profile,
            'load_config': self.load_config,
            'resolve_layout_spec': self.resolve_layout_spec,
            }
        )


class NativeRuntimeError(RuntimeError):
    """Indica quebra do contrato da primeira onda nativa."""


def _load_config(path: Path) -> dict[str, Any]:
    from app_bundle.scripts.pipeline.institution_profiles import (
        apply_institution_profile,
    )

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        payload = tomllib.load(stream)

    if not isinstance(payload, dict):
        raise NativeRuntimeError(
            f"Configuração TOML inválida: {config_path}"
        )

    payload["__config_path__"] = str(config_path)
    payload["__config_dir__"] = str(config_path.parent)
    profiled = apply_institution_profile(payload)

    if not isinstance(profiled, dict):
        raise NativeRuntimeError(
            "apply_institution_profile retornou configuração inválida"
        )
    return profiled


def default_runtime_context() -> RuntimeContext:
    """Carrega dependências por imports canônicos, sem ponte para o legado."""

    from app_bundle.scripts.pipeline.institution_explainer import (
        explain_profile,
    )
    from app_bundle.scripts.pipeline.institution_layouts import (
        available_layouts,
        resolve_layout_spec,
    )
    from app_bundle.scripts.pipeline.institution_profiles import (
        describe_institution_profiles,
    )

    return RuntimeContext(
        path_type=Path,
        load_config=_load_config,
        describe_institution_profiles=describe_institution_profiles,
        available_layouts=available_layouts,
        resolve_layout_spec=resolve_layout_spec,
        explain_profile=explain_profile,
    )


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    source = sys.argv[1:] if argv is None else argv
    return [str(item) for item in source]


def _matches_option(token: str, option: str) -> bool:
    return token == option or token.startswith(f"{option}=")


def select_runtime_route(argv: Sequence[str]) -> RuntimeRoute:
    """Seleciona a rota sem interpretar comandos operacionais legados."""

    for token in argv:
        if any(
            _matches_option(token, option)
            for option in NATIVE_TRIGGER_OPTIONS
        ):
            return RuntimeRoute.NATIVE_FIRST_WAVE
    return RuntimeRoute.LEGACY_FALLBACK


def _build_parser() -> argparse.ArgumentParser:
    parser = cli_parser.build_parser(pipeline_version=PIPELINE_VERSION)
    if not isinstance(parser, argparse.ArgumentParser):
        raise NativeRuntimeError(
            "build_parser não retornou ArgumentParser"
        )

    parser.prog = OFFICIAL_PROGRAM_NAME
    return parser


def _normalize_dispatch_code(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise NativeRuntimeError(
        f"Dispatcher retornou código inválido: {value!r}"
    )


def _run_native_first_wave(
    argv: Sequence[str],
    *,
    context: RuntimeContext | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    runtime = (context or default_runtime_context()).as_dispatch_mapping()

    for dispatch in (
        dispatch_stage_003,
        dispatch_stage_005,
        dispatch_stage_006,
        dispatch_stage_007,
    ):
        result = dispatch(args, runtime)
        if bool(getattr(result, 'handled')):
            return _normalize_dispatch_code(
                getattr(result, 'value')
            )

    raise NativeRuntimeError(
        "Uma opção da primeira onda foi selecionada, mas nenhum dispatcher "
        "a reconheceu."
    )


def run(
    argv: Sequence[str] | None = None,
    *,
    legacy_runner: LegacyRunner,
    context: RuntimeContext | None = None,
) -> int:
    """Executa a primeira onda nativamente ou usa fallback enumerado."""

    forwarded = _normalize_argv(argv)
    route = select_runtime_route(forwarded)

    if route is RuntimeRoute.NATIVE_FIRST_WAVE:
        return _run_native_first_wave(forwarded, context=context)

    return int(legacy_runner(forwarded))


__all__ = [
    "FIRST_WAVE_OPTIONS",
    "NATIVE_TRIGGER_OPTIONS",
    "NativeRuntimeError",
    "RuntimeContext",
    "RuntimeRoute",
    "default_runtime_context",
    "run",
    "select_runtime_route",
]

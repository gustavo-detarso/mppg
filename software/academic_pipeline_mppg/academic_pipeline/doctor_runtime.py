from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app_bundle.scripts.pipeline.diagnostics import (
    PIPELINE_VERSION,
    print_doctor_report,
    run_doctor,
)
from app_bundle.scripts.pipeline.prisma_busca_externa import (
    external_search_enabled,
)
from app_bundle.scripts.pipeline.utils import resolve_path, write_json

from . import cli_parser
from .document_orchestration import (
    apply_cli_path_overrides_impl,
    output_paths_impl,
)
from .prisma_generic_orchestration import (
    research_output_paths_impl_001,
)
from .runtime import default_runtime_context

OFFICIAL_PROGRAM_NAME = "academic-pipeline"
DOCTOR_OPTION = "--doctor"


class DoctorRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DoctorRuntimeContext:
    load_config: Callable[[Path], dict[str, Any]]
    apply_cli_path_overrides: Callable[
        [dict[str, Any], argparse.Namespace],
        dict[str, Any],
    ]
    output_paths: Callable[[dict[str, Any]], tuple[Path, str]]
    research_output_paths: Callable[
        [dict[str, Any]],
        tuple[Path, str],
    ]
    external_search_enabled: Callable[[dict[str, Any]], bool]
    run_doctor: Callable[
        [dict[str, Any] | None],
        Mapping[str, Any],
    ]
    print_doctor_report: Callable[[Mapping[str, Any]], Any]
    write_json: Callable[[Path, Any], Any]


def _section(
    cfg: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, dict) else {}


_EXPLICIT_RUNTIME = MappingProxyType(
    {
        "Path": Path,
        "_section": _section,
        "resolve_path": resolve_path,
    }
)


def _apply_cli_path_overrides(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = apply_cli_path_overrides_impl(
        _EXPLICIT_RUNTIME,
        cfg,
        args,
    )
    if not isinstance(result, dict):
        raise DoctorRuntimeError(
            "apply_cli_path_overrides_impl não retornou dict"
        )
    return result


def _output_paths(
    cfg: dict[str, Any],
) -> tuple[Path, str]:
    result = output_paths_impl(_EXPLICIT_RUNTIME, cfg)
    if not (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], Path)
        and isinstance(result[1], str)
    ):
        raise DoctorRuntimeError(
            "output_paths_impl retornou contrato inválido"
        )
    return result


def _research_output_paths(
    cfg: dict[str, Any],
) -> tuple[Path, str]:
    result = research_output_paths_impl_001(
        _EXPLICIT_RUNTIME,
        cfg,
    )
    if not (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], Path)
        and isinstance(result[1], str)
    ):
        raise DoctorRuntimeError(
            "research_output_paths_impl_001 retornou "
            "contrato inválido"
        )
    return result


def default_doctor_runtime_context() -> DoctorRuntimeContext:
    base = default_runtime_context()
    return DoctorRuntimeContext(
        load_config=base.load_config,
        apply_cli_path_overrides=_apply_cli_path_overrides,
        output_paths=_output_paths,
        research_output_paths=_research_output_paths,
        external_search_enabled=external_search_enabled,
        run_doctor=run_doctor,
        print_doctor_report=print_doctor_report,
        write_json=write_json,
    )


def _normalize_argv(
    argv: Sequence[str] | None,
) -> tuple[str, ...]:
    source = sys.argv[1:] if argv is None else argv
    return tuple(str(item) for item in source)


def _build_parser() -> argparse.ArgumentParser:
    parser = cli_parser.build_parser(
        pipeline_version=PIPELINE_VERSION,
    )
    if not isinstance(parser, argparse.ArgumentParser):
        raise DoctorRuntimeError(
            "build_parser não retornou ArgumentParser"
        )
    parser.prog = OFFICIAL_PROGRAM_NAME
    return parser


def run_doctor_command(
    argv: Sequence[str] | None = None,
    *,
    context: DoctorRuntimeContext | None = None,
) -> int:
    forwarded = _normalize_argv(argv)
    args = _build_parser().parse_args(list(forwarded))
    if not bool(args.doctor):
        raise DoctorRuntimeError(
            "run_doctor_command exige a opção --doctor"
        )

    active = context or default_doctor_runtime_context()
    cfg: dict[str, Any] | None = None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        cfg = active.load_config(config_path)
        if not isinstance(cfg, dict):
            raise DoctorRuntimeError(
                "load_config não retornou dict"
            )
        cfg = active.apply_cli_path_overrides(cfg, args)

    report = active.run_doctor(cfg)
    if not isinstance(report, Mapping):
        raise DoctorRuntimeError(
            "run_doctor não retornou Mapping"
        )
    active.print_doctor_report(report)

    if cfg:
        if active.external_search_enabled(cfg):
            out_dir, prefix = active.research_output_paths(cfg)
        else:
            out_dir, prefix = active.output_paths(cfg)
        active.write_json(
            out_dir / f"{prefix}.doctor_report.json",
            dict(report),
        )

    return 0 if bool(report.get("ok")) else 2


__all__ = [
    "DOCTOR_OPTION",
    "DoctorRuntimeContext",
    "DoctorRuntimeError",
    "default_doctor_runtime_context",
    "run_doctor_command",
]

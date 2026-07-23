from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app_bundle.scripts.pipeline.diagnostics import (
    PIPELINE_VERSION,
    check_config,
    print_check_config_report,
)

from . import cli_parser
from .command_dispatch import dispatch_stage_017
from .doctor_runtime import default_doctor_runtime_context

OFFICIAL_PROGRAM_NAME = "academic-pipeline"
CHECK_CONFIG_OPTION = "--check-config"


class CheckConfigRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CheckConfigRuntimeContext:
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
    check_config: Callable[[dict[str, Any]], dict[str, Any]]
    print_check_config_report: Callable[
        [dict[str, Any]],
        Any,
    ]
    write_json: Callable[[Path, Any], Any]


def default_check_config_runtime_context(
) -> CheckConfigRuntimeContext:
    base = default_doctor_runtime_context()
    return CheckConfigRuntimeContext(
        load_config=base.load_config,
        apply_cli_path_overrides=(
            base.apply_cli_path_overrides
        ),
        output_paths=base.output_paths,
        research_output_paths=base.research_output_paths,
        external_search_enabled=(
            base.external_search_enabled
        ),
        check_config=check_config,
        print_check_config_report=print_check_config_report,
        write_json=base.write_json,
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
        raise CheckConfigRuntimeError(
            "build_parser não retornou ArgumentParser"
        )
    parser.prog = OFFICIAL_PROGRAM_NAME
    return parser


def _prepare_config(
    args: argparse.Namespace,
    context: CheckConfigRuntimeContext,
) -> dict[str, Any]:
    if not args.config:
        return {}

    config_path = Path(args.config).expanduser().resolve()
    loaded = context.load_config(config_path)
    if not isinstance(loaded, dict):
        raise CheckConfigRuntimeError(
            "load_config não retornou dict"
        )

    prepared = context.apply_cli_path_overrides(
        loaded,
        args,
    )
    if not isinstance(prepared, dict):
        raise CheckConfigRuntimeError(
            "apply_cli_path_overrides não retornou dict"
        )
    return prepared


def run_check_config_command(
    argv: Sequence[str] | None = None,
    *,
    context: CheckConfigRuntimeContext | None = None,
) -> int:
    forwarded = _normalize_argv(argv)
    args = _build_parser().parse_args(list(forwarded))
    if not bool(args.check_config):
        raise CheckConfigRuntimeError(
            "run_check_config_command exige --check-config"
        )

    active = (
        context
        or default_check_config_runtime_context()
    )
    cfg = _prepare_config(args, active)
    dispatch_runtime = MappingProxyType(
        {
            "cfg": cfg,
            "check_config": active.check_config,
            "external_search_enabled": (
                active.external_search_enabled
            ),
            "output_paths": active.output_paths,
            "print_check_config_report": (
                active.print_check_config_report
            ),
            "research_output_paths": (
                active.research_output_paths
            ),
            "write_json": active.write_json,
        }
    )
    result = dispatch_stage_017(
        args,
        dispatch_runtime,
    )
    if not result.handled:
        raise CheckConfigRuntimeError(
            "dispatch_stage_017 não tratou --check-config"
        )
    return int(result.value)


__all__ = [
    "CHECK_CONFIG_OPTION",
    "CheckConfigRuntimeContext",
    "CheckConfigRuntimeError",
    "default_check_config_runtime_context",
    "run_check_config_command",
]

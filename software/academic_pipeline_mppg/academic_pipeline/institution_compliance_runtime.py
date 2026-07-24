from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app_bundle.scripts.pipeline.diagnostics import PIPELINE_VERSION
from app_bundle.scripts.pipeline.institution_compliance import (
    render_compliance_markdown,
    run_institution_compliance,
    write_compliance_reports,
)

from . import cli_parser
from .command_dispatch import dispatch_stage_015
from .doctor_runtime import default_doctor_runtime_context

OFFICIAL_PROGRAM_NAME = "academic-pipeline"
INSTITUTION_COMPLIANCE_OPTION = "--check-institution-compliance"


class InstitutionComplianceRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InstitutionComplianceRuntimeContext:
    load_config: Callable[[Path], dict[str, Any]]
    apply_cli_path_overrides: Callable[
        [dict[str, Any], argparse.Namespace],
        dict[str, Any],
    ]
    output_paths: Callable[[dict[str, Any]], tuple[Path, str]]
    run_institution_compliance: Callable[..., dict[str, Any]]
    write_compliance_reports: Callable[
        [dict[str, Any], Path],
        tuple[Path, Path],
    ]
    render_compliance_markdown: Callable[[dict[str, Any]], str]


def default_institution_compliance_runtime_context(
) -> InstitutionComplianceRuntimeContext:
    base = default_doctor_runtime_context()
    return InstitutionComplianceRuntimeContext(
        load_config=base.load_config,
        apply_cli_path_overrides=base.apply_cli_path_overrides,
        output_paths=base.output_paths,
        run_institution_compliance=run_institution_compliance,
        write_compliance_reports=write_compliance_reports,
        render_compliance_markdown=render_compliance_markdown,
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
        raise InstitutionComplianceRuntimeError(
            "build_parser não retornou ArgumentParser"
        )
    parser.prog = OFFICIAL_PROGRAM_NAME
    return parser


def _prepare_config(
    args: argparse.Namespace,
    context: InstitutionComplianceRuntimeContext,
) -> dict[str, Any]:
    if not args.config:
        return {}
    config_path = Path(args.config).expanduser().resolve()
    loaded = context.load_config(config_path)
    if not isinstance(loaded, dict):
        raise InstitutionComplianceRuntimeError(
            "load_config não retornou dict"
        )
    prepared = context.apply_cli_path_overrides(loaded, args)
    if not isinstance(prepared, dict):
        raise InstitutionComplianceRuntimeError(
            "apply_cli_path_overrides não retornou dict"
        )
    return prepared


def _dispatch_runtime(
    cfg: Mapping[str, Any],
    context: InstitutionComplianceRuntimeContext,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "Path": Path,
            "cfg": dict(cfg),
            "output_paths": context.output_paths,
            "run_institution_compliance": (
                context.run_institution_compliance
            ),
            "write_compliance_reports": (
                context.write_compliance_reports
            ),
            "render_compliance_markdown": (
                context.render_compliance_markdown
            ),
        }
    )


def run_institution_compliance_command(
    argv: Sequence[str] | None = None,
    *,
    context: InstitutionComplianceRuntimeContext | None = None,
) -> int:
    forwarded = _normalize_argv(argv)
    args = _build_parser().parse_args(list(forwarded))
    if not bool(args.check_institution_compliance):
        raise InstitutionComplianceRuntimeError(
            "run_institution_compliance_command exige "
            "--check-institution-compliance"
        )
    active = (
        context
        or default_institution_compliance_runtime_context()
    )
    cfg = _prepare_config(args, active)
    result = dispatch_stage_015(
        args,
        _dispatch_runtime(cfg, active),
    )
    if not result.handled:
        raise InstitutionComplianceRuntimeError(
            "dispatch_stage_015 não tratou "
            "--check-institution-compliance"
        )
    return int(result.value)


__all__ = [
    "INSTITUTION_COMPLIANCE_OPTION",
    "InstitutionComplianceRuntimeContext",
    "InstitutionComplianceRuntimeError",
    "default_institution_compliance_runtime_context",
    "run_institution_compliance_command",
]

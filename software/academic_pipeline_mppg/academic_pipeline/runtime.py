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
DOCTOR_OPTIONS = frozenset({"--doctor"})
DOCTOR_PRECEDING_TRIGGER_DESTS = frozenset(
    {
        'check_institution_compliance',
        'explain_profile',
        'forcar_regeneracao_mapa_mental',
        'gui',
        'init_project',
        'init_toml',
        'inspect_bib',
        'list_institutions',
        'list_layouts',
        'list_toml_profiles',
        'make_doi_manifest',
        'output',
        'reusar_mapa_mental',
        'show_prompts',
        'somente_mapa_mental',
        'somente_renderizar',
        'tui',
        'write_prompt_lock',
    }
)
DOCTOR_COMBINATION_ERROR = (
    "Erro de uso: --doctor aceita apenas opções de diagnóstico "
    "e não pode ser combinado com outros comandos."
)
CHECK_CONFIG_OPTIONS = frozenset({"--check-config"})
CHECK_CONFIG_PRECEDING_TRIGGER_DESTS = frozenset(
    {
        'check_institution_compliance',
        'doctor',
        'explain_profile',
        'forcar_regeneracao_mapa_mental',
        'gui',
        'init_project',
        'init_toml',
        'inspect_bib',
        'list_institutions',
        'list_layouts',
        'list_toml_profiles',
        'make_doi_manifest',
        'output',
        'reusar_mapa_mental',
        'show_prompts',
        'somente_mapa_mental',
        'somente_renderizar',
        'tui',
        'write_prompt_lock',
    }
)
CHECK_CONFIG_COMBINATION_ERROR = (
    "Erro de uso: --check-config aceita apenas opções de validação "
    "de configuração e não pode ser combinado com outros comandos."
)
LIST_PROFILES_OPTIONS = frozenset({"--list-profiles"})
LIST_PROFILES_COMBINATION_ERROR = (
    "Erro de uso: --list-profiles não aceita argumentos adicionais "
    "nem pode ser combinado com outros comandos."
)
INSTITUTION_COMPLIANCE_OPTIONS = frozenset({"--check-institution-compliance"})
INSTITUTION_COMPLIANCE_VALUE_OPTIONS = frozenset({
    "--config",
    "--org",
    "--academic-writing",
    "--latex-extra-path",
    "--pdf-engine",
    "--bib",
    "--docx",
    "--pdf",
    "--output",
    "--output-dir",
    "--work-dir",
    "--cache-dir",
    "--research-output-dir",
    "--output-prefix",
})

INSTITUTION_COMPLIANCE_COMBINATION_ERROR = (
    "Erro de uso: --check-institution-compliance aceita somente opções "
    "de conformidade institucional e não pode ser combinado com outros comandos."
)
DOI_MANIFEST_OPTIONS = frozenset({"--make-doi-manifest"})
DOI_MANIFEST_VALUE_OPTIONS = frozenset({
    "--input-dir",
    "--input-zip",
    "--output",
})
NATIVE_TRIGGER_OPTIONS = (
    FIRST_WAVE_OPTIONS
    | DOCTOR_OPTIONS
    | CHECK_CONFIG_OPTIONS
    | LIST_PROFILES_OPTIONS
    | INSTITUTION_COMPLIANCE_OPTIONS
    | DOI_MANIFEST_OPTIONS
    | {"-h"}
)



PARSER_BUILDER_NAME = 'build_parser'

LegacyRunner = Callable[[Sequence[str] | None], int]


class RuntimeRoute(str, Enum):
    """Rotas deliberadas do entrypoint durante a transição."""

    NATIVE_FIRST_WAVE = "native_first_wave"
    NATIVE_DOCTOR = "native_doctor"
    DOCTOR_COMBINATION_ERROR = "doctor_combination_error"
    NATIVE_CHECK_CONFIG = "native_check_config"
    CHECK_CONFIG_COMBINATION_ERROR = "check_config_combination_error"
    NATIVE_LIST_PROFILES = "native_list_profiles"
    LIST_PROFILES_COMBINATION_ERROR = "list_profiles_combination_error"
    NATIVE_INSTITUTION_COMPLIANCE = "native_institution_compliance"
    NATIVE_DOI_MANIFEST = "native_doi_manifest"
    INSTITUTION_COMPLIANCE_ERROR = "institution_compliance_error"
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


def _namespace_has_preceding_trigger(
    argv: Sequence[str],
) -> bool:
    args = _build_parser().parse_args(list(argv))
    return any(
        bool(getattr(args, dest, None))
        for dest in DOCTOR_PRECEDING_TRIGGER_DESTS
    )


def _namespace_has_check_config_preceding_trigger(
    argv: Sequence[str],
) -> bool:
    args = _build_parser().parse_args(list(argv))
    return any(
        bool(getattr(args, dest, None))
        for dest in CHECK_CONFIG_PRECEDING_TRIGGER_DESTS
    )


def _is_exact_list_profiles_invocation(
    argv: Sequence[str],
) -> bool:
    return bool(argv) and all(
        str(token) in LIST_PROFILES_OPTIONS
        for token in argv
    )


def _is_exact_institution_compliance_invocation(
    argv: Sequence[str],
) -> bool:
    seen_command = False
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token in INSTITUTION_COMPLIANCE_OPTIONS:
            if seen_command:
                return False
            seen_command = True
            index += 1
            continue

        matched_value_option = False
        for option in INSTITUTION_COMPLIANCE_VALUE_OPTIONS:
            if token == option:
                if index + 1 >= len(argv):
                    return False
                index += 2
                matched_value_option = True
                break
            if token.startswith(option + "="):
                index += 1
                matched_value_option = True
                break
        if matched_value_option:
            continue
        return False
    return seen_command


def _is_exact_doi_manifest_invocation(
    argv: Sequence[str],
) -> bool:
    seen_command = False
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token in DOI_MANIFEST_OPTIONS:
            if seen_command:
                return False
            seen_command = True
            index += 1
            continue

        matched_value_option = False
        for option in DOI_MANIFEST_VALUE_OPTIONS:
            if token == option:
                if index + 1 >= len(argv):
                    return False
                index += 2
                matched_value_option = True
                break
            if token.startswith(option + "="):
                if token == option + "=":
                    return False
                index += 1
                matched_value_option = True
                break
        if matched_value_option:
            continue
        return False
    return seen_command


def select_runtime_route(argv: Sequence[str]) -> RuntimeRoute:
    # Precedência conservadora: informativos; estágio 015; doctor; check-config;
    # primeira onda operacional; fallback legado.
    for token in argv:
        if any(
            _matches_option(token, option)
            for option in FIRST_WAVE_OPTIONS | {"-h"}
        ):
            return RuntimeRoute.NATIVE_FIRST_WAVE

    institution_selected = any(
        _matches_option(token, option)
        for token in argv
        for option in INSTITUTION_COMPLIANCE_OPTIONS
    )
    if institution_selected:
        if _is_exact_institution_compliance_invocation(argv):
            return RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE
        return RuntimeRoute.INSTITUTION_COMPLIANCE_ERROR

    doi_manifest_selected = any(
        _matches_option(token, option)
        for token in argv
        for option in DOI_MANIFEST_OPTIONS
    )
    if doi_manifest_selected:
        return RuntimeRoute.NATIVE_DOI_MANIFEST

    doctor_selected = any(
        _matches_option(token, option)
        for token in argv
        for option in DOCTOR_OPTIONS
    )
    if doctor_selected:
        if _namespace_has_preceding_trigger(argv):
            return RuntimeRoute.DOCTOR_COMBINATION_ERROR
        return RuntimeRoute.NATIVE_DOCTOR

    check_config_selected = any(
        _matches_option(token, option)
        for token in argv
        for option in CHECK_CONFIG_OPTIONS
    )
    if check_config_selected:
        if _namespace_has_check_config_preceding_trigger(argv):
            return RuntimeRoute.CHECK_CONFIG_COMBINATION_ERROR
        return RuntimeRoute.NATIVE_CHECK_CONFIG

    list_profiles_selected = any(
        _matches_option(token, option)
        for token in argv
        for option in LIST_PROFILES_OPTIONS
    )
    if list_profiles_selected:
        if not _is_exact_list_profiles_invocation(argv):
            return RuntimeRoute.LIST_PROFILES_COMBINATION_ERROR
        return RuntimeRoute.NATIVE_LIST_PROFILES

    return RuntimeRoute.LEGACY_FALLBACK




def _build_parser() -> argparse.ArgumentParser:
    parser = cli_parser.build_parser(pipeline_version=PIPELINE_VERSION)
    if not isinstance(parser, argparse.ArgumentParser):
        raise NativeRuntimeError(
            "build_parser não retornou ArgumentParser"
        )

    parser.prog = OFFICIAL_PROGRAM_NAME
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Lista os perfis TOML disponíveis.",
    )
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


def _run_native_doctor(argv: Sequence[str]) -> int:
    from .doctor_runtime import run_doctor_command

    return int(run_doctor_command(argv))


def _run_native_check_config(
    argv: Sequence[str],
) -> int:
    from .check_config_runtime import (
        run_check_config_command,
    )

    return int(run_check_config_command(argv))


def _run_check_config_combination_error() -> int:
    import sys as _sys

    print(CHECK_CONFIG_COMBINATION_ERROR, file=_sys.stderr)
    return 1


def _run_list_profiles_combination_error() -> int:
    import sys as _sys

    print(LIST_PROFILES_COMBINATION_ERROR, file=_sys.stderr)
    return 1


def _run_native_list_profiles(
    argv: Sequence[str],
) -> int:
    from .list_profiles_runtime import (
        run_list_profiles_command,
    )

    return int(run_list_profiles_command(argv))


def _run_native_institution_compliance(
    argv: Sequence[str],
) -> int:
    import sys as _sys

    from .institution_compliance_runtime import (
        InstitutionComplianceRuntimeError,
        run_institution_compliance_command,
    )

    try:
        return int(run_institution_compliance_command(argv))
    except InstitutionComplianceRuntimeError as exc:
        print(str(exc), file=_sys.stderr)
        return 1


def _run_native_doi_manifest(
    argv: Sequence[str],
) -> int:
    import sys as _sys

    from .doi_manifest_runtime import (
        DoiManifestRuntimeError,
        run_make_doi_manifest_command,
    )

    try:
        return int(run_make_doi_manifest_command(argv))
    except DoiManifestRuntimeError as exc:
        print(str(exc), file=_sys.stderr)
        return 1


def _run_doctor_combination_error() -> int:
    import sys as _sys

    print(DOCTOR_COMBINATION_ERROR, file=_sys.stderr)
    return 1


def _run_institution_compliance_error() -> int:
    import sys as _sys

    print(INSTITUTION_COMPLIANCE_COMBINATION_ERROR, file=_sys.stderr)
    return 1


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

    if route is RuntimeRoute.NATIVE_DOCTOR:
        return _run_native_doctor(forwarded)

    if route is RuntimeRoute.DOCTOR_COMBINATION_ERROR:
        return _run_doctor_combination_error()

    if route is RuntimeRoute.NATIVE_CHECK_CONFIG:
        return _run_native_check_config(forwarded)

    if route is RuntimeRoute.CHECK_CONFIG_COMBINATION_ERROR:
        return _run_check_config_combination_error()

    if route is RuntimeRoute.NATIVE_LIST_PROFILES:
        return _run_native_list_profiles(forwarded)

    if route is RuntimeRoute.LIST_PROFILES_COMBINATION_ERROR:
        return _run_list_profiles_combination_error()

    if route is RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE:
        return _run_native_institution_compliance(forwarded)

    if route is RuntimeRoute.NATIVE_DOI_MANIFEST:
        return _run_native_doi_manifest(forwarded)

    if route is RuntimeRoute.INSTITUTION_COMPLIANCE_ERROR:
        return _run_institution_compliance_error()

    return int(legacy_runner(forwarded))


__all__ = [
    "CHECK_CONFIG_OPTIONS",
    "CHECK_CONFIG_PRECEDING_TRIGGER_DESTS",
    "CHECK_CONFIG_COMBINATION_ERROR",
    "DOCTOR_OPTIONS",
    "DOCTOR_PRECEDING_TRIGGER_DESTS",
    "DOCTOR_COMBINATION_ERROR",
    "FIRST_WAVE_OPTIONS",
    "LIST_PROFILES_OPTIONS",
    "LIST_PROFILES_COMBINATION_ERROR",
    "INSTITUTION_COMPLIANCE_OPTIONS",
    "INSTITUTION_COMPLIANCE_VALUE_OPTIONS",
    "DOI_MANIFEST_OPTIONS",
    "DOI_MANIFEST_VALUE_OPTIONS",
    "INSTITUTION_COMPLIANCE_COMBINATION_ERROR",
    "NATIVE_TRIGGER_OPTIONS",
    "NativeRuntimeError",
    "RuntimeContext",
    "RuntimeRoute",
    "default_runtime_context",
    "run",
    "select_runtime_route",
]

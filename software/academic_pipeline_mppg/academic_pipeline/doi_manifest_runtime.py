from __future__ import annotations
import argparse
import json
import sys
from collections.abc import Sequence
import csv
import zipfile
from pathlib import Path
from typing import Any
from app_bundle.scripts.pipeline.diagnostics import PIPELINE_VERSION
from . import cli_parser

SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".org", ".rst", ".tex"}

def _iter_zip_sources(input_zip: Path) -> list[str]:
    if not input_zip.exists():
        raise FileNotFoundError(f"ZIP não encontrado: {input_zip}")
    names: list[str] = []
    with zipfile.ZipFile(input_zip, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix in SUPPORTED_SOURCE_SUFFIXES:
                names.append(name)
    return sorted(dict.fromkeys(names))

def _iter_dir_sources(input_dir: Path, recursive: bool = True) -> list[str]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    out = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES:
            try:
                out.append(str(path.relative_to(input_dir)))
            except Exception:
                out.append(path.name)
    return sorted(dict.fromkeys(out))

def make_doi_manifest(input_zip: Path | None, input_dir: Path | None, output: Path, overwrite: bool = True) -> dict[str, Any]:
    if not input_zip and not input_dir:
        raise ValueError("Informe --input-zip ou --input-dir.")
    if input_zip:
        files = _iter_zip_sources(input_zip)
        source = str(input_zip)
    else:
        files = _iter_dir_sources(input_dir)  # type: ignore[arg-type]
        source = str(input_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Arquivo já existe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arquivo", "doi"])
        for name in files:
            writer.writerow([name, ""])
    return {"source": source, "output": str(output), "total_files": len(files), "files": files}

OFFICIAL_PROGRAM_NAME = "academic-pipeline"
DOI_MANIFEST_OPTION = "--make-doi-manifest"


class DoiManifestRuntimeError(RuntimeError):
    pass


def _normalize_argv(argv: Sequence[str] | None) -> tuple[str, ...]:
    source = sys.argv[1:] if argv is None else argv
    return tuple(str(item) for item in source)


def _build_parser() -> argparse.ArgumentParser:
    parser = cli_parser.build_parser(pipeline_version=PIPELINE_VERSION)
    if not isinstance(parser, argparse.ArgumentParser):
        raise DoiManifestRuntimeError("build_parser não retornou ArgumentParser")
    parser.prog = OFFICIAL_PROGRAM_NAME
    return parser


def _validate_option_surface(argv: Sequence[str]) -> None:
    value_options = ("--input-dir", "--input-zip", "--output")
    seen_command = False
    seen_value_options: set[str] = set()
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token == DOI_MANIFEST_OPTION:
            if seen_command:
                raise DoiManifestRuntimeError("--make-doi-manifest duplicado")
            seen_command = True
            index += 1
            continue

        matched = False
        for option in value_options:
            if token == option:
                if option in seen_value_options:
                    raise DoiManifestRuntimeError(f"{option} duplicado")
                if (
                    index + 1 >= len(argv)
                    or str(argv[index + 1]).startswith("--")
                ):
                    raise DoiManifestRuntimeError(
                        f"valor ausente para {option}"
                    )
                seen_value_options.add(option)
                index += 2
                matched = True
                break

            if token.startswith(option + "="):
                if option in seen_value_options:
                    raise DoiManifestRuntimeError(f"{option} duplicado")
                if token == option + "=":
                    raise DoiManifestRuntimeError(
                        f"valor ausente para {option}"
                    )
                seen_value_options.add(option)
                index += 1
                matched = True
                break

        if matched:
            continue
        raise DoiManifestRuntimeError(
            f"argumento não suportado: {token}"
        )


def _request_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path]:
    if not bool(args.make_doi_manifest):
        raise DoiManifestRuntimeError(
            "run_make_doi_manifest_command exige --make-doi-manifest"
        )
    has_zip = bool(args.input_zip)
    has_dir = bool(args.input_dir)
    if has_zip == has_dir:
        raise DoiManifestRuntimeError(
            "informe exatamente uma origem: --input-zip ou --input-dir"
        )
    if not args.output:
        raise DoiManifestRuntimeError("--output é obrigatório")
    input_zip = Path(args.input_zip).expanduser() if has_zip else None
    input_dir = Path(args.input_dir).expanduser() if has_dir else None
    output = Path(args.output).expanduser()
    return input_zip, input_dir, output


def run_make_doi_manifest_command(argv: Sequence[str] | None = None) -> int:
    forwarded = _normalize_argv(argv)
    _validate_option_surface(forwarded)
    args = _build_parser().parse_args(list(forwarded))
    input_zip, input_dir, output = _request_paths(args)
    result = make_doi_manifest(
        input_zip,
        input_dir,
        output,
        overwrite=True,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "DOI_MANIFEST_OPTION",
    "DoiManifestRuntimeError",
    "make_doi_manifest",
    "run_make_doi_manifest_command",
]

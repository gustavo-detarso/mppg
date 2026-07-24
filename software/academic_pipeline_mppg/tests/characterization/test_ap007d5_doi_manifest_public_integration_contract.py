from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

import academic_pipeline
from academic_pipeline import cli, doi_manifest_runtime, runtime

TEST_FILE = Path(__file__).resolve()
REPO = TEST_FILE.parents[4]
MANIFEST = REPO / "docs/refactor/academic-pipeline/AP-007/ap007d5_doi_manifest_public_integration.json"


def _rows(path: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    inputs = tmp_path / "inputs"
    (inputs / "nested").mkdir(parents=True)
    (inputs / "nested" / "alpha.pdf").write_bytes(b"%PDF-1.4\n% integration fixture\n")
    (inputs / "beta.docx").write_bytes(b"PK\x03\x04integration")
    (inputs / "ignored.bin").write_bytes(b"ignored")
    source_zip = tmp_path / "sources.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(inputs / "nested" / "alpha.pdf", "nested/alpha.pdf")
        archive.write(inputs / "beta.docx", "beta.docx")
        archive.write(inputs / "ignored.bin", "ignored.bin")
    return inputs, source_zip


def test_manifest_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "ap007d5-doi-manifest-public-integration/v1"
    assert data["status"] == "doi_manifest_publicly_integrated"
    assert data["command"] == "--make-doi-manifest"
    assert data["public_route"] == "native_doi_manifest"
    assert data["candidate_path_count"] == 44
    assert len(data["candidate_paths"]) == 44
    assert data["legacy_defect_bypassed"] == "missing_legacy_bootstrap_dependency:dotenv"


def test_exact_doi_requests_are_native() -> None:
    assert runtime.select_runtime_route(("--make-doi-manifest", "--input-dir", "in", "--output", "out.csv")) is runtime.RuntimeRoute.NATIVE_DOI_MANIFEST
    assert runtime.select_runtime_route(("--make-doi-manifest", "--input-zip=z.zip", "--output=out.csv")) is runtime.RuntimeRoute.NATIVE_DOI_MANIFEST
    assert runtime.select_runtime_route(("--make-doi-manifest",)) is runtime.RuntimeRoute.NATIVE_DOI_MANIFEST


def test_competing_commands_remain_legacy_and_help_precedes() -> None:
    assert runtime.select_runtime_route(("--help", "--make-doi-manifest")) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE
    for competing in ("--doctor", "--check-config", "--list-profiles", "--check-institution-compliance", "--tui"):
        assert runtime.select_runtime_route((competing, "--make-doi-manifest")) is runtime.RuntimeRoute.LEGACY_FALLBACK


def test_existing_native_routes_are_preserved() -> None:
    assert runtime.select_runtime_route(("--list-profiles",)) is runtime.RuntimeRoute.NATIVE_LIST_PROFILES
    assert runtime.select_runtime_route(("--config", "x.toml", "--check-institution-compliance")) is runtime.RuntimeRoute.NATIVE_INSTITUTION_COMPLIANCE


def test_public_runner_uses_adapter_without_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    def fake(argv: Any) -> int:
        captured.append(list(argv))
        return 0
    def forbidden(argv: Any) -> int:
        raise AssertionError(f"fallback indevido: {argv}")
    monkeypatch.setattr(doi_manifest_runtime, "run_make_doi_manifest_command", fake)
    monkeypatch.setattr(cli, "run_legacy", forbidden)
    argv = ["--make-doi-manifest", "--input-dir", "in", "--output", "out.csv"]
    assert academic_pipeline.main(argv) == 0
    assert captured == [argv]


def test_public_runner_maps_usage_error_to_one(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake(argv: Any) -> int:
        raise doi_manifest_runtime.DoiManifestRuntimeError("informe exatamente uma origem")
    monkeypatch.setattr(doi_manifest_runtime, "run_make_doi_manifest_command", fake)
    assert academic_pipeline.main(["--make-doi-manifest"]) == 1
    assert "exatamente uma origem" in capsys.readouterr().err


def test_real_public_directory_and_zip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    inputs, source_zip = _fixtures(tmp_path)
    dir_output = tmp_path / "dir.csv"
    zip_output = tmp_path / "zip.csv"
    assert academic_pipeline.main(["--make-doi-manifest", "--input-dir", str(inputs), "--output", str(dir_output)]) == 0
    json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert academic_pipeline.main(["--make-doi-manifest", "--input-zip", str(source_zip), "--output", str(zip_output)]) == 0
    json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    expected = [["arquivo", "doi"], ["beta.docx", ""], ["nested/alpha.pdf", ""]]
    assert _rows(dir_output) == expected
    assert _rows(zip_output) == expected


def test_explicit_argv_preserves_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    before_argv = list(sys.argv)
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    monkeypatch.setattr(doi_manifest_runtime, "run_make_doi_manifest_command", lambda argv: 0)
    assert academic_pipeline.main(["--make-doi-manifest", "--input-dir", "in", "--output", "out.csv"]) == 0
    assert sys.argv == before_argv
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


def test_hashes_match_manifest() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in data["artifact_sha256"].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected


def test_runtime_has_no_implicit_legacy_bridge() -> None:
    source = (REPO / "software/academic_pipeline_mppg/academic_pipeline/runtime.py").read_text(encoding="utf-8")
    for forbidden in ("globals(", "locals(", "sys.path", "importlib", "academic_pipeline_rc10", "LEGACY_MODULE_NAME"):
        assert forbidden not in source

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from academic_pipeline import doi_manifest_runtime, runtime

TEST_FILE = Path(__file__).resolve()
REPO_ROOT = TEST_FILE.parents[4]
MANIFEST = REPO_ROOT / "docs/refactor/academic-pipeline/AP-007/ap007d5_doi_manifest_native_adapter.json"
SOURCE = REPO_ROOT / "software/academic_pipeline_mppg/app_bundle/scripts/pipeline/project_tools.py"
ADAPTER = REPO_ROOT / "software/academic_pipeline_mppg/academic_pipeline/doi_manifest_runtime.py"


def _rows(path: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))


def _normalized(path: Path) -> bytes:
    target = io.StringIO(newline="")
    csv.writer(target, lineterminator="\n").writerows(_rows(path))
    return target.getvalue().encode("utf-8")


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    inputs = tmp_path / "inputs"
    (inputs / "nested").mkdir(parents=True)
    (inputs / "nested" / "alpha.pdf").write_bytes(b"%PDF-1.4\n% adapter fixture\n")
    (inputs / "beta.docx").write_bytes(b"PK\x03\x04adapter")
    (inputs / "ignored.bin").write_bytes(b"ignored")
    source_zip = tmp_path / "sources.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(inputs / "nested" / "alpha.pdf", "nested/alpha.pdf")
        archive.write(inputs / "beta.docx", "beta.docx")
        archive.write(inputs / "ignored.bin", "ignored.bin")
    return inputs, source_zip


def test_adapter_manifest_and_source_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = ADAPTER.read_text(encoding="utf-8")
    ast.parse(source)
    assert data["schema"] == "ap007d5-doi-manifest-native-adapter/v1"
    assert data["status"] == "materialized_route_still_legacy"
    assert data["command"] == "--make-doi-manifest"
    assert data["characterization_payload_sha256"] == "bb7cf63340657a4bf0c0f0d25b4bf9c239780ebdd107029069b5bd95fece48e4"
    assert hashlib.sha256(ADAPTER.read_bytes()).hexdigest() == data["adapter"]["sha256"]
    assert data["adapter"]["closure_strategy"] == "minimal_same_module_ast_closure"
    assert data["legacy_defect"] == "missing_legacy_bootstrap_dependency:dotenv"
    for forbidden in (
        "project_tools", "bibliography_manager", "academic_pipeline_rc10",
        "dotenv", "pydantic", "run_legacy", "importlib", "globals(", "locals(",
    ):
        assert forbidden not in source


def test_canonical_function_segment_is_preserved() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_text = SOURCE.read_text(encoding="utf-8")
    adapter_text = ADAPTER.read_text(encoding="utf-8")
    source_tree = ast.parse(source_text)
    adapter_tree = ast.parse(adapter_text)
    source_fn = next(node for node in source_tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_doi_manifest")
    adapter_fn = next(node for node in adapter_tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_doi_manifest")
    source_segment = ast.get_source_segment(source_text, source_fn) or ""
    adapter_segment = ast.get_source_segment(adapter_text, adapter_fn) or ""
    assert adapter_segment == source_segment
    assert hashlib.sha256(adapter_segment.encode()).hexdigest() == data["source"]["function_sha256"]


def test_adapter_directory_and_zip_are_equivalent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    inputs, source_zip = _fixtures(tmp_path)
    dir_output = tmp_path / "dir.csv"
    zip_output = tmp_path / "zip.csv"
    assert doi_manifest_runtime.run_make_doi_manifest_command([
        "--make-doi-manifest", "--input-dir", str(inputs), "--output", str(dir_output),
    ]) == 0
    dir_stdout = capsys.readouterr().out
    assert doi_manifest_runtime.run_make_doi_manifest_command([
        "--make-doi-manifest", "--input-zip", str(source_zip), "--output", str(zip_output),
    ]) == 0
    zip_stdout = capsys.readouterr().out
    json.loads(dir_stdout.strip().splitlines()[-1])
    json.loads(zip_stdout.strip().splitlines()[-1])
    expected = [["arquivo", "doi"], ["beta.docx", ""], ["nested/alpha.pdf", ""]]
    assert _rows(dir_output) == expected
    assert _rows(zip_output) == expected
    assert _normalized(dir_output) == _normalized(zip_output)


def test_adapter_overwrites_existing_output(tmp_path: Path) -> None:
    inputs, _ = _fixtures(tmp_path)
    output = tmp_path / "manifest.csv"
    output.write_text("stale\n", encoding="utf-8")
    assert doi_manifest_runtime.run_make_doi_manifest_command([
        "--make-doi-manifest", "--input-dir", str(inputs), "--output", str(output),
    ]) == 0
    assert _rows(output)[0] == ["arquivo", "doi"]


@pytest.mark.parametrize(
    "argv,message",
    [
        ([], "exige --make-doi-manifest"),
        (["--make-doi-manifest", "--output", "out.csv"], "exatamente uma origem"),
        (["--make-doi-manifest", "--input-dir", "a"], "--output é obrigatório"),
        (["--make-doi-manifest", "--input-dir", "a", "--output", "out.csv", "--doctor"], "argumento não suportado"),
        (["--make-doi-manifest", "--make-doi-manifest", "--input-dir", "a", "--output", "out.csv"], "duplicado"),
    ],
)
def test_adapter_rejects_invalid_requests(argv: list[str], message: str) -> None:
    with pytest.raises(doi_manifest_runtime.DoiManifestRuntimeError, match=message):
        doi_manifest_runtime.run_make_doi_manifest_command(argv)


def test_request_paths_rejects_two_origins() -> None:
    import argparse
    args = argparse.Namespace(
        make_doi_manifest=True,
        input_dir="a",
        input_zip="b",
        output="out.csv",
    )
    with pytest.raises(doi_manifest_runtime.DoiManifestRuntimeError, match="exatamente uma origem"):
        doi_manifest_runtime._request_paths(args)


def test_explicit_argv_preserves_process_state(tmp_path: Path) -> None:
    inputs, _ = _fixtures(tmp_path)
    before_argv = list(sys.argv)
    before_path = list(sys.path)
    before_cwd = os.getcwd()
    assert doi_manifest_runtime.run_make_doi_manifest_command([
        "--make-doi-manifest", "--input-dir", str(inputs), "--output", str(tmp_path / "out.csv"),
    ]) == 0
    assert sys.argv == before_argv
    assert sys.path == before_path
    assert os.getcwd() == before_cwd


def test_adapter_surface_has_no_network_subprocess_or_dynamic_calls() -> None:
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                calls.append(ast.unparse(node.func))
            except Exception:
                pass
    assert not [name for name in calls if name.startswith(("requests.", "urllib.", "httpx.", "aiohttp.", "socket.", "subprocess."))]
    assert not [name for name in calls if name in {"eval", "exec", "compile", "__import__"}]


def test_route_remains_legacy_in_adapter_phase() -> None:
    route = runtime.select_runtime_route((
        "--make-doi-manifest", "--input-dir", "/tmp/input", "--output", "/tmp/output.csv",
    ))
    assert route is runtime.RuntimeRoute.LEGACY_FALLBACK

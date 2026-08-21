from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
import tokenize
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_RC10 = SOURCE_ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
LEGACY_MODULE = SOURCE_ROOT / "academic_pipeline/legacy.py"
RUNTIME = SOURCE_ROOT / "academic_pipeline/runtime.py"
PYPROJECT = SOURCE_ROOT / "pyproject.toml"
CANONICAL_MAIN = SOURCE_ROOT / "academic_pipeline/__main__.py"

AUXILIARIES = (
    SOURCE_ROOT / "atualizar_academic_pipeline_bundle.py",
    SOURCE_ROOT / "install_rc10.sh",
    SOURCE_ROOT / "setup_pipenv_env.sh",
)
PHYSICAL_RC10 = "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"


def test_ap008f5b_physical_rc10_and_legacy_module_are_absent() -> None:
    assert not HISTORICAL_RC10.exists()
    assert not LEGACY_MODULE.exists()
    assert CANONICAL_MAIN.is_file()


def test_ap008f5b_auxiliaries_have_no_physical_rc10_reference() -> None:
    for path in AUXILIARIES:
        assert path.is_file(), path
        assert PHYSICAL_RC10 not in path.read_text(encoding="utf-8"), path


def test_ap008f5b_runtime_has_no_legacy_seam() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNTIME))
    runs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    assert len(runs) == 1
    kwonly = {argument.arg for argument in runs[0].args.kwonlyargs}
    assert "legacy_runner" not in kwonly
    assert "LegacyRunner" not in source
    assert "run_legacy" not in source


def test_ap008f5b_packaging_metadata_exposes_only_canonical_console_entrypoint() -> None:
    source = PYPROJECT.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("academic-pipeline") == "academic_pipeline.cli:main"
    assert "academic_pipeline_rc10.py" not in source


def test_ap008f5b_canonical_module_help_remains_operational() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENAI_API_KEY"] = ""
    completed = subprocess.run(
        [sys.executable, "-m", "academic_pipeline", "--help"],
        cwd=SOURCE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()

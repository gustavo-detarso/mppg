from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
OLD_ROOT = REPO / "software/academic_pipeline_rc10_7_conformidade"
NEW_ROOT = REPO / "software/academic_pipeline_mppg"
RESOLVER_FILE = NEW_ROOT / "academic_pipeline/repository_paths.py"
VALIDATOR_FILE = REPO / "tools/refactor/ap006c_validate_physical_materialization.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ap006c_physical_topology_contract() -> None:
    _load("ap006c_validator", VALIDATOR_FILE).validate(REPO)


def test_ap006c_resolver_uses_structural_markers() -> None:
    resolver = _load("ap006c_resolver_structural", RESOLVER_FILE)
    assert resolver.repository_project_root(NEW_ROOT) == NEW_ROOT.resolve()
    assert resolver.repository_project_root(OLD_ROOT) == NEW_ROOT.resolve()


def test_ap006c_resolver_honors_valid_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _load("ap006c_resolver_override", RESOLVER_FILE)
    monkeypatch.setenv(resolver.PROJECT_ROOT_ENV, str(NEW_ROOT))
    assert resolver.repository_project_root() == NEW_ROOT.resolve()


def test_ap006c_resolver_rejects_invalid_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolver = _load("ap006c_resolver_invalid", RESOLVER_FILE)
    monkeypatch.setenv(resolver.PROJECT_ROOT_ENV, str(tmp_path))
    with pytest.raises(resolver.RepositoryRootError):
        resolver.repository_project_root()


def test_ap006c_repository_resource_prevents_escape() -> None:
    resolver = _load("ap006c_resolver_escape", RESOLVER_FILE)
    with pytest.raises(resolver.RepositoryRootError):
        resolver.repository_resource(
            "..",
            "..",
            "outside",
            start=NEW_ROOT,
            must_exist=False,
        )


def test_ap006c_public_packaging_surfaces_remain_declared() -> None:
    text = (NEW_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "academic-pipeline-mppg"' in text
    assert 'academic-pipeline = "academic_pipeline.cli:main"' in text
    assert (NEW_ROOT / "academic_pipeline/__init__.py").is_file()
    assert (NEW_ROOT / "app_bundle/__init__.py").is_file()

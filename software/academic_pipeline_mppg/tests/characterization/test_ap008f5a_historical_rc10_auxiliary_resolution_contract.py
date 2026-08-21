from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
UPDATER = SOURCE_ROOT / "atualizar_academic_pipeline_bundle.py"
INSTALL = SOURCE_ROOT / "install_rc10.sh"
SETUP = SOURCE_ROOT / "setup_pipenv_env.sh"
HISTORICAL_RC10 = SOURCE_ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PHYSICAL_RC10 = "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ap008f5a_auxiliary_scope_is_exact() -> None:
    targets = {UPDATER, INSTALL, SETUP}
    assert all(path.is_file() for path in targets)
    assert len(targets) == 3


def test_ap008f5a_no_auxiliary_references_physical_rc10_entrypoint() -> None:
    for path in (UPDATER, INSTALL, SETUP):
        assert PHYSICAL_RC10 not in _read(path), path


def test_ap008f5a_bundle_updater_copies_and_validates_canonical_package() -> None:
    source = _read(UPDATER)
    assert '"academic_pipeline/",' in source
    assert '"pyproject.toml",' in source
    assert 'p.parent / "academic_pipeline" / "__main__.py"' in source
    assert 'dst_root / "academic_pipeline" / "__main__.py"' in source
    assert 'dst_root / "pyproject.toml"' in source


def test_ap008f5a_bundle_updater_executes_canonical_module() -> None:
    source = _read(UPDATER)
    tree = ast.parse(source, filename=str(UPDATER))
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    assert "canonical_entrypoint" in assigned_names
    assert "main_script" not in assigned_names
    assert '["pipenv", "run", "python", "-m", "academic_pipeline"]' in source


def test_ap008f5a_shell_helpers_recommend_canonical_module() -> None:
    install_source = _read(INSTALL)
    setup_source = _read(SETUP)
    assert install_source.count("pipenv run python -m academic_pipeline") == 5
    assert "pipenv run python -m academic_pipeline --doctor" in setup_source
    assert PHYSICAL_RC10 not in install_source
    assert PHYSICAL_RC10 not in setup_source


def test_ap008f5a_historical_rc10_is_absent_after_ap008f5b() -> None:
    assert not HISTORICAL_RC10.exists()

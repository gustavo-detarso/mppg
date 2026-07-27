from __future__ import annotations

import ast
import tomllib
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = SOURCE_ROOT / "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py"
ORCHESTRATOR = SOURCE_ROOT / "app_bundle/scripts/pipeline/pipeline_orchestrator.py"
RUNTIME = SOURCE_ROOT / "academic_pipeline/runtime.py"
LEGACY_MODULE = SOURCE_ROOT / "academic_pipeline/legacy.py"
HISTORICAL_RC10 = SOURCE_ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
PYPROJECT = SOURCE_ROOT / "pyproject.toml"


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    result: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            result.add(id(first.value))
    return result


def _joined_template(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            parts.append("{...}")
    return "".join(parts)


def _semantic_string_templates(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    docstrings = _docstring_constant_ids(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    templates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            templates.append((node.lineno, _joined_template(node)))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and not isinstance(parents.get(node), ast.JoinedStr)
        ):
            templates.append((node.lineno, node.value))
    return templates


def test_ap008f2b_toml_generator_uses_canonical_module_entrypoint() -> None:
    templates = _semantic_string_templates(GENERATOR)
    legacy = [(line, value) for line, value in templates if "academic_pipeline_rc10" in value]
    assert legacy == [], f"Referências semânticas RC10 ainda ativas no gerador: {legacy!r}"

    canonical = [value for _, value in templates if "python -m academic_pipeline" in value]
    assert canonical, "O gerador ainda não produz o entrypoint canônico python -m academic_pipeline"
    assert any("--somente-renderizar" in value and "--document-json" in value for value in canonical)
    assert any("pipenv run python -m academic_pipeline" in value for value in canonical)
    assert any("--config" in value for value in canonical)


def test_ap008f2b_pipeline_orchestrator_does_not_resolve_historical_rc10() -> None:
    templates = _semantic_string_templates(ORCHESTRATOR)
    legacy = [(line, value) for line, value in templates if "academic_pipeline_rc10" in value]
    assert legacy == [], f"O orquestrador ainda resolve o script histórico RC10: {legacy!r}"


def test_ap008f2b_legacy_components_are_absent_after_physical_retirement() -> None:
    assert not LEGACY_MODULE.exists(), "legacy.py deve permanecer ausente após a AP-008F.4"
    assert not HISTORICAL_RC10.exists(), "academic_pipeline_rc10.py deve ser retirado na AP-008F.5B"

    runtime_source = RUNTIME.read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source, filename=str(RUNTIME))
    run_functions = [
        node for node in ast.walk(runtime_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    assert len(run_functions) == 1
    kwonly = {argument.arg for argument in run_functions[0].args.kwonlyargs}
    assert "legacy_runner" not in kwonly
    assert "LegacyRunner" not in runtime_source
    assert "run_legacy" not in runtime_source



def test_ap008f2b_public_entrypoint_remains_canonical() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("academic-pipeline") == "academic_pipeline.cli:main"
    assert (SOURCE_ROOT / "academic_pipeline/__main__.py").is_file()

from __future__ import annotations

import ast
import os
from pathlib import Path

ROOT = Path(os.environ["AP009D1_CANDIDATE_ROOT"])
PIPELINE_DIR = ROOT / "software/academic_pipeline/app_bundle/scripts/pipeline"
CONSUMER = PIPELINE_DIR / "recompilar_paper.py"
RC9 = PIPELINE_DIR / "academic_pipeline.py"
CANONICAL = PIPELINE_DIR / "academic_pipeline.py"

def semantic_ast(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return ast.dump(
        ast.parse(text, filename=str(path)),
        annotate_fields=True,
        include_attributes=False,
    )

def signature_dump(path: Path, symbol: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                return ast.dump(
                    node.args,
                    annotate_fields=True,
                    include_attributes=False,
                )
    raise AssertionError(f"symbol not found: {symbol}")

def assert_no_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    offenders = [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if line.endswith((" ", "\t"))
    ]
    assert offenders == []

def main() -> None:
    assert CANONICAL.exists()
    assert semantic_ast(CANONICAL) == semantic_ast(RC9)
    assert signature_dump(
        CANONICAL, "run_compile_sequence"
    ) == signature_dump(
        RC9, "run_compile_sequence"
    )
    assert_no_trailing_whitespace(CANONICAL)

    consumer_text = CONSUMER.read_text(encoding="utf-8")
    assert "academic_pipeline_rc7.py" not in consumer_text
    assert consumer_text.count("academic_pipeline.py") == 2
    tree = ast.parse(consumer_text, filename=str(CONSUMER))
    script_arguments = 0
    run_calls = 0
    canonical_defaults = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_argument":
                    literals = [
                        argument.value
                        for argument in node.args
                        if isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)
                    ]
                    if "--script" in literals:
                        script_arguments += 1
                if node.func.attr == "run_compile_sequence":
                    run_calls += 1
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if "academic_pipeline.py" in ast.unparse(node):
                canonical_defaults += 1

    assert script_arguments == 1
    assert run_calls == 1
    assert canonical_defaults == 1

if __name__ == "__main__":
    main()

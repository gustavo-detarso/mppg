#!/usr/bin/env python3
from pathlib import Path
import ast, json

ROOT=Path(__file__).resolve().parents[1]
py=ROOT/"mppg_orchestrator.py"
src=py.read_text(encoding="utf-8")
tree=ast.parse(src)

schema=json.loads((ROOT/"schemas/ai_resolution.schema.json").read_text())
assert schema["additionalProperties"] is False
assert set(schema["required"])==set(schema["properties"])

# No os.system / subprocess shell=True / eval / exec calls.
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id=="os" and node.func.attr=="system":
                raise AssertionError("os.system forbidden")
        for kw in node.keywords:
            if kw.arg=="shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                raise AssertionError("subprocess shell=True forbidden")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval","exec"}:
        raise AssertionError("eval/exec forbidden")

launcher=(ROOT/"launcher.sh").read_text()
assert "set -euo pipefail" not in launcher
assert "source " not in launcher
assert "exec python3 -S -u" in launcher

print("PERMANENT_ORCHESTRATOR_STATIC_TEST=PASS")
print("NO_UNRESTRICTED_SHELL_CANARY=PASS")
print("PARENT_SHELL_SAFETY_CANARY=PASS")
